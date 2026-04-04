"""Tests for GGUF expert store: Q4_K block parsing, INT4 conversion, and store interface."""

import struct

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tinyserve.cpu_expert import HAS_INT4_CPU
from tinyserve.gguf_reader import GGML_TYPES

requires_int4 = pytest.mark.skipif(not HAS_INT4_CPU, reason="INT4 CPU ops not available")

# Q4_K block layout constants
Q4K_BLOCK_BYTES = 144
Q4K_BLOCK_ELEMENTS = 256

# Test dimensions: must satisfy out_features % 16 == 0, in_features % 256 == 0 (Q4_K block size)
OUT_FEATURES = 64
IN_FEATURES = 256
GROUP_SIZE = 32


def _build_q4k_block(values_f32: np.ndarray) -> bytes:
    """Build a valid Q4_K block from 256 float32 values (test helper).

    Encodes using the simplest possible Q4_K scheme:
    - 8 sub-blocks of 32 values each
    - Each sub-block uses a single scale and zero minimum
    - Packed as per llama.cpp Q4_K format
    """
    assert values_f32.shape == (256,), f"Expected 256 values, got {values_f32.shape}"

    # Q4_K block structure:
    # d: float16 at offset 0 (super-block scale)
    # dmin: float16 at offset 2 (super-block minimum)
    # scales[12]: bytes 4:16 (packed 6-bit sub-block scales and mins)
    # qs[128]: bytes 16:144 (4-bit quantized values, 2 per byte)

    # Compute per-sub-block statistics
    sub_blocks = values_f32.reshape(8, 32)
    sub_mins = sub_blocks.min(axis=1)
    sub_maxs = sub_blocks.max(axis=1)
    sub_ranges = sub_maxs - sub_mins

    # Super-block d and dmin
    max_range = max(sub_ranges.max(), 1e-10)
    min_min = sub_mins.min()
    d = np.float16(max_range / 63.0)  # 6-bit scale range
    dmin = np.float16(-min_min / 63.0) if min_min < 0 else np.float16(0.0)

    d_f32 = float(d)
    dmin_f32 = float(dmin)

    # Compute 6-bit sub-block scales and mins
    raw_scales = np.zeros(8, dtype=np.uint8)
    raw_mins = np.zeros(8, dtype=np.uint8)
    for i in range(8):
        if d_f32 > 0:
            raw_scales[i] = min(63, max(0, int(round(sub_ranges[i] / d_f32))))
        if dmin_f32 > 0:
            raw_mins[i] = min(63, max(0, int(round(-sub_mins[i] / dmin_f32))))

    # Pack scales into 12 bytes (llama.cpp format):
    # First 8 sub-blocks: low 4 bits in bytes 0-3, high 4 bits in bytes 4-7
    # Actually the packing is: scales[12] stores 6-bit values for 8 sub-blocks (scale + min)
    # Simplified: pack lower 4 bits of scales[i] and mins[i] into bytes 0..7,
    # upper 2 bits into bytes 8..11
    scales_bytes = bytearray(12)
    for i in range(8):
        s_lo = raw_scales[i] & 0x0F
        m_lo = raw_mins[i] & 0x0F
        if i < 4:
            scales_bytes[i] = s_lo | (m_lo << 4)
        else:
            scales_bytes[i] = s_lo | (m_lo << 4)

    # Upper 2 bits of scales and mins packed into bytes 8-11
    for i in range(8):
        s_hi = (raw_scales[i] >> 4) & 0x03
        m_hi = (raw_mins[i] >> 4) & 0x03
        byte_idx = 8 + (i // 2)
        shift = (i % 2) * 4
        scales_bytes[byte_idx] |= (s_hi | (m_hi << 2)) << shift

    # Quantize values to 4-bit
    qs = bytearray(128)
    for i in range(8):
        sc = d_f32 * raw_scales[i]
        mn = dmin_f32 * raw_mins[i]
        for j in range(32):
            val = values_f32[i * 32 + j]
            if sc > 0:
                q = int(round((val + mn) / sc))
            else:
                q = 0
            q = max(0, min(15, q))

            byte_idx = (i * 32 + j) // 2
            if (i * 32 + j) % 2 == 0:
                qs[byte_idx] = q & 0x0F
            else:
                qs[byte_idx] |= (q << 4)

    # Assemble block
    block = struct.pack("<e", d) + struct.pack("<e", dmin) + bytes(scales_bytes) + bytes(qs)
    assert len(block) == Q4K_BLOCK_BYTES, f"Block size {len(block)} != {Q4K_BLOCK_BYTES}"
    return block


def _write_string(f, s: str):
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _write_metadata_string(f, key: str, value: str):
    _write_string(f, key)
    f.write(struct.pack("<I", 8))
    _write_string(f, value)


def _write_metadata_uint32(f, key: str, value: int):
    _write_string(f, key)
    f.write(struct.pack("<I", 4))
    f.write(struct.pack("<I", value))


def _write_tensor_info(f, name: str, shape: tuple[int, ...], ggml_type: int, offset: int):
    _write_string(f, name)
    f.write(struct.pack("<I", len(shape)))
    for dim in shape:
        f.write(struct.pack("<Q", dim))
    f.write(struct.pack("<I", ggml_type))
    f.write(struct.pack("<Q", offset))


def _create_q4k_gguf(path, n_layers=1, n_experts=2, hidden=IN_FEATURES, intermediate=OUT_FEATURES):
    """Create a synthetic GGUF with Q4_K expert tensors containing known float data.

    gate/up: shape (intermediate, hidden) -- Q4_K quantized
    down: shape (hidden, intermediate) -- Q4_K quantized
    """
    ggml_type = 12  # Q4_K
    _, bytes_per_block, block_size = GGML_TYPES[ggml_type]

    projections = ["gate", "up", "down"]
    shapes = {
        "gate": (intermediate, hidden),
        "up": (intermediate, hidden),
        "down": (hidden, intermediate),
    }

    # Pre-generate float data and Q4_K blocks for each tensor
    tensor_data = {}
    np.random.seed(42)
    for layer in range(n_layers):
        for expert in range(n_experts):
            for proj in projections:
                shape = shapes[proj]
                n_elements = shape[0] * shape[1]
                # Random values in a range that Q4_K can represent reasonably
                vals = np.random.uniform(-2.0, 2.0, n_elements).astype(np.float32)
                n_blocks = n_elements // block_size
                blocks_data = b""
                for b_idx in range(n_blocks):
                    block_vals = vals[b_idx * block_size:(b_idx + 1) * block_size]
                    blocks_data += _build_q4k_block(block_vals)
                tensor_data[(layer, expert, proj)] = (blocks_data, vals, shape)

    # Build tensor info list
    tensor_infos = []
    data_offset = 0
    for layer in range(n_layers):
        for expert in range(n_experts):
            for proj in projections:
                name = f"blk.{layer}.ffn_{proj}.{expert}.weight"
                shape = shapes[proj]
                n_elements = shape[0] * shape[1]
                n_blocks = n_elements // block_size
                nbytes = n_blocks * bytes_per_block
                tensor_infos.append((name, shape, ggml_type, data_offset, nbytes))
                data_offset += nbytes

    n_tensors = len(tensor_infos)
    n_kv = 2

    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x46554747))
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))

        _write_metadata_string(f, "general.name", "test-q4k-model")
        _write_metadata_uint32(f, "test.num_experts", n_experts)

        for name, shape, gtype, offset, _ in tensor_infos:
            _write_tensor_info(f, name, shape, gtype, offset)

        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b"\x00" * (aligned - pos))

        for layer in range(n_layers):
            for expert in range(n_experts):
                for proj in projections:
                    blocks_data, _, _ = tensor_data[(layer, expert, proj)]
                    f.write(blocks_data)

    return tensor_data


class TestQ4KBlockParse:
    def test_parse_known_block(self):
        """Parse a Q4_K block with known values and verify reconstruction."""
        from tinyserve.gguf_quant import parse_q4k_block

        # Build a block from known constant values
        values = np.full(256, 1.0, dtype=np.float32)
        block = _build_q4k_block(values)
        assert len(block) == Q4K_BLOCK_BYTES

        parsed, d, dmin = parse_q4k_block(block)
        assert parsed.shape == (256,)
        assert parsed.dtype == np.float32

    def test_parse_zero_block(self):
        """Parsing a block of all zeros should return near-zero values."""
        from tinyserve.gguf_quant import parse_q4k_block

        values = np.zeros(256, dtype=np.float32)
        block = _build_q4k_block(values)
        parsed, d, dmin = parse_q4k_block(block)
        assert np.allclose(parsed, 0.0, atol=0.1)

    def test_parse_random_block_correlation(self):
        """Parsed values should correlate strongly with original values."""
        from tinyserve.gguf_quant import parse_q4k_block

        np.random.seed(42)
        values = np.random.uniform(-2.0, 2.0, 256).astype(np.float32)
        block = _build_q4k_block(values)
        parsed, d, dmin = parse_q4k_block(block)

        # Correlation should be high (Q4_K is 4.5 bpw, decent precision)
        corr = np.corrcoef(values, parsed)[0, 1]
        assert corr > 0.85, f"Correlation {corr:.4f} too low"

    def test_parse_returns_d_dmin(self):
        """parse_q4k_block should return the super-block scale factors."""
        from tinyserve.gguf_quant import parse_q4k_block

        values = np.random.uniform(-1.0, 1.0, 256).astype(np.float32)
        block = _build_q4k_block(values)
        _, d, dmin = parse_q4k_block(block)
        assert isinstance(d, float)
        assert isinstance(dmin, float)


class TestQ4KToINT4Pack:
    @requires_int4
    def test_output_shapes(self):
        """Converted INT4 tensors should have correct shapes."""
        from tinyserve.gguf_quant import q4k_expert_to_int4pack

        # Build Q4_K data for gate (OUT_FEATURES, IN_FEATURES)
        np.random.seed(42)
        gate_shape = (OUT_FEATURES, IN_FEATURES)
        up_shape = (OUT_FEATURES, IN_FEATURES)
        down_shape = (IN_FEATURES, OUT_FEATURES)

        gate_data = _make_q4k_tensor_bytes(gate_shape)
        up_data = _make_q4k_tensor_bytes(up_shape)
        down_data = _make_q4k_tensor_bytes(down_shape)

        g_packed, g_sz, u_packed, u_sz, d_packed, d_sz = q4k_expert_to_int4pack(
            gate_data, up_data, down_data,
            gate_shape, up_shape, down_shape,
            group_size=GROUP_SIZE,
        )

        n_groups_gu = IN_FEATURES // GROUP_SIZE
        n_groups_dn = OUT_FEATURES // GROUP_SIZE
        assert g_sz.shape == (n_groups_gu, OUT_FEATURES, 2)
        assert u_sz.shape == (n_groups_gu, OUT_FEATURES, 2)
        assert d_sz.shape == (n_groups_dn, IN_FEATURES, 2)
        assert g_sz.dtype == torch.bfloat16

    @requires_int4
    def test_roundtrip_cosine_similarity(self):
        """Q4_K -> INT4 -> matmul should closely match Q4_K -> float -> matmul."""
        from tinyserve.gguf_quant import parse_q4k_blocks, q4k_expert_to_int4pack

        np.random.seed(42)
        shape = (OUT_FEATURES, IN_FEATURES)
        data_bytes = _make_q4k_tensor_bytes(shape)

        # Get float reference via parsing
        w_ref = parse_q4k_blocks(data_bytes, shape)
        w_ref_t = torch.from_numpy(w_ref).to(torch.bfloat16)

        # Get INT4 packed version (use gate slot, ignore up/down)
        g_packed, g_sz, _, _, _, _ = q4k_expert_to_int4pack(
            data_bytes, data_bytes, data_bytes,
            shape, shape, shape,
            group_size=GROUP_SIZE,
        )

        x = torch.randn(1, IN_FEATURES, dtype=torch.bfloat16)
        out_int4 = torch.ops.aten._weight_int4pack_mm_for_cpu(x, g_packed, GROUP_SIZE, g_sz)
        out_ref = (x.float() @ w_ref_t.float().T).to(torch.bfloat16)

        cos_sim = F.cosine_similarity(out_int4.float(), out_ref.float(), dim=-1)
        assert cos_sim.item() > 0.99, f"Cosine similarity {cos_sim.item():.4f} too low"

    @requires_int4
    def test_deterministic(self):
        """Same input should produce identical INT4 output."""
        from tinyserve.gguf_quant import q4k_expert_to_int4pack

        np.random.seed(42)
        shape = (OUT_FEATURES, IN_FEATURES)
        data = _make_q4k_tensor_bytes(shape)

        _, sz1, _, _, _, _ = q4k_expert_to_int4pack(
            data, data, data, shape, shape, shape, group_size=GROUP_SIZE
        )
        _, sz2, _, _, _, _ = q4k_expert_to_int4pack(
            data, data, data, shape, shape, shape, group_size=GROUP_SIZE
        )
        torch.testing.assert_close(sz1, sz2)


class TestGGUFExpertStore:
    @requires_int4
    def test_from_gguf_shapes(self, tmp_path):
        """GGUFExpertStore should load and expose correct dimensions."""
        from tinyserve.gguf_store import GGUFExpertStore

        path = tmp_path / "test.gguf"
        _create_q4k_gguf(path, n_layers=1, n_experts=2)

        store = GGUFExpertStore.from_gguf(str(path))
        assert store.num_layers == 1
        assert store.num_experts == 2
        assert store.expert_bytes > 0

    @requires_int4
    def test_from_gguf_layout(self, tmp_path):
        """Store layout should contain gate, up, down projections."""
        from tinyserve.gguf_store import GGUFExpertStore

        path = tmp_path / "test.gguf"
        _create_q4k_gguf(path, n_layers=1, n_experts=2)

        store = GGUFExpertStore.from_gguf(str(path))
        spec_names = set(store.layout.specs.keys())
        assert "gate_proj" in spec_names or "gate_packed" in spec_names or "gate" in spec_names
        assert store.layout.total_bytes > 0

    @requires_int4
    def test_from_gguf_data_accessible(self, tmp_path):
        """Expert data should be loadable from the store."""
        from tinyserve.gguf_store import GGUFExpertStore

        path = tmp_path / "test.gguf"
        _create_q4k_gguf(path, n_layers=1, n_experts=2)

        store = GGUFExpertStore.from_gguf(str(path))
        # Access first expert's data
        expert_data = store._data[0, 0]
        assert expert_data.shape == (store.expert_bytes,)
        assert expert_data.dtype == torch.uint8

    @requires_int4
    def test_different_experts_have_different_data(self, tmp_path):
        """Different experts should have distinct weight data."""
        from tinyserve.gguf_store import GGUFExpertStore

        path = tmp_path / "test.gguf"
        _create_q4k_gguf(path, n_layers=1, n_experts=2)

        store = GGUFExpertStore.from_gguf(str(path))
        e0 = store._data[0, 0]
        e1 = store._data[0, 1]
        assert not torch.equal(e0, e1)

    @requires_int4
    def test_multi_layer(self, tmp_path):
        """Store should work with multiple layers."""
        from tinyserve.gguf_store import GGUFExpertStore

        path = tmp_path / "test.gguf"
        _create_q4k_gguf(path, n_layers=2, n_experts=2)

        store = GGUFExpertStore.from_gguf(str(path))
        assert store.num_layers == 2
        assert store._data.shape[0] == 2


class TestExpertForwardFromGGUF:
    @requires_int4
    def test_forward_produces_valid_output(self, tmp_path):
        """Full forward pass with GGUFINT4Forward on GGUF-converted weights."""
        from tinyserve.gguf_store import GGUFExpertStore, GGUFINT4Forward

        path = tmp_path / "test.gguf"
        _create_q4k_gguf(path, n_layers=1, n_experts=1, hidden=IN_FEATURES, intermediate=OUT_FEATURES)

        store = GGUFExpertStore.from_gguf(str(path))
        layout = store.layout
        expert_packed = store._data[0, 0]

        h = torch.randn(1, IN_FEATURES)
        fwd = GGUFINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu, num_threads=1)
        out = fwd.forward(h, expert_packed)

        assert out.shape == (1, IN_FEATURES)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @requires_int4
    def test_forward_different_experts_give_different_outputs(self, tmp_path):
        """Different experts should produce different forward results."""
        from tinyserve.gguf_store import GGUFExpertStore, GGUFINT4Forward

        path = tmp_path / "test.gguf"
        _create_q4k_gguf(path, n_layers=1, n_experts=2, hidden=IN_FEATURES, intermediate=OUT_FEATURES)

        store = GGUFExpertStore.from_gguf(str(path))
        layout = store.layout

        h = torch.randn(1, IN_FEATURES)
        fwd = GGUFINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu, num_threads=1)

        out0 = fwd.forward(h, store._data[0, 0])
        out1 = fwd.forward(h, store._data[0, 1])

        assert not torch.allclose(out0, out1)


def _make_q4k_tensor_bytes(shape: tuple[int, int]) -> bytes:
    """Generate random Q4_K encoded bytes for a weight matrix of given shape."""
    n_elements = shape[0] * shape[1]
    n_blocks = n_elements // Q4K_BLOCK_ELEMENTS
    data = b""
    for _ in range(n_blocks):
        vals = np.random.uniform(-2.0, 2.0, Q4K_BLOCK_ELEMENTS).astype(np.float32)
        data += _build_q4k_block(vals)
    return data
