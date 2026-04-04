# ggml Kernel Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute directly on native GGUF quantized weights via ggml CUDA kernels — zero dequant, mmap storage, 3.6x GPU cache capacity. Unblocks Qwen 122B.

**Architecture:** Two parallel forward paths in ExpertPipeline, selected at construction time. The existing BF16/MXFP4 path is completely untouched. The new native-quant path (MmapExpertStore + ggml kernels) handles hits, misses, and batched dispatch independently. The ExpertCache is already format-agnostic (uint8 slots + LRU) — it doesn't care what bytes are in its slots.

**Tech Stack:** ggml (MIT, git submodule), PyTorch C++ extension, CMake, numpy mmap, city96 dequant (Apache-2.0 fallback)

**Spec:** `docs/superpowers/specs/2026-04-04-ggml-kernel-integration-design.md`

---

## Key Architectural Decision

The existing pipeline has 6+ callsites that assume cache slots and staging buffers contain BF16-interpretable data (`_inline_fwd`, `forward_from_packed`, `swap_weights_and_forward`, C++ expert loop, Cython forward, batched path). Rather than modifying all of them, we add a **parallel native-quant path** that is self-contained:

```python
class ExpertPipeline:
    def __init__(self, store, ...):
        self._native_quant = isinstance(store, MmapExpertStore)
        if self._native_quant:
            self._nq_forward = GGMLExpertForward(store.layout, store.ggml_types, ...)
        # ... existing BF16/MXFP4 setup unchanged ...

    def execute_layer_experts(self, hidden_states, layer_idx, expert_indices, routing_weights):
        if self._native_quant:
            return self._execute_layer_experts_native(...)
        # ... existing path unchanged ...

    def _execute_layer_experts_native(self, ...):
        # Complete native-quant implementation:
        # - cache lookup (same ExpertCache)
        # - cache hit: self._nq_forward.forward(packed, h)
        # - cache miss: store.copy_to_buffer → cache.store → forward
        # - batched: group by expert, forward each
```

**The existing BF16/MXFP4 path has zero changes.** The `_native_quant` flag routes everything at the top level.

---

## Task 1: city96 pure-PyTorch dequant fallback

**Files:**
- Create: `tinyserve/gguf_dequant_torch.py`
- Test: `tests/test_gguf_dequant_torch.py`

Port from [city96/ComfyUI-GGUF dequant.py](https://github.com/city96/ComfyUI-GGUF/blob/main/dequant.py) (Apache-2.0). Vectorized dequant for all K-quant types using torch ops (no Python loops over blocks). This is the fallback when ggml CUDA is unavailable AND the forward path for batch>1 prefill.

- [ ] **Step 1: Write tests**

```python
# tests/test_gguf_dequant_torch.py
import torch, numpy as np, pytest

def test_dequant_q8_0_known_values():
    from tinyserve.gguf_dequant_torch import dequant_tensor
    scale = np.float16(0.5)
    quants = np.arange(-16, 16, dtype=np.int8)
    data = scale.tobytes() + quants.tobytes()
    result = dequant_tensor(data, ggml_type=8, shape=(1, 32))
    expected = torch.tensor([0.5 * i for i in range(-16, 16)], dtype=torch.float32)
    torch.testing.assert_close(result.flatten(), expected, atol=0.01, rtol=0.01)

def test_dequant_q4_k_matches_reference():
    from tinyserve.gguf_dequant_torch import dequant_tensor
    from tinyserve.gguf_quant import parse_q4k_blocks
    from tests.test_gguf_store import _build_q4k_block
    vals = np.linspace(-1.0, 1.0, 256).astype(np.float32)
    data = _build_q4k_block(vals)
    ref = torch.from_numpy(parse_q4k_blocks(data, (16, 16)))
    result = dequant_tensor(data, ggml_type=12, shape=(16, 16))
    torch.testing.assert_close(result, ref, atol=0.01, rtol=0.05)

def test_dequant_unsupported_raises():
    from tinyserve.gguf_dequant_torch import dequant_tensor
    with pytest.raises(ValueError, match="Unsupported"):
        dequant_tensor(b"\x00" * 100, ggml_type=99, shape=(10, 10))
```

- [ ] **Step 2: Implement**

Port city96's vectorized dequant. Function signature: `dequant_tensor(data: bytes, ggml_type: int, shape: tuple) -> torch.Tensor`. Support types: Q4_0 (2), Q4_1 (3), Q5_0 (6), Q5_1 (7), Q8_0 (8), Q4_K (12), Q5_K (13), Q6_K (14). All use torch tensor ops — no Python loops over individual blocks.

Include Apache-2.0 attribution comment at top of file.

- [ ] **Step 3: Run tests, commit**

```bash
pytest tests/test_gguf_dequant_torch.py -v
git add tinyserve/gguf_dequant_torch.py tests/test_gguf_dequant_torch.py
git commit -m "feat: pure-PyTorch GGUF dequant fallback (city96 port, Apache-2.0)"
```

---

## Task 2: MmapExpertStore

**Files:**
- Create: `tinyserve/mmap_store.py`
- Test: `tests/test_mmap_store.py`

mmap GGUF files. Build per-expert offset table. Provides `copy_to_buffer` that does: mmap read → pinned staging → GPU async H2D. Raw quant bytes, no dequant.

**Key interface requirements** (checked against ExpertPipeline and _model_hooks.py callsites):
- `num_layers`, `num_experts` — int
- `expert_bytes` — bytes per expert in native quant format
- `buffer_expert_bytes` — same as `expert_bytes` (no BF16 expansion)
- `layout` — TensorLayout with specs for projections as `(shape, torch.uint8)`
- `_bf16_layout` — set to `layout` (the native-quant path never uses BF16 layout)
- `_fp8` property — returns `False`
- `ggml_types` — dict `{"gate": 12, "up": 12, "down": 13}` etc.
- `allocate_buffer(device)` — returns ExpertBuffer sized for native quant bytes
- `copy_to_buffer(buf, layer, expert, non_blocking)` — mmap → pinned staging → GPU
- `get_expert_data(layer, expert)` — returns pinned staging tensor (for compatibility with miss paths that read `store._data[layer, expert]`)
- `copy_to_buffer_slot(cache, slot, layer, expert)` — direct mmap → cache slot

- [ ] **Step 1: Write tests**

```python
# tests/test_mmap_store.py
def test_mmap_store_from_per_expert_gguf():
    """MmapExpertStore from per-expert GGUF, returns raw bytes."""
    from tinyserve.mmap_store import MmapExpertStore
    from tests.test_gguf_reader import _create_synthetic_gguf
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
    try:
        _create_synthetic_gguf(path, n_layers=2, n_experts=4, ggml_type=0)
        store = MmapExpertStore.from_gguf(path)
        assert store.num_layers == 2
        assert store.num_experts == 4
        assert store.expert_bytes > 0
        assert not store._fp8
        assert hasattr(store, 'ggml_types')
        store.close()
    finally:
        os.unlink(path)

@requires_cuda
def test_mmap_store_copy_to_buffer():
    """copy_to_buffer transfers raw bytes to GPU."""
    ...  # Create store, allocate buffer, copy, verify buf.packed is non-zero

def test_mmap_store_directory_input():
    """from_gguf accepts directory path (multi-shard)."""
    ...
```

- [ ] **Step 2: Implement MmapExpertStore**

Core implementation:
```python
class MmapExpertStore:
    def __init__(self, mmaps, offsets, layout, ggml_types, num_layers, num_experts):
        self._mmaps = mmaps
        self._offsets = offsets  # {(layer, expert): (file_idx, byte_offset)}
        self.layout = layout
        self._bf16_layout = layout  # Native quant path, no BF16 expansion
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_bytes = layout.total_bytes
        self.buffer_expert_bytes = layout.total_bytes
        self.ggml_types = ggml_types
        # Pinned staging buffer for async H2D (one per store, reused)
        self._pinned_staging = torch.empty(layout.total_bytes, dtype=torch.uint8).pin_memory()

    @property
    def _fp8(self):
        return False

    def copy_to_buffer(self, buf, layer_idx, expert_idx, non_blocking=False):
        # 1. Read from mmap into pinned staging
        offset_info = self._offsets[(layer_idx, expert_idx)]
        mm = self._mmaps[offset_info.file_idx]
        self._pinned_staging[:].copy_(
            torch.frombuffer(mm[offset_info.offset:offset_info.offset + self.expert_bytes],
                           dtype=torch.uint8))
        # 2. Async H2D from pinned staging to GPU buffer
        buf.packed.copy_(self._pinned_staging, non_blocking=non_blocking)

    def allocate_buffer(self, device):
        return ExpertBuffer(self.layout, device)

    def get_expert_data(self, layer_idx, expert_idx):
        """Return pinned tensor with raw quant bytes (for miss path compatibility)."""
        self.copy_to_buffer_staging(layer_idx, expert_idx)
        return self._pinned_staging
```

- [ ] **Step 3: Run tests, commit**

```bash
pytest tests/test_mmap_store.py -v
git add tinyserve/mmap_store.py tests/test_mmap_store.py
git commit -m "feat: MmapExpertStore — zero-copy GGUF expert storage"
```

---

## Task 3: Fused-to-per-expert conversion (.experts file)

**Files:**
- Modify: `tinyserve/mmap_store.py` (add `from_fused_gguf`, Q8_0 requant)
- Test: `tests/test_mmap_store.py`

Fused tensors can't be byte-sliced per expert. One-time conversion: dequant fused → slice per expert → re-quant to Q8_0 → save `.experts` file. Vectorized Q8_0 quantization (no Python loops).

- [ ] **Step 1: Write vectorized Q8_0 quantizer**

```python
def _quantize_to_q8_0(tensor: torch.Tensor) -> bytes:
    """Vectorized Q8_0 quantization: 34 bytes per 32 elements."""
    flat = tensor.flatten().float()
    n_blocks = flat.shape[0] // 32
    blocks = flat[:n_blocks * 32].reshape(n_blocks, 32)
    scales = blocks.abs().amax(dim=1) / 127.0
    scales = scales.clamp(min=1e-10)
    quants = torch.round(blocks / scales.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
    # Pack: [f16_scale, int8[32]] × n_blocks
    scales_f16 = scales.to(torch.float16)
    result = torch.empty(n_blocks * 34, dtype=torch.uint8)
    for i in range(n_blocks):
        result[i*34:i*34+2] = torch.frombuffer(scales_f16[i].numpy().tobytes(), dtype=torch.uint8)
        result[i*34+2:i*34+34] = quants[i].view(torch.uint8)
    return bytes(result.numpy())
```

Note: the inner loop is over blocks (small), not over elements. For Qwen 122B (~3M elements per expert projection), this is ~12K blocks — fast enough. Can optimize with `torch.cat` later if needed.

- [ ] **Step 2: Implement from_fused_gguf with atomic file write**

```python
@classmethod
def from_fused_gguf(cls, gguf_path, cache_dir=None):
    experts_path = _experts_path(gguf_path, cache_dir)
    if experts_path.exists():
        return cls._from_experts_file(experts_path)
    cls._convert_fused_to_per_expert(gguf_path, experts_path)
    return cls._from_experts_file(experts_path)
```

`.experts` file format: 4-byte header length + JSON header + contiguous Q8_0 expert data. Atomic write via `.tmp` + `os.rename()`.

Conversion streams one layer at a time:
1. Dequant fused gate/up/down via existing `_dequant_fused_tensor` (one layer, ~3 GB peak)
2. Slice per expert
3. Quantize each projection to Q8_0
4. Write contiguously

- [ ] **Step 3: Write tests**

```python
def test_fused_to_experts_conversion_creates_file():
    """from_fused_gguf creates .experts file on first call."""
    ...

def test_fused_to_experts_reuses_cached_file():
    """Second call skips conversion, loads from .experts."""
    ...

def test_q8_0_quantize_roundtrip():
    """Q8_0 quantize → dequant roundtrip within int8 precision."""
    ...
```

- [ ] **Step 4: Run tests, commit**

```bash
pytest tests/test_mmap_store.py -v
git add tinyserve/mmap_store.py tests/test_mmap_store.py
git commit -m "feat: fused-to-per-expert Q8_0 conversion for Qwen-style GGUF"
```

---

## Task 4: ggml submodule + PyTorch C++ extension

**Files:**
- Add: `third_party/ggml/` (git submodule)
- Create: `tinyserve/csrc/ggml_ops.cu`
- Create: `build_ggml.py` (build script for ggml + extension)
- Test: `tests/test_ggml_ops.py`

This is the hardest task. The MMVQ kernels are `static` in ggml, so we copy the kernel dispatch logic (~200-300 LOC of CUDA) into our extension file, with include paths to ggml headers for type definitions.

- [ ] **Step 1: Add ggml submodule**

```bash
git submodule add https://github.com/ggml-org/ggml.git third_party/ggml
git commit -m "chore: add ggml submodule (MIT)"
```

- [ ] **Step 2: Write build_ggml.py**

Separate build script (not in setup.py) to avoid conflicts with Cython build:

```python
# build_ggml.py
"""Build ggml CUDA backend + tinyserve ggml_ops extension.

Usage: python build_ggml.py
"""
import subprocess, os
from torch.utils.cpp_extension import load

# Step 1: Build ggml-cuda shared library via CMake
subprocess.run([
    "cmake", "-S", "third_party/ggml", "-B", "build/ggml",
    "-DGGML_CUDA=ON", "-DBUILD_SHARED_LIBS=ON",
    "-DGGML_BUILD_EXAMPLES=OFF", "-DGGML_BUILD_TESTS=OFF",
], check=True)
subprocess.run(["cmake", "--build", "build/ggml", "--target", "ggml-cuda",
                f"-j{os.cpu_count()}"], check=True)

# Step 2: JIT-compile PyTorch extension with ggml includes
ggml_ops = load(
    name="ggml_ops",
    sources=["tinyserve/csrc/ggml_ops.cu"],
    extra_include_paths=["third_party/ggml/src/ggml-cuda", "third_party/ggml/src",
                         "third_party/ggml/include"],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2", "--use_fast_math"],
    verbose=True,
)
```

- [ ] **Step 3: Write ggml_ops.cu**

The extension copies the minimal kernel dispatch code from ggml's mmvq.cu. Key pieces:
1. Include `ggml-common.h` for block type structs
2. Copy `quantize_row_q8_1_cuda` kernel (or link against ggml-cuda.so)
3. Copy `mul_mat_vec_q` kernel template instantiation for each quant type
4. Expose as `torch::Tensor ggml_mul_mat_vec(activation, weight_data, ggml_type, N, K)`

The wrapper:
- Takes BF16 activation, casts to F32
- Allocates q8_1 buffer, quantizes activation
- Dispatches to type-specific MMVQ kernel
- Returns result as BF16

Register as `torch.ops.tinyserve.ggml_mul_mat_vec`.

- [ ] **Step 4: Write test**

```python
@requires_cuda
def test_ggml_mul_mat_vec_q8_0():
    """ggml MMVQ: Q8_0 matmul produces correct-ish output."""
    try:
        torch.ops.load_library("build/ggml_ops.so")  # or however it loads
    except:
        pytest.skip("ggml extension not built")
    # Create known Q8_0 data, verify matmul output shape + non-NaN
    ...

@requires_cuda
def test_ggml_mul_mat_vec_matches_dequant_fallback():
    """ggml MMVQ output matches city96 dequant + F.linear within quant noise."""
    ...
```

- [ ] **Step 5: Build, test, commit**

```bash
python build_ggml.py
pytest tests/test_ggml_ops.py -v
git add third_party/ggml tinyserve/csrc/ggml_ops.cu build_ggml.py tests/test_ggml_ops.py
git commit -m "feat: ggml MMVQ kernel as PyTorch custom op"
```

---

## Task 5: GGMLExpertForward

**Files:**
- Create: `tinyserve/ggml_forward.py`
- Test: `tests/test_ggml_forward.py`

Expert forward with two internal paths:
- **batch=1 + ggml available:** 3 MMVQ kernel calls (gate, up, down)
- **batch>1 or no ggml:** city96 dequant → F.linear (fallback)

- [ ] **Step 1: Write tests**

```python
@requires_cuda
def test_ggml_forward_batch_1():
    """GGMLExpertForward: batch=1 produces valid output."""
    ...

def test_ggml_forward_batch_gt_1_uses_fallback():
    """GGMLExpertForward: batch>1 uses dequant+F.linear."""
    ...

@requires_cuda
def test_ggml_forward_matches_dequant_fallback():
    """ggml kernel output ≈ dequant+F.linear output (within quant noise)."""
    ...
```

- [ ] **Step 2: Implement**

```python
class GGMLExpertForward:
    def __init__(self, layout, ggml_types, act_fn, proj_shapes):
        self._layout = layout
        self._ggml_types = ggml_types
        self._act_fn = act_fn
        self._proj_shapes = proj_shapes  # {"gate": (N, K), "up": (N, K), "down": (K, N)}
        self._has_ggml = self._check_ggml()

    def forward(self, packed, h):
        if h.shape[0] == 1 and self._has_ggml:
            return self._ggml_forward(packed, h)
        return self._fallback_forward(packed, h)

    def _ggml_forward(self, packed, h):
        gate = torch.ops.tinyserve.ggml_mul_mat_vec(h, packed[g_off:g_end], g_type, g_N, g_K)
        up = torch.ops.tinyserve.ggml_mul_mat_vec(h, packed[u_off:u_end], u_type, u_N, u_K)
        hidden = self._act_fn(gate) * up
        return torch.ops.tinyserve.ggml_mul_mat_vec(hidden, packed[d_off:d_end], d_type, d_N, d_K)

    def _fallback_forward(self, packed, h):
        from .gguf_dequant_torch import dequant_tensor
        gate_w = dequant_tensor(packed[g_off:g_end], g_type, g_shape).to(h.device, h.dtype)
        up_w = dequant_tensor(packed[u_off:u_end], u_type, u_shape).to(h.device, h.dtype)
        down_w = dequant_tensor(packed[d_off:d_end], d_type, d_shape).to(h.device, h.dtype)
        gate = F.linear(h, gate_w)
        up = F.linear(h, up_w)
        hidden = self._act_fn(gate) * up
        return F.linear(hidden, down_w)
```

- [ ] **Step 3: Run tests, commit**

```bash
pytest tests/test_ggml_forward.py -v
git add tinyserve/ggml_forward.py tests/test_ggml_forward.py
git commit -m "feat: GGMLExpertForward — native quant expert compute with fallback"
```

---

## Task 6: Native-quant path in ExpertPipeline

**Files:**
- Modify: `tinyserve/expert_pipeline.py`
- Test: `tests/test_native_quant_pipeline.py`

Add a complete parallel forward path for native-quant stores. Selected at construction time. The existing BF16/MXFP4 path has **zero changes**.

- [ ] **Step 1: Write test**

```python
@requires_cuda
def test_native_quant_pipeline_cache_hit():
    """Native-quant pipeline: cache hit uses GGMLExpertForward."""
    from tinyserve.mmap_store import MmapExpertStore
    from tinyserve.expert_pipeline import ExpertPipeline
    from tinyserve.expert_cache import ExpertCache
    # Create synthetic MmapExpertStore with Q8_0 data
    # Build pipeline, execute, verify output shape + non-NaN
    ...

@requires_cuda
def test_native_quant_pipeline_cache_miss():
    """Native-quant pipeline: miss loads from mmap, fills cache, forwards."""
    ...

@requires_cuda
def test_native_quant_pipeline_batched():
    """Native-quant pipeline: batched prefill uses dequant fallback."""
    ...
```

- [ ] **Step 2: Add _native_quant detection and routing**

In `ExpertPipeline.__init__`:
```python
self._native_quant = hasattr(store, 'ggml_types')
if self._native_quant:
    from .ggml_forward import GGMLExpertForward
    self._nq_forward = GGMLExpertForward(
        store.layout, store.ggml_types, self._act_fn, store.proj_shapes)
```

- [ ] **Step 3: Add _execute_layer_experts_native**

Complete implementation: cache lookup, hit forward via `_nq_forward.forward(packed, h)`, miss via `store.copy_to_buffer` → cache store → forward. Same eviction, same stats tracking.

```python
def _execute_layer_experts_native(self, hidden_states, layer_idx, expert_indices, routing_weights):
    """Complete native-quant forward path. Parallel to execute_layer_experts."""
    output = torch.zeros_like(hidden_states)
    cache = self.cache

    for tok in range(hidden_states.shape[0]):
        h = hidden_states[tok:tok+1]
        for k in range(expert_indices.shape[1]):
            eid = expert_indices[tok, k].item()
            w = routing_weights[tok, k]

            slot = cache.lookup(layer_idx, eid) if cache else None
            if slot is not None:
                packed = cache.get_packed(slot)
                out = self._nq_forward.forward(packed, h)
            else:
                # Miss: load from store to staging buffer
                buf = self.staging_buffer_a
                self.store.copy_to_buffer(buf, layer_idx, eid, non_blocking=False)
                torch.cuda.synchronize()
                out = self._nq_forward.forward(buf.packed, h)
                # Fill cache
                if cache is not None:
                    slot = cache.allocate(layer_idx, eid)
                    cache.get_packed(slot).copy_(buf.packed)

            output[tok] += w * out.squeeze(0)

    if cache is not None:
        cache.flush_slot_updates()
    return output
```

- [ ] **Step 4: Add _execute_layer_experts_batched_native for prefill**

Group by expert, use dequant fallback for batch>1:
```python
def _execute_layer_experts_batched_native(self, hidden_states, layer_idx, expert_indices, routing_weights):
    """Batched native-quant path for prefill. Uses dequant fallback."""
    # Same grouping logic as execute_layer_experts_batched
    # But forward via self._nq_forward.forward(packed, h_batch)
    # which internally uses dequant+F.linear for batch>1
    ...
```

- [ ] **Step 5: Route at top level**

```python
def execute_layer_experts(self, hidden_states, layer_idx, expert_indices, routing_weights):
    if self._native_quant:
        return self._execute_layer_experts_native(
            hidden_states, layer_idx, expert_indices, routing_weights)
    # ... existing path unchanged ...

def execute_layer_experts_batched(self, hidden_states, layer_idx, expert_indices, routing_weights):
    if self._native_quant:
        return self._execute_layer_experts_batched_native(
            hidden_states, layer_idx, expert_indices, routing_weights)
    # ... existing path unchanged ...
```

- [ ] **Step 6: Run tests, commit**

```bash
pytest tests/test_native_quant_pipeline.py -v
git add tinyserve/expert_pipeline.py tests/test_native_quant_pipeline.py
git commit -m "feat: parallel native-quant path in ExpertPipeline"
```

---

## Task 7: Wire into gguf_loader

**Files:**
- Modify: `tinyserve/gguf_loader.py`
- Modify: `tinyserve/_model_hooks.py` (handle MmapExpertStore in from_module)

- [ ] **Step 1: Update load_from_gguf**

Replace the dequant→ExpertStore path with MmapExpertStore:

```python
from .mmap_store import MmapExpertStore

reader = open_gguf(gguf_path)
if reader.list_fused_expert_tensors():
    expert_store = MmapExpertStore.from_fused_gguf(gguf_path)
elif reader.list_expert_tensors():
    expert_store = MmapExpertStore.from_gguf(gguf_path)
else:
    raise ValueError(f"No expert tensors in {gguf_path}")
reader.close()
```

- [ ] **Step 2: Handle MmapExpertStore in _model_hooks.py**

In `OffloadedModel.from_module`, the pipeline construction accesses `store._fp8`, `store._bf16_layout`, `store.allocate_buffer`. MmapExpertStore provides all of these (Task 2). Verify no breakage by reading the construction code and tracing all store attribute accesses.

The key callsites:
- `store._fp8` → returns `False` ✓
- `store._bf16_layout` → returns `store.layout` ✓
- `store.allocate_buffer(device)` → returns native-quant-sized buffer ✓
- `store.layout` → TensorLayout with uint8 specs ✓

The `_install_offloaded_forward` hook doesn't access the store directly — it calls `pipeline.execute_layer_experts`. Since the pipeline routes based on `_native_quant`, no hook changes needed.

- [ ] **Step 3: Test with GPT-OSS-20B (regression)**

```bash
python3 -m scripts.test_contexts --contexts 256 --max-seq-len 2048
```

Verify MXFP4 path still works (untouched).

- [ ] **Step 4: Commit**

```bash
git add tinyserve/gguf_loader.py tinyserve/_model_hooks.py
git commit -m "feat: wire MmapExpertStore into GGUF loader"
```

---

## Task 8: Qwen 122B end-to-end test

- [ ] **Step 1: Test MmapExpertStore on Qwen 122B**

```bash
python3 -c "
from tinyserve.mmap_store import MmapExpertStore
store = MmapExpertStore.from_fused_gguf('/home/elnur/models/Qwen3.5-122B-A10B-GGUF/Q4_K_M/')
print(f'Layers: {store.num_layers}, Experts: {store.num_experts}')
print(f'Expert bytes: {store.expert_bytes}')
import psutil; print(f'RSS: {psutil.Process().memory_info().rss / 1e9:.1f} GB')
"
```

Expected: RSS < 5 GB.

- [ ] **Step 2: Test inference**

```bash
python3 -m scripts.test_contexts \
    --model Qwen/Qwen3.5-122B-A10B \
    --gguf /home/elnur/models/Qwen3.5-122B-A10B-GGUF/Q4_K_M/ \
    --contexts 256 --max-seq-len 2048
```

- [ ] **Step 3: Full test suite (no regressions)**

```bash
pytest tests/ -q 2>&1 | tail -5
```

Target: 414+ passed.

- [ ] **Step 4: ruff + commit**

```bash
ruff check tinyserve/ tests/
ruff format tinyserve/ tests/
git add -A
git commit -m "test: Qwen 122B end-to-end via ggml native quant compute"
```
