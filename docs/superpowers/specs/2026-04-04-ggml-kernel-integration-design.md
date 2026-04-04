# ggml Kernel Integration — Native Quantized Compute for GGUF Models

*2026-04-04*

---

## Problem

tinyserve cannot load Qwen 122B or any large GGUF model. The current path dequants all expert weights to BF16, requiring 232 GB RAM (OOMs on 64 GB). Even if it fit, BF16 in the GPU cache wastes 3.6x bandwidth vs native Q4_K — storing 16 bits per weight when only 4.5 bits of information exist.

## Solution

Integrate ggml's battle-tested CUDA kernels (MIT, ggml-org/ggml) via git submodule. Store and compute on native GGUF quant format everywhere — zero dequant, zero conversion, zero wasted bytes.

## Architecture

```
GGUF on SSD (Q4_K/Q5_K/Q6_K native bytes)
    │
    ▼ mmap (zero RAM)
MmapExpertStore — offset table per (layer, expert, projection)
    │
    ▼ cache miss: memcpy raw quant bytes → GPU (H2D, no dequant)
ExpertCache — slots hold native quant bytes (3.6x more experts than BF16)
    │
    ▼ cache hit: ggml MMVQ kernel (fused dequant+matvec, one VRAM read)
Output tensor
```

Peak RAM: ~2 GB (model skeleton + KV cache). Expert data entirely on SSD via mmap.

### Data flow — what format at each stage

| Stage | Format | Bytes/weight | Conversion |
|-------|--------|-------------|------------|
| Disk (GGUF file) | Q4_K native | 0.56 | None |
| CPU (mmap) | Q4_K native | 0 (page cache) | None |
| H2D transfer | Q4_K native | 0.56 | None (raw memcpy) |
| GPU cache slot | Q4_K native | 0.56 | None |
| Compute | MMVQ kernel | — | Activation → q8_1 on-the-fly (~5us) |

Zero conversion at any stage. Bits in GPU cache are byte-identical to bits on disk.

### Fused tensor handling (Qwen-style)

Fused GGUF tensors (`[out, in, n_experts]`) have elements interleaved across experts within Q4_K blocks — cannot be byte-sliced per expert. Solution: one-time conversion at first load:

1. Dequant fused tensor → float32 (one layer at a time, ~3 GB peak)
2. Slice per expert
3. Re-quantize each expert to Q4_K independently
4. Save as tinyserve-native `.experts` file (per-expert Q4_K, contiguous)
5. Subsequent loads: instant mmap of the `.experts` file

This conversion runs once. The `.experts` file is cached on disk alongside the GGUF.

---

## Components

### 1. ggml submodule + build

```
third_party/ggml/    ← git submodule of ggml-org/ggml
```

Build target: `ggml-cuda` shared library via CMake.

```bash
cmake -S third_party/ggml -B build/ggml \
    -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
    -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF
cmake --build build/ggml --target ggml-cuda
```

Produces `libggml-cuda.so`. Linked at runtime by the PyTorch C++ extension.

### 2. PyTorch C++ extension (`tinyserve/csrc/ggml_ops.cpp`)

~100 LOC thin wrapper. Exposes one op:

```python
torch.ops.tinyserve.ggml_mul_mat_vec(
    activation: Tensor,     # [1, K] BF16 on GPU
    weight_data: Tensor,    # [nbytes] uint8 on GPU (raw Q4_K/Q5_K/Q6_K bytes)
    ggml_type: int,         # GGML_TYPE_Q4_K = 12, etc.
    out_features: int,      # N dimension
    in_features: int,       # K dimension
) -> Tensor                 # [1, N] BF16 on GPU
```

Internally:
1. Quantize activation from BF16 → q8_1 via `quantize_row_q8_1_cuda()`
2. Call `mul_mat_vec_q` kernel template with raw weight pointer + ggml_type
3. Return f32 output cast to BF16

The wrapper handles CUDA stream from PyTorch's current stream. No ggml_tensor abstraction exposed to Python.

### 3. MmapExpertStore (`tinyserve/mmap_store.py`)

```python
class MmapExpertStore:
    """Zero-copy expert store backed by mmap'd GGUF or .experts files."""

    def __init__(self, mmaps, expert_offsets, shapes, ggml_types, num_layers, num_experts):
        self._mmaps = mmaps           # list of mmap objects
        self._offsets = expert_offsets # {(layer, expert, proj): (file_idx, offset, nbytes)}
        self._shapes = shapes         # {proj: (out_features, in_features)}
        self._ggml_types = ggml_types # {proj: ggml_type_id}

    @classmethod
    def from_gguf(cls, path) -> MmapExpertStore:
        """Build from per-expert GGUF format."""
        ...

    @classmethod
    def from_fused_gguf(cls, path, cache_dir=None) -> MmapExpertStore:
        """Build from fused GGUF. Converts to .experts file on first load."""
        experts_path = Path(cache_dir or path).with_suffix(".experts")
        if experts_path.exists():
            return cls._from_experts_file(experts_path)
        # One-time conversion: fused Q4K → per-expert Q4K
        cls._convert_fused_to_per_expert(path, experts_path)
        return cls._from_experts_file(experts_path)

    def get_expert_bytes(self, layer_idx, expert_idx) -> dict[str, memoryview]:
        """Return raw quant bytes for one expert's projections. Zero copy."""
        ...

    def copy_to_buffer(self, buf, layer_idx, expert_idx, non_blocking=False):
        """H2D copy raw quant bytes to GPU buffer. No dequant."""
        for proj in ("gate", "up", "down"):
            src = self.get_expert_bytes(layer_idx, expert_idx)[proj]
            dst_offset = self._layout.offsets[proj]
            buf.packed[dst_offset:dst_offset + len(src)].copy_(
                torch.frombuffer(src, dtype=torch.uint8), non_blocking=non_blocking
            )
```

Interface matches `ExpertStore.copy_to_buffer` so ExpertPipeline works unchanged.

### 4. Expert forward (`tinyserve/ggml_forward.py`)

```python
class GGMLExpertForward:
    """Expert forward using ggml MMVQ kernels on native quant data."""

    def __init__(self, layout, ggml_types, act_fn):
        self._layout = layout
        self._ggml_types = ggml_types  # {"gate": 12, "up": 12, "down": 13}
        self._act_fn = act_fn

    def forward(self, packed: Tensor, h: Tensor) -> Tensor:
        """Forward pass on native quant expert data."""
        gate_out = torch.ops.tinyserve.ggml_mul_mat_vec(
            h, packed[gate_off:gate_off+gate_sz], self._ggml_types["gate"],
            gate_out_features, gate_in_features,
        )
        up_out = torch.ops.tinyserve.ggml_mul_mat_vec(
            h, packed[up_off:up_off+up_sz], self._ggml_types["up"],
            up_out_features, up_in_features,
        )
        hidden = self._act_fn(gate_out) * up_out
        return torch.ops.tinyserve.ggml_mul_mat_vec(
            hidden, packed[down_off:down_off+down_sz], self._ggml_types["down"],
            down_out_features, down_in_features,
        )
```

Three kernel launches per expert (gate, up, down). Could fuse gate+up if ggml's GLU fusion is wired, but that's an optimization for later.

### 5. Fallback: pure-PyTorch dequant (`tinyserve/gguf_dequant_torch.py`)

Ported from city96/ComfyUI-GGUF `dequant.py` (Apache-2.0, ~300 LOC). Pure `torch` ops, no compilation needed. Covers Q2_K through Q6_K, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0.

Used when ggml-cuda is not available (no CUDA toolkit, AMD GPU, CI). Dequants to BF16, then standard `F.linear`.

### 6. Integration into ExpertPipeline

```python
# In ExpertPipeline.__init__:
if isinstance(store, MmapExpertStore):
    self._ggml_fwd = GGMLExpertForward(store.layout, store.ggml_types, act_fn)
else:
    self._ggml_fwd = None

# In _forward_cache_hits:
if self._ggml_fwd is not None:
    out = self._ggml_fwd.forward(packed, h)
elif self._inline_fwd is not None:
    out = self._inline_fwd(packed, h)
...
```

### 7. ExpertCache changes

```python
# ExpertCache.__init__ already takes expert_bytes as a parameter.
# For GGUF native quant, expert_bytes = sum of Q4_K bytes for gate+up+down.
# No code change needed — just pass the smaller byte count.
#
# Example: Qwen 122B expert
#   BF16: gate_up(2*3072*1024*2) + down(1024*3072*2) = 18.9 MB
#   Q4_K: gate(3072*1024*0.56) + up(same) + down(Q5_K: 1024*3072*0.69) = ~5.6 MB
#   3.4x more experts in same VRAM
```

---

## File manifest

| File | Action | LOC est. |
|------|--------|----------|
| `third_party/ggml/` | git submodule add | — |
| `tinyserve/csrc/ggml_ops.cpp` | Create | ~100 |
| `tinyserve/ggml_forward.py` | Create | ~60 |
| `tinyserve/mmap_store.py` | Create | ~200 |
| `tinyserve/gguf_dequant_torch.py` | Create (port from city96) | ~300 |
| `tinyserve/expert_pipeline.py` | Modify (add ggml_fwd routing) | ~10 |
| `tinyserve/expert_cache.py` | No change (already parameterized) | 0 |
| `tinyserve/gguf_loader.py` | Modify (use MmapExpertStore) | ~30 |
| `setup.py` / `pyproject.toml` | Modify (build ggml + extension) | ~30 |
| `tests/test_ggml_forward.py` | Create | ~80 |
| `tests/test_mmap_store.py` | Create | ~80 |
| **Total new code** | | **~890** |

---

## What does NOT change

- GPT-OSS-20B path (MXFP4 via Triton dot_scaled) — untouched
- ExpertCache eviction policies (LRU/LFRU/etc.) — untouched
- FATE prefetch, StreamingLLM, buddy experts — untouched
- StaticKVCache, VRAMBudget — untouched
- CLI, server, benchmark scripts — untouched (auto-detect store type)

---

## Supported GGUF quant types (day one)

All types supported by ggml MMVQ kernels:

| Type | bpw | Status |
|------|-----|--------|
| Q4_K | 4.5 | Primary target (most popular) |
| Q5_K | 5.5 | Primary target (quality sweet spot) |
| Q6_K | 6.6 | Used internally by Q4_K_M for attn layers |
| Q8_0 | 8.5 | Near-lossless |
| Q4_0 | 4.5 | Legacy |
| Q4_1 | 5.0 | Legacy |
| Q5_0 | 5.5 | Legacy |
| Q5_1 | 6.0 | Legacy |
| Q2_K | 3.2 | Extreme compression |
| Q3_K | 3.5 | Low-VRAM |
| IQ types | 2-4 | Via ggml kernels |

---

## Memory comparison

| Metric | Current (BF16 dequant) | New (native quant) |
|--------|------------------------|-------------------|
| Qwen 122B load peak RAM | 232 GB (OOM) | ~2 GB (mmap) |
| Expert store size | 232 GB pinned | 0 (mmap, OS page cache) |
| GPU cache slot (Qwen expert) | 18.9 MB | 5.6 MB |
| Experts in cache (8 GB GPU) | ~300 | ~1,000 |
| Cache hit rate (estimated) | — | Higher (3.4x more experts fit) |
| H2D transfer per miss | 18.9 MB BF16 | 5.6 MB native (3.4x faster) |

---

## Design decisions from review

### D1: Fused re-quant uses Q8_0 (not Q4_K)
Q4_K re-quantization is complex (hierarchical scales). Q8_0 is trivial: `scale = max(abs(x))/127`, round to int8, store as 34 bytes per 32 elements. ~2x larger than Q4_K but still 4x smaller than BF16. MMVQ supports Q8_0 natively. Can optimize to Q4_K later.

### D2: `.experts` file format
Simple binary format:
```
Header (JSON, length-prefixed):
  {"num_layers": 48, "num_experts": 256, "ggml_type": 8,
   "gate_shape": [3072, 1024], "up_shape": [3072, 1024], "down_shape": [1024, 3072],
   "expert_bytes": 174080, "projections": ["gate", "up", "down"]}
Data:
  [layer_0_expert_0_gate][layer_0_expert_0_up][layer_0_expert_0_down]
  [layer_0_expert_1_gate]...
```
Fixed `expert_bytes` per expert. Offset = header_size + (layer * num_experts + expert) * expert_bytes. Projections contiguous within each expert.

### D3: Activation must be cast BF16→F32 before q8_1 quantization
ggml's `quantize_row_q8_1_cuda` expects `float*`. The wrapper casts BF16→F32 before quantization (~5us for [1, K] vector). Output F32→BF16 cast on return.

### D4: mmap H2D needs pinned staging buffer
mmap pages are not pinned — `non_blocking=True` silently falls back to synchronous copy. Fix: `MmapExpertStore` allocates one small pinned CPU staging buffer (~20 MB, covers largest expert). `copy_to_buffer` does: mmap → pinned staging (CPU memcpy) → GPU (async H2D). Reuses existing double-buffering in ExpertPipeline.

### D5: MmapExpertStore conforms to ExpertStore interface
Must provide: `_data` (not used, set to None), `layout`, `_bf16_layout` (same as layout for native quant), `_fp8` (False), `expert_bytes`, `buffer_expert_bytes`, `allocate_buffer(device)`, `copy_to_buffer(buf, layer, expert, non_blocking)`. The `allocate_buffer` returns an ExpertBuffer sized for native quant bytes.

### D6: Batch>1 fallback for prefill
`GGMLExpertForward.forward` checks `h.shape[0]`. If >1 (prefill), falls back to city96 dequant + F.linear. MMVQ is matvec-only. The batched path (`execute_layer_experts_batched`) uses the same fallback. Prefill is not the hot path for MoE offloading — experts are loaded once per batch.

### D7: Atomic .experts file creation
`_convert_fused_to_per_expert` writes to `{path}.experts.tmp`, then `os.rename()` to final path. Atomic on POSIX. Second process sees either complete file or no file.

---

## Risks

1. **ggml CUDA/PyTorch stream sharing:** The C++ extension must use PyTorch's current CUDA stream via `at::cuda::getCurrentCUDAStream()`. Pass to ggml kernel launch.

2. **MMVQ kernel is `static`:** Use ggml's backend API (`ggml_backend_cuda_buffer_type`, graph compute) or copy the kernel dispatch logic (~50 LOC) into `ggml_ops.cpp`. Copying is simpler and doesn't require forking. Track ggml version updates by diffing the dispatch code on submodule bumps.

3. **Build complexity:** Users need CUDA toolkit + CMake to build ggml. Mitigated by the pure-PyTorch fallback (city96 dequant) which works without any compilation.

4. **GGUF block alignment:** `MmapExpertStore.__init__` validates all computed offsets are within file bounds and aligned to quant block boundaries.

5. **Mixed quant types within GGUF:** Expert projections can have different quant types (e.g., gate=Q4_K, down=Q5_K as in Qwen 122B). `MmapExpertStore._ggml_types` is per-projection, which handles the common case. Per-layer variation within experts is not supported (not observed in practice).
