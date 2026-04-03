# Elite Refactor Spec — tinyserve

**Date:** 2026-04-03
**Goal:** Bring tinyserve from "working" to "a principal engineer says 'well-built' in 30 seconds."
**Constraint:** All 59 existing tests must pass after each phase. No behavioral changes.

---

## Codebase Snapshot (pre-refactor)

```
tinyserve/
  generic_pipeline.py   855 LOC  (→ expert_pipeline.py)
  generic_store.py      823 LOC  (→ expert_store.py + expert_cache.py)
  offloaded_model.py    589 LOC  (→ _model_hooks.py)
  offload.py            512 LOC
  server.py             509 LOC
  cache_policy.py       411 LOC
  cpu_expert.py         331 LOC
  ... 21 more files
scripts/                20 files, 3427 LOC total
tests/                  36 test files
docs/                   4 internal planning docs + superpowers/
```

**Files that reference `generic_pipeline` or `generic_store` (imports):**
- `tinyserve/`: generic_pipeline.py, generic_store.py, offloaded_model.py, offload.py, expert_batcher.py, gpu_int4.py, gguf_loader.py, gguf_store.py, imatrix.py, cache_policy.py
- `tests/`: test_generic_pipeline.py, test_generic_store.py, test_e2e_offload.py, test_imatrix.py, test_vram_budget.py, test_cache_stats.py, test_cpu_miss_fallback.py, test_batched_prefill.py, test_gpu_int4.py, test_cpp_expert_loop.py, test_expert_batcher.py, test_disk_offload_integration.py, test_cpu_expert.py, test_int4_expert.py
- `scripts/`: benchmark.py

**Files that reference `offloaded_model`:**
- `tinyserve/`: offload.py, generic_pipeline.py
- `tests/`: test_offloaded_model.py, test_hf_models.py, test_mxfp4.py
- `scripts/`: benchmark.py, comprehensive_bench.py, calibrate_buddies.py

---

## Phase 1 — Independent Tasks (all parallelizable)

These tasks have ZERO dependencies on each other. Each can be executed by a separate subagent.

---

### Task 1: Rename `generic_*` files and classes to `expert_*`

**What:** Rename two files and all class/import references across the entire codebase.

**Files to rename:**
- `tinyserve/generic_pipeline.py` → `tinyserve/expert_pipeline.py`
- `tinyserve/generic_store.py` → `tinyserve/expert_store.py`

**Classes to rename (in the new files AND all importers):**
- `GenericExpertPipeline` → `ExpertPipeline`
- `GenericExpertStore` → `ExpertStore`
- `GenericExpertBuffer` → `ExpertBuffer`
- `GenericLRUCache` → `ExpertCache`

**Test files to rename:**
- `tests/test_generic_pipeline.py` → `tests/test_expert_pipeline.py`
- `tests/test_generic_store.py` → `tests/test_expert_store.py`

**All files requiring import updates (exhaustive list):**

Source files (update `from .generic_store`/`from .generic_pipeline` to `from .expert_store`/`from .expert_pipeline`):
- `tinyserve/expert_pipeline.py` (the renamed file itself): line 12 `from .generic_store import ...` → `from .expert_store import ...`
- `tinyserve/offloaded_model.py`: lines 13-14
- `tinyserve/offload.py`: line 323
- `tinyserve/expert_batcher.py`: line 14
- `tinyserve/gpu_int4.py`: line 15
- `tinyserve/gguf_loader.py`: lines 509, 813
- `tinyserve/gguf_store.py`: line 13
- `tinyserve/imatrix.py` (if it imports GenericExpertStore/GenericLRUCache — verify)

Test files (update `from tinyserve.generic_store`/`from tinyserve.generic_pipeline`):
- `tests/test_expert_pipeline.py` (renamed): ~6 import sites
- `tests/test_expert_store.py` (renamed): ~8 import sites
- `tests/test_imatrix.py`: lines 163, 205, 229
- `tests/test_vram_budget.py`: line 7
- `tests/test_cache_stats.py`: line 8
- `tests/test_cpu_miss_fallback.py`: lines 27-28
- `tests/test_batched_prefill.py`: lines 27-28
- `tests/test_gpu_int4.py`: lines 9, 288, 292, 299
- `tests/test_cpp_expert_loop.py`: lines 8, 207, 222
- `tests/test_expert_batcher.py`: ~14 import sites
- `tests/test_disk_offload_integration.py`: lines 8, 94-95
- `tests/test_cpu_expert.py`: line 9
- `tests/test_int4_expert.py`: line 14
- `tests/test_e2e_offload.py`: lines 90-91, 110, 157-158, 183

Script files:
- `scripts/benchmark.py`: line 274

**Backward compat:** Add `tinyserve/generic_pipeline.py` and `tinyserve/generic_store.py` as thin re-export shims with `warnings.warn("Deprecated: use tinyserve.expert_pipeline / tinyserve.expert_store", DeprecationWarning, stacklevel=2)` so external code does not break immediately. These shim files should be ~10 lines each.

**How to verify:** `python -m pytest tests/ -x --tb=short` — all 59 tests pass.

**Estimated LOC changed:** ~120 lines of import edits across ~25 files, +20 lines for shims. Net new: ~20 LOC.

---

### Task 2: Rename `offloaded_model.py` → `_model_hooks.py`

**What:** Rename the file and update all imports. The underscore prefix signals "internal module."

**File to rename:**
- `tinyserve/offloaded_model.py` → `tinyserve/_model_hooks.py`

**All files requiring import updates:**
- `tinyserve/offload.py`: line 17 `from .offloaded_model import OffloadedModel` → `from ._model_hooks import OffloadedModel`
- `tinyserve/expert_pipeline.py` (or `generic_pipeline.py` if Task 1 hasn't run): line 220 `from .offloaded_model import _mxfp4_linear` → `from ._model_hooks import _mxfp4_linear`
- `tests/test_offloaded_model.py`: lines 12, 55, 111 — update to `from tinyserve._model_hooks import OffloadedModel`
- `tests/test_hf_models.py`: line 23
- `tests/test_mxfp4.py`: line 107
- `scripts/benchmark.py`: lines 104, 182, 421, 520, 811
- `scripts/comprehensive_bench.py`: line 87
- `scripts/calibrate_buddies.py`: line 27

**Rename test file:**
- `tests/test_offloaded_model.py` → `tests/test_model_hooks.py`

**Backward compat:** Add `tinyserve/offloaded_model.py` as a thin re-export shim with deprecation warning (~10 lines).

**How to verify:** `python -m pytest tests/ -x --tb=short`

**Estimated LOC changed:** ~15 import edits + 10-line shim. Net new: ~10 LOC.

---

### Task 3: Create `RoutingSpec` namedtuple

**What:** Replace the raw tuple values in `_ROUTING_MAP` with a named type.

**File to modify:** `tinyserve/offload.py`

**Current code (lines 221-234):**
```python
_ROUTING_MAP = {
    "mixtral": ("router_native", False, "gate"),
    "qwen3_moe": ("router_native", False, "gate"),
    ...
}
```

**Target code:**
```python
from typing import NamedTuple

class RoutingSpec(NamedTuple):
    softmax_order: str       # "router_native" | "softmax_then_topk"
    returns_logits: bool     # whether router returns raw logits
    router_attr: str         # attribute name on MoE block ("gate" | "router")

_ROUTING_MAP: dict[str, RoutingSpec] = {
    "mixtral": RoutingSpec("router_native", False, "gate"),
    "qwen3_moe": RoutingSpec("router_native", False, "gate"),
    ...
}
```

**Update the unpacking site (line 294):**
```python
# Before:
softmax_order, returns_logits, router_attr = _ROUTING_MAP.get(model_type, ("softmax_then_topk", True, "gate"))
# After:
spec = _ROUTING_MAP.get(model_type, RoutingSpec("softmax_then_topk", True, "gate"))
softmax_order, returns_logits, router_attr = spec
```

**Files affected:** Only `tinyserve/offload.py`.

**How to verify:** `python -m pytest tests/test_e2e_offload.py tests/test_offloaded_model.py -x --tb=short`

**Estimated LOC changed:** +8 (NamedTuple class), ~15 lines modified. Net new: ~8 LOC.

---

### Task 4: Create `AttentionBackend` enum

**What:** Replace magic strings `"flex"`, `"sdpa"`, `"flashinfer"`, `"eager"`, `"flash_attention_2"` with a string enum.

**File to modify:** `tinyserve/offload.py`

**Add at top of file (after imports):**
```python
from enum import Enum

class AttentionBackend(str, Enum):
    EAGER = "eager"
    SDPA = "sdpa"
    FLEX = "flex"
    FLASHINFER = "flashinfer"
    FLASH_ATTENTION_2 = "flash_attention_2"
```

Using `str, Enum` so the values pass directly to HuggingFace's `attn_implementation` parameter (which expects strings).

**Update sites in `offload.py`:**
- `_register_flex_attention()` return values: `return "flex"` → `return AttentionBackend.FLEX`, `return "eager"` → `return AttentionBackend.EAGER`
- `_register_sdpa_attention()` return values: same pattern
- `_register_flashinfer_attention()` return values: same pattern
- `load_and_offload()` body: all string comparisons like `if attn_impl == "flex"` → `if attn_impl == AttentionBackend.FLEX`
- `offload_model()` body: `use_flex = attn_implementation == "flex"` → `use_flex = attn_implementation == AttentionBackend.FLEX`
- Type hints: `attn_implementation: str | None` → `attn_implementation: str | AttentionBackend | None`

**Export:** Add `AttentionBackend` to `tinyserve/__init__.py`.

**Files affected:** `tinyserve/offload.py`, `tinyserve/__init__.py`.

**How to verify:** `python -m pytest tests/ -x --tb=short` (string enum is backward-compatible with string comparisons)

**Estimated LOC changed:** +10 (enum class), ~20 lines modified. Net new: ~10 LOC.

---

### Task 5: Rename `buf_a`/`buf_b` → `staging_buffer_a`/`staging_buffer_b`

**What:** Rename the cryptic buffer variables throughout the pipeline.

**Files to modify:**
- `tinyserve/generic_pipeline.py` (or `expert_pipeline.py` post-Task 1): constructor params (lines 348-349), attributes (lines 361-362), usage at lines 473, 714
- `tinyserve/offloaded_model.py` (or `_model_hooks.py` post-Task 2): lines 209-210 (`shared_buf_a`, `shared_buf_b`), lines 233-234 (kwargs)
- `tinyserve/expert_batcher.py`: line 96

**Rename map:**
- `buf_a` → `staging_buffer_a` (or `transfer_buffer` / `staging_buffer` — pick one)
- `buf_b` → `staging_buffer_b`
- `shared_buf_a` → `shared_staging_a`
- `shared_buf_b` → `shared_staging_b`

**Note:** The local variable `buf` inside methods (e.g., line 473 `buf = self.buf_a`) can stay as `buf` — it's a short-lived local. Only the constructor params, attributes, and cross-file references need renaming.

**How to verify:** `python -m pytest tests/ -x --tb=short`

**Estimated LOC changed:** ~15 lines modified. Net new: 0 LOC.

---

### Task 6: Rename `_cy_classify` and `_cy_group`

**What:** Replace abbreviated names with descriptive ones.

**File to modify:** `tinyserve/generic_pipeline.py` (or `expert_pipeline.py` post-Task 1)

**Rename map:**
- `_cy_classify` → `_cython_classify_hits` (lines 23-24, 525-526)
- `_cy_group` → `_cython_group_tokens` (lines 27-28, 441-442)
- `_TEMPLATE_STORAGE` → `_template_weight_storage` (lines 32, 42, 43, 51, 52)

**All usage sites (within the same file):**
- `_cy_classify`: lines 23-24 (import), 525 (if check)
- `_cy_group`: lines 27-28 (import), 441 (if check)
- `_TEMPLATE_STORAGE`: lines 32, 42, 43, 51, 52

**How to verify:** `python -m pytest tests/test_generic_pipeline.py -x --tb=short`

**Estimated LOC changed:** ~10 lines modified. Net new: 0 LOC.

---

### Task 7: Add `logger.warning()` to all silent fallback sites

**What:** Every `except Exception` that silently swallows errors gets a `logger.warning()` call.

**Sites to add warnings (file:line):**

In `tinyserve/offload.py`:
1. Line 115: `except Exception: pass` in `_register_flex_attention` (inner) → `logger.warning("FlexAttention mask interface registration failed", exc_info=True)`
2. Line 121: `except Exception: return "eager"` in `_register_flex_attention` (outer) → `logger.warning("FlexAttention registration failed, falling back to eager", exc_info=True)`
3. Line 156: `except Exception: pass` in `_register_sdpa_attention` (inner) → same pattern
4. Line 162: `except Exception: return "eager"` in `_register_sdpa_attention` (outer)
5. Line 211: `except Exception: pass` in `_register_flashinfer_attention` (inner)
6. Line 217: `except Exception: return "eager"` in `_register_flashinfer_attention` (outer)

In `tinyserve/generic_pipeline.py`:
7. Line 18: `except Exception: _get_expert_loop = lambda: None` → `logger.warning("C++ expert loop extension not available")`
8. Lines 23-24: `except ImportError: _cy_classify = None` — already `ImportError`, acceptable. Add: `logger.debug("Cython classify_hits_misses not available, using Python fallback")`
9. Lines 27-28: same pattern for `_cy_group`

In `tinyserve/offloaded_model.py`:
10. Line 23: `except Exception:` for triton_dot_scaled import → `logger.debug("triton dot_scaled not available")`
11. Line 28: `except Exception:` for triton_dequant import → `logger.debug("triton fused_dequant not available")`

In `tinyserve/generic_store.py`:
12. Line 406: `except Exception:` → need to read what this does

In `tinyserve/server.py`:
13. Line 221: `except Exception:` → `logger.warning("...", exc_info=True)`
14. Line 377: `except Exception:` → `logger.warning("...", exc_info=True)`
15. Line 404: `except Exception:` → `logger.warning("...", exc_info=True)`

In `tinyserve/gguf_reader.py`:
16. Line 85: `except Exception:` → `logger.warning("...", exc_info=True)`

In `tinyserve/int4_cache.py`:
17. Line 127: `except Exception:` → `logger.warning("...", exc_info=True)`

**Rule:** Use `logger.warning()` for runtime fallbacks that affect behavior. Use `logger.debug()` for optional-extension-not-available at import time.

**Ensure each file has:** `import logging` and `logger = logging.getLogger(__name__)` at the top (most already do).

**How to verify:** `python -m pytest tests/ -x --tb=short` (logging doesn't change behavior)

**Estimated LOC changed:** ~20 lines added/modified. Net new: ~20 LOC.

---

### Task 8: Remove dead code and `#noqa` suppressions

**What:** Clean up suppressions that mask real issues vs. necessary ones.

**Current `#noqa` sites:**
1. `tinyserve/offload.py:480` — `import flash_attn  # noqa: F401` — **KEEP** (side-effect import, F401 is correct)
2. `tinyserve/generic_pipeline.py:19` — `_get_expert_loop = lambda: None  # noqa: E731` — **FIX** by converting to `def _get_expert_loop(): return None`
3. `tinyserve/generic_pipeline.py:81` — `layout: "TensorLayout",  # noqa: F821` — **FIX** by adding proper `TYPE_CHECKING` import
4. `tinyserve/offloaded_model.py:325` — `# noqa: F821` — **FIX** same pattern
5. `tinyserve/offloaded_model.py:327` — `from .generic_store import TensorLayout  # noqa: F401` — **FIX** by moving to TYPE_CHECKING block

**Fix for #3 and #4:** Add at the top of each file:
```python
from __future__ import annotations  # already present in generic_store.py
```
OR use `TYPE_CHECKING`:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .generic_store import TensorLayout
```

**Dead code to look for:** Run `python -m pytest tests/ --collect-only 2>&1 | grep "no tests"` and check for unused functions. Also check: any functions in `offloaded_model.py` that are only used in tests (`_record_fate_prediction`, `_record_fate_outcome`, `reset_fate_stats`, `get_fate_accuracy_by_layer`) — these are diagnostic/test utilities, **KEEP** but could be behind `if TYPE_CHECKING` guard.

**How to verify:** `python -m pytest tests/ -x --tb=short && python -m flake8 tinyserve/ --select=E731,F821,F401 --exclude=__pycache__`

**Estimated LOC changed:** ~15 lines modified. Net new: ~5 LOC.

---

### Task 9: Prune scripts/ to essential files

**What:** Move the 4 essential scripts to the front, archive the rest.

**Essential scripts (KEEP in `scripts/`):**
1. `scripts/benchmark.py` (935 LOC) — the main benchmark
2. `scripts/autotune.py` (216 LOC) — auto-configuration
3. `scripts/calibrate_buddies.py` (81 LOC) — buddy expert calibration
4. `scripts/prompts.py` (94 LOC) — shared prompt data for benchmarks

**Move to `scripts/archived/`:**
- `bench_attention.py`, `bench_disk_offload.py`, `bench_flex_only.py`, `bench_kv_configs.py`, `bench_long_context.py`, `bench_qwen35.py`, `bench_with_buddies.py`, `cache_benchmark.py`, `comprehensive_bench.py`, `debug_bench.py`, `expert_similarity.py`, `sweep_cache_bias.py`, `validate_context_scaling.py`

**Move to `tests/` (see Task 14):**
- `scripts/test_dynamic_rebalance.py` → `tests/test_dynamic_rebalance.py`
- `scripts/test_qwen_122b.py` → `tests/test_qwen_122b.py`

**Delete:** `scripts/__init__.py` (scripts directory should not be a Python package)

**How to verify:** `python -m pytest tests/ -x --tb=short && python scripts/benchmark.py --help`

**Estimated LOC changed:** 0 code changes. File moves only. ~2600 LOC moved to archived/.

---

### Task 10: Move internal docs from `docs/` to `notes/`

**What:** `docs/` should contain only user-facing documentation. Internal planning docs go to `notes/`.

**Files to move:**
- `docs/gpt-oss-20b-architecture.md` → `notes/gpt-oss-20b-architecture.md`
- `docs/llama_cpp_rfc_draft.md` → `notes/llama_cpp_rfc_draft.md`
- `docs/quant_format_plan.md` → `notes/quant_format_plan.md`
- `docs/throughput_roadmap.md` → `notes/throughput_roadmap.md`
- `docs/throughput_roadmap_v2.md` → `notes/throughput_roadmap_v2.md`

**Keep in `docs/`:** `superpowers/` directory (it contains specs like this one).

**Create:** `notes/` directory at repo root. Add a one-line `notes/README.md`: "Internal planning and architecture notes. Not user-facing documentation."

**How to verify:** `ls docs/` shows only `superpowers/` + user-facing docs. `ls notes/` shows moved files.

**Estimated LOC changed:** 0 code changes. File moves only.

---

### Task 11: Trim `__init__.py` exports

**What:** Export only the public API. Remove internal class exports.

**Current exports (10 symbols):**
```python
from .chunked import chunked_prefill, generate_chunked
from .gguf_loader import load_from_gguf
from .offload import load_and_offload, offload_model
from .paged_kv_cache import PagedKVPool, PagedRequestKVCache
from .static_kv_cache import StaticKVCache
```

**Target exports (6 symbols — public API only):**
```python
from .offload import load_and_offload, offload_model
from .gguf_loader import load_from_gguf
from .chunked import chunked_prefill, generate_chunked
from .static_kv_cache import StaticKVCache
```

**Remove from `__init__.py`:**
- `PagedKVPool` — internal, used only by server.py
- `PagedRequestKVCache` — internal, used only by server.py

**Add (from Task 4):**
- `from .offload import AttentionBackend` (after Task 4 creates it)

**Add `__all__`:**
```python
__all__ = [
    "load_and_offload",
    "offload_model",
    "load_from_gguf",
    "chunked_prefill",
    "generate_chunked",
    "StaticKVCache",
    "AttentionBackend",
]
```

**Check for breakage:** Grep entire codebase for `from tinyserve import PagedKVPool` or `from tinyserve import PagedRequestKVCache`. If any exist outside `tinyserve/server.py`, update them to `from tinyserve.paged_kv_cache import ...`.

**How to verify:** `python -c "from tinyserve import load_and_offload, offload_model, load_from_gguf, StaticKVCache, chunked_prefill, generate_chunked"` succeeds.

**Estimated LOC changed:** ~5 lines modified, +8 for `__all__`. Net new: ~8 LOC.

**Dependency:** Partially depends on Task 4 (AttentionBackend export). Can be done without Task 4 — just leave AttentionBackend out of `__all__` and add it in Phase 2.

---

## Phase 2 — Depends on Phase 1

These tasks depend on one or more Phase 1 tasks being complete. Within Phase 2, tasks are independent of each other and can run in parallel.

---

### Task 12: Extract `ExpertCache` from `expert_store.py` into `expert_cache.py`

**Depends on:** Task 1 (files renamed to `expert_store.py`)

**What:** The `GenericLRUCache` class (now `ExpertCache` after Task 1) is 150+ lines starting at line 574 of `generic_store.py`. Extract it into its own file.

**Steps:**
1. Create `tinyserve/expert_cache.py` containing the `ExpertCache` class (everything from line 574 to end of file, ~250 LOC)
2. Move required imports: `torch`, `numpy`, `make_policy` from `cache_policy`
3. In `tinyserve/expert_store.py`, add: `from .expert_cache import ExpertCache` for backward compat
4. Update all direct importers of `GenericLRUCache` / `ExpertCache` from `expert_store` to import from `expert_cache` instead:
   - `tinyserve/expert_pipeline.py`: line 12
   - `tinyserve/offload.py`: line 323
   - All test files that import it (~10 files)
   - `scripts/benchmark.py`: line 274

**How to verify:** `python -m pytest tests/ -x --tb=short`

**Estimated LOC changed:** ~250 LOC moved (not new), ~15 import edits. Net new file: `expert_cache.py` (~250 LOC).

---

### Task 13: Create `TinyserveConfig` dataclass

**Depends on:** Task 3 (RoutingSpec), Task 4 (AttentionBackend)

**What:** Replace the 15+ parameters on `offload_model()` and `load_and_offload()` with a single config object.

**Create file:** `tinyserve/config.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
import torch

from .offload import AttentionBackend


@dataclass
class TinyserveConfig:
    device: str | torch.device = "cuda"
    cache_capacity: int = 0
    cache_policy: str = "lfru"
    cache_bias: float = 0.0
    fp8: bool = True
    adaptive_fate: bool = True
    max_seq_len: int = 0
    kv_dtype: torch.dtype = torch.bfloat16
    gpu_memory_utilization: float = 0.90
    attn_implementation: str | AttentionBackend | None = None
    disk_offload: bool = False
    ram_cache_gb: float = 0.0
    kv_offload: bool = False
    buddy_table_path: str | None = None
    imatrix_path: str | None = None
```

**Update `offload_model()` signature:**
```python
def offload_model(
    model: torch.nn.Module,
    config: TinyserveConfig | None = None,
    *,
    # Keep all old kwargs for backward compat
    device: str | torch.device = "cuda",
    cache_capacity: int = 0,
    ...
) -> torch.nn.Module:
```

The function body creates a `TinyserveConfig` from kwargs if `config` is None, then uses `config.device`, `config.cache_capacity`, etc.

**Same pattern for `load_and_offload()`** — add `config` as second positional arg, keep kwargs.

**Export:** Add `TinyserveConfig` to `__init__.py` and `__all__`.

**How to verify:** `python -m pytest tests/ -x --tb=short` (backward compatible — existing kwargs still work)

**Estimated LOC changed:** New file ~40 LOC, ~30 lines modified in `offload.py`. Net new: ~70 LOC.

---

### Task 14: Move `scripts/test_*.py` into `tests/`

**Depends on:** Task 9 (scripts pruning, so we don't double-move)

**What:** Test files belong in the test directory.

**Files to move:**
- `scripts/test_dynamic_rebalance.py` → `tests/test_dynamic_rebalance.py`
- `scripts/test_qwen_122b.py` → `tests/test_qwen_122b.py`

**After moving:** Check imports in each file. They likely use `from tinyserve import ...` which will work from `tests/`.

**How to verify:** `python -m pytest tests/test_dynamic_rebalance.py tests/test_qwen_122b.py --collect-only`

**Estimated LOC changed:** 0 code changes. File moves only.

---

### Task 15: Split `_execute_token_experts` into submethods

**Depends on:** Task 1 (file renamed to `expert_pipeline.py`), Task 5 (buf names), Task 6 (variable names)

**What:** The `_execute_token_experts` method (lines 493-702, ~210 lines) has 4 distinct phases. Extract each into a named method.

**Current structure (in `expert_pipeline.py`):**
1. **Lines 493-509:** No-cache fast path → extract as `_execute_uncached()`
2. **Lines 511-564:** Cache lookup + classify hits/misses → extract as `_classify_cache_hits()`
3. **Lines 566-609:** C++ fast path (all hits) → extract as `_execute_cpp_fast_path()`
4. **Lines 611-626:** Process hits → stays inline (short)
5. **Lines 628-696:** Process misses (CPU fallback, buddy, RAM cache) → extract as `_execute_cache_misses()`

**Method signatures:**

```python
def _classify_cache_hits(
    self, layer_idx: int, expert_ids: torch.Tensor | list[int],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Classify experts into cache hits and misses. Returns (hits, misses, expert_ids_list)."""

def _execute_cache_misses(
    self, h: torch.Tensor, output: torch.Tensor, tok_idx: int,
    layer_idx: int, expert_ids_list: list[int], weights: torch.Tensor,
    misses: list[int],
) -> None:
    """Handle cache misses via buddy substitution, CPU compute, or GPU pipeline."""
```

**Rule:** Each extracted method should be 30-60 lines. The parent `_execute_token_experts` becomes ~40 lines of orchestration.

**How to verify:** `python -m pytest tests/test_expert_pipeline.py tests/test_e2e_offload.py tests/test_cpu_miss_fallback.py -x --tb=short`

**Estimated LOC changed:** ~210 lines refactored (moved within file). Net new: ~15 LOC (method signatures + docstrings).

---

### Task 16: Create `OffloadedLM` wrapper class

**Depends on:** Task 2 (file renamed to `_model_hooks.py`), Task 13 (TinyserveConfig)

**What:** Replace monkey-patched attributes (`model._kv_cache`, `model._vram_budget`, `model._offload_pipelines`) with a typed wrapper.

**Create in `tinyserve/_model_hooks.py` (or new file `tinyserve/offloaded_lm.py`):**

```python
class OffloadedLM:
    """Typed wrapper around an offloaded HuggingFace model.
    
    Replaces monkey-patched attributes with proper typed access.
    Delegates __call__ and generate() to the wrapped model.
    """
    
    def __init__(self, model: torch.nn.Module):
        self._model = model
        self.kv_cache: StaticKVCache | None = None
        self.vram_budget: VRAMBudget | None = None
        self.pipelines: list[ExpertPipeline] = []
    
    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)
    
    @property
    def config(self):
        return self._model.config
    
    def cache_stats(self) -> dict:
        ...
```

**Update `offload_model()` return type:** `-> OffloadedLM` instead of `-> torch.nn.Module`.

**Update attribute assignments in `offload_model()`:**
```python
# Before:
model._kv_cache = kv_cache
model._vram_budget = budget
model._offload_pipelines = offloaded.pipelines

# After:
wrapper = OffloadedLM(model)
wrapper.kv_cache = kv_cache
wrapper.vram_budget = budget
wrapper.pipelines = offloaded.pipelines
return wrapper
```

**Backward compat:** `OffloadedLM.__getattr__` delegates to `self._model` for any attribute not on the wrapper, so existing code like `model.config` still works.

**How to verify:** `python -m pytest tests/ -x --tb=short`

**Estimated LOC changed:** New class ~60 LOC, ~20 lines modified in `offload.py`. Net new: ~80 LOC.

---

## Phase 3 — Depends on Phase 2

---

### Task 17: Rename tests to behavior-describe pattern

**Depends on:** Tasks 1, 2 (files renamed — test imports must be stable)

**What:** Rename test functions from implementation-description to behavior-description.

**Examples (from current test files):**

In `tests/test_expert_pipeline.py` (formerly `test_generic_pipeline.py`):
- `test_swap_weights_and_forward` → `test_expert_weights_swap_into_template_and_produce_output`
- `test_pipeline_runs_forward` → `test_pipeline_processes_single_token_through_experts`
- `test_pipeline_with_cache` → `test_cache_hit_skips_host_to_device_transfer`
- `test_pipeline_cache_eviction` → `test_cache_evicts_least_recently_used_when_full`

In `tests/test_expert_store.py` (formerly `test_generic_store.py`):
- `test_is_qtensor` → `test_qtensor_detection_for_mxfp4_weights`
- `test_store_basic` → `test_store_packs_and_retrieves_expert_weights`
- `test_lru_cache_basic` → `test_cache_returns_none_on_miss_and_slot_on_hit`

**Rule:** Each test name should answer "what behavior does this verify?" not "what function does this call?"

**Scope:** Rename functions in ALL test files. Do NOT change test logic — only the `def test_...` name.

**How to verify:** `python -m pytest tests/ -x --tb=short` (renaming functions doesn't change behavior)

**Estimated LOC changed:** ~100 lines modified (function names only). Net new: 0 LOC.

---

### Task 18: Add `tests/test_quickstart.py`

**Depends on:** Tasks 1-4 (stable API surface)

**What:** Create a test file that mirrors the README quickstart examples, serving as executable documentation.

**File to create:** `tests/test_quickstart.py`

**Contents should test (CPU-only, no GPU required):**
1. `from tinyserve import load_and_offload, offload_model, StaticKVCache` — import succeeds
2. `TinyserveConfig` can be instantiated with defaults
3. `AttentionBackend.EAGER` is a valid value
4. `RoutingSpec` is a valid namedtuple (if exported)
5. `offload_model` raises `ValueError` on `cache_capacity=-1`
6. `offload_model` raises `RuntimeError` when CUDA unavailable (mock torch.cuda.is_available)

**Rule:** These tests must run on CPU-only CI (no `@pytest.mark.skipif(not torch.cuda.is_available())`).

**How to verify:** `python -m pytest tests/test_quickstart.py -x --tb=short`

**Estimated LOC changed:** New file ~60 LOC.

---

### Task 19: Create `docs/architecture.md`

**Depends on:** Tasks 1, 2, 12 (file names must be final)

**What:** User-facing architecture overview with data flow description.

**File to create:** `docs/architecture.md`

**Contents:**
1. **Data Flow Diagram** (ASCII art):
   ```
   User code
     → load_and_offload() / offload_model()    [offload.py]
       → OffloadedModel.from_module()           [_model_hooks.py]
         → ExpertStore (CPU weight storage)     [expert_store.py]
         → ExpertPipeline (GPU dispatch)        [expert_pipeline.py]
         → ExpertCache (GPU VRAM LRU)           [expert_cache.py]
       → StaticKVCache (attention cache)        [static_kv_cache.py]
       → VRAMBudget (dynamic rebalancing)       [vram_budget.py]
   
   Inference (per token):
     hidden → Router → top-k expert IDs
       → ExpertCache.lookup()
         HIT  → forward_from_packed() (zero-copy)
         MISS → ExpertStore.copy_to_buffer() (H2D)
              → swap_weights_and_forward()
       → weighted sum → output
   ```

2. **Module Responsibilities** — one sentence per file
3. **Key Design Decisions** — why template-based dispatch, why FP8 on GPU, why CPU fallback

**How to verify:** File exists, renders as valid markdown.

**Estimated LOC:** ~80 LOC new file.

---

### Task 20: Create `docs/troubleshooting.md`

**Depends on:** Task 7 (warning messages are final — troubleshooting references them)

**What:** Document the 3 most common failures and their solutions.

**File to create:** `docs/troubleshooting.md`

**Contents:**
1. **"CUDA not available" RuntimeError** — install CUDA-enabled PyTorch
2. **OOM during cache allocation** — reduce `gpu_memory_utilization`, reduce `max_seq_len`, use `kv_offload=True`
3. **Slow first token (cold cache)** — use `imatrix_path` for cache seeding, enable `adaptive_fate=True`

**How to verify:** File exists, renders as valid markdown.

**Estimated LOC:** ~60 LOC new file.

---

## Summary

| Phase | Tasks | Parallelizable? | Total LOC impact |
|-------|-------|-----------------|------------------|
| 1     | 1-11  | All 11 in parallel | ~120 net new, ~250 modified |
| 2     | 12-16 | All 5 in parallel  | ~275 net new, ~80 modified |
| 3     | 17-20 | All 4 in parallel  | ~200 net new, ~100 modified |

**Total estimated effort:** ~595 net new LOC, ~430 lines modified, ~2600 LOC moved to archived scripts.

**Verification after each phase:**
```bash
python -m pytest tests/ -x --tb=short  # all tests pass
python -c "import tinyserve; print(tinyserve.__version__)"  # import works
python -m flake8 tinyserve/ --max-line-length=120  # no new lint errors
```

---

## Dependency Graph

```
Phase 1 (all independent):
  T1: rename generic_* → expert_*
  T2: rename offloaded_model → _model_hooks
  T3: RoutingSpec namedtuple
  T4: AttentionBackend enum
  T5: rename buf_a/buf_b
  T6: rename _cy_classify etc.
  T7: add logger.warning()
  T8: remove dead code / #noqa
  T9: prune scripts/
  T10: move docs → notes/
  T11: trim __init__.py exports

Phase 2 (depends on Phase 1):
  T12: extract ExpertCache         ← T1
  T13: TinyserveConfig dataclass   ← T3, T4
  T14: move scripts/test_*         ← T9
  T15: split _execute_token_experts ← T1, T5, T6
  T16: OffloadedLM wrapper         ← T2, T13

Phase 3 (depends on Phase 2):
  T17: rename test functions       ← T1, T2
  T18: test_quickstart.py          ← T1, T2, T3, T4
  T19: docs/architecture.md        ← T1, T2, T12
  T20: docs/troubleshooting.md     ← T7
```
