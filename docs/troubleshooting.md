# Troubleshooting

## 1. OOM During Model Loading

**Symptom:** `torch.cuda.OutOfMemoryError` or the process is killed while calling
`load_and_offload` or `offload_model`.

**Cause:** tinyserve reserves `gpu_memory_utilization × total_VRAM` for experts and
KV cache. If the non-expert parts of the model (embeddings, attention, LM head) already
consume more VRAM than expected, the expert cache allocation pushes total usage over the
hardware limit.

**Fixes:**

- Lower `gpu_memory_utilization` (default `0.90`):

  ```python
  from tinyserve import TinyserveConfig, load_and_offload

  cfg = TinyserveConfig(gpu_memory_utilization=0.75)
  model = load_and_offload("path/to/model", cfg)
  ```

- Enable disk offload to eliminate RAM pressure from the expert store:

  ```python
  cfg = TinyserveConfig(disk_offload=True, gpu_memory_utilization=0.80)
  ```

- Use FP8 for experts (enabled by default, `fp8=True`) — if disabled, re-enable it to
  halve expert storage.

---

## 2. Slow Inference / Low Token Rate

**Symptom:** Tokens per second are far below the expected range (9–27 tok/s on an 8 GB
card) or drop significantly after the first few tokens.

**Cause:** A low cache hit rate means most experts are transferred from CPU RAM (or
disk) every token, saturating the PCIe bus. Each miss costs ~20 ms on a cold mmap page.

**Diagnosis:** Check hit rate via the cache stats:

```python
stats = model.pipelines[0].cache.get_layer_stats()
hits   = sum(s["hits"]   for s in stats.values())
misses = sum(s["misses"] for s in stats.values())
print(f"Hit rate: {hits / max(hits + misses, 1):.1%}")
```

A hit rate below ~50% on diverse text or below ~70% on repetitive text indicates the
cache is too small.

**Fixes:**

- Increase `cache_capacity` (number of expert weight blocks to keep in VRAM):

  ```python
  cfg = TinyserveConfig(cache_capacity=200)
  ```

  Set `cache_capacity=0` (default) to let tinyserve auto-size the cache to fill
  available VRAM after non-expert modules are placed.

- Keep the default `cache_policy="lfru"`. LFRU outperforms `"lru"` on decode-heavy
  workloads because it does not evict frequently reused experts that were displaced
  briefly during prefill.

- Enable RAM caching to avoid mmap page faults (useful when expert store is on disk):

  ```python
  cfg = TinyserveConfig(disk_offload=True, ram_cache_gb=4.0)
  ```

---

## 3. Unsupported Model

**Symptom:** `KeyError`, `AttributeError`, or `NotImplementedError` when loading a new
model family, or expert hooks are not applied (all weights stay on CPU and no offloading
occurs).

**Cause:** tinyserve uses `model_registry.py` to map HuggingFace model config class
names to the hooks that intercept MoE router and expert calls. A model not in the
registry will not have its experts offloaded.

**Fix:** Check which architectures are registered:

```python
from tinyserve.model_registry import MODEL_REGISTRY
print(list(MODEL_REGISTRY.keys()))
```

To add support for a new model, register its config class name and the extractor
functions that identify its MoE layers:

```python
from tinyserve.model_registry import register_model

register_model(
    config_class="YourModelConfig",
    moe_layer_cls="YourMoELayer",
    expert_attr="mlp",        # attribute on the MoE layer holding experts
    router_attr="gate",       # attribute holding the router
)
```

Refer to existing entries in `tinyserve/model_registry.py` for the required signature.
Open an issue on the repository if the model is a common open-source architecture that
should be supported out of the box.
