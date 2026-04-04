"""Unified offloaded MoE model wrapper.

Takes any nn.Module with MoE layers, extracts expert weights to CPU,
installs offloaded dispatch hooks, moves non-expert weights to GPU.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert_forward import _QUICK_GELU_COEFF
from .expert_pipeline import ExpertPipeline
from .expert_store import ExpertStore, _is_qtensor
from .mxfp4 import dequant_mxfp4_no_transpose

if TYPE_CHECKING:
    from .expert_store import TensorLayout
from .profiler import OffloadProfiler

logger = logging.getLogger(__name__)

_MXFP4_BACKEND = "pytorch"
try:
    from .triton_dot_scaled import dot_scaled_vecmat as _dot_scaled_vecmat

    _MXFP4_BACKEND = "dot_scaled"
except ImportError:
    logger.debug("triton dot_scaled not available")
    try:
        from .triton_dequant import fused_dequant_vecmat as _fused_dequant_vecmat

        _MXFP4_BACKEND = "triton_sw"
    except ImportError:
        logger.debug("triton fused_dequant not available, using pytorch backend")


# Per-layer FATE accuracy tracking.
# Maps layer_idx -> {"predictions": int, "hits": int}.
# Hits counts how many actual top-k experts appeared in the predicted set.
# Single-threaded inference — no locks needed.
_fate_stats: dict[int, dict] = {}
# Stores the last FATE prediction made BY layer N FOR layer N+1.
# Key = target layer idx (N+1), value = predicted set.
_fate_pending: dict[int, set] = {}

# Temporal routing cache: layer_idx -> expert_ids chosen on the previous token.
# Populated during offloaded_forward after routing; cleared by reset_temporal_routing().
_last_routing: dict[int, list[int]] = {}


def reset_temporal_routing() -> None:
    """Clear the temporal routing cache. Call before each benchmark run."""
    _last_routing.clear()


def reset_fate_stats() -> None:
    """Clear all FATE accuracy statistics. Call before each timed run."""
    _fate_stats.clear()
    _fate_pending.clear()


def get_fate_accuracy_by_layer() -> dict[int, dict]:
    """Return per-layer FATE accuracy.

    Returns:
        {layer_idx: {"predictions": int, "hits": int, "accuracy": float}}
    """
    result = {}
    for layer_idx, stats in _fate_stats.items():
        predictions = stats["predictions"]
        hits = stats["hits"]
        accuracy = hits / predictions if predictions > 0 else 0.0
        result[layer_idx] = {
            "predictions": predictions,
            "hits": hits,
            "accuracy": accuracy,
        }
    return result


def _record_fate_prediction(target_layer_idx: int, predicted_set: set) -> None:
    """Store a FATE prediction for target_layer_idx made by the previous layer."""
    _fate_pending[target_layer_idx] = predicted_set


def _record_fate_outcome(layer_idx: int, actual_indices: list[int]) -> None:
    """Compare actual expert indices against the pending FATE prediction for layer_idx."""
    if layer_idx not in _fate_pending:
        return
    predicted = _fate_pending.pop(layer_idx)
    actual_set = set(actual_indices)
    hits = len(actual_set & predicted)
    if layer_idx not in _fate_stats:
        _fate_stats[layer_idx] = {"predictions": 0, "hits": 0}
    _fate_stats[layer_idx]["predictions"] += len(actual_set)
    _fate_stats[layer_idx]["hits"] += hits


def _mxfp4_linear(
    x: torch.Tensor,
    blocks: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.shape[0] == 1 and _MXFP4_BACKEND == "dot_scaled":
        return _dot_scaled_vecmat(x, blocks, scales, bias)
    if x.shape[0] == 1 and _MXFP4_BACKEND == "triton_sw":
        return _fused_dequant_vecmat(x, blocks, scales, bias)
    w = dequant_mxfp4_no_transpose(blocks, scales, x.dtype)
    return F.linear(x, w, bias.to(x.dtype) if bias is not None else None)


class OffloadedModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        pipelines: list[ExpertPipeline],
    ):
        super().__init__()
        self.model = model
        self.pipelines = pipelines

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def cache_stats(self) -> dict:
        total_hits = 0
        total_misses = 0
        for pipeline in self.pipelines:
            if pipeline.cache is not None:
                total_hits += pipeline.cache.hits
                total_misses += pipeline.cache.misses
        return {"total_hits": total_hits, "total_misses": total_misses}

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        moe_block_attr: str,
        expert_list_attr: str,
        router_attr: str,
        top_k: int,
        device: torch.device,
        cache_capacity: int = 0,
        returns_router_logits: bool = False,
        softmax_order: str = "topk_then_softmax",
        first_moe_layer: int = 0,
        model_id: str | None = None,
        cache_policy: str = "lfru",
        fp8: bool = True,
        adaptive_fate: bool = True,
        profiler: OffloadProfiler | None = None,
        disk_offload: bool = False,
        ram_cache_gb: float = 0,
    ) -> OffloadedModel:
        model.eval()
        layers = model.layers

        moe_layers = [
            (li, layer)
            for li, layer in enumerate(layers)
            if li >= first_moe_layer and hasattr(getattr(layer, moe_block_attr, None), expert_list_attr)
        ]

        first_container = getattr(getattr(moe_layers[0][1], moe_block_attr), expert_list_attr)

        ram_cache = None
        cpu_expert = None
        if model_id is not None:
            # Native quantized path: load expert weights directly from safetensors.
            # Non-expert weights (attention, norms, etc.) are taken from `model`.
            layer_indices = [li for li, _ in moe_layers]

            # Phase 1: Zero HF expert weights BEFORE loading MXFP4 store.
            # HF loads all weights including experts as BF16 (~8 GB). Zeroing
            # them first eliminates peak RAM overlap with the MXFP4 store.
            for _, layer in moe_layers:
                container = getattr(getattr(layer, moe_block_attr), expert_list_attr)
                for param in container.parameters():
                    param.data = torch.empty(0, device="cpu")
            import gc

            gc.collect()

            if disk_offload:
                store, _, ram_cache = ExpertStore.from_safetensors(
                    model_id,
                    moe_block_attr,
                    expert_list_attr,
                    layer_indices,
                    disk_offload=True,
                    ram_cache_gb=ram_cache_gb,
                )
                # Create CPUExpertForward from the store layout.
                from .cpu_expert import CPUExpertForward

                try:
                    cpu_expert = CPUExpertForward(store.layout)
                except ValueError:
                    cpu_expert = None
            else:
                store, _ = ExpertStore.from_safetensors(
                    model_id,
                    moe_block_attr,
                    expert_list_attr,
                    layer_indices,
                )
            template = _FusedExpertTemplate.from_layout(store.layout, first_container)
            template = template.to(device)
            if not template._is_mxfp4:
                template = template.to(torch.bfloat16)
            else:
                for name in template._param_names:
                    param = getattr(template, name)
                    if param.dtype.is_floating_point:
                        param.data = param.data.to(torch.bfloat16)
        else:
            # Standard path: extract weights from the model's current parameters.
            template = _make_template(first_container, device)
            store, _ = ExpertStore.build(moe_layers, moe_block_attr, expert_list_attr, fp8=fp8)

        shared_staging_a = store.allocate_buffer(device)  # always BF16 layout
        shared_staging_b = store.allocate_buffer(device)
        transfer_stream = torch.cuda.Stream(device)
        compute_stream = torch.cuda.Stream(device)
        shared_stream = torch.cuda.Stream(device)
        # One shared FP8 staging buffer for FATE prefetch — 12MB, one active at a time.
        shared_prefetch_fp8_stage = (
            torch.empty(store.expert_bytes, dtype=torch.uint8, device=device) if store._fp8 else None
        )

        # Cache is attached later (after the full model.to(device) in offload_model)
        # so that auto-sizing sees the real remaining VRAM. Pipelines start with cache=None.
        # Pass 1: build pipelines and extract route functions (needed for FATE).
        pipelines = []
        routes = []
        fate_routes = []
        moe_blocks = []
        fate_top_k = top_k + 1
        for store_idx, (li, layer) in enumerate(moe_layers):
            moe_block = getattr(layer, moe_block_attr)
            pipeline = ExpertPipeline(
                store,
                template,
                device,
                staging_buffer_a=shared_staging_a,
                staging_buffer_b=shared_staging_b,
                transfer_stream=transfer_stream,
                compute_stream=compute_stream,
                cache=None,
                shared_stream=shared_stream,
                ram_cache=ram_cache,
                cpu_expert=cpu_expert,
            )
            pipeline._prefetch_fp8_stage = shared_prefetch_fp8_stage
            pipeline.profiler = profiler
            if cpu_expert is not None:
                pipeline.cpu_on_miss = True
            pipelines.append(pipeline)
            moe_blocks.append(moe_block)
            routes.append(_extract_routing_fn(moe_block, router_attr, top_k, softmax_order))
            fate_routes.append(_extract_fate_fn(moe_block, router_attr, fate_top_k, softmax_order))

        # Pass 2: install forwards — each layer gets a reference to the NEXT layer's
        # route so it can predict and prefetch the next layer's experts (FATE).
        # fate_fn uses top_k+1 to capture border-case experts and improve hit rate.
        for store_idx, moe_block in enumerate(moe_blocks):
            next_pipeline = pipelines[store_idx + 1] if store_idx + 1 < len(pipelines) else None
            next_route = routes[store_idx + 1] if store_idx + 1 < len(routes) else None
            next_fate_fn = fate_routes[store_idx + 1] if store_idx + 1 < len(fate_routes) else None
            _install_offloaded_forward(
                moe_block,
                pipelines[store_idx],
                store_idx,
                router_attr,
                top_k,
                returns_router_logits=returns_router_logits,
                softmax_order=softmax_order,
                next_pipeline=next_pipeline,
                next_route=next_route,
                next_fate_fn=next_fate_fn,
                adaptive_fate=adaptive_fate,
                is_first_layer=(store_idx == 0),
                all_pipelines=pipelines,
            )

        return cls(model, pipelines), store, cache_capacity, cache_policy


def _make_template(expert_container, device):
    """Create a template expert module for weight swapping."""
    if isinstance(expert_container, nn.ModuleList):
        return copy.deepcopy(expert_container[0]).to(device).to(torch.bfloat16)
    template = _FusedExpertTemplate(expert_container).to(device)
    if template._is_mxfp4:
        # uint8 blocks/scales must stay uint8; only cast floating-point params.
        for name in template._param_names:
            param = getattr(template, name)
            if param.dtype.is_floating_point:
                param.data = param.data.to(torch.bfloat16)
    else:
        template = template.to(torch.bfloat16)
    return template


class _FusedExpertTemplate(nn.Module):
    """Template for fused-parameter experts (GPT-OSS, Qwen3.5).

    Creates nn.Parameter placeholders matching the per-expert slice shapes.
    Detects activation type and weight layout from the original container.
    Supports both bf16 and MXFP4-quantized weight layouts.
    """

    def __init__(self, fused_container: nn.Module):
        super().__init__()
        self._param_names = []
        for name, param in fused_container.named_parameters():
            if _is_qtensor(param):
                # Expand QTensor into int_data (blocks) and scale components.
                int_data = param.int_data
                scale = param.scale
                self.register_parameter(name, nn.Parameter(torch.zeros(int_data.shape[1:], dtype=int_data.dtype)))
                self.register_parameter(
                    name + "_scales",
                    nn.Parameter(torch.zeros(scale.shape[1:], dtype=scale.dtype)),
                )
                self._param_names.extend([name, name + "_scales"])
            else:
                per_expert_shape = param.shape[1:]
                self.register_parameter(name, nn.Parameter(torch.zeros(per_expert_shape, dtype=param.dtype)))
                self._param_names.append(name)

        self._act_fn = getattr(fused_container, "act_fn", None)
        self._has_bias = "gate_up_proj_bias" in self._param_names
        self._is_mxfp4 = "gate_up_proj_scales" in self._param_names

    @classmethod
    def from_layout(cls, layout: TensorLayout, fused_container: nn.Module) -> _FusedExpertTemplate:
        """Create template from a TensorLayout (used with native safetensors loading)."""

        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj._param_names = []
        for name, (shape, dtype) in layout.specs.items():
            data = torch.zeros(shape, dtype=dtype)
            if dtype.is_floating_point:
                obj.register_parameter(name, nn.Parameter(data))
            else:
                obj.register_parameter(name, nn.Parameter(data, requires_grad=False))
            obj._param_names.append(name)
        obj._act_fn = getattr(fused_container, "act_fn", None)
        obj._has_bias = "gate_up_proj_bias" in obj._param_names
        obj._is_mxfp4 = "gate_up_proj_scales" in obj._param_names
        return obj

    def forward(self, hidden_states):
        params = {name: getattr(self, name) for name in self._param_names}

        if self._is_mxfp4:
            gate_up = _mxfp4_linear(
                hidden_states,
                params["gate_up_proj"],
                params["gate_up_proj_scales"],
                params.get("gate_up_proj_bias"),
            )
            gate = gate_up[..., ::2].clamp(max=7.0)
            up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
            gated = (up + 1) * gate * torch.sigmoid(gate * _QUICK_GELU_COEFF)
            return _mxfp4_linear(
                gated,
                params["down_proj"],
                params["down_proj_scales"],
                params.get("down_proj_bias"),
            )

        w_gu = params["gate_up_proj"]
        if w_gu.shape[0] == hidden_states.shape[-1]:
            w_gu = w_gu.t()
        gate_up = nn.functional.linear(hidden_states, w_gu, params.get("gate_up_proj_bias"))

        if self._act_fn is not None:
            gate, up = gate_up.chunk(2, dim=-1)
            gated = self._act_fn(gate) * up
        else:
            gate = gate_up[..., ::2].clamp(max=7.0)
            up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
            gated = (up + 1) * gate * torch.sigmoid(gate * _QUICK_GELU_COEFF)

        w_dn = params["down_proj"]
        if w_dn.shape[0] == gated.shape[-1]:
            w_dn = w_dn.t()
        return nn.functional.linear(gated, w_dn, params.get("down_proj_bias"))


def _extract_fate_fn(moe_block, router_attr, fate_top_k, softmax_order):
    """Build a FATE prediction function using fate_top_k (typically top_k + 1).

    Identical to _extract_routing_fn but substitutes fate_top_k for top_k so
    that one extra candidate expert is included in each prefetch batch.
    """
    return _extract_routing_fn(moe_block, router_attr, fate_top_k, softmax_order)


def _extract_routing_fn(moe_block, router_attr, top_k, softmax_order):
    """Build a routing function matching the model's original routing logic."""
    router = getattr(moe_block, router_attr)
    renormalize = getattr(moe_block, "norm_topk_prob", True)

    if softmax_order == "router_native":

        def route(hidden_states):
            result = router(hidden_states)
            route.last_logits = None

            if isinstance(result, torch.Tensor):
                router_logits = result
                routing_weights = torch.sigmoid(router_logits)
                routing_weights, top_idx = torch.topk(routing_weights, top_k, dim=-1)
                return top_idx, routing_weights.to(hidden_states.dtype)

            if len(result) == 3:
                router_logits, routing_weights, top_idx = result
                route.last_logits = router_logits
                return top_idx, routing_weights.to(hidden_states.dtype)

            first, second = result
            if second.dtype in (torch.int32, torch.int64):
                routing_weights_full, top_idx = first, second
                if routing_weights_full.shape[-1] > top_k:
                    routing_weights = routing_weights_full.gather(-1, top_idx)
                else:
                    routing_weights = routing_weights_full
            else:
                top_idx, routing_weights = first, second
            return top_idx, routing_weights.to(hidden_states.dtype)
    elif softmax_order == "softmax_then_topk":

        def route(hidden_states):
            router_logits = router(hidden_states)
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_weights, top_idx = torch.topk(routing_weights, top_k, dim=-1)
            if renormalize:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            route.last_logits = router_logits
            return top_idx, routing_weights.to(hidden_states.dtype)
    else:

        def route(hidden_states):
            router_logits = router(hidden_states)
            top_vals, top_idx = torch.topk(router_logits, top_k, dim=-1)
            routing_weights = F.softmax(top_vals, dim=-1).to(hidden_states.dtype)
            route.last_logits = router_logits
            return top_idx, routing_weights

    route.last_logits = None
    return route


def _install_offloaded_forward(
    moe_block,
    pipeline,
    layer_idx,
    router_attr,
    top_k,
    returns_router_logits: bool = False,
    softmax_order: str = "topk_then_softmax",
    next_pipeline=None,
    next_route=None,
    next_fate_fn=None,
    adaptive_fate: bool = True,
    is_first_layer: bool = False,
    all_pipelines=None,
):
    route = _extract_routing_fn(moe_block, router_attr, top_k, softmax_order)

    shared_expert = getattr(moe_block, "shared_experts", None) or getattr(moe_block, "shared_expert", None)

    import contextlib as _contextlib

    @_contextlib.contextmanager
    def _maybe_phase(prof, name):
        if prof is not None:
            with prof.phase(name):
                yield
        else:
            yield

    def offloaded_forward(hidden_states, **_kwargs):
        _prof = pipeline.profiler

        if hidden_states.dim() == 3:
            batch, seq_len, hidden = hidden_states.shape
            flat = hidden_states.view(-1, hidden)
        else:
            flat = hidden_states
            batch, seq_len = None, None

        # Notify Least-Stale policy that a new token pass is starting.
        # Only on the first MoE layer (layer_idx == 0) for single-token decode
        # so the rotation happens exactly once per token.
        if flat.shape[0] == 1 and layer_idx == 0 and pipeline.cache is not None:
            pipeline.cache.begin_pass()

        top_idx, routing_weights = route(flat)

        # FATE accuracy recording — diagnostic only, skipped in production.
        # _record_fate_outcome requires .tolist() (CUDA sync). Only enable
        # when _fate_stats is actively being collected (--fate-diagnostic).
        if _fate_pending and flat.shape[0] == 1:
            _record_fate_outcome(layer_idx, top_idx[0].tolist())

        # Cache-aware routing bias (ExpertFlow, arxiv 2510.26730):
        # Re-route using biased logits that favour GPU-resident experts, then
        # re-compute weights. Only for single-token decode; skip if no logits
        # are available (router_native path that returns indices directly).
        if (
            flat.shape[0] == 1
            and pipeline.cache is not None
            and pipeline.cache_bias > 0.0
            and route.last_logits is not None
        ):
            logits = route.last_logits  # [1, num_experts]
            bias = torch.zeros_like(logits)
            for eid in range(logits.shape[-1]):
                if pipeline.cache.contains(layer_idx, eid):
                    bias[0, eid] = pipeline.cache_bias
            biased = logits + bias
            # Re-derive top_idx and routing_weights from biased logits using
            # the same softmax/topk order the original route function used.
            routing_weights, top_idx = torch.topk(F.softmax(biased, dim=-1, dtype=torch.float), top_k, dim=-1)
            routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).to(flat.dtype)

        shared_event = None
        shared_out = None
        if shared_expert is not None:
            with torch.cuda.stream(pipeline.shared_stream):
                shared_out = shared_expert(flat)
                shared_event = torch.cuda.Event()
                shared_event.record(pipeline.shared_stream)

        if flat.shape[0] > 1:
            output = pipeline.execute_layer_experts_batched(flat, layer_idx, top_idx, routing_weights)
        else:
            output = pipeline.execute_layer_experts(flat, layer_idx, top_idx, routing_weights)

        if shared_event is not None:
            torch.cuda.current_stream().wait_event(shared_event)
            output = output + shared_out

        # Store actual routing for temporal locality prediction (used when adaptive_fate=True).
        # Must happen before the prefetch decision below so _last_routing is always current.
        # Store routing as GPU tensor — no .tolist() sync on the hot path.
        # schedule_prefetch handles conversion when it runs (overlapped with attention).
        if flat.shape[0] == 1:
            _last_routing[layer_idx] = top_idx[0]

        # FATE cross-layer prefetch (arxiv 2502.12224): adjacent layer gate inputs
        # have >83% cosine similarity, so running the NEXT layer's gate on the
        # CURRENT hidden states predicts next-layer expert selection with ~97%
        # accuracy. Prefetch those experts NOW — H2D overlaps with this layer's
        # attention compute and the next layer's attention compute (~15-30ms window).
        # next_fate_fn uses top_k+1 to cover border-case experts and reduce miss rate.
        #
        # Adaptive mode (adaptive_fate=True): whenever prior-token routing is available
        # for the target layer, use it (temporal locality). Fall back to FATE only on
        # the very first token (when _last_routing is empty for that layer). Temporal
        # prediction is at least as good as FATE everywhere: ~99%+ on layers where FATE
        # is 100%, and clearly better on layers where FATE degrades.
        _fate_fn = next_fate_fn if next_fate_fn is not None else next_route
        if flat.shape[0] == 1 and next_pipeline is not None and _fate_fn is not None:
            target_layer = layer_idx + 1
            use_temporal = adaptive_fate and target_layer in _last_routing
            if use_temporal:
                predicted = _last_routing[target_layer]  # GPU tensor
            else:
                with _maybe_phase(_prof, "fate_gate"), torch.no_grad():
                    fate_idx, _ = _fate_fn(flat)
                predicted = fate_idx[0]  # GPU tensor
            with _maybe_phase(_prof, "schedule_prefetch"):
                next_pipeline.schedule_prefetch(target_layer, predicted)
            if _fate_pending:
                predicted_list = predicted.tolist() if isinstance(predicted, torch.Tensor) else predicted
                _record_fate_prediction(target_layer, set(predicted_list))

        # Multi-hop prefetch: layers N+2, N+3 using temporal routing.
        # Only fires during single-token decode when prior routing exists.
        if flat.shape[0] == 1 and all_pipelines is not None:
            for hop in range(2, 4):
                target = layer_idx + hop
                if target < len(all_pipelines) and target in _last_routing:
                    with _maybe_phase(_prof, "schedule_prefetch"):
                        all_pipelines[target].schedule_prefetch(target, _last_routing[target])

        if batch is not None:
            output = output.view(batch, seq_len, -1)

        if returns_router_logits:
            return output, route.last_logits
        return output

    moe_block.forward = offloaded_forward
