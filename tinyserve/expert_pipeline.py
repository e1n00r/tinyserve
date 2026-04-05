# Backward-compat shim — removed in Task 12
from .expert_execution import (  # noqa: F401
    ExpertPipeline,
    _build_cpp_layout_args,
    _build_gpu_int4_forward,
    _build_inline_forward,
    forward_from_packed,
    swap_weights_and_forward,
)
