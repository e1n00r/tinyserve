# Backward-compat shim — removed in Task 12
from .gguf_model_loader import *  # noqa: F401, F403
from .gguf_model_loader import (  # noqa: F401 (re-export underscore names not covered by *)
    _build_expert_store_from_fused_reader,
    _dequant_fused_tensor,
    _dequant_tensor,
    _find_tensor_info,
    _get_param,
    _set_param,
)
