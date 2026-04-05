# Backward-compat shim — removed in Task 12
from .model_hooks import *  # noqa: F401, F403
from .model_hooks import (  # noqa: F401
    _mxfp4_linear,
    _FusedExpertTemplate,
    _record_fate_outcome,
    _record_fate_prediction,
    _install_offloaded_forward,
    _extract_routing_fn,
    _extract_fate_fn,
    _make_template,
    _fate,
    _MXFP4_BACKEND,
)
