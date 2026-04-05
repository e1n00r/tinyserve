# Backward-compat shim — replaced by engine.py. Removed in Task 12.
from .engine import *  # noqa: F401, F403
from .engine import (  # noqa: F401 - explicit re-export of private symbols accessed by tests
    _ROUTING_MAP,
    _register_flex_attention,
    _register_sdpa_attention,
    _register_flashinfer_attention,
)
