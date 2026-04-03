"""Deprecated: use tinyserve.expert_store instead."""
import warnings
warnings.warn(
    "tinyserve.generic_store is deprecated, use tinyserve.expert_store",
    DeprecationWarning, stacklevel=2,
)
from tinyserve.expert_store import *  # noqa: F401,F403
