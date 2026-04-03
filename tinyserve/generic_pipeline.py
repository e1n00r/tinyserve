"""Deprecated: use tinyserve.expert_pipeline instead."""
import warnings
warnings.warn(
    "tinyserve.generic_pipeline is deprecated, use tinyserve.expert_pipeline",
    DeprecationWarning, stacklevel=2,
)
from tinyserve.expert_pipeline import *  # noqa: F401,F403
