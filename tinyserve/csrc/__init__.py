"""JIT-compiled C++ extensions for tinyserve."""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)

_expert_loop = None
HAS_CPP_LOOP: bool = False


def _load_expert_loop():
    global _expert_loop, HAS_CPP_LOOP
    if _expert_loop is not None:
        return _expert_loop

    try:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(__file__), "expert_loop.cpp")
        _expert_loop = load(
            name="expert_loop",
            sources=[src],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
        HAS_CPP_LOOP = True
        _logger.info("C++ expert_loop extension compiled and loaded")
    except Exception as exc:
        _logger.debug("C++ expert_loop extension unavailable: %s", exc)
        HAS_CPP_LOOP = False
        _expert_loop = None

    return _expert_loop


def get_expert_loop():
    """Return the compiled expert_loop module, or None if unavailable."""
    if _expert_loop is None and not HAS_CPP_LOOP:
        _load_expert_loop()
    return _expert_loop
