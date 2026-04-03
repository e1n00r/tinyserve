# SPDX-License-Identifier: Apache-2.0
"""Quickstart smoke tests — mirrors README examples, no GPU required."""
import pytest


def test_public_api_importable():
    from tinyserve import load_and_offload, offload_model, load_from_gguf
    assert callable(load_and_offload)
    assert callable(offload_model)
    assert callable(load_from_gguf)


def test_config_defaults():
    from tinyserve import TinyserveConfig
    cfg = TinyserveConfig()
    assert cfg.cache_capacity == 0
    assert cfg.cache_policy == "lfru"
    assert cfg.gpu_memory_utilization == 0.90


def test_offloaded_lm_importable():
    from tinyserve import OffloadedLM
    assert callable(OffloadedLM)
