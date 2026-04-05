# SPDX-License-Identifier: Apache-2.0
"""Quickstart smoke tests — mirrors README examples, no GPU required."""


def test_public_api_importable():
    from tinyserve import load_and_offload, load_from_gguf, offload_model

    assert callable(load_and_offload)
    assert callable(offload_model)
    assert callable(load_from_gguf)


def test_config_defaults():
    from tinyserve import OffloadConfig, TinyserveConfig

    cfg = OffloadConfig()
    assert cfg.expert_cache_slots == 0
    assert cfg.eviction_policy == "lfru"
    assert cfg.vram_fraction == 0.90

    assert TinyserveConfig is OffloadConfig


def test_offloaded_lm_importable():
    from tinyserve import OffloadedLM

    assert callable(OffloadedLM)
