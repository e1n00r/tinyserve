# SPDX-License-Identifier: Apache-2.0
"""Vocabulary test: engine.py OffloadConfig and field renames."""

from tinyserve.engine import OffloadConfig, TinyserveConfig


def test_offload_config_new_fields():
    cfg = OffloadConfig(
        expert_cache_slots=32,
        eviction_policy="lfru",
        temporal_prefetch=True,
        vram_fraction=0.90,
        compress_weights=True,
        max_context_tokens=4096,
        sliding_window_kv=False,
        kv_window_tokens=1024,
        kv_sink_tokens=4,
    )
    assert cfg.expert_cache_slots == 32
    assert cfg.eviction_policy == "lfru"


def test_tinyserve_config_alias():
    cfg = TinyserveConfig(expert_cache_slots=8)
    assert isinstance(cfg, OffloadConfig)
