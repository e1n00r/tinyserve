"""Tests for imatrix parsing and expert cache seeding."""

from __future__ import annotations

import struct
import tempfile

import torch

from tests.conftest import requires_cuda


def _make_imatrix_dat(entries: list[tuple[str, int, int]]) -> bytes:
    """Build a synthetic imatrix .dat binary blob.

    Args:
        entries: list of (name, ncall, nval). nval float32 values are zeros.
    """
    buf = struct.pack("<i", len(entries))
    for name, ncall, nval in entries:
        name_bytes = name.encode("utf-8")
        buf += struct.pack("<i", len(name_bytes))
        buf += name_bytes
        buf += struct.pack("<i", ncall)
        buf += struct.pack("<i", nval)
        buf += struct.pack(f"<{nval}f", *([0.0] * nval))
    return buf


def _write_imatrix_dat(entries: list[tuple[str, int, int]]) -> str:
    """Write synthetic imatrix .dat to a temp file, return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
    tmp.write(_make_imatrix_dat(entries))
    tmp.flush()
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# test_parse_imatrix_dat
# ---------------------------------------------------------------------------


def test_parse_imatrix_dat_basic():
    """Parsing returns correct name-to-ncall mapping."""
    from tinyserve.imatrix import parse_imatrix_dat

    entries = [
        ("blk.0.ffn_gate.0.weight", 100, 4),
        ("blk.0.ffn_gate.1.weight", 50, 4),
        ("blk.1.ffn_gate.0.weight", 200, 8),
    ]
    path = _write_imatrix_dat(entries)
    counts = parse_imatrix_dat(path)

    assert counts["blk.0.ffn_gate.0.weight"] == 100
    assert counts["blk.0.ffn_gate.1.weight"] == 50
    assert counts["blk.1.ffn_gate.0.weight"] == 200
    assert len(counts) == 3


def test_parse_imatrix_dat_empty():
    """Empty imatrix file (zero entries) parses to empty dict."""
    from tinyserve.imatrix import parse_imatrix_dat

    path = _write_imatrix_dat([])
    counts = parse_imatrix_dat(path)
    assert counts == {}


def test_parse_imatrix_dat_multiple_projections():
    """gate, up, down projections all appear independently."""
    from tinyserve.imatrix import parse_imatrix_dat

    entries = [
        ("blk.0.ffn_gate.0.weight", 10, 2),
        ("blk.0.ffn_up.0.weight", 20, 2),
        ("blk.0.ffn_down.0.weight", 30, 2),
    ]
    path = _write_imatrix_dat(entries)
    counts = parse_imatrix_dat(path)
    assert len(counts) == 3
    assert counts["blk.0.ffn_up.0.weight"] == 20


# ---------------------------------------------------------------------------
# test_rank_experts_per_layer
# ---------------------------------------------------------------------------


def test_rank_experts_per_layer_basic():
    """Experts are sorted by activation count descending."""
    from tinyserve.imatrix import rank_experts_from_imatrix

    counts = {
        "blk.0.ffn_gate.0.weight": 100,
        "blk.0.ffn_gate.1.weight": 50,
        "blk.0.ffn_gate.2.weight": 200,
        "blk.0.ffn_gate.3.weight": 10,
    }
    ranking = rank_experts_from_imatrix(counts, num_layers=1, num_experts=4)
    assert ranking[0] == [2, 0, 1, 3]


def test_rank_experts_accumulates_projections():
    """gate + up + down ncalls are summed per (layer, expert)."""
    from tinyserve.imatrix import rank_experts_from_imatrix

    counts = {
        "blk.0.ffn_gate.0.weight": 10,
        "blk.0.ffn_up.0.weight": 20,
        "blk.0.ffn_down.0.weight": 30,
        "blk.0.ffn_gate.1.weight": 5,
        "blk.0.ffn_up.1.weight": 5,
        "blk.0.ffn_down.1.weight": 5,
    }
    ranking = rank_experts_from_imatrix(counts, num_layers=1, num_experts=2)
    # expert 0: 10+20+30=60, expert 1: 5+5+5=15
    assert ranking[0] == [0, 1]


def test_rank_experts_fused_format():
    """Fused blk.L.ffn_gate_exps.weight spreads count to all experts equally."""
    from tinyserve.imatrix import rank_experts_from_imatrix

    # All experts get the same fused count → tie → sorted by index
    counts = {"blk.0.ffn_gate_exps.weight": 100}
    ranking = rank_experts_from_imatrix(counts, num_layers=1, num_experts=3)
    assert set(ranking[0]) == {0, 1, 2}  # order may vary on tie, all present


def test_rank_experts_missing_layer_uses_zero():
    """Experts with no counts default to 0 and appear last."""
    from tinyserve.imatrix import rank_experts_from_imatrix

    counts = {"blk.0.ffn_gate.2.weight": 50}
    ranking = rank_experts_from_imatrix(counts, num_layers=1, num_experts=4)
    assert ranking[0][0] == 2  # expert 2 has highest count
    assert set(ranking[0][1:]) == {0, 1, 3}  # the rest have zero


def test_rank_experts_multi_layer():
    """Rankings are computed independently per layer."""
    from tinyserve.imatrix import rank_experts_from_imatrix

    counts = {
        "blk.0.ffn_gate.0.weight": 100,
        "blk.0.ffn_gate.1.weight": 200,
        "blk.1.ffn_gate.0.weight": 300,
        "blk.1.ffn_gate.1.weight": 100,
    }
    ranking = rank_experts_from_imatrix(counts, num_layers=2, num_experts=2)
    assert ranking[0] == [1, 0]
    assert ranking[1] == [0, 1]


# ---------------------------------------------------------------------------
# test_seed_cache_from_ranking
# ---------------------------------------------------------------------------


@requires_cuda
def test_seed_cache_from_ranking_experts_cached():
    """Top experts land in cache; contains() returns True after seeding."""
    from tinyserve.expert_store import ExpertCache, ExpertStore
    from tinyserve.imatrix import rank_experts_from_imatrix, seed_cache_from_ranking

    num_layers, num_experts = 2, 4
    expert_weights = {
        (li, ei): {"w.weight": torch.full((8, 8), float(li * num_experts + ei), dtype=torch.bfloat16)}
        for li in range(num_layers)
        for ei in range(num_experts)
    }
    store = ExpertStore.from_dict(expert_weights, num_layers, num_experts)

    # Seed with 2 experts per layer (4 total), cache has 8 slots
    device = torch.device("cuda")
    cache = ExpertCache(8, store.buffer_expert_bytes, device, num_layers=num_layers, num_experts=num_experts)

    counts = {
        "blk.0.ffn_gate.0.weight": 200,
        "blk.0.ffn_gate.1.weight": 100,
        "blk.0.ffn_gate.2.weight": 50,
        "blk.0.ffn_gate.3.weight": 10,
        "blk.1.ffn_gate.0.weight": 300,
        "blk.1.ffn_gate.1.weight": 150,
        "blk.1.ffn_gate.2.weight": 75,
        "blk.1.ffn_gate.3.weight": 25,
    }
    ranking = rank_experts_from_imatrix(counts, num_layers=num_layers, num_experts=num_experts)
    n_seeded = seed_cache_from_ranking(cache, store, ranking, slots_per_layer=2)

    assert n_seeded == 4
    # Top-2 per layer must be cached
    assert cache.contains(0, 0)
    assert cache.contains(0, 1)
    assert cache.contains(1, 0)
    assert cache.contains(1, 1)
    # Bottom-2 per layer must NOT be cached
    assert not cache.contains(0, 2)
    assert not cache.contains(0, 3)


@requires_cuda
def test_seed_cache_respects_capacity():
    """seed_cache_from_ranking never overflows cache capacity."""
    from tinyserve.expert_store import ExpertCache, ExpertStore
    from tinyserve.imatrix import seed_cache_from_ranking

    num_layers, num_experts = 4, 4
    expert_weights = {
        (li, ei): {"w.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}
        for li in range(num_layers)
        for ei in range(num_experts)
    }
    store = ExpertStore.from_dict(expert_weights, num_layers, num_experts)

    # Only 3 slots — far fewer than num_layers * num_experts
    device = torch.device("cuda")
    cache = ExpertCache(3, store.buffer_expert_bytes, device, num_layers=num_layers, num_experts=num_experts)

    ranking = {li: list(range(num_experts)) for li in range(num_layers)}
    n_seeded = seed_cache_from_ranking(cache, store, ranking)

    assert n_seeded == 3


@requires_cuda
def test_seed_cache_slot_map_usable_after_seeding():
    """After seeding, lookup_slots returns valid slot indices for cached experts."""
    from tinyserve.expert_store import ExpertCache, ExpertStore
    from tinyserve.imatrix import seed_cache_from_ranking

    num_layers, num_experts = 1, 4
    expert_weights = {
        (0, ei): {"w.weight": torch.full((4, 4), float(ei), dtype=torch.bfloat16)} for ei in range(num_experts)
    }
    store = ExpertStore.from_dict(expert_weights, num_layers, num_experts)

    device = torch.device("cuda")
    cache = ExpertCache(4, store.buffer_expert_bytes, device, num_layers=num_layers, num_experts=num_experts)
    ranking = {0: [0, 1, 2, 3]}
    seed_cache_from_ranking(cache, store, ranking)

    expert_ids = torch.tensor([0, 1, 2, 3], device=device)
    slots = cache.gpu_slots_for(0, expert_ids)
    assert (slots >= 0).all(), f"Expected all slots valid, got {slots}"
