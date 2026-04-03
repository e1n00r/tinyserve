"""Parse imatrix activation data for expert cache seeding."""
from __future__ import annotations

import re
import struct
from pathlib import Path


def parse_imatrix_dat(path: str) -> dict[str, int]:
    """Parse llama.cpp imatrix .dat file.

    Returns dict mapping tensor name to ncall (activation count).
    """
    counts: dict[str, int] = {}
    with open(path, "rb") as f:
        (n_entries,) = struct.unpack("<i", f.read(4))
        for _ in range(n_entries):
            (name_len,) = struct.unpack("<i", f.read(4))
            name = f.read(name_len).decode("utf-8")
            (ncall,) = struct.unpack("<i", f.read(4))
            (nval,) = struct.unpack("<i", f.read(4))
            f.seek(nval * 4, 1)  # skip float32 values
            counts[name] = ncall
    return counts


def rank_experts_from_imatrix(
    counts: dict[str, int],
    num_layers: int,
    num_experts: int,
) -> dict[int, list[int]]:
    """Rank experts per layer by activation count (descending).

    Returns dict mapping layer_idx to list of expert_ids sorted by activation count.
    """
    expert_counts: dict[tuple[int, int], int] = {}

    per_expert = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)\.(\d+)\.weight")
    fused = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight")

    for name, ncall in counts.items():
        m = per_expert.match(name)
        if m:
            layer, expert = int(m.group(1)), int(m.group(3))
            key = (layer, expert)
            expert_counts[key] = expert_counts.get(key, 0) + ncall
            continue
        m = fused.match(name)
        if m:
            layer = int(m.group(1))
            for e in range(num_experts):
                key = (layer, e)
                expert_counts[key] = expert_counts.get(key, 0) + ncall

    ranking: dict[int, list[int]] = {}
    for layer in range(num_layers):
        layer_experts = [(e, expert_counts.get((layer, e), 0)) for e in range(num_experts)]
        layer_experts.sort(key=lambda x: x[1], reverse=True)
        ranking[layer] = [e for e, _ in layer_experts]

    return ranking


def seed_cache_from_ranking(
    cache,
    store,
    ranking: dict[int, list[int]],
    slots_per_layer: int | None = None,
) -> int:
    """Pre-load top experts into GPU cache based on imatrix ranking.

    Args:
        cache: ExpertCache
        store: ExpertStore
        ranking: {layer_idx: [expert_ids sorted by importance descending]}
        slots_per_layer: max experts to seed per layer (None = fill cache evenly)

    Returns:
        Number of experts loaded into cache.
    """
    num_layers = len(ranking)
    if slots_per_layer is None:
        slots_per_layer = max(1, cache.capacity // num_layers)

    loaded = 0
    for layer_idx in sorted(ranking.keys()):
        experts = ranking[layer_idx][:slots_per_layer]
        for expert_idx in experts:
            if loaded >= cache.capacity:
                cache.flush_slot_updates()
                return loaded
            slot = cache.allocate(layer_idx, expert_idx)
            store.copy_to_buffer_slot(cache, slot, layer_idx, expert_idx)
            loaded += 1

    cache.flush_slot_updates()
    return loaded
