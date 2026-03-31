"""Tests for buddy expert co-activation profiling and substitution."""
import torch
import pytest


def test_coactivation_matrix_from_routing():
    from tinyserve.buddy_experts import build_coactivation_matrix

    # 10 tokens, 4 experts, top_k=2
    routing = torch.tensor([
        [0, 1], [0, 2], [1, 2], [0, 1], [2, 3],
        [0, 1], [1, 3], [0, 2], [2, 3], [0, 1],
    ])
    coact = build_coactivation_matrix(routing, num_experts=4)
    # Experts 0 and 1 co-activate 4 times
    assert coact[0, 1] == coact[1, 0]
    assert coact[0, 1] > coact[0, 3]


def test_buddy_lookup():
    from tinyserve.buddy_experts import BuddyTable

    # 4 experts, buddy of expert 0 is expert 1 (highest co-activation)
    coact = torch.tensor([
        [0, 5, 2, 1],
        [5, 0, 3, 1],
        [2, 3, 0, 4],
        [1, 1, 4, 0],
    ], dtype=torch.float32)
    table = BuddyTable.from_coactivation(coact, max_buddies=2)

    buddies = table.get_buddies(0)
    assert buddies[0] == 1  # highest co-activation with expert 0


def test_buddy_substitution_uses_cached_expert():
    from tinyserve.buddy_experts import BuddyTable

    coact = torch.tensor([
        [0, 5, 2, 1],
        [5, 0, 3, 1],
        [2, 3, 0, 4],
        [1, 1, 4, 0],
    ], dtype=torch.float32)
    table = BuddyTable.from_coactivation(coact, max_buddies=2)

    cached_experts = {1, 3}  # experts currently in GPU cache
    buddy = table.find_cached_buddy(0, cached_experts)
    assert buddy == 1  # expert 1 is buddy of 0 and is cached
