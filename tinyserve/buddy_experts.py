"""Buddy expert co-activation profiling and substitution.

Based on BuddyMoE (arxiv 2511.10054): when a cache miss occurs, substitute
a co-activation-similar cached expert for zero-stall inference. Small
accuracy cost bounded by co-activation similarity.
"""

import torch


def build_coactivation_matrix(
    routing_decisions: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Build expert co-activation matrix from routing decisions.

    Args:
        routing_decisions: [num_tokens, top_k] expert indices
        num_experts: total number of experts

    Returns:
        [num_experts, num_experts] symmetric co-activation count matrix
    """
    coact = torch.zeros(num_experts, num_experts, dtype=torch.float32)
    for token_experts in routing_decisions:
        experts = token_experts.tolist()
        for i in range(len(experts)):
            for j in range(i + 1, len(experts)):
                coact[experts[i], experts[j]] += 1
                coact[experts[j], experts[i]] += 1
    return coact


class BuddyTable:
    """Pre-computed buddy expert lookup table."""

    def __init__(self, buddies: dict[int, list[int]]):
        self._buddies = buddies

    @classmethod
    def from_coactivation(cls, coact: torch.Tensor, max_buddies: int = 3) -> "BuddyTable":
        """Build buddy table from co-activation matrix."""
        n = coact.shape[0]
        buddies = {}
        for eid in range(n):
            scores = coact[eid].clone()
            scores[eid] = -1  # exclude self
            top = scores.topk(min(max_buddies, n - 1)).indices.tolist()
            buddies[eid] = top
        return cls(buddies)

    def get_buddies(self, expert_id: int) -> list[int]:
        return self._buddies.get(expert_id, [])

    def find_cached_buddy(self, expert_id: int, cached_set: set[int]) -> int | None:
        for buddy in self.get_buddies(expert_id):
            if buddy in cached_set:
                return buddy
        return None
