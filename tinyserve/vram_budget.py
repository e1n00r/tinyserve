"""Dynamic VRAM rebalancing between expert cache and KV cache.

Demand-driven: KV cache raises KVCacheOverflow when it runs out of space.
VRAMBudget catches it, frees expert slots, extends KV, retries.
When a request completes, expert cache grows back.

The tradeoff: 1 expert slot ≈ N tokens of KV (model-dependent).
For GPT-OSS-20B: 1 slot = 13.2 MB / 48 KB = 269 tokens.
"""

import logging
import math

logger = logging.getLogger(__name__)


class VRAMBudget:
    """Demand-driven expert↔KV VRAM rebalancing."""

    def __init__(
        self,
        expert_cache,
        kv_cache,
        expert_bytes: int,
        kv_bytes_per_token: int,
        max_expert_capacity: int | None = None,
        min_expert_capacity: int = 4,
    ):
        self.expert_cache = expert_cache
        self.kv_cache = kv_cache
        self.expert_bytes = expert_bytes
        self.kv_bytes_per_token = kv_bytes_per_token
        self.max_expert_capacity = max_expert_capacity or expert_cache.capacity
        self.min_expert_capacity = min_expert_capacity
        self.tokens_per_expert_slot = expert_bytes // max(1, kv_bytes_per_token)
        self._rebalance_count = 0

    def handle_overflow(self, tokens_needed: int) -> bool:
        """Called when KV cache overflows. Frees expert slots to extend KV.

        Args:
            tokens_needed: how many additional KV tokens are required.

        Returns:
            True if KV was successfully extended, False if no expert slots
            could be freed (expert cache already at minimum).
        """
        slots_needed = math.ceil(tokens_needed / max(1, self.tokens_per_expert_slot))
        slots_available = self.expert_cache.capacity - self.min_expert_capacity
        slots_to_free = min(slots_needed, slots_available)

        if slots_to_free <= 0:
            logger.warning(
                "KV overflow: need %d tokens but expert cache at minimum (%d slots)",
                tokens_needed, self.expert_cache.capacity,
            )
            return False

        freed_bytes = self.expert_cache.shrink(slots_to_free)
        kv_tokens = freed_bytes // max(1, self.kv_bytes_per_token)
        self.kv_cache.extend(max(kv_tokens, tokens_needed))
        self._rebalance_count += 1

        logger.info(
            "Rebalance #%d: freed %d expert slots → +%d KV tokens "
            "(experts: %d/%d, KV: %d tokens)",
            self._rebalance_count, slots_to_free, kv_tokens,
            self.expert_cache.capacity, self.max_expert_capacity,
            self.kv_cache.max_seq_len,
        )
        return True

    def release_kv(self) -> None:
        """Called when a request completes. Grows expert cache back from freed KV space."""
        expert_cap = self.expert_cache.capacity
        if expert_cap >= self.max_expert_capacity:
            return

        slots_to_grow = self.max_expert_capacity - expert_cap
        self.expert_cache.grow(slots_to_grow)
        logger.info(
            "KV released: grew expert cache %d → %d slots",
            expert_cap, self.expert_cache.capacity,
        )

    # Keep check() for backward compat with tests, but simplified
    def check(self) -> dict:
        """Check if rebalancing is needed (for monitoring/tests)."""
        kv_util = self.kv_utilization()
        expert_cap = self.expert_cache.capacity

        if kv_util >= 0.85 and expert_cap > self.min_expert_capacity:
            slots = min(
                max(1, int(self.kv_cache.max_seq_len * 0.25) // max(1, self.tokens_per_expert_slot)),
                expert_cap - self.min_expert_capacity,
            )
            return {"should_rebalance": True, "direction": "shrink_experts",
                    "expert_slots_to_free": slots,
                    "kv_tokens_gained": slots * self.tokens_per_expert_slot}

        if kv_util <= 0.10 and expert_cap < self.max_expert_capacity:
            return {"should_rebalance": True, "direction": "grow_experts",
                    "expert_slots_to_free": -(self.max_expert_capacity - expert_cap),
                    "kv_tokens_gained": 0}

        return {"should_rebalance": False, "direction": None,
                "expert_slots_to_free": 0, "kv_tokens_gained": 0}

    def execute(self, action: dict) -> None:
        """Execute a check() action (backward compat)."""
        if not action["should_rebalance"]:
            return
        if action["direction"] == "shrink_experts":
            n = action["expert_slots_to_free"]
            freed = self.expert_cache.shrink(n)
            kv_tokens = freed // max(1, self.kv_bytes_per_token)
            self.kv_cache.extend(kv_tokens)
        elif action["direction"] == "grow_experts":
            self.release_kv()

    def kv_utilization(self) -> float:
        if self.kv_cache is None or self.kv_cache.max_seq_len == 0:
            return 0.0
        max_seq = max(self.kv_cache._seq_lens) if self.kv_cache._seq_lens else 0
        return max_seq / self.kv_cache.max_seq_len
