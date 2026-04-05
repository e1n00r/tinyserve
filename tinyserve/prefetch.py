"""Token-to-token routing locality prefetch (FATE temporal prefetch).

Expert routing is highly stable token-to-token (same experts fire 85-99%
of steps). FATEPrefetcher tracks which experts fired last token and uses
that access_history to predict what the next token will need, overlapping
h2d_transfer with attention compute.
"""


class FATEPrefetcher:
    """Temporal prefetch engine: tracks access_history to predict predicted_expert_keys.

    One instance per offloaded model, held by model_hooks.py as a module-level singleton.
    """

    def __init__(self) -> None:
        self.access_history: dict[int, list[int]] = {}
        self.pending_predictions: dict[int, set] = {}
        self.stats: dict[int, dict] = {}

    def record(self, layer_idx: int, expert_ids: list[int]) -> None:
        """Record which expert_ids fired at layer_idx this token."""
        self.access_history[layer_idx] = expert_ids

    def predicted_expert_keys(self, layer_idx: int) -> set[tuple[int, int]]:
        """Return predicted expert_keys for layer_idx based on last token's access_history."""
        return {(layer_idx, eid) for eid in self.access_history.get(layer_idx, [])}

    def reset(self) -> None:
        """Clear access_history. Call before each benchmark run."""
        self.access_history.clear()
        self.pending_predictions.clear()
