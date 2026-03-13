"""Phase 3: Double-buffered expert pipeline with CUDA streams.

Overlaps expert PCIe transfer with expert GPU compute using two streams
and two GPU buffers. Within each layer:

  Stream transfer:  [load E0] [load E1]  [load E2]  [load E3]
  Stream compute:          [compute E0] [compute E1] [compute E2] [compute E3]

The first expert per layer has no overlap (cold start). Experts 1-3
overlap their load with the previous expert's compute.
"""

import torch

from .config import HIDDEN_SIZE, NUM_EXPERTS_PER_TOK
from .expert_store import ExpertBuffer, ExpertStore
from .experts import expert_forward


class ExpertPipeline:
    """Double-buffered expert execution with overlapped transfer and compute."""

    def __init__(self, expert_store: ExpertStore, device: torch.device):
        self.store = expert_store
        self.device = device
        self.dtype = torch.bfloat16

        # Two GPU buffers for double-buffering
        self.buf_a = ExpertBuffer(device)
        self.buf_b = ExpertBuffer(device)

        # Two CUDA streams
        self.transfer_stream = torch.cuda.Stream(device)
        self.compute_stream = torch.cuda.Stream(device)

    def execute_layer_experts(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run top-k experts for one layer with overlapped transfer/compute.

        Args:
            hidden_states: [num_tokens, hidden_size] input to experts
            layer_idx: which layer
            expert_indices: [num_tokens, top_k] expert indices
            routing_weights: [num_tokens, top_k] softmax weights

        Returns:
            [num_tokens, hidden_size] weighted expert output
        """
        num_tokens = hidden_states.shape[0]
        expert_output = torch.zeros_like(hidden_states)

        for tok in range(num_tokens):
            self._execute_token_experts(
                hidden_states[tok:tok + 1],
                expert_output,
                tok,
                layer_idx,
                expert_indices[tok].tolist(),
                routing_weights[tok],
            )

        return expert_output

    def _execute_token_experts(
        self,
        h: torch.Tensor,
        output: torch.Tensor,
        tok_idx: int,
        layer_idx: int,
        expert_ids: list[int],
        weights: torch.Tensor,
    ):
        """Pipeline 4 experts for a single token with double buffering.

        Synchronization:
        - compute_stream waits for load_done before reading a buffer
        - transfer_stream waits for compute_done before overwriting a buffer
        """
        k = NUM_EXPERTS_PER_TOK
        bufs = [self.buf_a, self.buf_b]

        # Track when compute finishes with each buffer so transfer doesn't
        # overwrite a buffer that's still being read
        compute_done = [torch.cuda.Event() for _ in range(2)]

        # Kick off first expert load into buf_a
        load_done = torch.cuda.Event()
        with torch.cuda.stream(self.transfer_stream):
            self.store.copy_to_buffer(bufs[0], layer_idx, expert_ids[0], non_blocking=True)
            load_done.record(self.transfer_stream)

        for i in range(k):
            cur = i % 2
            nxt = (i + 1) % 2

            # Wait for current buffer's transfer to complete
            self.compute_stream.wait_event(load_done)

            # Start loading next expert into the OTHER buffer
            if i < k - 1:
                next_load_done = torch.cuda.Event()
                with torch.cuda.stream(self.transfer_stream):
                    # Wait for compute to finish with nxt buffer before overwriting
                    if i >= 1:
                        self.transfer_stream.wait_event(compute_done[nxt])
                    self.store.copy_to_buffer(
                        bufs[nxt], layer_idx, expert_ids[i + 1], non_blocking=True
                    )
                    next_load_done.record(self.transfer_stream)
                load_done = next_load_done

            # Compute current expert on compute_stream
            with torch.cuda.stream(self.compute_stream):
                out = expert_forward(
                    h,
                    bufs[cur].gate_up_blocks,
                    bufs[cur].gate_up_scales,
                    bufs[cur].gate_up_bias,
                    bufs[cur].down_blocks,
                    bufs[cur].down_scales,
                    bufs[cur].down_bias,
                    dtype=self.dtype,
                )
                output[tok_idx] += weights[i] * out.squeeze(0)
                compute_done[cur].record(self.compute_stream)

        # Ensure all compute is done before returning
        self.compute_stream.synchronize()
