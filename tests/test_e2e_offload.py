"""End-to-end test: build a real MoE model, offload experts, verify output matches."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.conftest import requires_cuda


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyMoEBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int, num_experts: int, top_k: int):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(hidden, intermediate) for _ in range(num_experts)
        ])
        self.top_k = top_k

    def forward(self, hidden_states):
        batch_seq, hidden = hidden_states.shape
        logits = self.gate(hidden_states)
        top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
        routing_weights = F.softmax(top_vals, dim=-1)

        output = torch.zeros_like(hidden_states)
        for i in range(self.top_k):
            expert_mask = top_idx[:, i]
            for expert_idx in range(len(self.experts)):
                mask = expert_mask == expert_idx
                if mask.any():
                    expert_out = self.experts[expert_idx](hidden_states[mask])
                    output[mask] += routing_weights[mask, i:i + 1] * expert_out
        return output


class TinyMoELayer(nn.Module):
    def __init__(self, hidden, intermediate, num_experts, top_k):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.Linear(hidden, hidden, bias=False)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = TinyMoEBlock(hidden, intermediate, num_experts, top_k)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x).view(-1, x.shape[-1])).view_as(x)
        return x


class TinyMoEModel(nn.Module):
    def __init__(self, vocab, hidden, intermediate, num_layers, num_experts, top_k):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([
            TinyMoELayer(hidden, intermediate, num_experts, top_k)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.norm(h))


def _extract_expert_weights(model, num_layers, num_experts):
    weights = {}
    for li in range(num_layers):
        for ei in range(num_experts):
            expert = model.layers[li].mlp.experts[ei]
            weights[(li, ei)] = {
                "gate_proj.weight": expert.gate_proj.weight.data.detach().cpu().clone(),
                "up_proj.weight": expert.up_proj.weight.data.detach().cpu().clone(),
                "down_proj.weight": expert.down_proj.weight.data.detach().cpu().clone(),
            }
    return weights


@requires_cuda
def test_offloaded_model_matches_reference():
    """Full MoE model with offloaded experts produces identical logits."""
    from src.generic_store import GenericExpertStore
    from src.generic_pipeline import GenericExpertPipeline

    torch.manual_seed(42)
    hidden, intermediate = 64, 128
    num_layers, num_experts, top_k = 2, 8, 2
    vocab = 256

    model = TinyMoEModel(vocab, hidden, intermediate, num_layers, num_experts, top_k)
    device = torch.device("cuda")
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    input_ids = torch.tensor([[1, 42, 100, 7]], device=device)

    with torch.no_grad():
        ref_logits = model(input_ids)

    expert_weights = _extract_expert_weights(model, num_layers, num_experts)
    store = GenericExpertStore.from_dict(expert_weights, num_layers, num_experts)

    for li in range(num_layers):
        moe = model.layers[li].mlp
        template = SwiGLUExpert(hidden, intermediate).to(device).to(torch.bfloat16)
        pipeline = GenericExpertPipeline(store, template, device, cache_capacity=16)

        def make_offloaded_forward(layer_idx, pipe, gate):
            def offloaded_forward(hidden_states):
                logits = gate(hidden_states)
                top_vals, top_idx = torch.topk(logits, top_k, dim=-1)
                routing_weights = F.softmax(top_vals, dim=-1)
                return pipe.execute_layer_experts(
                    hidden_states, layer_idx, top_idx, routing_weights.to(hidden_states.dtype),
                )
            return offloaded_forward

        moe.forward = make_offloaded_forward(li, pipeline, moe.gate)

    with torch.no_grad():
        offloaded_logits = model(input_ids)

    torch.testing.assert_close(offloaded_logits, ref_logits, rtol=0, atol=0)


@requires_cuda
def test_offloaded_autoregressive_matches():
    """Multi-step autoregressive generation produces identical tokens."""
    from src.generic_store import GenericExpertStore
    from src.generic_pipeline import GenericExpertPipeline

    torch.manual_seed(123)
    hidden, intermediate = 64, 128
    num_layers, num_experts, top_k = 2, 8, 2
    vocab = 256

    model = TinyMoEModel(vocab, hidden, intermediate, num_layers, num_experts, top_k)
    device = torch.device("cuda")
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    input_ids = torch.tensor([[10, 20, 30]], device=device)

    with torch.no_grad():
        ref_tokens = []
        ids = input_ids.clone()
        for _ in range(5):
            logits = model(ids)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ref_tokens.append(next_tok.item())
            ids = torch.cat([ids, next_tok], dim=1)

    expert_weights = _extract_expert_weights(model, num_layers, num_experts)
    store = GenericExpertStore.from_dict(expert_weights, num_layers, num_experts)
    template = SwiGLUExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    for li in range(num_layers):
        moe = model.layers[li].mlp
        pipeline = GenericExpertPipeline(store, template, device, cache_capacity=16)

        def make_offloaded(layer_idx, pipe, gate):
            def fwd(hidden_states):
                logits = gate(hidden_states)
                top_vals, top_idx = torch.topk(logits, top_k, dim=-1)
                rw = F.softmax(top_vals, dim=-1)
                return pipe.execute_layer_experts(hidden_states, layer_idx, top_idx, rw.to(hidden_states.dtype))
            return fwd

        moe.forward = make_offloaded(li, pipeline, moe.gate)

    with torch.no_grad():
        offloaded_tokens = []
        ids = input_ids.clone()
        for _ in range(5):
            logits = model(ids)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            offloaded_tokens.append(next_tok.item())
            ids = torch.cat([ids, next_tok], dim=1)

    assert offloaded_tokens == ref_tokens
