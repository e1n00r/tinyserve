"""Per-model metadata for locating and intercepting MoE expert layers.

Each model family has different class names, weight layouts, and expert
access patterns. This registry provides a uniform interface for the
offloading infrastructure to find and replace expert weights.
"""

from dataclasses import dataclass, field
from transformers import PretrainedConfig


@dataclass
class ExpertLayout:
    """Describes how one expert's weights are organized."""
    weight_names: list[str]
    bias_names: list[str] = field(default_factory=list)


@dataclass
class ModelProfile:
    """Everything the offloader needs to know about a model family."""
    moe_block_attr: str
    expert_list_attr: str
    expert_layout: ExpertLayout
    num_experts: int
    num_experts_per_tok: int
    num_layers: int
    first_moe_layer: int = 0
    shared_expert_attr: str | None = None


def profile_from_config(config: PretrainedConfig) -> ModelProfile:
    model_type = config.model_type

    if model_type == "gpt_oss":
        return ModelProfile(
            moe_block_attr="mlp",
            expert_list_attr="experts",
            expert_layout=ExpertLayout(
                weight_names=["gate_up_proj", "down_proj"],
                bias_names=["gate_up_proj_bias", "down_proj_bias"],
            ),
            num_experts=getattr(config, "num_local_experts", getattr(config, "num_experts", 128)),
            num_experts_per_tok=config.num_experts_per_tok,
            num_layers=config.num_hidden_layers,
        )

    if model_type in ("qwen3_moe", "qwen2_moe"):
        return ModelProfile(
            moe_block_attr="mlp",
            expert_list_attr="experts",
            expert_layout=ExpertLayout(
                weight_names=["gate_proj", "up_proj", "down_proj"],
            ),
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            num_layers=config.num_hidden_layers,
        )

    if model_type == "mixtral":
        return ModelProfile(
            moe_block_attr="mlp",
            expert_list_attr="experts",
            expert_layout=ExpertLayout(
                weight_names=["w1", "w2", "w3"],
            ),
            num_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            num_layers=config.num_hidden_layers,
        )

    if model_type == "deepseek_v3":
        return ModelProfile(
            moe_block_attr="mlp",
            expert_list_attr="experts",
            expert_layout=ExpertLayout(
                weight_names=["gate_proj", "up_proj", "down_proj"],
            ),
            num_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            num_layers=config.num_hidden_layers,
            first_moe_layer=config.first_k_dense_replace,
            shared_expert_attr="shared_experts",
        )

    if model_type in ("qwen3_5_moe", "qwen3_5_moe_text"):
        return ModelProfile(
            moe_block_attr="mlp",
            expert_list_attr="experts",
            expert_layout=ExpertLayout(
                weight_names=["gate_up_proj", "down_proj"],
            ),
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            num_layers=config.num_hidden_layers,
            shared_expert_attr="shared_expert",
        )

    if model_type == "olmoe":
        return ModelProfile(
            moe_block_attr="mlp",
            expert_list_attr="experts",
            expert_layout=ExpertLayout(
                weight_names=["gate_proj", "up_proj", "down_proj"],
            ),
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            num_layers=config.num_hidden_layers,
        )

    raise ValueError(f"Unsupported model type: {model_type}")
