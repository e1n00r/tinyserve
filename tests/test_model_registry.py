"""Parametrized tests for model registry profile_from_config."""

from types import SimpleNamespace

import pytest

from tinyserve.model_registry import ModelProfile, profile_from_config
from tinyserve.offload import _ROUTING_MAP


def _make_config(**kwargs):
    return SimpleNamespace(**kwargs)


@pytest.mark.parametrize(
    "model_type, config_kwargs, expected",
    [
        pytest.param(
            "llama4",
            dict(num_local_experts=16, num_experts_per_tok=2, num_hidden_layers=48),
            dict(
                moe_block_attr="feed_forward",
                expert_list_attr="experts",
                weight_names=["gate_proj", "up_proj", "down_proj"],
                num_experts=16,
                num_experts_per_tok=2,
                num_layers=48,
                first_moe_layer=0,
                shared_expert_attr="shared_expert",
            ),
            id="llama4",
        ),
        pytest.param(
            "kimi_k2",
            dict(
                n_routed_experts=256,
                num_experts_per_tok=8,
                num_hidden_layers=61,
                first_k_dense_replace=3,
            ),
            dict(
                moe_block_attr="mlp",
                expert_list_attr="experts",
                weight_names=["gate_proj", "up_proj", "down_proj"],
                num_experts=256,
                num_experts_per_tok=8,
                num_layers=61,
                first_moe_layer=3,
                shared_expert_attr="shared_experts",
            ),
            id="kimi_k2",
        ),
        pytest.param(
            "dbrx",
            dict(
                ffn_config=SimpleNamespace(moe_num_experts=16, moe_top_k=4),
                n_layers=40,
            ),
            dict(
                moe_block_attr="ffn",
                expert_list_attr="experts",
                weight_names=["w1", "v1", "w2"],
                num_experts=16,
                num_experts_per_tok=4,
                num_layers=40,
                first_moe_layer=0,
                shared_expert_attr=None,
            ),
            id="dbrx",
        ),
        pytest.param(
            "phimoe",
            dict(num_local_experts=16, num_experts_per_tok=2, num_hidden_layers=32),
            dict(
                moe_block_attr="block_sparse_moe",
                expert_list_attr="experts",
                weight_names=["w1", "w2", "w3"],
                num_experts=16,
                num_experts_per_tok=2,
                num_layers=32,
                first_moe_layer=0,
                shared_expert_attr=None,
            ),
            id="phimoe",
        ),
        pytest.param(
            "mixtral",
            dict(num_local_experts=8, num_experts_per_tok=2, num_hidden_layers=32),
            dict(
                moe_block_attr="mlp",
                expert_list_attr="experts",
                weight_names=["w1", "w2", "w3"],
                num_experts=8,
                num_experts_per_tok=2,
                num_layers=32,
                first_moe_layer=0,
                shared_expert_attr=None,
            ),
            id="mixtral",
        ),
        pytest.param(
            "deepseek_v3",
            dict(
                n_routed_experts=256,
                num_experts_per_tok=8,
                num_hidden_layers=61,
                first_k_dense_replace=1,
            ),
            dict(
                moe_block_attr="mlp",
                expert_list_attr="experts",
                weight_names=["gate_proj", "up_proj", "down_proj"],
                num_experts=256,
                num_experts_per_tok=8,
                num_layers=61,
                first_moe_layer=1,
                shared_expert_attr="shared_experts",
            ),
            id="deepseek_v3",
        ),
        pytest.param(
            "olmoe",
            dict(num_experts=64, num_experts_per_tok=8, num_hidden_layers=16),
            dict(
                moe_block_attr="mlp",
                expert_list_attr="experts",
                weight_names=["gate_proj", "up_proj", "down_proj"],
                num_experts=64,
                num_experts_per_tok=8,
                num_layers=16,
                first_moe_layer=0,
                shared_expert_attr=None,
            ),
            id="olmoe",
        ),
    ],
)
def test_profile_from_config(model_type, config_kwargs, expected):
    config = _make_config(model_type=model_type, **config_kwargs)
    profile = profile_from_config(config)

    assert isinstance(profile, ModelProfile)
    assert profile.moe_block_attr == expected["moe_block_attr"]
    assert profile.expert_list_attr == expected["expert_list_attr"]
    assert profile.expert_layout.weight_names == expected["weight_names"]
    assert profile.num_experts == expected["num_experts"]
    assert profile.num_experts_per_tok == expected["num_experts_per_tok"]
    assert profile.num_layers == expected["num_layers"]
    assert profile.first_moe_layer == expected["first_moe_layer"]
    assert profile.shared_expert_attr == expected["shared_expert_attr"]


def test_unsupported_model_type_raises():
    config = _make_config(model_type="unknown_model_xyz")
    with pytest.raises(ValueError, match="Unsupported model type"):
        profile_from_config(config)


@pytest.mark.parametrize(
    "model_type",
    ["llama4", "kimi_k2", "dbrx", "phimoe"],
)
def test_routing_map_entries(model_type):
    assert model_type in _ROUTING_MAP
    softmax_order, returns_logits, router_attr = _ROUTING_MAP[model_type]
    assert softmax_order in ("router_native", "softmax_then_topk")
    assert isinstance(returns_logits, bool)
    assert isinstance(router_attr, str)


def test_routing_map_phimoe_is_softmax_then_topk():
    assert _ROUTING_MAP["phimoe"][0] == "softmax_then_topk"


def test_routing_map_llama4_is_router_native():
    assert _ROUTING_MAP["llama4"][0] == "router_native"


def test_kimi_k2_matches_deepseek_v3_structure():
    ds_config = _make_config(
        model_type="deepseek_v3",
        n_routed_experts=256,
        num_experts_per_tok=8,
        num_hidden_layers=61,
        first_k_dense_replace=3,
    )
    kimi_config = _make_config(
        model_type="kimi_k2",
        n_routed_experts=256,
        num_experts_per_tok=8,
        num_hidden_layers=61,
        first_k_dense_replace=3,
    )
    ds_profile = profile_from_config(ds_config)
    kimi_profile = profile_from_config(kimi_config)

    assert ds_profile.moe_block_attr == kimi_profile.moe_block_attr
    assert ds_profile.expert_list_attr == kimi_profile.expert_list_attr
    assert ds_profile.expert_layout.weight_names == kimi_profile.expert_layout.weight_names
    assert ds_profile.first_moe_layer == kimi_profile.first_moe_layer
    assert ds_profile.shared_expert_attr == kimi_profile.shared_expert_attr
