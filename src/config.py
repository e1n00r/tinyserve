import torch

NUM_LAYERS = 36
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 4
HIDDEN_SIZE = 2880
INTERMEDIATE_SIZE = 2880
NUM_ATTENTION_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 64
VOCAB_SIZE = 201088
MAX_POSITION_EMBEDDINGS = 131072
SLIDING_WINDOW = 128
ROPE_THETA = 150000.0
RMS_NORM_EPS = 1e-5
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0

ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 32.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "truncate": False,
    "original_max_position_embeddings": 4096,
}

LAYER_TYPES = [
    "sliding_attention" if bool((i + 1) % 2) else "full_attention"
    for i in range(NUM_LAYERS)
]

# MXFP4 tensor shapes per layer (all 128 experts)
GATE_UP_BLOCKS_SHAPE = (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 32, 16)
GATE_UP_SCALES_SHAPE = (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 32)
GATE_UP_BIAS_SHAPE = (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE)
DOWN_BLOCKS_SHAPE = (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 32, 16)
DOWN_SCALES_SHAPE = (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 32)
DOWN_BIAS_SHAPE = (NUM_EXPERTS, HIDDEN_SIZE)

# Per-expert byte sizes (for buffer allocation)
_gu_b = GATE_UP_BLOCKS_SHAPE[1] * GATE_UP_BLOCKS_SHAPE[2] * GATE_UP_BLOCKS_SHAPE[3]
_gu_s = GATE_UP_SCALES_SHAPE[1] * GATE_UP_SCALES_SHAPE[2]
_gu_bias = GATE_UP_BIAS_SHAPE[1] * 4  # float32
_dn_b = DOWN_BLOCKS_SHAPE[1] * DOWN_BLOCKS_SHAPE[2] * DOWN_BLOCKS_SHAPE[3]
_dn_s = DOWN_SCALES_SHAPE[1] * DOWN_SCALES_SHAPE[2]
_dn_bias = DOWN_BIAS_SHAPE[1] * 4  # float32
EXPERT_BYTES = _gu_b + _gu_s + _gu_bias + _dn_b + _dn_s + _dn_bias  # 13,253,760

FP4_LUT = torch.tensor(
    [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.bfloat16,
)

MODEL_ID = "openai/gpt-oss-120b"

# Weight key patterns for classification
EXPERT_KEY_SUFFIXES = (
    "gate_up_proj_blocks", "gate_up_proj_scales", "gate_up_proj_bias",
    "down_proj_blocks", "down_proj_scales", "down_proj_bias",
)


def is_expert_key(key: str) -> bool:
    return ".mlp.experts." in key and any(key.endswith(s) for s in EXPERT_KEY_SUFFIXES)


def parse_expert_key(key: str) -> tuple[int, str]:
    """Returns (layer_idx, tensor_name) from a key like
    'model.layers.5.mlp.experts.gate_up_proj_blocks'."""
    parts = key.split(".")
    layer_idx = int(parts[2])
    tensor_name = parts[-1]  # e.g. 'gate_up_proj_blocks'
    return layer_idx, tensor_name
