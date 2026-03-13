"""YaRN RoPE implementation for GPT-OSS-120B."""

import math

import torch

from .config import (
    HEAD_DIM,
    MAX_POSITION_EMBEDDINGS,
    ROPE_SCALING,
    ROPE_THETA,
)


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float) -> float:
    return dim * math.log(num_rotations / (2 * math.pi)) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: int, high: int, dim: int) -> torch.Tensor:
    if low == high:
        high += 0.001
    t = torch.arange(dim, dtype=torch.float32)
    return ((t - low) / (high - low)).clamp(0, 1)


def build_rope_cache(
    device: torch.device,
    dtype: torch.dtype,
    max_seq_len: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build YaRN RoPE cos/sin caches.

    Returns:
        (cos_cache, sin_cache) each of shape [max_seq_len, head_dim//2]
    """
    factor = ROPE_SCALING["factor"]
    beta_fast = ROPE_SCALING["beta_fast"]
    beta_slow = ROPE_SCALING["beta_slow"]
    original_max_pos = ROPE_SCALING["original_max_position_embeddings"]

    dim = HEAD_DIM
    half_dim = dim // 2

    # Base inverse frequencies
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))

    # YaRN correction
    low, high = _yarn_find_correction_range(beta_slow, beta_fast, dim, ROPE_THETA)
    inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, half_dim).to(torch.float64)

    inv_freq_extrapolated = inv_freq
    inv_freq_interpolated = inv_freq / factor
    inv_freq = inv_freq_interpolated * (1 - inv_freq_mask) + inv_freq_extrapolated * inv_freq_mask

    # Attention scaling
    _mscale = 0.1 * math.log(factor) + 1.0
    attention_scaling = _mscale * _mscale

    # Build cache
    positions = torch.arange(max_seq_len, dtype=torch.float64)
    freqs = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim//2]

    cos_cache = (freqs.cos() * attention_scaling).to(dtype).to(device)
    sin_cache = (freqs.sin() * attention_scaling).to(dtype).to(device)

    return cos_cache, sin_cache
