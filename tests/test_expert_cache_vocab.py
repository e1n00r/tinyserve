import torch
from tinyserve.expert_cache import ExpertCache


def _make_cache():
    return ExpertCache(capacity=4, expert_bytes=64, device=torch.device("cpu"), policy="lru")


def test_locate_returns_slot_on_hit():
    cache = _make_cache()
    slot = cache.claim_slot_for((0, 1))
    assert cache.locate((0, 1)) == slot


def test_locate_returns_none_on_miss():
    cache = _make_cache()
    assert cache.locate((0, 99)) is None


def test_gpu_slots_for_returns_tensor():
    cache = _make_cache()
    cache.claim_slot_for((0, 0))
    ids = torch.tensor([0, 1], dtype=torch.int32)
    result = cache.gpu_slots_for(0, ids)
    assert result.shape == ids.shape
