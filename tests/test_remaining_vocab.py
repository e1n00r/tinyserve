from tinyserve.expert_store import ExpertStore
from tinyserve.mmap_store import MmapExpertStore


def test_expert_store_packed_weights_for():
    assert hasattr(ExpertStore, "packed_weights_for"), \
        "ExpertStore must expose packed_weights_for"


def test_mmap_store_copy_packed_weights():
    assert hasattr(MmapExpertStore, "copy_packed_weights"), \
        "MmapExpertStore must expose copy_packed_weights"
