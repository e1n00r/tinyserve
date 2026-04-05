import pytest
from tinyserve import gguf_model_loader


def test_gguf_model_loader_has_load_from_gguf():
    assert hasattr(gguf_model_loader, "load_from_gguf")


def test_gguf_model_loader_has_open_gguf():
    assert hasattr(gguf_model_loader, "open_gguf")
