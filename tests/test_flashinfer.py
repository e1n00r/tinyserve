"""Tests for FlashInfer attention backend registration."""



def test_register_flashinfer_returns_string():
    """_register_flashinfer_attention returns 'flashinfer' or 'eager'."""
    from tinyserve.offload import _register_flashinfer_attention

    result = _register_flashinfer_attention()
    assert result in ("flashinfer", "eager")


def test_register_flashinfer_graceful_fallback():
    """If flashinfer not installed, falls back to 'eager'."""
    import sys

    saved = sys.modules.get("flashinfer")
    sys.modules["flashinfer"] = None
    try:
        from tinyserve.offload import _register_flashinfer_attention

        result = _register_flashinfer_attention()
        assert result == "eager"
    finally:
        if saved is not None:
            sys.modules["flashinfer"] = saved
        else:
            sys.modules.pop("flashinfer", None)
