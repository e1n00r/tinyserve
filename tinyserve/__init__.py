import os

# Enable persistent torch.compile kernel cache — eliminates
# recompilation VRAM spikes on subsequent runs.
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "tinyserve", "inductor"),
)

__version__ = "0.1.0"

__all__ = [
    "load_from_gguf",
    "load_and_offload",
    "offload_model",
    "TinyserveConfig",
    "OffloadedLM",
]

from .gguf_loader import load_from_gguf as load_from_gguf
from .offload import OffloadedLM as OffloadedLM
from .offload import TinyserveConfig as TinyserveConfig
from .offload import load_and_offload as load_and_offload
from .offload import offload_model as offload_model
