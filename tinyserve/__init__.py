import os

# Enable persistent torch.compile kernel cache — eliminates
# recompilation VRAM spikes on subsequent runs.
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "tinyserve", "inductor"),
)

__version__ = "0.1.0"

from .gguf_loader import load_from_gguf as load_from_gguf
from .offload import load_and_offload as load_and_offload
from .offload import offload_model as offload_model
from .offload import TinyserveConfig as TinyserveConfig
