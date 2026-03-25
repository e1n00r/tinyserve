import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
