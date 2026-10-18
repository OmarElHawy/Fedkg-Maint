import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def check_gpu() -> None:
    """Print whether CUDA is available and the GPU name."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: True")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU count: {torch.cuda.device_count()}")
        else:
            print("CUDA available: False")
    except ImportError:
        print("PyTorch not installed. Cannot check GPU.")