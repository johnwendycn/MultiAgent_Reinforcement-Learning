"""Deterministic seed utility for reproducible MARL experimentation."""

import os
import random
from typing import Optional
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic_cuda: bool = True) -> int:
    """Sets random seeds across Python, NumPy, PyTorch, and CUDA.

    Args:
        seed: The integer seed to set. Defaults to 42.
        deterministic_cuda: Whether to enforce deterministic CUDA backend operations.
            Setting this to True guarantees bitwise reproducibility at a minor performance cost.

    Returns:
        The integer seed configured.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    return seed
