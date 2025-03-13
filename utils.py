"""
Utility functions and miscellaneous helpers.
"""

import datetime
import random
import numpy as np
import torch


def thai_time() -> datetime.datetime:
    """
    Returns current UTC time shifted by +7 hours (Thai time).
    """
    return datetime.datetime.utcnow() + datetime.timedelta(hours=7)


def set_random_seed(seed: int = 666) -> None:
    """
    Set random seeds for reproducibility across numpy, torch, and random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_obj(obj_path: str):
    """
    Dynamically load a Python object from a string path, e.g. "torch.optim.AdamW".
    """
    parts = obj_path.split(".")
    module_path = ".".join(parts[:-1])
    obj_name = parts[-1]
    module = __import__(module_path, fromlist=[obj_name])
    return getattr(module, obj_name)
