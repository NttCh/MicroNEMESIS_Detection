#!/usr/bin/env python
"""Utility functions."""

import datetime
import random
import numpy as np
import torch
import importlib


def thai_time():
    """Return the current time in Thailand (UTC+7)."""
    return datetime.datetime.utcnow() + datetime.timedelta(hours=7)


def set_seed(seed: int = 666) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_obj(obj_path: str):
    """
    Dynamically load an object by its string path.

    Args:
        obj_path (str): The module path in dot notation.

    Returns:
        Any: The loaded object (class, function, etc.).
    """
    parts = obj_path.split(".")
    module_path = ".".join(parts[:-1])
    obj_name = parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)
