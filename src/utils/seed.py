from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set deterministic seed for Python and NumPy."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
