import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)  # imagecorruptions draws its noise from NumPy's global RNG
    torch.manual_seed(seed)
