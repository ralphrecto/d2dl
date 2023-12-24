from typing import Tuple
from torch import Tensor
import torch

def gen_lin_data(w: Tensor, b: float, num_samples: int, noise: float = 0.01) -> Tuple[Tensor, Tensor]:
    rand_x = torch.rand(num_samples, len(w))
    base_y = rand_x.matmul(w.reshape(-1, 1))
    noise_y = torch.randn(num_samples, 1) * noise

    return rand_x, base_y + b + noise_y