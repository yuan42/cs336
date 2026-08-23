from __future__ import annotations
import math

from einops import einsum
import torch
from torch import nn

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        temp = torch.empty((out_features, in_features), device=device, dtype=dtype)
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(temp, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
        self.weight = nn.Parameter(temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")