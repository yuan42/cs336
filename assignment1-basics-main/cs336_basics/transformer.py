from __future__ import annotations

import torch
from torch import nn

from einops import rearrange


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()

        self.theta = theta
        assert d_k % 2 == 0
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, device=device)
        pair_indices = torch.arange(d_k // 2, device=device)
        frequencies = theta ** (-2 * pair_indices / d_k)
        angles = positions[:, None] * frequencies[None, :]
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        x_even, x_odd = rearrange(x, "... seq (width group) -> group ... seq width", group=2)
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return rearrange([rotated_even, rotated_odd], "group ... seq width -> ... seq (width group)")
