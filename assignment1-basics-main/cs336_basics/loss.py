from __future__ import annotations

from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import rearrange, reduce
import torch


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:

    max_values = torch.amax(inputs, dim=-1, keepdim=True)

    log_sum_exp = torch.log(torch.sum(torch.exp(inputs - max_values), dim=-1)) + max_values.squeeze(-1)

    target_logits = torch.gather(
        inputs,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    return torch.mean(log_sum_exp - target_logits)
