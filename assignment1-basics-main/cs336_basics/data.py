import numpy.typing as npt
import torch
import numpy as np


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str = "mps"
) -> tuple[torch.Tensor, torch.Tensor]:

    starts = np.random.randint(
        0,
        len(dataset) - context_length,
        size=batch_size,
    )

    offsets = np.arange(context_length)
    indices = starts[:, None] + offsets[None, :]

    inputs = torch.as_tensor(
        dataset[indices],
        dtype=torch.long,
        device=device,
    )
    targets = torch.as_tensor(
        dataset[indices + 1],
        dtype=torch.long,
        device=device,
    )

    return inputs, targets
