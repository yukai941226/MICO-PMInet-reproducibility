from __future__ import annotations

import numpy as np
import torch


def mask_exact(
    x: torch.Tensor,
    missing_count: int,
    rng: np.random.RandomState | np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_organs = x.shape[1]
    if not 0 <= missing_count < num_organs:
        raise ValueError("missing_count must retain at least one organ")
    masked = x.clone()
    mask = torch.ones(x.shape[0], num_organs, dtype=torch.float32, device=x.device)
    if missing_count:
        for sample_index in range(x.shape[0]):
            missing = rng.choice(num_organs, missing_count, replace=False)
            masked[sample_index, missing] = 0
            mask[sample_index, missing] = 0
    return masked, mask


def expand_missing_levels(
    x: torch.Tensor,
    y: torch.Tensor,
    repeats: int,
    deterministic: bool,
    rng: np.random.RandomState | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rng is None:
        rng = np.random.RandomState()
    xs: list[torch.Tensor] = [x.clone()]
    masks: list[torch.Tensor] = [
        torch.ones(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    ]
    labels: list[torch.Tensor] = [y]
    levels: list[torch.Tensor] = [torch.zeros(x.shape[0], dtype=torch.long, device=x.device)]

    for repeat in range(repeats):
        for missing_count in range(1, x.shape[1]):
            current_rng = (
                np.random.RandomState(1000 + 100 * repeat + missing_count)
                if deterministic
                else rng
            )
            masked, mask = mask_exact(x, missing_count, current_rng)
            xs.append(masked)
            masks.append(mask)
            labels.append(y)
            levels.append(
                torch.full((x.shape[0],), missing_count, dtype=torch.long, device=x.device)
            )
    return (
        torch.cat(xs),
        torch.cat(masks),
        torch.cat(labels),
        torch.cat(levels),
    )
