from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .masking import mask_exact


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_retention(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    repeats: int = 10,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    model.eval()
    device = torch.device(device)
    rows: list[dict[str, float | int | str]] = []
    combined_predictions: list[np.ndarray] = []
    combined_truth: list[np.ndarray] = []
    with torch.no_grad():
        for missing_count in range(x.shape[1]):
            masked_list: list[torch.Tensor] = []
            mask_list: list[torch.Tensor] = []
            truth_list: list[torch.Tensor] = []
            for repeat in range(repeats):
                rng = np.random.RandomState(1000 * missing_count + repeat)
                masked, mask = mask_exact(x, missing_count, rng)
                masked_list.append(masked)
                mask_list.append(mask)
                truth_list.append(y)
            masked_all = torch.cat(masked_list).to(device)
            mask_all = torch.cat(mask_list).to(device)
            y_all = torch.cat(truth_list).cpu().numpy()
            predictions = model(masked_all, mask_all)[0].cpu().numpy()
            metrics = regression_metrics(y_all, predictions)
            rows.append({"reserve_organs": x.shape[1] - missing_count, **metrics})
            combined_predictions.append(predictions)
            combined_truth.append(y_all)

    overall = regression_metrics(
        np.concatenate(combined_truth), np.concatenate(combined_predictions)
    )
    rows.append({"reserve_organs": "Test", **overall})
    return pd.DataFrame(rows)

