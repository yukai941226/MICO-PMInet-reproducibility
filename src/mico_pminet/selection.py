from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def validation_only_selection(
    fold_results: Path,
    output_dir: Path,
    criterion: str = "Val_RMSE",
    tie_breaker: str = "Val_MAE",
    target_model: str = "OSB-WMHA-AWA-MORM",
) -> dict[str, object]:
    """Select beta for the manuscript model without reading test metrics.

    The input may contain either one row per split (``Val_RMSE`` plus ``fold``)
    or the released ten-split beta summaries (``Val_RMSE_mean`` and
    ``Val_RMSE_std``).  In both cases, only model identifiers and validation
    columns are copied into the selection frame.
    """
    data = pd.read_csv(fold_results)
    identifiers = {"model_name", "beta"}
    missing_identifiers = identifiers - set(data.columns)
    if missing_identifiers:
        raise ValueError(f"Missing model-selection columns: {sorted(missing_identifiers)}")

    if {"fold", criterion, tie_breaker}.issubset(data.columns):
        # Deliberately construct a validation-only frame. Test columns cannot affect ranking.
        validation = data[["model_name", "beta", "fold", criterion, tie_breaker]].copy()
        beta_summary = validation.groupby(["model_name", "beta"], as_index=False).agg(
            mean_validation_criterion=(criterion, "mean"),
            sd_validation_criterion=(criterion, "std"),
            mean_validation_tie_breaker=(tie_breaker, "mean"),
            n_splits=("fold", "nunique"),
        )
        input_granularity = "one row per validation split"
    else:
        criterion_mean = f"{criterion}_mean"
        criterion_sd = f"{criterion}_std"
        tie_mean = f"{tie_breaker}_mean"
        required_summary = {criterion_mean, criterion_sd, tie_mean, "n_folds"}
        missing = required_summary - set(data.columns)
        if missing:
            raise ValueError(f"Missing model-selection columns: {sorted(missing)}")
        validation = data[
            [
                "model_name",
                "beta",
                criterion_mean,
                criterion_sd,
                tie_mean,
                "n_folds",
            ]
        ].copy()
        beta_summary = validation.rename(
            columns={
                criterion_mean: "mean_validation_criterion",
                criterion_sd: "sd_validation_criterion",
                tie_mean: "mean_validation_tie_breaker",
                "n_folds": "n_splits",
            }
        )
        input_granularity = "aggregated validation summaries"

    beta_summary = beta_summary.sort_values(
        ["model_name", "mean_validation_criterion", "mean_validation_tie_breaker", "beta"],
        kind="mergesort",
    )
    model_rows = beta_summary[beta_summary["model_name"] == target_model].reset_index(drop=True)
    if model_rows.empty:
        raise ValueError(f"Target model not found in selection table: {target_model}")
    winner = model_rows.iloc[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    beta_summary.to_csv(output_dir / "validation_beta_ranking.csv", index=False)
    payload: dict[str, object] = {
        "selection_scope": "beta selection for the manuscript model using internal validation",
        "input_granularity": input_granularity,
        "candidate_rows": int(len(beta_summary)),
        "candidate_models": int(beta_summary["model_name"].nunique()),
        "criterion": criterion,
        "tie_breaker": tie_breaker,
        "selected_model": target_model,
        "selected_beta": float(winner["beta"]),
        "mean_validation_criterion": float(winner["mean_validation_criterion"]),
        "mean_validation_tie_breaker": float(winner["mean_validation_tie_breaker"]),
        "test_metrics_used_for_selection": False,
    }
    (output_dir / "beta_selection.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return payload
