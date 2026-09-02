from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    VARIANT_TO_COMPONENTS,
    Protocol,
    model_name,
)
from .data import TensorData
from .evaluation import evaluate_retention
from .masking import expand_missing_levels
from .models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def stratified_repeated_holdout(
    y: np.ndarray,
    n_splits: int,
    test_size: float,
    random_state: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(random_state)
    label_to_indices: dict[float, np.ndarray] = {}
    grouped: dict[float, list[int]] = defaultdict(list)
    for index, label in enumerate(y):
        grouped[float(label)].append(index)
    for label, indices in grouped.items():
        label_to_indices[label] = np.asarray(indices)
    for _ in range(n_splits):
        validation: list[int] = []
        for indices in label_to_indices.values():
            count = max(1, int(np.floor(len(indices) * test_size)))
            validation.extend(indices[rng.permutation(len(indices))[:count]].tolist())
        val_idx = np.asarray(sorted(validation))
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx, assume_unique=True)
        yield train_idx, val_idx


def joint_loss(
    final_prediction: torch.Tensor,
    organ_predictions: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    main = nn.functional.mse_loss(final_prediction, y_true)
    branches = organ_predictions.squeeze(-1)
    target = y_true.unsqueeze(1).expand_as(branches)
    branch = (((branches - target) ** 2) * mask).mean()
    count = mask.sum(dim=1, keepdim=True).clamp(min=1)
    # The article's Equation (17) contains only the sample-level main loss and
    # organ-level auxiliary loss.  gamma is retained solely so the explicitly
    # named archived-checkpoint protocol can still be inspected; it is zero in
    # the default manuscript protocol and therefore adds no third loss term.
    if gamma == 0:
        return alpha * main + beta * branch
    branch_mean = (branches * mask).sum(dim=1, keepdim=True) / count
    consistency = (((branches - branch_mean) ** 2) * mask).mean()
    return alpha * main + beta * branch + gamma * consistency


def _metrics(y: torch.Tensor, prediction: np.ndarray) -> dict[str, float]:
    truth = y.detach().cpu().numpy()
    return {
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "MAE": float(mean_absolute_error(truth, prediction)),
        "R2": float(r2_score(truth, prediction)),
    }


def train_one_fold(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    strategy: str,
    beta: float,
    protocol: Protocol,
    device: torch.device,
    run_seed: int,
) -> tuple[nn.Module, dict[str, float]]:
    seed_everything(run_seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=protocol.scheduler_factor,
        patience=protocol.scheduler_patience,
    )
    generator = torch.Generator().manual_seed(run_seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=protocol.batch_size,
        shuffle=True,
        generator=generator,
    )
    rng = np.random.RandomState(run_seed)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    epochs_trained = 0

    x_val_device = x_val.to(device)
    y_val_device = y_val.to(device)
    for epoch in range(protocol.max_epochs):
        model.train()
        for batch_x, batch_y in loader:
            if strategy == "random_masking":
                expanded_x, mask, expanded_y, _ = expand_missing_levels(
                    batch_x,
                    batch_y,
                    repeats=protocol.morm_training_repeats,
                    deterministic=False,
                    rng=rng,
                )
            else:
                expanded_x, expanded_y = batch_x, batch_y
                mask = torch.ones(batch_x.shape[0], protocol.num_organs)
            expanded_x = expanded_x.to(device)
            expanded_y = expanded_y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            final, branches = model(expanded_x, mask)
            loss = joint_loss(
                final,
                branches,
                expanded_y,
                mask,
                protocol.alpha,
                beta,
                protocol.gamma,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), protocol.gradient_clip)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if strategy == "random_masking":
                val_x, val_mask, val_y, _ = expand_missing_levels(
                    x_val_device,
                    y_val_device,
                    repeats=protocol.morm_validation_repeats,
                    deterministic=True,
                )
            else:
                val_x, val_y = x_val_device, y_val_device
                val_mask = torch.ones(x_val.shape[0], protocol.num_organs, device=device)
            final, branches = model(val_x, val_mask)
            val_loss = joint_loss(
                final,
                branches,
                val_y,
                val_mask,
                protocol.alpha,
                beta,
                protocol.gamma,
            ).item()
        scheduler.step(val_loss)
        epochs_trained = epoch + 1
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= protocol.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if strategy == "random_masking":
            train_x, train_mask, train_y, _ = expand_missing_levels(
                x_train,
                y_train,
                repeats=protocol.morm_training_repeats,
                deterministic=True,
            )
            metric_val_x, metric_val_mask, metric_val_y, _ = expand_missing_levels(
                x_val,
                y_val,
                repeats=protocol.morm_validation_repeats,
                deterministic=True,
            )
        else:
            train_x, train_y = x_train, y_train
            train_mask = torch.ones(x_train.shape[0], protocol.num_organs)
            # This follows the archived analysis: MOCE validation metrics were
            # evaluated over retention masks although early stopping used full input.
            metric_val_x, metric_val_mask, metric_val_y, _ = expand_missing_levels(
                x_val,
                y_val,
                repeats=protocol.morm_validation_repeats,
                deterministic=True,
            )
        train_prediction = model(train_x.to(device), train_mask.to(device))[0].cpu().numpy()
        val_prediction = model(
            metric_val_x.to(device), metric_val_mask.to(device)
        )[0].cpu().numpy()
    train_metrics = _metrics(train_y, train_prediction)
    val_metrics = _metrics(metric_val_y, val_prediction)
    return model, {
        "best_val_loss": best_loss,
        "epochs_trained": epochs_trained,
        **{f"Train_{key}": value for key, value in train_metrics.items()},
        **{f"Val_{key}": value for key, value in val_metrics.items()},
    }


def run_experiment(
    data: TensorData,
    protocol: Protocol,
    output_dir: Path,
    profile: str,
    device_name: str,
    resume: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if profile == "smoke":
        variants = ["no_attention"]
        strategies = ["random_masking"]
        betas = [0.2]
        splits = 1
        protocol = Protocol(**{**asdict(protocol), "max_epochs": 2, "early_stopping_patience": 2})
    elif profile == "optimal":
        variants = ["no_attention"]
        strategies = ["random_masking"]
        betas = [0.2]
        splits = protocol.cv_splits
    elif profile == "paper":
        variants = list(VARIANT_TO_COMPONENTS)
        strategies = ["no_masking", "random_masking"]
        betas = list(protocol.beta_candidates)
        splits = protocol.cv_splits
    else:
        raise ValueError(f"Unknown profile: {profile}")

    fold_path = output_dir / "fold_metrics_all_betas.csv"
    existing = pd.read_csv(fold_path) if resume and fold_path.exists() else pd.DataFrame()
    records = existing.to_dict("records")
    completed = {
        (row["variant"], row["strategy"], float(row["beta"]), int(row["fold"]))
        for row in records
    }
    splits_list = list(
        stratified_repeated_holdout(
            data.y_train.numpy(),
            n_splits=splits,
            test_size=protocol.cv_validation_fraction_per_pmi,
            random_state=protocol.split_seed,
        )
    )

    run_index = 0
    for strategy in strategies:
        for variant in variants:
            branch, attention, aggregation = VARIANT_TO_COMPONENTS[variant]
            for beta in betas:
                for fold_index, (train_idx, val_idx) in enumerate(splits_list, start=1):
                    # Increment for every scheduled job, including jobs skipped
                    # by --resume, so an interrupted run keeps the same seed as
                    # an uninterrupted run.
                    run_index += 1
                    key = (variant, strategy, float(beta), fold_index)
                    if key in completed:
                        continue
                    run_seed = protocol.seed + run_index
                    seed_everything(run_seed)
                    print(
                        f"[{model_name(variant, strategy)}] beta={beta} fold={fold_index}/{splits}"
                    )
                    model = build_model(
                        branch,
                        attention,
                        aggregation,
                        protocol.num_organs,
                        protocol.input_dim,
                        protocol.hidden_dim,
                        protocol.dropout,
                    )
                    model, train_metrics = train_one_fold(
                        model,
                        data.x_train[train_idx],
                        data.y_train[train_idx],
                        data.x_train[val_idx],
                        data.y_train[val_idx],
                        strategy,
                        beta,
                        protocol,
                        device,
                        run_seed,
                    )
                    retention = evaluate_retention(
                        model,
                        data.x_test,
                        data.y_test,
                        protocol.retention_test_repeats,
                        device,
                    )
                    row: dict[str, float | int | str] = {
                        "fold": fold_index,
                        "variant": variant,
                        "strategy": strategy,
                        "model_name": model_name(variant, strategy),
                        "beta": beta,
                        **train_metrics,
                    }
                    overall = retention[retention["reserve_organs"] == "Test"].iloc[0]
                    for metric in ("RMSE", "MAE", "R2"):
                        row[f"Test_{metric}"] = float(overall[metric])
                    for _, retention_row in retention.iterrows():
                        if retention_row["reserve_organs"] == "Test":
                            continue
                        retained = int(retention_row["reserve_organs"])
                        for metric in ("R2", "RMSE", "MAE"):
                            row[f"reserveorgan-{retained}-{metric}"] = float(retention_row[metric])
                    records.append(row)
                    pd.DataFrame(records).to_csv(fold_path, index=False)

                    if (
                        variant == "no_attention"
                        and strategy == "random_masking"
                        and float(beta) == 0.2
                        and fold_index == 5
                    ):
                        torch.save(model.state_dict(), output_dir / "mico_pminet_fold5.pt")

    all_folds = pd.DataFrame(records)
    best_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []
    for (variant, strategy), group in all_folds.groupby(["variant", "strategy"]):
        beta_scores = group.groupby("beta")["Val_RMSE"].mean()
        best_beta = float(beta_scores.idxmin())
        selected = group[group["beta"] == best_beta].copy()
        selected["best_beta"] = best_beta
        best_rows.append(selected)
        summary: dict[str, float | int | str] = {
            "model_name": model_name(variant, strategy),
            "variant": variant,
            "strategy": strategy,
            "best_beta": best_beta,
            "n_folds": len(selected),
        }
        metric_columns = [
            column
            for column in selected.columns
            if column.startswith(("Train_", "Val_", "Test_", "reserveorgan-"))
            and column != "epochs_trained"
        ]
        for column in metric_columns:
            summary[column] = float(selected[column].mean())
            summary[f"{column}_std"] = float(selected[column].std(ddof=1))
        summaries.append(summary)
    pd.concat(best_rows, ignore_index=True).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "model_summary.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps({"profile": profile, "device": str(device), "protocol": asdict(protocol)}, indent=2),
        encoding="utf-8",
    )
