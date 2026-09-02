from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .config import ORGAN_ORDER
from .data import load_tensor_data, resolve_table
from .evaluation import evaluate_retention
from .models import build_from_name


def _load_model(checkpoint: Path, model_name: str) -> torch.nn.Module:
    model = build_from_name(model_name)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return model


def evaluate_checkpoint_by_cohort(
    checkpoint: Path,
    model_name: str,
    processed_dir: Path,
    output_dir: Path,
    repeats: int = 10,
) -> None:
    """Report modeled-time and unseen-time tests separately."""
    data = load_tensor_data(processed_dir)
    model = _load_model(checkpoint, model_name)
    cohorts = {
        "modeled_time_test": (data.x_test_modeled, data.y_test_modeled),
        "unseen_time_test": (data.x_test_unseen, data.y_test_unseen),
        "combined_test": (data.x_test, data.y_test),
    }
    rows: list[pd.DataFrame] = []
    for cohort, (x, y) in cohorts.items():
        result = evaluate_retention(model, x, y, repeats=repeats)
        result.insert(0, "cohort", cohort)
        result.insert(1, "n_animals", len(y))
        rows.append(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = pd.concat(rows, ignore_index=True)
    report.to_csv(output_dir / "checkpoint_metrics_by_cohort.csv", index=False)
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "checkpoint": str(checkpoint),
                "retention_repeats": repeats,
                "selection_warning": (
                    "This file reports checkpoint performance by held-out cohort and organ "
                    "retention level; see the manuscript for the interpretation of these results."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report.to_string(index=False))


def export_awa_weights(
    checkpoints: list[Path],
    model_name: str,
    processed_dir: Path,
    output_dir: Path,
) -> None:
    """Export actual AWA weights; these are distinct from SHAP importance."""
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    data = load_tensor_data(processed_dir)

    cohort_tensors = {
        "modeled_time_test": (data.x_test_modeled, data.y_test_modeled, "test_modeled"),
        "unseen_time_test": (data.x_test_unseen, data.y_test_unseen, "test_unseen"),
    }
    sample_rows: list[dict[str, float | int | str]] = []
    for checkpoint in checkpoints:
        model = _load_model(checkpoint, model_name)
        if getattr(model, "aggregation", None) != "AWA":
            raise ValueError(f"{model_name} is not an AWA model")
        with torch.no_grad():
            for cohort, (x, y, table_stem) in cohort_tensors.items():
                weights = model.normalized_organ_weights(x).cpu().numpy()
                table = pd.read_csv(resolve_table(processed_dir, table_stem))
                metadata = (
                    table[["PMI", "label_4"]]
                    .drop_duplicates()
                    .sort_values(["PMI", "label_4"])
                    .reset_index(drop=True)
                )
                if len(metadata) != len(weights):
                    raise ValueError(f"Metadata/tensor mismatch for {cohort}")
                for sample_index, row in metadata.iterrows():
                    record: dict[str, float | int | str] = {
                        "checkpoint": checkpoint.name,
                        "cohort": cohort,
                        "PMI": float(row["PMI"]),
                        "animal_within_PMI": int(row["label_4"]),
                    }
                    record.update(
                        {
                            organ: float(weights[sample_index, organ_index])
                            for organ_index, organ in enumerate(ORGAN_ORDER)
                        }
                    )
                    sample_rows.append(record)

    samples = pd.DataFrame(sample_rows)
    long = samples.melt(
        id_vars=["checkpoint", "cohort", "PMI", "animal_within_PMI"],
        value_vars=list(ORGAN_ORDER),
        var_name="organ",
        value_name="AWA_weight",
    )
    by_checkpoint = (
        long.groupby(["checkpoint", "cohort", "organ"])["AWA_weight"]
        .mean()
        .reset_index()
    )
    summary = (
        by_checkpoint.groupby(["cohort", "organ"])["AWA_weight"]
        .agg(checkpoint_mean="mean", checkpoint_sd="std", checkpoint_min="min", checkpoint_max="max")
        .reset_index()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(output_dir / "awa_weights_per_sample_and_checkpoint.csv", index=False)
    by_checkpoint.to_csv(output_dir / "awa_weights_by_checkpoint.csv", index=False)
    summary.to_csv(output_dir / "awa_weights_summary_across_checkpoints.csv", index=False)
    print(summary.to_string(index=False))
