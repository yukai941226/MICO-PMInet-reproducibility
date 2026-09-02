from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ORGAN_ORDER, load_protocol
from .data import load_tensor_data, resolve_table
from .models import build_from_name
from .preprocess import spectral_columns


EXPECTED_FINAL = {
    "Test_R2": 0.90,
    "Test_RMSE": 2.56,
    "Test_MAE": 1.62,
    "reserveorgan-8-R2": 0.93,
    "reserveorgan-8-RMSE": 2.08,
    "reserveorgan-8-MAE": 1.36,
    "reserveorgan-1-R2": 0.78,
    "reserveorgan-1-RMSE": 3.71,
    "reserveorgan-1-MAE": 2.37,
}

def _check_frame(frame: pd.DataFrame, expected_rows: int, expected_animals: int, name: str) -> None:
    if len(frame) != expected_rows:
        raise AssertionError(f"{name}: expected {expected_rows} rows, found {len(frame)}")
    if len(spectral_columns(frame)) != 467:
        raise AssertionError(f"{name}: expected 467 spectral columns")
    counts = frame.groupby(["PMI", "label_4"])["label_2"].agg(set)
    if len(counts) != expected_animals:
        raise AssertionError(f"{name}: expected {expected_animals} animals, found {len(counts)}")
    if not counts.map(lambda organs: organs == set(ORGAN_ORDER)).all():
        raise AssertionError(f"{name}: at least one animal lacks an organ")


def verify(
    processed_dir: Path,
    reference_dir: Path,
    deep_results: Path | None = None,
) -> None:
    protocol = load_protocol("manuscript_protocol")
    article_settings = {
        "max_epochs": 1000,
        "early_stopping_patience": 200,
        "cv_validation_fraction_per_pmi": 0.25,
        "morm_training_repeats": 2,
        "morm_validation_repeats": 10,
        "gamma": 0.0,
    }
    for field, expected in article_settings.items():
        if not np.isclose(float(getattr(protocol, field)), float(expected)):
            raise AssertionError(f"Article protocol mismatch for {field}")
    train = pd.read_csv(resolve_table(processed_dir, "train_dataset"))
    test = pd.read_csv(resolve_table(processed_dir, "test_dataset"))
    _check_frame(train, 512, 64, "train_dataset")
    _check_frame(test, 192, 24, "test_dataset")
    train_ids = set(zip(train["PMI"], train["label_4"]))
    test_modeled = test[test["PMI"].isin(train["PMI"].unique())]
    modeled_test_ids = set(zip(test_modeled["PMI"], test_modeled["label_4"]))
    if train_ids & modeled_test_ids:
        raise AssertionError("Animal-level leakage detected between train and modeled test")

    data = load_tensor_data(processed_dir)
    if data.x_train.shape != (64, 8, 467) or data.x_test.shape != (24, 8, 467):
        raise AssertionError("Unexpected tensor shapes")
    if data.x_test_modeled.shape != (16, 8, 467):
        raise AssertionError("Unexpected modeled-time test tensor shape")
    if data.x_test_unseen.shape != (8, 8, 467):
        raise AssertionError("Unexpected unseen-time test tensor shape")
    if not np.isclose(data.wavenumbers[0], 900.5939) or not np.isclose(
        data.wavenumbers[-1], 1799.259
    ):
        raise AssertionError("Unexpected wavenumber range")

    checkpoint = reference_dir / "mico_pminet_fold5.pt"
    model = build_from_name("OSB-WMHA-AWA-MORM")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        prediction = model(data.x_test[:2], torch.ones(2, 8))[0]
    if prediction.shape != (2,) or not torch.isfinite(prediction).all():
        raise AssertionError("Final checkpoint inference failed")

    metrics_path = deep_results or reference_dir / "published_model_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    row = metrics[metrics["model_name"] == "OSB-WMHA-AWA-MORM"].iloc[0]
    for metric, expected in EXPECTED_FINAL.items():
        candidate_columns = (metric, f"{metric}_mean")
        available = next((column for column in candidate_columns if column in row.index), None)
        if available is None or not np.isclose(float(row[available]), expected, atol=0.02):
            raise AssertionError(f"{metric}: expected approximately {expected}")

    beta_summary = pd.read_csv(reference_dir / "published_beta_summary.csv")
    if len(beta_summary) != 128 or beta_summary["model_name"].nunique() != 16:
        raise AssertionError("Expected 128 validation-only beta summaries for 16 models")
    osb_betas = beta_summary[beta_summary["model_name"] == "OSB-WMHA-AWA-MORM"].sort_values(
        ["Val_RMSE_mean", "Val_MAE_mean", "beta"], kind="mergesort"
    )
    if osb_betas.empty or not np.isclose(float(osb_betas.iloc[0]["beta"]), 0.2):
        raise AssertionError("OSB manuscript-model beta selection changed unexpectedly")

    shap_expected = pd.read_csv(reference_dir / "published_organ_importance.csv")
    expected_order = ["Liver", "Kidney", "Heart", "Spleen", "Lung", "Brain", "Muscle", "VH"]
    if shap_expected.sort_values("Rank")["Organ"].tolist() != expected_order:
        raise AssertionError("Unexpected SHAP organ order")

    summary = {
        "train_tensor": list(data.x_train.shape),
        "test_tensor": list(data.x_test.shape),
        "modeled_time_test_tensor": list(data.x_test_modeled.shape),
        "unseen_time_test_tensor": list(data.x_test_unseen.shape),
        "wavenumber_range_cm-1": [float(data.wavenumbers[0]), float(data.wavenumbers[-1])],
        "checkpoint_inference": "passed",
        "article_protocol": "48/16, 1000/200, two-term loss passed",
        "published_reference_metric_table": "passed",
        "manuscript_model_beta_selection": "OSB beta 0.2 passed",
        "beta_grid": "128 summaries passed",
        "shap_order": "passed",
    }
    print(json.dumps(summary, indent=2))
    print("All reproducibility checks passed.")
