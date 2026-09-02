from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ORGAN_ORDER
from .data import load_tensor_data
from .models import build_from_name


def _normalize_shap_output(values: object, samples: int, features: int) -> np.ndarray:
    if isinstance(values, list):
        values = values[0]
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape != (samples, features):
        raise ValueError(f"Unexpected SHAP shape: {array.shape}")
    return array


def summarize_shap(
    feature_shap_3d: np.ndarray,
    wavenumbers: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    organ_per_sample = np.abs(feature_shap_3d).sum(axis=2)
    importance = organ_per_sample.mean(axis=0)
    organ_frame = pd.DataFrame({"Organ": ORGAN_ORDER, "Importance": importance})
    organ_frame["Importance_%"] = 100 * organ_frame["Importance"] / organ_frame["Importance"].sum()
    organ_frame = organ_frame.sort_values("Importance", ascending=False).reset_index(drop=True)
    organ_frame.insert(1, "Rank", np.arange(1, len(organ_frame) + 1))
    organ_frame.to_csv(output_dir / "organ_importance.csv", index=False)

    feature_rows: list[dict[str, float | int | str]] = []
    for organ_index, organ in enumerate(ORGAN_ORDER):
        values = np.abs(feature_shap_3d[:, organ_index, :]).mean(axis=0)
        order = np.argsort(values)[::-1][:30]
        for rank, feature_index in enumerate(order, start=1):
            feature_rows.append(
                {
                    "Organ": organ,
                    "Rank": rank,
                    "Feature_index": int(feature_index),
                    "Wavenumber_cm-1": float(wavenumbers[feature_index]),
                    "Mean_abs_SHAP": float(values[feature_index]),
                }
            )
    pd.DataFrame(feature_rows).to_csv(output_dir / "top30_features_per_organ.csv", index=False)
    print(organ_frame.round(4).to_string(index=False))


def run_cached_shap(cache_path: Path, processed_dir: Path, output_dir: Path) -> None:
    archive = np.load(cache_path)
    data = load_tensor_data(processed_dir)
    summarize_shap(archive["feature_shap_3d"], data.wavenumbers, output_dir)


def recompute_shap(
    checkpoint: Path,
    processed_dir: Path,
    output_dir: Path,
    seed: int = 225,
    model_name: str = "OSB-WMHA-AWA-MORM",
) -> None:
    import shap

    data = load_tensor_data(processed_dir)
    x = torch.cat([data.x_train, data.x_test]).numpy()
    flat = x.reshape(len(x), -1)
    model = build_from_name(model_name)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    def predictor(values: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(values, dtype=torch.float32).reshape(-1, 8, 467)
        mask = torch.ones(tensor.shape[0], 8)
        with torch.no_grad():
            return model(tensor, mask)[0].numpy().reshape(-1, 1)

    rng = np.random.RandomState(seed)
    # Use training samples only for the background distribution. The test cohort
    # remains excluded from explainer construction.
    train_flat = data.x_train.numpy().reshape(len(data.x_train), -1)
    background = train_flat[
        rng.choice(len(train_flat), min(50, len(train_flat)), replace=False)
    ]
    explainer = shap.KernelExplainer(predictor, background)
    values = _normalize_shap_output(explainer.shap_values(flat), len(flat), flat.shape[1])
    feature_shap_3d = values.reshape(len(flat), 8, 467)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "shap_values.npz", feature_shap_3d=feature_shap_3d)
    (output_dir / "shap_metadata.json").write_text(
        __import__("json").dumps(
            {
                "model_name": model_name,
                "checkpoint": str(checkpoint),
                "explainer": "shap.KernelExplainer",
                "background_source": "training cohort only",
                "background_size": len(background),
                "explained_samples": len(flat),
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summarize_shap(feature_shap_3d, data.wavenumbers, output_dir)
