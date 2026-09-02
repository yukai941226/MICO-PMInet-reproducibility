from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, diags, eye
from scipy.sparse.linalg import spsolve

from .config import MODELED_PMIS, UNSEEN_PMIS


ORGAN_REPLACEMENTS = {
    "B": "Brain",
    "H": "Heart",
    "K": "Kidney",
    "L": "Liver",
    "Lung": "Lung",
    "Muscle": "Muscle",
    "S": "Spleen",
    "VH": "VH",
}


def spectral_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        try:
            float(column)
        except (TypeError, ValueError):
            continue
        columns.append(column)
    return columns


def snv(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, keepdims=True)
    if np.any(stds == 0):
        raise ValueError("SNV cannot process a constant spectrum.")
    return (values - means) / stds


def _whittaker_smooth(y: np.ndarray, weights: np.ndarray, lam: float) -> np.ndarray:
    size = y.size
    identity = eye(size, format="csc")
    difference = identity[1:] - identity[:-1]
    weight_matrix = diags(weights, 0, shape=(size, size))
    system = csc_matrix(weight_matrix + lam * difference.T * difference)
    rhs = csc_matrix(weight_matrix @ np.asmatrix(y).T)
    return np.asarray(spsolve(system, rhs))


def airpls(y: np.ndarray, lam: float = 100.0, max_iter: int = 15) -> np.ndarray:
    """Match the ZhangFit implementation used for the reported analysis."""
    original = np.asarray(y, dtype=float)
    weights = np.ones(original.size)
    baseline = np.zeros_like(original)
    for iteration in range(1, max_iter + 1):
        baseline = _whittaker_smooth(original, weights, lam)
        residual = original - baseline
        negative_sum = abs(residual[residual < 0].sum())
        if negative_sum < 0.001 * abs(original).sum() or iteration == max_iter:
            break
        weights[residual >= 0] = 0
        weights[residual < 0] = np.exp(
            iteration * abs(residual[residual < 0]) / negative_sum
        )
        edge_weight = np.exp(iteration * residual[residual < 0].max() / negative_sum)
        weights[0] = edge_weight
        weights[-1] = edge_weight
    return original - baseline


def preprocess_spectra(raw: pd.DataFrame) -> pd.DataFrame:
    feature_columns = spectral_columns(raw)
    wavenumbers = np.asarray([float(column) for column in feature_columns])
    selected = (wavenumbers >= 900.0) & (wavenumbers <= 1800.0)
    selected_columns = [column for column, keep in zip(feature_columns, selected) if keep]
    if len(selected_columns) != 467:
        raise ValueError(f"Expected 467 fingerprint features, found {len(selected_columns)}")

    transformed = snv(raw[selected_columns].to_numpy(dtype=float))
    corrected = np.vstack([airpls(row) for row in transformed])
    metadata = raw.drop(columns=feature_columns).reset_index(drop=True)
    corrected_frame = pd.DataFrame(corrected, columns=selected_columns)
    return pd.concat([metadata, corrected_frame], axis=1)


def average_technical_spectra(corrected: pd.DataFrame) -> pd.DataFrame:
    feature_columns = spectral_columns(corrected)
    identifiers = corrected["label_all"].astype(str).str.split("-")
    average_ids = identifiers.map(lambda parts: "-".join(parts[:4]))

    values = corrected[feature_columns].copy()
    values.insert(0, "label_average", average_ids)
    averaged = values.groupby("label_average", sort=True).mean().reset_index()

    parts = averaged["label_average"].str.split("-")
    for index in range(4):
        averaged.insert(index + 1, f"label_{index + 1}", parts.str[index])

    averaged["label_2"] = averaged["label_2"].replace(ORGAN_REPLACEMENTS)
    pmi_text = averaged["label_1"].replace({"20min": "0.33333h"})
    averaged.insert(1, "PMI", pmi_text.str.removesuffix("h").astype(float))
    averaged = averaged.drop(columns="label_1")
    averaged["label_4"] = averaged["label_4"].astype(int)
    # Preserve the exact row ordering used by the archived preprocessing notebook.
    averaged = averaged.sort_values(["PMI", "label_2"]).reset_index(drop=True)

    if len(averaged) != 704:
        raise ValueError(f"Expected 704 averaged organ samples, found {len(averaged)}")
    return averaged


def split_by_animal(
    averaged: pd.DataFrame,
    split_seed: int = 22,
    test_animals_per_pmi: int = 2,
) -> dict[str, pd.DataFrame]:
    modeled = averaged[averaged["PMI"].isin(MODELED_PMIS)].copy()
    unseen = averaged[averaged["PMI"].isin(UNSEEN_PMIS)].copy()
    rng = np.random.RandomState(split_seed)
    test_indices: list[int] = []

    for pmi in sorted(modeled["PMI"].unique()):
        chosen_animals = rng.choice(
            np.arange(1, 11), size=test_animals_per_pmi, replace=False
        )
        pmi_rows = modeled[(modeled["PMI"] == pmi) & modeled["label_4"].isin(chosen_animals)]
        test_indices.extend(pmi_rows.index.tolist())

    test_modeled = modeled.loc[test_indices].copy()
    train = modeled.drop(index=test_indices).copy()
    test_combined = pd.concat([test_modeled, unseen], ignore_index=True)

    outputs = {
        "train_dataset": train.reset_index(drop=True),
        "test_modeled": test_modeled.reset_index(drop=True),
        "test_unseen": unseen.reset_index(drop=True),
        "test_dataset": test_combined.sort_values(["PMI", "label_2"]).reset_index(drop=True),
    }
    expected_rows = {
        "train_dataset": 512,
        "test_modeled": 128,
        "test_unseen": 64,
        "test_dataset": 192,
    }
    for name, frame in outputs.items():
        if len(frame) != expected_rows[name]:
            raise ValueError(f"{name}: expected {expected_rows[name]} rows, found {len(frame)}")
    return outputs


def run_preprocessing(raw_path: Path, output_dir: Path, split_seed: int = 22) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading raw spectra: {raw_path}")
    raw = pd.read_csv(raw_path)
    if len(raw) != 6336:
        raise ValueError(f"Expected 6,336 raw spectra, found {len(raw)}")

    # Author-confirmed executed order: preprocess every technical spectrum
    # independently, then average the nine corrected spectra into the sample's
    # representative spectrum.
    print("Applying 1800-900 cm^-1 selection, SNV, and airPLS...")
    corrected = preprocess_spectra(raw)
    print("Averaging nine preprocessed technical spectra per animal-organ sample...")
    averaged = average_technical_spectra(corrected)
    partitions = split_by_animal(averaged, split_seed=split_seed)

    deterministic_gzip = {"method": "gzip", "mtime": 0}
    averaged.to_csv(
        output_dir / "data_all_average.csv.gz",
        index=False,
        compression=deterministic_gzip,
    )
    for name, frame in partitions.items():
        frame.to_csv(
            output_dir / f"{name}.csv.gz",
            index=False,
            compression=deterministic_gzip,
        )
    print(f"Processed datasets written to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild MICO-PMInet processed data")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=22)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_preprocessing(args.raw, args.output_dir, args.split_seed)


if __name__ == "__main__":
    main()
