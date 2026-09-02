from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .data import resolve_table
from .preprocess import spectral_columns


def stratified_splits(
    y: np.ndarray,
    n_splits: int = 10,
    test_size: float = 0.3,
    random_state: int = 225,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(random_state)
    grouped: dict[float, list[int]] = defaultdict(list)
    for index, label in enumerate(y):
        grouped[float(label)].append(index)
    arrays = {label: np.asarray(indices) for label, indices in grouped.items()}
    for _ in range(n_splits):
        test: list[int] = []
        for indices in arrays.values():
            count = max(1, int(np.floor(len(indices) * test_size)))
            test.extend(indices[rng.permutation(len(indices))[:count]].tolist())
        test_idx = np.asarray(sorted(test))
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=True)
        yield train_idx, test_idx


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    prediction = np.maximum(np.asarray(y_pred).reshape(-1), 0)
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "R2": float(r2_score(y_true, prediction)),
    }


def run_pls(processed_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(resolve_table(processed_dir, "train_dataset"))
    test = pd.read_csv(resolve_table(processed_dir, "test_dataset"))
    features = spectral_columns(train)
    result_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[dict[str, float | str]] = []

    for organ in sorted(train["label_2"].unique()):
        organ_train = train[train["label_2"] == organ].reset_index(drop=True)
        organ_test = test[test["label_2"] == organ].reset_index(drop=True)
        x_train = organ_train[features].to_numpy()
        y_train = organ_train["PMI"].to_numpy()
        x_test = organ_test[features].to_numpy()
        y_test = organ_test["PMI"].to_numpy()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        splits = list(stratified_splits(y_train))

        best_components = 1
        best_mse = float("inf")
        for components in range(1, 31):
            fold_mse: list[float] = []
            for train_idx, val_idx in splits:
                model = PLSRegression(n_components=components)
                model.fit(x_train[train_idx], y_train[train_idx])
                prediction = np.maximum(model.predict(x_train[val_idx]).reshape(-1), 0)
                fold_mse.append(mean_squared_error(y_train[val_idx], prediction))
            score = float(np.mean(fold_mse))
            if score < best_mse:
                best_mse = score
                best_components = components

        cv_values = {"MAE": [], "RMSE": [], "R2": []}
        for train_idx, val_idx in splits:
            model = PLSRegression(n_components=best_components)
            model.fit(x_train[train_idx], y_train[train_idx])
            fold = metrics(y_train[val_idx], model.predict(x_train[val_idx]))
            for key in cv_values:
                cv_values[key].append(fold[key])

        model = PLSRegression(n_components=best_components).fit(x_train, y_train)
        train_prediction = np.maximum(model.predict(x_train).reshape(-1), 0)
        test_prediction = np.maximum(model.predict(x_test).reshape(-1), 0)
        train_metrics = metrics(y_train, train_prediction)
        test_metrics = metrics(y_test, test_prediction)
        row: dict[str, float | int | str] = {
            "organ": organ,
            "n_components": best_components,
        }
        row.update({f"Train_{key}": value for key, value in train_metrics.items()})
        row.update({f"CV_{key}": float(np.mean(value)) for key, value in cv_values.items()})
        row.update({f"Test_{key}": value for key, value in test_metrics.items()})
        result_rows.append(row)
        for true, prediction in zip(y_test, test_prediction):
            prediction_rows.append(
                {"organ": organ, "PMI": float(true), "prediction": float(prediction)}
            )

    pd.DataFrame(result_rows).to_csv(output_dir / "pls_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "pls_predictions.csv", index=False)
    print(pd.DataFrame(result_rows).round(4).to_string(index=False))

