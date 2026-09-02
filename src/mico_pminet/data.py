from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import ORGAN_ORDER
from .preprocess import spectral_columns


@dataclass
class TensorData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_test_modeled: torch.Tensor
    y_test_modeled: torch.Tensor
    x_test_unseen: torch.Tensor
    y_test_unseen: torch.Tensor
    scaler: StandardScaler
    wavenumbers: np.ndarray


def resolve_table(directory: Path, stem: str) -> Path:
    candidates = (directory / f"{stem}.csv.gz", directory / f"{stem}.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {stem}.csv[.gz] in {directory}")


def dataframe_to_tensor(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    features = spectral_columns(frame)
    samples: list[np.ndarray] = []
    labels: list[float] = []
    for (pmi, animal), group in frame.groupby(["PMI", "label_4"], sort=True):
        if set(group["label_2"]) != set(ORGAN_ORDER):
            raise ValueError(f"Incomplete original sample at PMI={pmi}, animal={animal}")
        ordered = group.set_index("label_2").loc[list(ORGAN_ORDER)]
        samples.append(ordered[features].to_numpy(dtype=np.float32))
        labels.append(float(pmi))
    return torch.tensor(np.stack(samples)), torch.tensor(labels, dtype=torch.float32)


def load_tensor_data(processed_dir: Path) -> TensorData:
    train = pd.read_csv(resolve_table(processed_dir, "train_dataset"))
    test = pd.read_csv(resolve_table(processed_dir, "test_dataset"))
    try:
        test_modeled = pd.read_csv(resolve_table(processed_dir, "test_modeled"))
        test_unseen = pd.read_csv(resolve_table(processed_dir, "test_unseen"))
    except FileNotFoundError:
        modeled_pmis = set(train["PMI"].unique())
        test_modeled = test[test["PMI"].isin(modeled_pmis)].copy()
        test_unseen = test[~test["PMI"].isin(modeled_pmis)].copy()
    feature_columns = spectral_columns(train)
    if feature_columns != spectral_columns(test):
        raise ValueError("Training and test spectral columns differ")
    if len(feature_columns) != 467:
        raise ValueError(f"Expected 467 spectral features, found {len(feature_columns)}")

    scaler = StandardScaler()
    train = train.copy()
    test = test.copy()
    test_modeled = test_modeled.copy()
    test_unseen = test_unseen.copy()
    train.loc[:, feature_columns] = scaler.fit_transform(train[feature_columns])
    test.loc[:, feature_columns] = scaler.transform(test[feature_columns])
    test_modeled.loc[:, feature_columns] = scaler.transform(test_modeled[feature_columns])
    test_unseen.loc[:, feature_columns] = scaler.transform(test_unseen[feature_columns])
    x_train, y_train = dataframe_to_tensor(train)
    x_test, y_test = dataframe_to_tensor(test)
    x_test_modeled, y_test_modeled = dataframe_to_tensor(test_modeled)
    x_test_unseen, y_test_unseen = dataframe_to_tensor(test_unseen)
    return TensorData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_test_modeled=x_test_modeled,
        y_test_modeled=y_test_modeled,
        x_test_unseen=x_test_unseen,
        y_test_unseen=y_test_unseen,
        scaler=scaler,
        wavenumbers=np.asarray([float(column) for column in feature_columns]),
    )
