from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ORGAN_ORDER = (
    "Brain",
    "Heart",
    "Kidney",
    "Liver",
    "Lung",
    "Muscle",
    "Spleen",
    "VH",
)

MODELED_PMIS = (0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 18.0, 24.0)
UNSEEN_PMIS = (0.33333, 4.0, 11.0, 20.0)

VARIANT_TO_COMPONENTS = {
    "learned_weights": ("OSB", "MHA", "AWA"),
    "no_attention": ("OSB", "WMHA", "AWA"),
    "no_attention_simple_avg": ("OSB", "WMHA", "APA"),
    "simple_avg": ("OSB", "MHA", "APA"),
    "shared_branches": ("USB", "MHA", "AWA"),
    "no_attention_shared": ("USB", "WMHA", "AWA"),
    "minimal_model": ("USB", "WMHA", "APA"),
    "shared_simple_avg": ("USB", "MHA", "APA"),
}

STRATEGY_TO_COMPONENT = {
    "no_masking": "MOCE",
    "random_masking": "MORM",
}


@dataclass(frozen=True)
class Protocol:
    seed: int
    split_seed: int
    num_organs: int
    input_dim: int
    hidden_dim: int
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    scheduler_factor: float
    scheduler_patience: int
    gradient_clip: float
    alpha: float
    gamma: float
    beta_candidates: tuple[float, ...]
    cv_splits: int
    cv_validation_fraction_per_pmi: float
    morm_training_repeats: int
    morm_validation_repeats: int
    retention_test_repeats: int

    @classmethod
    def from_json(cls, path: Path) -> "Protocol":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("description", None)
        payload["beta_candidates"] = tuple(payload["beta_candidates"])
        return cls(**payload)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_protocol(name: str) -> Protocol:
    path = repository_root() / "configs" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown protocol: {name!r}. Expected {path}")
    return Protocol.from_json(path)


def model_name(variant: str, strategy: str) -> str:
    branch, attention, aggregation = VARIANT_TO_COMPONENTS[variant]
    return "-".join((branch, attention, aggregation, STRATEGY_TO_COMPONENT[strategy]))

