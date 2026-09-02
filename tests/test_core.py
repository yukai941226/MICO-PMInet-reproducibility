import numpy as np
import pandas as pd
import torch

from mico_pminet.config import VARIANT_TO_COMPONENTS, load_protocol
from mico_pminet.masking import expand_missing_levels
from mico_pminet.models import build_model
from mico_pminet.preprocess import snv
from mico_pminet.selection import validation_only_selection
from mico_pminet.training import joint_loss, stratified_repeated_holdout


def test_snv_centers_and_scales_each_spectrum():
    values = np.asarray([[1.0, 2.0, 4.0], [10.0, 11.0, 15.0]])
    transformed = snv(values)
    np.testing.assert_allclose(transformed.mean(axis=1), 0.0, atol=1e-7)
    np.testing.assert_allclose(transformed.std(axis=1), 1.0, atol=1e-7)


def test_mask_expansion_matches_manuscript_counts():
    x = torch.randn(48, 8, 12)
    y = torch.arange(48, dtype=torch.float32)
    expanded, mask, labels, levels = expand_missing_levels(
        x, y, repeats=2, deterministic=True
    )
    assert expanded.shape == (720, 8, 12)
    assert mask.shape == (720, 8)
    assert labels.shape == (720,)
    assert set(levels.tolist()) == set(range(8))
    assert torch.all(mask.sum(dim=1) >= 1)

    x_val = torch.randn(16, 8, 12)
    y_val = torch.arange(16, dtype=torch.float32)
    expanded_val, _, _, _ = expand_missing_levels(
        x_val, y_val, repeats=10, deterministic=True
    )
    assert expanded_val.shape == (1136, 8, 12)


def test_manuscript_protocol_matches_article_proof():
    protocol = load_protocol("manuscript_protocol")
    assert protocol.max_epochs == 1000
    assert protocol.early_stopping_patience == 200
    assert protocol.batch_size == 32
    assert protocol.learning_rate == 0.001
    assert protocol.weight_decay == 0.0001
    assert protocol.gradient_clip == 1.0
    assert protocol.alpha == 1.0
    assert protocol.gamma == 0.0
    assert protocol.cv_splits == 10
    assert protocol.cv_validation_fraction_per_pmi == 0.25
    assert protocol.morm_training_repeats == 2
    assert protocol.morm_validation_repeats == 10


def test_manuscript_split_is_48_train_16_validation():
    y = np.repeat(np.asarray([0.5, 1, 2, 3, 6, 12, 18, 24], dtype=float), 8)
    splits = list(
        stratified_repeated_holdout(y, n_splits=10, test_size=0.25, random_state=225)
    )
    assert len(splits) == 10
    for train_idx, val_idx in splits:
        assert len(train_idx) == 48
        assert len(val_idx) == 16
        labels, counts = np.unique(y[val_idx], return_counts=True)
        assert len(labels) == 8
        assert np.all(counts == 2)


def test_article_joint_loss_has_only_main_and_organ_terms():
    final = torch.tensor([2.0, 5.0])
    branches = torch.tensor([[[1.0], [3.0]], [[4.0], [8.0]]])
    truth = torch.tensor([2.0, 6.0])
    mask = torch.ones(2, 2)
    alpha = 1.0
    beta = 0.2
    expected = alpha * torch.nn.functional.mse_loss(final, truth)
    target = truth.unsqueeze(1).expand(2, 2)
    expected = expected + beta * (((branches.squeeze(-1) - target) ** 2) * mask).mean()
    actual = joint_loss(final, branches, truth, mask, alpha, beta, gamma=0.0)
    torch.testing.assert_close(actual, expected)


def test_all_eight_architectures_forward():
    x = torch.randn(2, 8, 20)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0]])
    for branch, attention, aggregation in VARIANT_TO_COMPONENTS.values():
        model = build_model(
            branch, attention, aggregation, num_organs=8, input_dim=20, hidden_dim=32
        )
        final, branches = model(x, mask)
        assert final.shape == (2,)
        assert branches.shape == (2, 8, 1)
        assert torch.isfinite(final).all()


def test_awa_weights_are_normalized_and_masked():
    model = build_model("OSB", "WMHA", "AWA", num_organs=8, input_dim=20, hidden_dim=32)
    x = torch.randn(3, 8, 20)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    weights = model.normalized_organ_weights(x, mask)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
    assert torch.all(weights[mask == 0] == 0)


def test_selection_ignores_better_test_score(tmp_path):
    rows = []
    for fold in range(1, 4):
        rows.extend(
            [
                {
                    "model_name": "validation-winner",
                    "beta": 0.2,
                    "fold": fold,
                    "Val_RMSE": 1.0,
                    "Val_MAE": 0.8,
                    "Test_RMSE": 9.0,
                },
                {
                    "model_name": "test-winner",
                    "beta": 0.2,
                    "fold": fold,
                    "Val_RMSE": 2.0,
                    "Val_MAE": 1.0,
                    "Test_RMSE": 0.1,
                },
            ]
        )
    source = tmp_path / "folds.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    selected = validation_only_selection(
        source, tmp_path / "selection", target_model="validation-winner"
    )
    assert selected["selected_model"] == "validation-winner"
    assert selected["test_metrics_used_for_selection"] is False


def test_selection_accepts_validation_summary(tmp_path):
    source = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "model_name": "model-a",
                "beta": 0.1,
                "n_folds": 10,
                "Val_RMSE_mean": 1.2,
                "Val_RMSE_std": 0.2,
                "Val_MAE_mean": 0.8,
                "Test_RMSE_mean": 9.0,
            },
            {
                "model_name": "model-a",
                "beta": 0.2,
                "n_folds": 10,
                "Val_RMSE_mean": 1.0,
                "Val_RMSE_std": 0.1,
                "Val_MAE_mean": 0.7,
                "Test_RMSE_mean": 10.0,
            },
            {
                "model_name": "model-b",
                "beta": 0.1,
                "n_folds": 10,
                "Val_RMSE_mean": 2.0,
                "Val_RMSE_std": 0.1,
                "Val_MAE_mean": 1.0,
                "Test_RMSE_mean": 0.1,
            },
        ]
    ).to_csv(source, index=False)
    selected = validation_only_selection(
        source, tmp_path / "selection-summary", target_model="model-a"
    )
    assert selected["selected_model"] == "model-a"
    assert selected["selected_beta"] == 0.2
    assert selected["candidate_rows"] == 3
