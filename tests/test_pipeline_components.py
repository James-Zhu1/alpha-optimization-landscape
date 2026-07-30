import json

import numpy as np
import pandas as pd
import pytest

from src.landscape import run_clustering, run_pca
from src.search import (
    chronological_splits,
    evaluate_out_of_sample,
    run_bayesian_search,
    run_random_search,
)
from src.trajectory import build_feature_matrix


def _panels():
    rng = np.random.default_rng(7)
    index = pd.date_range("2017-01-01", periods=500, freq="B")
    columns = [f"T{i}" for i in range(15)]
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.012, (len(index), len(columns))), index=index, columns=columns
    )
    prices = 100 * (1 + returns).cumprod()
    volume = pd.DataFrame(rng.lognormal(14, 0.4, returns.shape), index=index, columns=columns)
    return prices, returns, volume


def test_search_logging_feature_matrix_and_landscape(tmp_path):
    trajectory = run_random_search(
        20,
        11,
        output_path=tmp_path / "trajectory.parquet",
        panels=_panels(),
        checkpoint_every=5,
    )
    assert len(trajectory) == 20
    assert set(trajectory["optimizer"]) == {"random"}
    assert {"train_sharpe", "validation_sharpe"}.issubset(trajectory.columns)
    assert trajectory["sharpe"].equals(trajectory["validation_sharpe"])
    assert (
        trajectory["expression"].map(json.loads).map(lambda value: value["primitive"]).notna().all()
    )
    features = build_feature_matrix(trajectory)
    expression_features = build_feature_matrix(trajectory, include_metrics=False)
    assert features.shape[0] == 20
    assert expression_features.shape[1] < features.shape[1]
    assert np.isfinite(features.to_numpy()).all()
    assert run_pca(features).shape == (20, 3)
    labels = run_clustering(features)
    assert set(labels) == {"kmeans", "agglomerative", "hdbscan"}
    assert all(len(value) == 20 for value in labels.values())


def test_chronological_splits_are_disjoint_and_purge_forward_horizon():
    index = pd.date_range("2020-01-01", periods=100, freq="B")
    splits = chronological_splits(index, 5, 0.6, 0.2, 0.2)
    assert len(splits["train"]) == 56
    assert len(splits["validation"]) == 16
    assert len(splits["test"]) == 16
    assert splits["train"].max() < splits["validation"].min() < splits["test"].min()
    assert splits["train"].max() == index[55]
    assert splits["validation"].max() == index[75]


def test_holdout_evaluates_only_validation_selected_winner(tmp_path):
    panels = _panels()
    trajectory = run_random_search(8, 13, output_path=tmp_path / "trajectory.parquet", panels=panels)
    # Combined optimizer logs use a global display order but retain the original
    # optimizer-specific evaluation number for traceability.
    trajectory["source_iteration"] = np.arange(len(trajectory)) + 100
    result = evaluate_out_of_sample(
        trajectory, output_path=tmp_path / "holdout.parquet", panels=panels
    )
    assert len(result) == 1
    winner_index = trajectory["validation_sharpe"].idxmax()
    assert result.loc[0, "source_iteration"] == trajectory.loc[winner_index, "source_iteration"]
    assert result.loc[0, "test_n_observations"] > 0
    assert np.isfinite(result.loc[0, "test_sharpe"])
    assert np.isfinite(result.loc[0, "test_net_sharpe"])
    assert 0 <= result.loc[0, "test_probabilistic_sharpe_ratio"] <= 1
    assert result.loc[0, "n_selection_trials"] == len(trajectory)


def test_holdout_flags_same_expression_selected_by_two_optimizers(tmp_path):
    panels = _panels()
    trajectory = run_random_search(4, 13, output_path=tmp_path / "trajectory.parquet", panels=panels)
    duplicate = trajectory.iloc[[0, 0]].copy()
    duplicate["optimizer"] = ["random", "bayesian"]
    result = evaluate_out_of_sample(
        duplicate, output_path=tmp_path / "holdout.parquet", panels=panels
    )
    assert len(result) == 2
    assert result["winner_expression_duplicated"].all()


def test_bayesian_search_does_not_log_duplicate_expressions(tmp_path):
    pytest.importorskip("optuna")
    trajectory = run_bayesian_search(
        10,
        21,
        output_path=tmp_path / "bayesian.parquet",
        panels=_panels(),
        checkpoint_every=5,
    )
    assert len(trajectory) == 10
    assert trajectory["expression"].is_unique
