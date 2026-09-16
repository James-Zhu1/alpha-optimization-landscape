import json

import numpy as np
import pandas as pd
import pytest

from src import forward_test
from src.forward_test import (
    evaluate_forward_window,
    format_summary,
    frozen_expressions,
    power_verdict,
)

BOUNDARY = pd.Timestamp("2018-01-01")


def _panels():
    """Warmup history plus a post-boundary window, mirroring a refreshed panel."""
    rng = np.random.default_rng(3)
    index = pd.date_range("2016-06-01", periods=600, freq="B")
    columns = [f"T{i}" for i in range(12)]
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.011, (len(index), len(columns))), index=index, columns=columns
    )
    prices = 100 * (1 + returns).cumprod()
    volume = pd.DataFrame(rng.lognormal(14, 0.3, returns.shape), index=index, columns=columns)
    return prices, returns, volume


def _write_holdout(path, optimizers=("bayesian", "random"), lookback=20):
    expression = json.dumps(
        {"lookback": lookback, "normalize_by": None, "primitive": "momentum"}, sort_keys=True
    )
    pd.DataFrame(
        {"optimizer": list(optimizers), "expression": [expression] * len(optimizers)}
    ).to_parquet(path, index=False)
    return path


@pytest.fixture
def frozen_study(tmp_path, monkeypatch):
    """Point the module at a synthetic study boundary and holdout result."""
    holdout_path = _write_holdout(tmp_path / "out_of_sample_results.parquet")
    monkeypatch.setattr(forward_test, "OUT_OF_SAMPLE_PATH", holdout_path)
    monkeypatch.setattr(forward_test, "study_boundary", lambda: BOUNDARY)
    return tmp_path


def test_converged_optimizers_collapse_to_one_scored_candidate(tmp_path):
    path = _write_holdout(tmp_path / "holdout.parquet", optimizers=("bayesian", "random"))
    candidates = frozen_expressions(path)
    # One expression chosen twice must not become two forward results, which
    # would misread a single candidate as two independent confirmations.
    assert len(candidates) == 1
    assert candidates.loc[0, "optimizer"] == "bayesian+random"


def test_distinct_winners_are_scored_separately(tmp_path):
    path = tmp_path / "holdout.parquet"
    pd.DataFrame(
        {
            "optimizer": ["bayesian", "random"],
            "expression": [
                json.dumps({"lookback": 20, "primitive": "momentum"}, sort_keys=True),
                json.dumps({"lookback": 60, "primitive": "volatility"}, sort_keys=True),
            ],
        }
    ).to_parquet(path, index=False)
    assert len(frozen_expressions(path)) == 2


def test_forward_window_excludes_every_study_date(frozen_study):
    results = evaluate_forward_window(
        panels=_panels(), output_path=frozen_study / "forward.parquet"
    )
    assert len(results) == 1
    # The whole point of the harness: nothing at or before the boundary is scored.
    assert pd.Timestamp(results.loc[0, "forward_start"]) > BOUNDARY


def test_warmup_history_is_used_but_not_scored(frozen_study):
    """A 20-day lookback must not shrink the forward window by 20 observations."""
    prices, returns, volume = _panels()
    results = evaluate_forward_window(
        panels=(prices, returns, volume), output_path=frozen_study / "forward.parquet"
    )
    available = int((returns.index > BOUNDARY).sum())
    scored = int(results.loc[0, "forward_n_observations"])
    # Only the forward-return horizon should be lost at the tail, not the lookback.
    assert available - scored < 20


def test_short_window_reports_insufficient_power(frozen_study):
    results = evaluate_forward_window(
        panels=_panels(), output_path=frozen_study / "forward.parquet"
    )
    width = float(results.loc[0, "forward_sharpe_ci_width"])
    lower = float(results.loc[0, "forward_sharpe_ci_lower"])
    upper = float(results.loc[0, "forward_sharpe_ci_upper"])
    assert width == pytest.approx(upper - lower)
    assert "non-overlapping observations" in str(results.loc[0, "power_verdict"])


def test_costs_never_improve_the_forward_result(frozen_study):
    results = evaluate_forward_window(
        panels=_panels(), output_path=frozen_study / "forward.parquet"
    )
    sensitivity = json.loads(str(results.loc[0, "cost_sensitivity"]))
    by_cost = [sensitivity[key] for key in sorted(sensitivity, key=float)]
    assert by_cost == sorted(by_cost, reverse=True)
    assert results.loc[0, "forward_net_sharpe"] <= results.loc[0, "forward_sharpe"]


def test_power_verdict_escalates_with_evidence():
    assert power_verdict(40, 5, 6.1).startswith("Insufficient")
    assert power_verdict(500, 5, 1.8).startswith("Weak")
    assert power_verdict(2000, 5, 0.6).startswith("Usable")


def test_summary_states_the_interval_and_its_limits(frozen_study):
    results = evaluate_forward_window(
        panels=_panels(), output_path=frozen_study / "forward.parquet"
    )
    summary = format_summary(results)
    assert "95% interval" in summary
    assert "inconclusive" in summary
    assert "bayesian+random" in summary
