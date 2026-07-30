import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    backtest_alpha,
    cross_sectional_weights,
    max_drawdown,
    net_portfolio_returns,
    probabilistic_sharpe_ratio,
)


def _frame(values, columns=None):
    values = np.asarray(values, dtype=float)
    if columns is None:
        columns = [f"T{i}" for i in range(values.shape[1])]
    return pd.DataFrame(
        values,
        index=pd.date_range("2020-01-01", periods=values.shape[0], freq="B"),
        columns=columns,
    )


def test_lookahead_shift_changes_result_meaningfully():
    rng = np.random.default_rng(1)
    returns = _frame(rng.normal(0, 0.02, size=(300, 20)))
    signal = returns.copy()  # deliberately leaks same-day returns
    correct = backtest_alpha(signal, returns, horizon=1)["sharpe"]

    leaked_weights = cross_sectional_weights(signal)
    leaked_portfolio = (leaked_weights * returns).sum(axis=1)
    leaked_sharpe = leaked_portfolio.mean() / leaked_portfolio.std() * np.sqrt(252)
    assert leaked_sharpe - correct > 3.0


def test_perfect_future_signal_has_high_sharpe_and_ic():
    rng = np.random.default_rng(2)
    returns = _frame(rng.normal(0, 0.015, size=(350, 30)))
    # Row t contains the return that begins on t+1; the engine shifts it once.
    signal = returns.shift(-1)
    result = backtest_alpha(signal, returns, horizon=1)
    assert result["sharpe"] > 3.0
    assert result["ic"] > 0.99


def test_turnover_stable_is_zero_and_reshuffled_is_high():
    rng = np.random.default_rng(3)
    returns = _frame(rng.normal(0, 0.01, size=(250, 40)))
    base = np.linspace(-1, 1, returns.shape[1])
    stable = _frame(np.tile(base, (len(returns), 1)))
    shuffled = _frame(np.vstack([rng.permutation(base) for _ in range(len(returns))]))
    stable_turnover = backtest_alpha(stable, returns, 1)["turnover"]
    shuffled_turnover = backtest_alpha(shuffled, returns, 1)["turnover"]
    assert stable_turnover == pytest.approx(0.0, abs=1e-12)
    assert shuffled_turnover > 0.5
    assert shuffled_turnover < 2.0  # unit-gross L1 theoretical maximum


def test_drawdown_matches_manual_series_through_backtest():
    portfolio = pd.Series([0.10, -0.05, -0.10, 0.04, 0.02, -0.03])
    wealth = (1 + portfolio).cumprod()
    expected = -float((wealth / wealth.cummax() - 1).min())
    assert max_drawdown(portfolio) == pytest.approx(expected, abs=1e-4)

    signal = _frame(np.tile([-1.0, 1.0], (len(portfolio) + 1, 1)))
    returns = _frame(np.column_stack([-np.r_[0.0, portfolio], np.r_[0.0, portfolio]]))
    result = backtest_alpha(signal, returns, 1)
    assert result["max_drawdown"] == pytest.approx(expected, abs=1e-4)
    assert max_drawdown(pd.Series([-0.10, 0.05])) == pytest.approx(0.10, abs=1e-4)


def test_costs_reduce_returns_in_proportion_to_turnover():
    returns = pd.Series([0.01, 0.02, 0.03])
    weights = pd.DataFrame([[0.5, -0.5], [-0.5, 0.5], [-0.5, 0.5]])
    net = net_portfolio_returns(returns, weights, cost_bps=10)
    assert net.iloc[0] == pytest.approx(0.01)
    assert net.iloc[1] == pytest.approx(0.018)
    assert net.iloc[2] == pytest.approx(0.03)


def test_probabilistic_sharpe_ratio_is_monotonic():
    low = probabilistic_sharpe_ratio(0.05, 0.0, 250, 0.0, 3.0)
    high = probabilistic_sharpe_ratio(0.15, 0.0, 250, 0.0, 3.0)
    assert 0.5 < low < high < 1.0


def test_block_bootstrap_sharpe_interval_is_reproducible():
    from src.backtest import block_bootstrap_sharpe_ci

    rng = np.random.default_rng(9)
    returns = pd.Series(rng.normal(0.001, 0.01, 300))
    first = block_bootstrap_sharpe_ci(returns, 5, n_bootstrap=200, seed=7)
    second = block_bootstrap_sharpe_ci(returns, 5, n_bootstrap=200, seed=7)
    assert first == second
    assert first[0] < first[1]
