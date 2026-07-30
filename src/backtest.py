"""Backtest cross-sectional signals with explicit timing and research diagnostics.

The implementation favors auditable assumptions over strategy-specific
complexity: signals are lagged, portfolios are unit gross, missing returns
cannot reduce exposure silently, and the returned series support later
holdout, cost, and uncertainty analysis.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from statistics import NormalDist
from typing import Any, TypedDict

import numpy as np
import pandas as pd

TRADING_DAYS = 252


class BacktestResult(TypedDict):
    """Metrics and diagnostic series produced by an alpha backtest."""

    sharpe: float
    sortino: float
    max_drawdown: float
    annualized_volatility: float
    ic: float
    turnover: float
    n_observations: int
    runtime_seconds: float
    portfolio_returns: pd.Series
    daily_ic: pd.Series
    weights: pd.DataFrame


def probabilistic_sharpe_ratio(
    sharpe: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """Estimate P(SR > benchmark), using non-annualized Sharpe values."""
    inputs = np.array([sharpe, benchmark_sharpe, skewness, kurtosis], dtype=float)
    if n_observations < 2 or not np.isfinite(inputs).all():
        return np.nan
    variance_term = 1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    if variance_term <= 0:
        return np.nan
    z_score = (sharpe - benchmark_sharpe) * np.sqrt(n_observations - 1) / np.sqrt(
        variance_term
    )
    return float(NormalDist().cdf(float(z_score)))


def sharpe_statistics(
    portfolio_returns: pd.Series,
    horizon: int,
    benchmark_annualized_sharpe: float = 0.0,
) -> dict[str, float]:
    """Return sample moments and the probabilistic Sharpe ratio."""
    observations = portfolio_returns.dropna()
    volatility = float(observations.std(ddof=1))
    periodic_sharpe = float(observations.mean() / volatility) if volatility > 0 else np.nan
    skewness = float(observations.skew())
    kurtosis = float(observations.kurt() + 3.0)  # pandas reports excess kurtosis.
    benchmark = benchmark_annualized_sharpe / np.sqrt(TRADING_DAYS / horizon)
    return {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "probabilistic_sharpe_ratio": probabilistic_sharpe_ratio(
            periodic_sharpe, benchmark, len(observations), skewness, kurtosis
        ),
    }


def expected_maximum_sharpe(trial_sharpes: pd.Series) -> float:
    """Expected best annualized Sharpe under repeated noisy trials (DSR benchmark)."""
    values = trial_sharpes.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    n_trials = len(values)
    if n_trials < 2:
        return 0.0
    sharpe_std = float(values.std(ddof=1))
    if not np.isfinite(sharpe_std) or sharpe_std == 0:
        return 0.0
    euler_gamma = 0.5772156649015329
    normal = NormalDist()
    expected_standard_max = (1 - euler_gamma) * normal.inv_cdf(1 - 1 / n_trials)
    expected_standard_max += euler_gamma * normal.inv_cdf(1 - 1 / (n_trials * np.e))
    return float(values.mean() + sharpe_std * expected_standard_max)


def net_portfolio_returns(
    portfolio_returns: pd.Series, weights: pd.DataFrame, cost_bps: float
) -> pd.Series:
    """Deduct linear transaction costs per unit of daily L1 turnover."""
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    turnover = weights.diff().abs().sum(axis=1, min_count=1).fillna(0.0)
    return portfolio_returns - turnover.reindex(portfolio_returns.index) * cost_bps / 10_000.0


def block_bootstrap_sharpe_ci(
    portfolio_returns: pd.Series,
    horizon: int,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 2_000,
    block_size: int | None = None,
    seed: int = 42,
) -> tuple[float, float]:
    """Moving-block bootstrap confidence interval for annualized Sharpe.

    Circular blocks preserve local serial dependence, which matters because the
    project's forward-return observations overlap. The default block length is
    at least the return horizon and grows slowly with sample size.
    """
    observations = portfolio_returns.dropna().to_numpy(dtype=float)
    if horizon < 1 or n_bootstrap < 1:
        raise ValueError("horizon and n_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if len(observations) < 2:
        return np.nan, np.nan
    size = block_size or max(horizon, int(round(len(observations) ** (1 / 3))))
    if size < 1:
        raise ValueError("block_size must be positive")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(observations) / size))
    offsets = np.arange(size)
    estimates = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        starts = rng.integers(0, len(observations), size=n_blocks)
        indices = (starts[:, None] + offsets) % len(observations)
        sample = observations[indices.ravel()[: len(observations)]]
        volatility = sample.std(ddof=1)
        estimates[iteration] = (
            sample.mean() / volatility * np.sqrt(TRADING_DAYS / horizon)
            if volatility > 0
            else np.nan
        )
    alpha = (1 - confidence) / 2
    lower, upper = np.nanquantile(estimates, [alpha, 1 - alpha])
    return float(lower), float(upper)


def cross_sectional_weights(signal: pd.DataFrame) -> pd.DataFrame:
    """Rank each date, z-score the ranks, and scale to unit gross exposure."""
    ranks = signal.rank(axis=1, method="average", na_option="keep")
    centered = ranks.sub(ranks.mean(axis=1), axis=0)
    scale = ranks.std(axis=1, ddof=0).replace(0.0, np.nan)
    zscores = centered.div(scale, axis=0)
    gross = zscores.abs().sum(axis=1).replace(0.0, np.nan)
    return zscores.div(gross, axis=0)


def forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compound returns from the row's date through the next ``horizon-1`` dates."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    gross = 1.0 + returns
    compounded = gross.rolling(horizon, min_periods=horizon).apply(np.prod, raw=True)
    return compounded.shift(-(horizon - 1)) - 1.0


def max_drawdown(portfolio_returns: pd.Series) -> float:
    """Return the non-negative peak-to-trough maximum drawdown magnitude."""
    wealth = (1.0 + portfolio_returns.fillna(0.0)).cumprod()
    # Include initial capital so a loss on the first observation is a drawdown.
    initial = pd.Series([1.0], index=["initial"])
    wealth = pd.concat([initial, wealth.reset_index(drop=True)])
    drawdown = wealth.div(wealth.cummax()) - 1.0
    return float(-drawdown.min()) if not drawdown.empty else np.nan


def _metrics(portfolio_returns: pd.Series, horizon: int) -> dict[str, float]:
    observations = portfolio_returns.dropna()
    if observations.empty:
        return {
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "annualized_volatility": np.nan,
        }
    periods_per_year = TRADING_DAYS / horizon
    mean = float(observations.mean())
    volatility = float(observations.std(ddof=1))
    downside = observations.clip(upper=0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    sharpe = mean / volatility * np.sqrt(periods_per_year) if volatility > 0 else np.inf
    sortino = (
        mean / downside_deviation * np.sqrt(periods_per_year) if downside_deviation > 0 else np.inf
    )
    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown(observations),
        "annualized_volatility": float(volatility * np.sqrt(periods_per_year)),
    }


def summarize_backtest(
    result: Mapping[str, Any],
    horizon: int,
    evaluation_dates: pd.Index,
) -> dict[str, Any]:
    """Recompute metrics on an explicit subset of a full-panel backtest."""
    dates = pd.Index(evaluation_dates)
    portfolio_returns = result["portfolio_returns"].reindex(dates)
    daily_ic = result["daily_ic"].reindex(dates)
    weights = result["weights"]
    turnover = weights.diff().abs().sum(axis=1, min_count=1).reindex(dates)
    summary = _metrics(portfolio_returns, horizon)
    summary.update(
        {
            "ic": float(daily_ic.mean()),
            "turnover": float(turnover.mean()),
            "n_observations": int(portfolio_returns.notna().sum()),
            "runtime_seconds": float(result.get("runtime_seconds", np.nan)),
            "portfolio_returns": portfolio_returns,
            "daily_ic": daily_ic,
            "weights": weights.reindex(dates),
        }
    )
    return summary


def backtest_alpha(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    horizon: int,
) -> BacktestResult:
    """Backtest a daily cross-sectional signal without same-day lookahead.

    Signal values observed at the close of day t are shifted to row t+1. The
    realized return on row t+1 compounds returns from t+1 through t+h, so no
    return used by the test predates the first tradable day.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    started = time.perf_counter()
    signal, returns = signal.align(returns, join="inner", axis=0)
    signal, returns = signal.align(returns, join="inner", axis=1)
    if signal.empty or returns.empty:
        raise ValueError("signal and returns must overlap on dates and tickers")

    contemporaneous_weights = cross_sectional_weights(signal)
    # A close-of-t observation first becomes a portfolio weight for t+1.
    weights = contemporaneous_weights.shift(1)
    realized_forward = forward_returns(returns, horizon)
    valid = weights.notna() & realized_forward.notna()
    tradable_weights = weights.where(valid)
    # Missing realized returns must not silently reduce gross portfolio exposure.
    tradable_gross = tradable_weights.abs().sum(axis=1).replace(0.0, np.nan)
    tradable_weights = tradable_weights.div(tradable_gross, axis=0)

    # Unit-gross weights make the cross-sectional dot product a portfolio return.
    portfolio_returns = (tradable_weights * realized_forward.where(valid)).sum(axis=1, min_count=1)

    # Spearman IC is Pearson correlation of cross-sectional ranks, vectorized by date.
    signal_rank = weights.rank(axis=1, method="average")
    return_rank = realized_forward.rank(axis=1, method="average")
    signal_centered = signal_rank.sub(signal_rank.mean(axis=1), axis=0)
    return_centered = return_rank.sub(return_rank.mean(axis=1), axis=0)
    numerator = (signal_centered * return_centered).sum(axis=1, min_count=2)
    denominator = np.sqrt(
        signal_centered.pow(2).sum(axis=1) * return_centered.pow(2).sum(axis=1)
    ).replace(0.0, np.nan)
    daily_ic = numerator / denominator

    turnover_series = tradable_weights.diff().abs().sum(axis=1, min_count=1)
    metrics = _metrics(portfolio_returns, horizon)
    return BacktestResult(
        sharpe=metrics["sharpe"],
        sortino=metrics["sortino"],
        max_drawdown=metrics["max_drawdown"],
        annualized_volatility=metrics["annualized_volatility"],
        ic=float(daily_ic.mean()),
        turnover=float(turnover_series.mean()),
        n_observations=int(portfolio_returns.notna().sum()),
        runtime_seconds=float(time.perf_counter() - started),
        portfolio_returns=portfolio_returns,
        daily_ic=daily_ic,
        weights=tradable_weights,
    )
