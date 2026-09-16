"""Search the alpha space, preserve every trial, and enforce honest selection.

Training performance supplies optimizer feedback, validation performance selects
one candidate per optimizer, and only those fixed winners reach the test window.
The complete trial log is retained because the search path—not just the winner—
is the object of study.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd

from src.alpha_lib import (
    LONG_LOOKBACKS,
    LOOKBACKS,
    PRIMITIVES,
    SHORT_LOOKBACKS,
    AlphaExpression,
    PrimitiveName,
    build_signal,
    sample_alpha_expression,
)
from src.backtest import (
    backtest_alpha,
    block_bootstrap_sharpe_ci,
    expected_maximum_sharpe,
    net_portfolio_returns,
    sharpe_statistics,
    summarize_backtest,
)
from src.data_pipeline import PROCESSED_DIR, RESULTS_DIR, ROOT, load_config

TRAJECTORY_PATH = RESULTS_DIR / "trajectory_random.parquet"
BAYESIAN_TRAJECTORY_PATH = RESULTS_DIR / "trajectory_bayesian.parquet"
COMBINED_TRAJECTORY_PATH = RESULTS_DIR / "trajectory_combined.parquet"
OUT_OF_SAMPLE_PATH = RESULTS_DIR / "out_of_sample_results.parquet"
PUBLIC_METRICS = (
    "sharpe",
    "sortino",
    "max_drawdown",
    "annualized_volatility",
    "ic",
    "turnover",
    "n_observations",
    "runtime_seconds",
)
LOGGER = logging.getLogger(__name__)
Panels: TypeAlias = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

if TYPE_CHECKING:
    import optuna


def chronological_splits(
    index: pd.Index,
    horizon: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, pd.Index]:
    """Create disjoint chronological windows and purge boundary-crossing targets."""
    fractions = np.array([train_fraction, validation_fraction, test_fraction], dtype=float)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train, validation, and test fractions must be positive and sum to 1")
    dates = pd.Index(index).sort_values().unique()
    train_stop = int(np.floor(len(dates) * train_fraction))
    validation_stop = int(np.floor(len(dates) * (train_fraction + validation_fraction)))
    boundaries = ((0, train_stop), (train_stop, validation_stop), (validation_stop, len(dates)))
    splits: dict[str, pd.Index] = {}
    for name, (start, stop) in zip(("train", "validation", "test"), boundaries, strict=True):
        safe_stop = stop - (horizon - 1)
        splits[name] = dates[start:max(start, safe_stop)]
        if splits[name].empty:
            raise ValueError(f"{name} split is empty after purging the {horizon}-day horizon")
    return splits


def _configured_splits(index: pd.Index, horizon: int) -> dict[str, pd.Index]:
    config = load_config()
    return chronological_splits(
        index,
        horizon,
        config.train_fraction,
        config.validation_fraction,
        config.test_fraction,
    )


def load_panels() -> Panels:
    """Load processed price, return, and volume panels."""
    paths = [PROCESSED_DIR / f"{name}.parquet" for name in ("prices", "returns", "volume")]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Processed data missing: {missing}. Run src.data_pipeline first.")
    prices, returns, volume = (pd.read_parquet(path) for path in paths)
    return prices, returns, volume


def _metric_row(
    iteration: int,
    optimizer: str,
    expression: AlphaExpression,
    metrics: Mapping[str, Any],
    wall: float,
) -> dict[str, object]:
    row = {
        "iteration": int(iteration),
        "optimizer": optimizer,
        "expression": json.dumps(expression, sort_keys=True),
        "wall_clock_seconds": float(wall),
    }
    row.update({key: metrics.get(key, np.nan) for key in PUBLIC_METRICS})
    return row


def _split_metric_row(
    iteration: int,
    optimizer: str,
    expression: AlphaExpression,
    full_result: Mapping[str, Any],
    splits: dict[str, pd.Index],
    horizon: int,
    wall: float,
) -> dict[str, object]:
    train = summarize_backtest(dict(full_result), horizon, splits["train"])
    validation = summarize_backtest(dict(full_result), horizon, splits["validation"])
    # Unprefixed metrics remain available and now represent validation results.
    row = _metric_row(iteration, optimizer, expression, validation, wall)
    for split_name, metrics in (("train", train), ("validation", validation)):
        for key in PUBLIC_METRICS:
            row[f"{split_name}_{key}"] = metrics.get(key, np.nan)
    return row


def _write(rows: list[dict[str, object]], output_path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)
    return frame


def _validate_search_inputs(n_iters: int, checkpoint_every: int) -> None:
    if n_iters < 1:
        raise ValueError("n_iters must be positive")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")


def run_random_search(
    n_iters: int,
    seed: int,
    *,
    output_path: Path = TRAJECTORY_PATH,
    panels: Panels | None = None,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """Run and incrementally persist every random-search evaluation."""
    _validate_search_inputs(n_iters, checkpoint_every)
    prices, returns, volume = panels or load_panels()
    horizon = load_config().forward_return_horizon
    splits = _configured_splits(returns.index, horizon)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for iteration in range(n_iters):
        started = time.perf_counter()
        expression = sample_alpha_expression(rng)
        signal = build_signal(expression, prices, returns, volume)
        metrics = backtest_alpha(signal, returns, horizon)
        rows.append(
            _split_metric_row(
                iteration,
                "random",
                expression,
                metrics,
                splits,
                horizon,
                time.perf_counter() - started,
            )
        )
        if (iteration + 1) % checkpoint_every == 0:
            _write(rows, output_path)
    return _write(rows, output_path)


def _suggest_expression(trial: optuna.trial.Trial) -> AlphaExpression:
    primitive = cast(
        PrimitiveName,
        trial.suggest_categorical("primitive", list(PRIMITIVES)),
    )
    expression: AlphaExpression = {
        "primitive": primitive,
        "normalize_by": None,
    }
    if primitive == "ma_crossover":
        pairs = [
            f"{short}:{long}"
            for short in SHORT_LOOKBACKS
            for long in LONG_LOOKBACKS
            if short < long
        ]
        short, long = map(int, trial.suggest_categorical("ma_pair", pairs).split(":"))
        expression["short"] = short
        expression["long"] = long
    else:
        expression["lookback"] = int(trial.suggest_categorical("lookback", LOOKBACKS.tolist()))
    normalize = cast(
        "Literal['volatility'] | None",
        trial.suggest_categorical("normalize_by", [None, "volatility"]),
    )
    expression["normalize_by"] = normalize
    if normalize:
        if primitive == "volatility":
            pairs = [
                f"{lookback}:{normalizer}"
                for lookback in LOOKBACKS
                for normalizer in LOOKBACKS[LOOKBACKS <= 120]
                if lookback != normalizer
            ]
            lookback, normalizer = map(
                int, trial.suggest_categorical("volatility_pair", pairs).split(":")
            )
            expression["lookback"] = lookback
            expression["normalize_lookback"] = normalizer
        else:
            expression["normalize_lookback"] = int(
                trial.suggest_categorical(
                    "normalize_lookback", LOOKBACKS[LOOKBACKS <= 120].tolist()
                )
            )
    return expression


def run_bayesian_search(
    n_iters: int,
    seed: int,
    *,
    output_path: Path = BAYESIAN_TRAJECTORY_PATH,
    panels: Panels | None = None,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """Use Optuna's TPE sampler and log the same schema as random search."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install optuna to run Bayesian search") from exc
    _validate_search_inputs(n_iters, checkpoint_every)
    prices, returns, volume = panels or load_panels()
    horizon = load_config().forward_return_horizon
    splits = _configured_splits(returns.index, horizon)
    rows: list[dict[str, object]] = []
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    seen_expressions: set[str] = set()
    attempts = 0
    max_attempts = n_iters * 100
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # The research domain is discrete and a deterministic backtest has no value in
    # repeated evaluations. Prune duplicates and keep asking until the budget
    # contains n_iters distinct alpha expressions.
    while len(rows) < n_iters:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Bayesian sampler could not find enough unique expressions")
        trial = study.ask()
        expression = _suggest_expression(trial)
        expression_key = json.dumps(expression, sort_keys=True)
        if expression_key in seen_expressions:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        started = time.perf_counter()
        metrics = backtest_alpha(
            build_signal(expression, prices, returns, volume), returns, horizon
        )
        rows.append(
            _split_metric_row(
                trial.number,
                "bayesian",
                expression,
                metrics,
                splits,
                horizon,
                time.perf_counter() - started,
            )
        )
        seen_expressions.add(expression_key)
        if len(rows) % checkpoint_every == 0:
            _write(rows, output_path)
        train_metrics = summarize_backtest(metrics, horizon, splits["train"])
        sharpe = float(train_metrics["sharpe"])
        study.tell(trial, sharpe if np.isfinite(sharpe) else -1e6)
    return _write(rows, output_path)


def combine_trajectories(
    random_path: Path = TRAJECTORY_PATH,
    bayesian_path: Path = BAYESIAN_TRAJECTORY_PATH,
    output_path: Path = COMBINED_TRAJECTORY_PATH,
) -> pd.DataFrame:
    """Combine optimizer logs while retaining origin and a unique evaluation order."""
    frames: list[pd.DataFrame] = []
    for path in (random_path, bayesian_path):
        if not path.exists():
            raise RuntimeError(f"Trajectory file does not exist: {path}")
        frame = pd.read_parquet(path).copy()
        frame["source_iteration"] = frame["iteration"]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["iteration"] = np.arange(len(combined), dtype=int)
    return _write(combined.to_dict(orient="records"), output_path)


def evaluate_out_of_sample(
    trajectory: pd.DataFrame,
    *,
    output_path: Path = OUT_OF_SAMPLE_PATH,
    panels: Panels | None = None,
) -> pd.DataFrame:
    """Test the validation-selected winner from each optimizer exactly once.

    Cost sensitivity and statistical diagnostics are computed only after
    selection. They audit a fixed candidate rather than turn the test window
    into another search stage.
    """
    if trajectory.empty:
        raise ValueError("trajectory is empty")
    # Selection must use a unique row index. Parquet concatenation or caller-side
    # slicing can preserve duplicate labels, causing ``loc[idxmax]`` to expand a
    # single winner into multiple rows.
    trajectory = trajectory.reset_index(drop=True)
    selection_column = "validation_sharpe" if "validation_sharpe" in trajectory else "sharpe"
    prices, returns, volume = panels or load_panels()
    horizon = load_config().forward_return_horizon
    test_dates = _configured_splits(returns.index, horizon)["test"]
    winner_indices = trajectory.groupby("optimizer")[selection_column].idxmax()
    winners = trajectory.loc[winner_indices].copy()
    canonical_expressions = winners["expression"].map(
        lambda value: json.dumps(json.loads(value), sort_keys=True)
        if isinstance(value, str)
        else json.dumps(value, sort_keys=True)
    )
    duplicate_expressions = canonical_expressions.duplicated(keep=False)
    rows: list[dict[str, object]] = []
    for position, (_, winner) in enumerate(winners.iterrows()):
        raw_expression = winner["expression"]
        expression = cast(
            AlphaExpression,
            json.loads(raw_expression) if isinstance(raw_expression, str) else raw_expression,
        )
        full_result = backtest_alpha(
            build_signal(expression, prices, returns, volume), returns, horizon
        )
        test_metrics = summarize_backtest(full_result, horizon, test_dates)
        cost_bps = load_config().transaction_cost_bps
        net_result = dict(full_result)
        net_result["portfolio_returns"] = net_portfolio_returns(
            full_result["portfolio_returns"], full_result["weights"], cost_bps
        )
        net_metrics = summarize_backtest(net_result, horizon, test_dates)
        optimizer_trials = trajectory.loc[
            trajectory["optimizer"] == winner["optimizer"], selection_column
        ]
        dsr_benchmark = expected_maximum_sharpe(optimizer_trials)
        statistics = sharpe_statistics(
            test_metrics["portfolio_returns"], horizon, dsr_benchmark
        )
        bootstrap_lower, bootstrap_upper = block_bootstrap_sharpe_ci(
            test_metrics["portfolio_returns"], horizon, seed=load_config().random_seed
        )
        cost_sensitivity: dict[str, float] = {}
        for scenario_bps in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0):
            scenario_result = dict(full_result)
            scenario_result["portfolio_returns"] = net_portfolio_returns(
                full_result["portfolio_returns"], full_result["weights"], scenario_bps
            )
            scenario_metrics = summarize_backtest(scenario_result, horizon, test_dates)
            cost_sensitivity[f"{scenario_bps:g}"] = float(scenario_metrics["sharpe"])
        is_duplicate = bool(duplicate_expressions.iloc[position])
        LOGGER.info(
            "Selected %s winner (iteration %s, %s=%.6f): %s%s",
            winner["optimizer"],
            winner.get("source_iteration", winner["iteration"]),
            selection_column,
            winner[selection_column],
            canonical_expressions.iloc[position],
            " [same expression selected by another optimizer]" if is_duplicate else "",
        )
        row: dict[str, object] = {
            "optimizer": winner["optimizer"],
            "source_iteration": int(winner.get("source_iteration", winner["iteration"])),
            "expression": json.dumps(expression, sort_keys=True),
            "selection_metric": selection_column,
            "selection_sharpe": float(winner[selection_column]),
            "selection_turnover": float(winner.get("validation_turnover", winner["turnover"])),
            "winner_expression_duplicated": is_duplicate,
            "transaction_cost_bps": float(cost_bps),
            "test_net_sharpe": float(net_metrics["sharpe"]),
            "n_selection_trials": int(len(optimizer_trials)),
            "deflated_sharpe_benchmark": dsr_benchmark,
            "test_sharpe_ci_lower": bootstrap_lower,
            "test_sharpe_ci_upper": bootstrap_upper,
            "bootstrap_method": "circular_moving_block",
            "bootstrap_replications": 2_000,
            "cost_sensitivity": json.dumps(cost_sensitivity, sort_keys=True),
            "test_start": str(test_dates.min().date()),
            "test_end": str(test_dates.max().date()),
        }
        row.update({f"test_{key}": test_metrics.get(key, np.nan) for key in PUBLIC_METRICS})
        row.update({f"test_{key}": value for key, value in statistics.items()})
        rows.append(row)
    return _write(rows, output_path)


def timing_check(seed: int = 42) -> pd.DataFrame:
    """Compare 5/10/20 iteration timings; ratios should be approximately linear."""
    panels = load_panels()
    records = []
    for n_iters in (5, 10, 20):
        started = time.perf_counter()
        run_random_search(
            n_iters, seed, output_path=ROOT / f".timing_{n_iters}.parquet", panels=panels
        )
        elapsed = time.perf_counter() - started
        records.append(
            {"iterations": n_iters, "seconds": elapsed, "seconds_per_iteration": elapsed / n_iters}
        )
    return pd.DataFrame(records)


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["random", "bayesian", "timing", "combine", "holdout"])
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trajectory", type=Path, default=COMBINED_TRAJECTORY_PATH)
    args = parser.parse_args()
    config = load_config()
    seed = args.seed if args.seed is not None else config.random_seed
    if args.mode == "timing":
        LOGGER.info("Timing results:\n%s", timing_check(seed).to_string(index=False))
    elif args.mode == "combine":
        count = len(combine_trajectories())
        LOGGER.info("Wrote %d combined evaluations to %s", count, COMBINED_TRAJECTORY_PATH)
    elif args.mode == "holdout":
        result = evaluate_out_of_sample(pd.read_parquet(args.trajectory))
        LOGGER.info("Wrote %d held-out evaluations to %s", len(result), OUT_OF_SAMPLE_PATH)
    elif args.mode == "random":
        run_random_search(args.iterations or config.n_random_search_iters, seed)
    else:
        run_bayesian_search(args.iterations or config.n_bayesian_iters, seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
