"""Re-evaluate the frozen study winner on data that did not exist at selection time.

The main study ends at the configured ``end_date``. Every observation after that
boundary is genuinely out of time: it was unavailable when the expression was
searched, selected, and audited, so it cannot have leaked into any of those
decisions. This module downloads that window into an isolated cache, rebuilds a
panel with the study's own construction rules, and scores the fixed expression
on it.

The harness is deliberately rerunnable. Each execution extends the forward
window to the present, so evidence accumulates over calendar time instead of
requiring a new experiment. It is an accumulating audit, not a second search:
nothing here selects, tunes, or reranks anything.

A short forward window cannot support a confident Sharpe estimate. The reported
diagnostics therefore include the number of non-overlapping observations and an
explicit power verdict, so a wide interval is read as insufficient evidence
rather than as a weak result.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd

from src.alpha_lib import AlphaExpression, build_signal
from src.backtest import (
    backtest_alpha,
    block_bootstrap_sharpe_ci,
    net_portfolio_returns,
    sharpe_statistics,
    summarize_backtest,
)
from src.data_pipeline import (
    PROCESSED_DIR,
    RESULTS_DIR,
    ROOT,
    SP100_TICKERS,
    build_panel,
    download_universe,
    load_config,
)
from src.search import OUT_OF_SAMPLE_PATH, PUBLIC_METRICS

FORWARD_RAW_DIR = ROOT / "data" / "forward_raw"
FORWARD_PROCESSED_DIR = ROOT / "data" / "forward_processed"
FORWARD_RESULTS_PATH = RESULTS_DIR / "forward_test.parquet"
COST_SCENARIOS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)

# The expression grammar's longest lookback is 252 trading days. Two calendar
# years of warmup covers that for any candidate the search could have produced,
# so the first forward date is scored from a fully-formed signal.
WARMUP_DAYS = 730
# Below this many non-overlapping observations, a Sharpe point estimate carries
# no practical information and the verdict says so outright.
MIN_INFORMATIVE_OBSERVATIONS = 60

LOGGER = logging.getLogger(__name__)
Panels: TypeAlias = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def study_boundary() -> pd.Timestamp:
    """Return the last date present in the frozen study panel."""
    path = PROCESSED_DIR / "prices.parquet"
    if not path.exists():
        raise RuntimeError(f"Study panel missing: {path}. Run src.data_pipeline first.")
    return cast(pd.Timestamp, pd.read_parquet(path).index.max())


def frozen_expressions(path: Path = OUT_OF_SAMPLE_PATH) -> pd.DataFrame:
    """Load the expressions the study selected, deduplicated to unique candidates.

    Both optimizers converged on one expression in the published run. Scoring it
    once and recording which optimizers chose it keeps the forward evidence from
    appearing to be two independent confirmations of the same thing.
    """
    if not path.exists():
        raise RuntimeError(f"Holdout results missing: {path}. Run 'src.search holdout' first.")
    results = pd.read_parquet(path)
    if results.empty:
        raise ValueError("holdout results are empty")
    canonical = results["expression"].map(
        lambda value: json.dumps(
            json.loads(value) if isinstance(value, str) else value, sort_keys=True
        )
    )
    grouped = (
        pd.DataFrame({"expression": canonical, "optimizer": results["optimizer"]})
        .groupby("expression")["optimizer"]
        .apply(lambda names: "+".join(sorted(names)))
        .reset_index()
    )
    return grouped


def refresh_forward_panel(end: date | None = None) -> Panels:
    """Download and build an isolated panel spanning warmup plus the forward window.

    The study's own cache is never written to, so the published experiment stays
    reproducible regardless of how often this runs.
    """
    boundary = study_boundary()
    start = (boundary - timedelta(days=WARMUP_DAYS)).date()
    stop = end or date.today()
    if stop <= boundary.date():
        raise ValueError(
            f"No forward data available: study ends {boundary.date()}, requested end {stop}"
        )
    LOGGER.info("Downloading forward panel %s through %s", start, stop)
    download_universe(
        SP100_TICKERS, start.isoformat(), stop.isoformat(), raw_dir=FORWARD_RAW_DIR
    )
    build_panel(raw_dir=FORWARD_RAW_DIR, processed_dir=FORWARD_PROCESSED_DIR)
    return load_forward_panels()


def load_forward_panels() -> Panels:
    """Load the isolated forward panel."""
    paths = [FORWARD_PROCESSED_DIR / f"{name}.parquet" for name in ("prices", "returns", "volume")]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Forward panel missing: {missing}. Run with --refresh first.")
    prices, returns, volume = (pd.read_parquet(path) for path in paths)
    return prices, returns, volume


def power_verdict(n_observations: int, horizon: int, ci_width: float) -> str:
    """State plainly whether the window can support a Sharpe claim."""
    independent = n_observations / horizon if horizon else 0.0
    if independent < MIN_INFORMATIVE_OBSERVATIONS / horizon:
        return (
            f"Insufficient: {independent:.0f} non-overlapping observations give a "
            f"{ci_width:.2f}-wide 95% interval. Directional only; no Sharpe claim is supported."
        )
    if ci_width > 1.0:
        return (
            f"Weak: {independent:.0f} non-overlapping observations still leave a "
            f"{ci_width:.2f}-wide 95% interval spanning both signs."
        )
    return (
        f"Usable: {independent:.0f} non-overlapping observations give a "
        f"{ci_width:.2f}-wide 95% interval."
    )


def evaluate_forward_window(
    *,
    panels: Panels | None = None,
    output_path: Path = FORWARD_RESULTS_PATH,
) -> pd.DataFrame:
    """Score each frozen expression on dates strictly after the study boundary."""
    config = load_config()
    horizon = config.forward_return_horizon
    boundary = study_boundary()
    prices, returns, volume = panels or load_forward_panels()

    forward_dates = returns.index[returns.index > boundary]
    if forward_dates.empty:
        raise ValueError(f"Forward panel contains no dates after {boundary.date()}")

    rows: list[dict[str, object]] = []
    for _, candidate in frozen_expressions().iterrows():
        expression = cast(AlphaExpression, json.loads(str(candidate["expression"])))
        # The signal is built on warmup + forward history so lookbacks are fully
        # formed, then scored only on post-boundary dates.
        full_result = backtest_alpha(
            build_signal(expression, prices, returns, volume), returns, horizon
        )
        gross = summarize_backtest(full_result, horizon, forward_dates)

        net_result = dict(full_result)
        net_result["portfolio_returns"] = net_portfolio_returns(
            full_result["portfolio_returns"], full_result["weights"], config.transaction_cost_bps
        )
        net = summarize_backtest(net_result, horizon, forward_dates)

        lower, upper = block_bootstrap_sharpe_ci(
            gross["portfolio_returns"], horizon, seed=config.random_seed
        )
        statistics = sharpe_statistics(gross["portfolio_returns"], horizon)

        cost_sensitivity = {}
        for scenario_bps in COST_SCENARIOS:
            scenario = dict(full_result)
            scenario["portfolio_returns"] = net_portfolio_returns(
                full_result["portfolio_returns"], full_result["weights"], scenario_bps
            )
            cost_sensitivity[f"{scenario_bps:g}"] = float(
                summarize_backtest(scenario, horizon, forward_dates)["sharpe"]
            )

        observations = int(gross["n_observations"])
        ci_width = float(upper - lower) if np.isfinite(upper - lower) else float("nan")
        row: dict[str, object] = {
            "expression": json.dumps(expression, sort_keys=True),
            "selected_by": candidate["optimizer"],
            "study_boundary": str(boundary.date()),
            "forward_start": str(forward_dates.min().date()),
            "forward_end": str(forward_dates.max().date()),
            "evaluated_on": date.today().isoformat(),
            "transaction_cost_bps": float(config.transaction_cost_bps),
            "forward_net_sharpe": float(net["sharpe"]),
            "forward_sharpe_ci_lower": lower,
            "forward_sharpe_ci_upper": upper,
            "forward_sharpe_ci_width": ci_width,
            "non_overlapping_observations": observations / horizon,
            "bootstrap_method": "circular_moving_block",
            "bootstrap_replications": 2_000,
            "cost_sensitivity": json.dumps(cost_sensitivity, sort_keys=True),
            "power_verdict": power_verdict(observations, horizon, ci_width),
        }
        row.update({f"forward_{key}": gross.get(key, np.nan) for key in PUBLIC_METRICS})
        row.update({f"forward_{key}": value for key, value in statistics.items()})
        rows.append(row)

    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)
    return frame


def format_summary(results: pd.DataFrame) -> str:
    """Render the forward result as a short, self-limiting markdown section."""
    lines = [
        "## Forward test on data after the study window",
        "",
        "The study's holdout ended at the configured `end_date`. The rows below score the",
        "already-selected expression on observations that did not exist when it was chosen,",
        "so no part of this window informed the search, the selection, or the audit.",
        "",
        "| Selected expression | Chosen by | Forward window | Obs. | Gross Sharpe | "
        "Net Sharpe | 95% interval |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in results.iterrows():
        expression: dict[str, Any] = json.loads(str(row["expression"]))
        label = f"{expression.get('lookback')}-day {expression.get('primitive')}"
        lines.append(
            f"| {label} | {row['selected_by']} | {row['forward_start']} to {row['forward_end']} "
            f"| {int(row['forward_n_observations'])} | {row['forward_sharpe']:.3f} "
            f"| {row['forward_net_sharpe']:.3f} "
            f"| [{row['forward_sharpe_ci_lower']:.3f}, {row['forward_sharpe_ci_upper']:.3f}] |"
        )
    lines += ["", "**Statistical power:**", ""]
    for _, row in results.iterrows():
        lines.append(f"- {row['power_verdict']}")
    lines += [
        "",
        "This window grows every time the harness is rerun. Until the interval narrows enough",
        "to exclude zero, the honest reading is that the forward evidence is inconclusive —",
        "which is a statement about sample size, not a verdict on the signal.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="download the forward window before scoring"
    )
    parser.add_argument(
        "--end", type=date.fromisoformat, default=None, help="forward window end (YYYY-MM-DD)"
    )
    args = parser.parse_args()
    panels = refresh_forward_panel(args.end) if args.refresh else None
    results = evaluate_forward_window(panels=panels)
    LOGGER.info("Wrote %d forward evaluations to %s", len(results), FORWARD_RESULTS_PATH)
    print(format_summary(results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
