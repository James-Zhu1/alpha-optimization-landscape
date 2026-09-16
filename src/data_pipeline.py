"""Download and align the OHLCV panel used by the alpha-search case study.

The price observations are dated, but the configured universe is a fixed
large-cap snapshot rather than point-in-time constituent history. The resulting
survivorship bias is an explicit study limitation.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import pandas as pd
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "config.yaml"
CACHE_MAX_AGE_SECONDS: Final = 24 * 60 * 60
MAX_DOWNLOAD_ATTEMPTS: Final = 5
MAX_FAILURE_RATE: Final = 0.10
LOGGER = logging.getLogger(__name__)

# A stable research-universe snapshot keeps the experiment internally consistent.
# It is not point-in-time membership data; the resulting survivorship bias is
# documented in the portfolio report.
SP100_TICKERS: Final = (
    "AAPL",
    "ABBV",
    "ABT",
    "ACN",
    "ADBE",
    "AIG",
    "AMD",
    "AMGN",
    "AMT",
    "AMZN",
    "AVGO",
    "AXP",
    "BA",
    "BAC",
    "BK",
    "BKNG",
    "BLK",
    "BMY",
    "BRK-B",
    "C",
    "CAT",
    "CHTR",
    "CL",
    "CMCSA",
    "COF",
    "COP",
    "COST",
    "CRM",
    "CSCO",
    "CVS",
    "CVX",
    "DE",
    "DHR",
    "DIS",
    "DUK",
    "EMR",
    "EXC",
    "F",
    "FDX",
    "GD",
    "GE",
    "GILD",
    "GM",
    "GOOG",
    "GOOGL",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "INTU",
    "JNJ",
    "JPM",
    "KO",
    "LIN",
    "LLY",
    "LMT",
    "LOW",
    "MA",
    "MCD",
    "MDLZ",
    "MDT",
    "MET",
    "META",
    "MMM",
    "MO",
    "MRK",
    "MS",
    "MSFT",
    "NEE",
    "NFLX",
    "NKE",
    "NOW",
    "NVDA",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "PM",
    "PYPL",
    "QCOM",
    "RTX",
    "SBUX",
    "SCHW",
    "SO",
    "SPG",
    "T",
    "TGT",
    "TMO",
    "TMUS",
    "TSLA",
    "TXN",
    "UNH",
    "UNP",
    "UPS",
    "USB",
    "V",
    "VZ",
    "WFC",
    "WMT",
    "XOM",
)


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """Validated configuration for data preparation and alpha searches."""

    universe: str
    start_date: str
    end_date: str
    forward_return_horizon: int
    n_random_search_iters: int
    n_bayesian_iters: int
    random_seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    transaction_cost_bps: float = 5.0

    def __post_init__(self) -> None:
        try:
            start = date.fromisoformat(self.start_date)
            end = date.fromisoformat(self.end_date)
        except ValueError as exc:
            raise ValueError("start_date and end_date must use ISO format YYYY-MM-DD") from exc
        if start >= end:
            raise ValueError("start_date must precede end_date")
        for name in (
            "forward_return_horizon",
            "n_random_search_iters",
            "n_bayesian_iters",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("train, validation, and test fractions must be positive and sum to 1")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative")


def load_config(path: Path = CONFIG_PATH) -> ResearchConfig:
    """Load and validate the project configuration from YAML."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    try:
        return ResearchConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"Configuration has missing or unsupported fields: {path}") from exc


def _cache_is_fresh(path: Path) -> bool:
    return path.exists() and time.time() - path.stat().st_mtime < CACHE_MAX_AGE_SECONDS


def download_universe(
    tickers: Iterable[str], start: str, end: str, *, raw_dir: Path = RAW_DIR
) -> None:
    """Download one adjusted OHLCV parquet per ticker, with retry/backoff.

    ``raw_dir`` lets callers stage an independent panel (for example a
    post-study forward-test window) without disturbing the frozen study cache.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional network stack
        raise RuntimeError("Install project dependencies before downloading data") from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    ticker_list = tuple(dict.fromkeys(tickers))
    if not ticker_list:
        raise ValueError("tickers must not be empty")
    failures: list[str] = []
    for ticker in ticker_list:
        path = raw_dir / f"{ticker}.parquet"
        if _cache_is_fresh(path):
            continue
        for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
            try:
                frame = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if frame.empty:
                    raise ValueError("empty response")
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = frame.columns.get_level_values(0)
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                frame.index.name = "date"
                frame.to_parquet(path)
                break
            except (OSError, RuntimeError, ValueError) as exc:
                if attempt == MAX_DOWNLOAD_ATTEMPTS - 1:
                    LOGGER.warning("Download failed for %s: %s", ticker, exc)
                    failures.append(ticker)
                else:
                    time.sleep(2**attempt)
    if len(failures) > max(1, int(MAX_FAILURE_RATE * len(ticker_list))):
        raise RuntimeError(f"More than 10% of tickers failed to download: {failures}")
    if failures:
        warnings.warn(
            f"Continuing without {len(failures)} unavailable tickers: {failures}",
            RuntimeWarning,
            stacklevel=2,
        )


def _load_field(files: Iterable[Path], field: str) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for path in files:
        frame = pd.read_parquet(path)
        lookup = {str(column).lower(): column for column in frame.columns}
        if field.lower() not in lookup:
            continue
        series[path.stem] = frame[lookup[field.lower()]].rename(path.stem)
    if not series:
        raise RuntimeError(f"No raw files contained the {field!r} field")
    return pd.concat(series.values(), axis=1).sort_index()


def build_panel(*, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> None:
    """Align raw ticker files and write clean close, return, and volume panels."""
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError("No raw parquet files found; run download_universe first")

    prices = _load_field(files, "Close")
    volume = _load_field(files, "Volume")
    common_columns = prices.columns.intersection(volume.columns)
    prices, volume = prices[common_columns], volume[common_columns]

    # Keep dates on which at least 90% of the downloaded universe has observations.
    eligible_dates = prices.notna().mean(axis=1) >= 0.90
    prices, volume = prices.loc[eligible_dates], volume.loc[eligible_dates]
    prices = prices.ffill(limit=1)
    volume = volume.ffill(limit=1)

    keep = (prices.isna().mean() <= 0.05) & (volume.isna().mean() <= 0.05)
    prices, volume = prices.loc[:, keep], volume.loc[:, keep]
    returns = prices.pct_change(fill_method=None)

    processed_dir.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(processed_dir / "prices.parquet")
    returns.to_parquet(processed_dir / "returns.parquet")
    volume.to_parquet(processed_dir / "volume.parquet")


def _max_missing_run(series: pd.Series) -> int:
    missing = series.isna()
    if not missing.any():
        return 0
    groups = missing.ne(missing.shift()).cumsum()
    return int(missing.groupby(groups).sum().max())


def sanity_check_panel() -> dict[str, object]:
    """Return the processed-panel validation summary."""
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    volume = pd.read_parquet(PROCESSED_DIR / "volume.parquet")
    result = {
        "start_date": str(prices.index.min().date()),
        "end_date": str(prices.index.max().date()),
        "n_dates": int(len(prices)),
        "n_tickers": int(prices.shape[1]),
        "max_consecutive_missing_days": {
            column: _max_missing_run(prices[column]) for column in prices.columns
        },
        "tickers_over_5pct_missing": prices.columns[prices.isna().mean() > 0.05].tolist(),
        "zero_or_negative_prices": bool((prices <= 0).any().any()),
        "negative_volume": bool((volume < 0).any().any()),
    }
    return result


def main() -> None:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "build", "sanity", "all"])
    args = parser.parse_args()
    config = load_config()
    if args.command in {"download", "all"}:
        download_universe(SP100_TICKERS, config.start_date, config.end_date)
    if args.command in {"build", "all"}:
        build_panel()
    if args.command in {"sanity", "all"}:
        LOGGER.info("Panel validation:\n%s", yaml.safe_dump(sanity_check_panel(), sort_keys=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
