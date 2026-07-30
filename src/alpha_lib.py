"""Define a small, interpretable alpha space for controlled search experiments.

The primitives are deliberately familiar and the parameters are discrete. That
keeps the landscape explainable: differences between candidates can be traced
to signal family, horizon, or normalization instead of an opaque model.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict, cast

import numpy as np
import pandas as pd

LOOKBACKS = np.array([5, 10, 20, 40, 60, 120, 180, 252], dtype=int)
SHORT_LOOKBACKS = np.array([5, 10, 20, 40], dtype=int)
LONG_LOOKBACKS = np.array([40, 60, 120, 180, 252], dtype=int)
PRIMITIVES = ("momentum", "mean_reversion", "volatility", "volume_ratio", "ma_crossover")

PrimitiveName = Literal["momentum", "mean_reversion", "volatility", "volume_ratio", "ma_crossover"]


class AlphaExpression(TypedDict):
    """Serializable definition of a supported alpha signal."""

    primitive: PrimitiveName
    lookback: NotRequired[int]
    short: NotRequired[int]
    long: NotRequired[int]
    normalize_by: Literal["volatility"] | None
    normalize_lookback: NotRequired[int]


def _validate_lookback(lookback: int, name: str = "lookback") -> None:
    if lookback < 1:
        raise ValueError(f"{name} must be positive")


def momentum(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Calculate compounded trailing returns over ``lookback`` observations."""
    _validate_lookback(lookback)
    return (1.0 + returns).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1.0


def mean_reversion(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Calculate price displacement from its trailing arithmetic mean."""
    _validate_lookback(lookback)
    return prices - prices.rolling(lookback, min_periods=lookback).mean()


def volatility(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Calculate trailing sample volatility."""
    _validate_lookback(lookback)
    return returns.rolling(lookback, min_periods=lookback).std(ddof=1)


def volume_ratio(volume: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Calculate volume relative to its trailing arithmetic mean."""
    _validate_lookback(lookback)
    average = volume.rolling(lookback, min_periods=lookback).mean().replace(0.0, np.nan)
    return volume / average


def ma_crossover(prices: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    """Calculate the difference between fast exponential and slow simple averages."""
    _validate_lookback(short, "short")
    _validate_lookback(long, "long")
    if short >= long:
        raise ValueError("short lookback must be less than long lookback")
    fast = prices.ewm(span=short, min_periods=short, adjust=False).mean()
    slow = prices.rolling(long, min_periods=long).mean()
    return fast - slow


def sample_alpha_expression(rng: np.random.Generator) -> AlphaExpression:
    """Sample a JSON-serializable expression from the fixed research space."""
    primitive = cast(PrimitiveName, str(rng.choice(PRIMITIVES)))
    expression: AlphaExpression = {
        "primitive": primitive,
        "normalize_by": None,
    }
    if primitive == "ma_crossover":
        short = int(rng.choice(SHORT_LOOKBACKS))
        eligible_longs = LONG_LOOKBACKS[short < LONG_LOOKBACKS]
        expression["short"] = short
        expression["long"] = int(rng.choice(eligible_longs))
    else:
        expression["lookback"] = int(rng.choice(LOOKBACKS))
    if bool(rng.integers(0, 2)):
        normalization_choices = LOOKBACKS[LOOKBACKS <= 120]
        if primitive == "volatility":
            normalization_choices = normalization_choices[
                normalization_choices != int(expression["lookback"])
            ]
        expression["normalize_by"] = "volatility"
        expression["normalize_lookback"] = int(rng.choice(normalization_choices))
    return expression


def build_signal(
    expression: AlphaExpression,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    """Build one expression, preserving the input date/ticker panel shape."""
    primitive = expression.get("primitive")
    if primitive == "momentum":
        signal = momentum(returns, int(expression["lookback"]))
    elif primitive == "mean_reversion":
        signal = mean_reversion(prices, int(expression["lookback"]))
    elif primitive == "volatility":
        signal = volatility(returns, int(expression["lookback"]))
    elif primitive == "volume_ratio":
        signal = volume_ratio(volume, int(expression["lookback"]))
    elif primitive == "ma_crossover":
        signal = ma_crossover(prices, int(expression["short"]), int(expression["long"]))
    else:
        raise ValueError(f"Unknown primitive: {primitive!r}")

    if expression.get("normalize_by") == "volatility":
        denominator = volatility(returns, int(expression["normalize_lookback"]))
        signal = signal / denominator.replace(0.0, np.nan)
    elif expression.get("normalize_by") not in {None, "none"}:
        raise ValueError(f"Unknown normalizer: {expression.get('normalize_by')!r}")
    return signal.replace([np.inf, -np.inf], np.nan)
