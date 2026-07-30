"""Represent each alpha evaluation as a comparable point in feature space.

Two views are supported: expression-only features for structural comparisons,
and expression-plus-outcome features for describing the realized search
landscape. Keeping both prevents performance metrics from silently determining
the structural conclusion.
"""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.alpha_lib import PRIMITIVES, AlphaExpression

PARAMETER_COLUMNS = ["lookback", "short", "long", "normalize_lookback"]
METRIC_COLUMNS = [
    "sharpe",
    "sortino",
    "ic",
    "turnover",
    "annualized_volatility",
    "max_drawdown",
]


def _parse_expression(value: object) -> AlphaExpression:
    if isinstance(value, dict):
        expression = value
    if isinstance(value, str):
        expression = json.loads(value)
    elif not isinstance(value, dict):
        raise TypeError(f"Unsupported expression value: {type(value).__name__}")
    if not isinstance(expression, dict) or expression.get("primitive") not in PRIMITIVES:
        raise ValueError("expression must contain a supported primitive")
    return cast(AlphaExpression, expression)


def build_feature_matrix(
    trajectory_db: pd.DataFrame, *, include_metrics: bool = True
) -> pd.DataFrame:
    """Create standardized candidate features, optionally including outcomes.

    Categorical primitives are one-hot encoded, inactive parameters use zero,
    and every column is standardized before distance-based analysis.
    """
    if trajectory_db.empty:
        raise ValueError("trajectory_db is empty")
    if "expression" not in trajectory_db:
        raise ValueError("trajectory_db must contain an expression column")
    expressions = trajectory_db["expression"].map(_parse_expression)
    parameters = pd.DataFrame(
        [
            {column: expression.get(column, 0) for column in PARAMETER_COLUMNS}
            for expression in expressions
        ],
        index=trajectory_db.index,
        dtype=float,
    )
    primitive = pd.Series(
        pd.Categorical(
            [expression["primitive"] for expression in expressions], categories=PRIMITIVES
        ),
        index=trajectory_db.index,
        name="primitive",
    )
    one_hot = pd.get_dummies(primitive, prefix="primitive", dtype=float)
    normalized = pd.DataFrame(
        {
            "normalize_by_volatility": [
                float(expression.get("normalize_by") == "volatility") for expression in expressions
            ]
        },
        index=trajectory_db.index,
    )
    pieces = [parameters, one_hot, normalized]
    if include_metrics:
        missing_metrics = [column for column in METRIC_COLUMNS if column not in trajectory_db]
        if missing_metrics:
            raise ValueError(f"trajectory_db missing metrics: {missing_metrics}")
        pieces.append(trajectory_db[METRIC_COLUMNS].astype(float))
    raw = pd.concat(pieces, axis=1)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.fillna(raw.median(numeric_only=True)).fillna(0.0)
    scaled = StandardScaler().fit_transform(raw)
    result = pd.DataFrame(scaled, index=trajectory_db.index, columns=raw.columns)
    result.attrs["unscaled_features"] = raw
    return result
