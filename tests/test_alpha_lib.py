import numpy as np
import pandas as pd

from src.alpha_lib import build_signal, sample_alpha_expression


def test_twenty_sampled_expressions_produce_valid_columns():
    rng = np.random.default_rng(42)
    index = pd.date_range("2018-01-01", periods=600, freq="B")
    columns = [f"T{i}" for i in range(12)]
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, (len(index), len(columns))), index=index, columns=columns
    )
    prices = 100 * (1 + returns).cumprod()
    volume = pd.DataFrame(rng.lognormal(14, 0.3, returns.shape), index=index, columns=columns)
    for _ in range(20):
        expression = sample_alpha_expression(rng)
        signal = build_signal(expression, prices, returns, volume)
        assert signal.shape == prices.shape
        assert not signal.iloc[300:].isna().all(axis=0).any()
