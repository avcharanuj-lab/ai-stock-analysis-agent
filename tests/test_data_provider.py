"""
Unit tests for Data Provider, Demo Mode, and Risk Assessment.
"""

import pandas as pd
import pytest

from stock_agent.data_provider import get_stock_data
from stock_agent.risk import calculate_risk_metrics
from stock_agent.sample_data import generate_market_data


def test_demo_mode_data_structure():
    df, info, is_demo, msg = get_stock_data(symbol="AAPL", period="1y", force_demo=True)
    assert is_demo is True
    assert not df.empty
    assert len(df) >= 200
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert info["currency"] == "USD"
    assert info["last_price"] > 0


def test_custom_symbol_demo():
    df, info, is_demo, msg = get_stock_data(symbol="UNKNOWN_XYZ", period="6mo", force_demo=True)
    assert is_demo is True
    assert not df.empty
    assert len(df) >= 50
    assert info["last_price"] > 0


def test_risk_metrics_calculation():
    df = generate_market_data(symbol="SPY", days=252, seed=101)
    metrics = calculate_risk_metrics(df["Close"])

    assert metrics["annualized_volatility"] > 0
    assert metrics["max_drawdown"] <= 0
    assert metrics["var_95_daily"] > 0
    assert metrics["risk_rating"] in [
        "Low Risk / Defensive",
        "Moderate Risk / Standard Equity",
        "High Risk / Volatile Growth",
        "Very High / Speculative",
    ]
