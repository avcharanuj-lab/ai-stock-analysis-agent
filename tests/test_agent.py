"""
Unit tests for AI Market Analysis Agent module.
"""

import pandas as pd
import pytest

from stock_agent.analysis_agent import analyze_stock
from stock_agent.indicators import add_all_indicators
from stock_agent.sample_data import generate_market_data


def test_agent_with_valid_data():
    df = generate_market_data(symbol="AAPL", days=100, seed=123)
    df_ind = add_all_indicators(df)
    report = analyze_stock(df_ind, symbol="AAPL")

    assert report.symbol == "AAPL"
    assert report.current_price > 0
    assert report.overall_verdict in [
        "Strong Bullish", "Moderate Bullish", "Neutral / Consolidation",
        "Moderate Bearish", "Strong Bearish"
    ]
    assert -100.0 <= report.sentiment_score <= 100.0
    assert 0.0 <= report.confidence_score <= 100.0
    assert len(report.rules) > 0
    assert "EDUCATIONAL DISCLAIMER" in report.disclaimer

    # Verify trade plan fields
    assert "stop_loss" in report.trade_plan
    assert "take_profit" in report.trade_plan
    assert report.trade_plan["stop_loss"] < report.current_price
    assert report.trade_plan["take_profit"] > report.current_price


def test_agent_with_short_data():
    # Only 2 rows of data
    short_df = pd.DataFrame({
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.0, 100.0],
        "Close": [101.0, 102.0],
        "Volume": [1000, 1100],
    }, index=pd.date_range("2024-01-01", periods=2))
    
    report = analyze_stock(short_df, symbol="TEST")
    assert report.overall_verdict == "Insufficient Data"
    assert report.confidence_score == 0.0


def test_agent_rules_integrity():
    df = generate_market_data(symbol="NVDA", days=150, seed=456)
    df_ind = add_all_indicators(df)
    report = analyze_stock(df_ind, symbol="NVDA")

    categories = {r.category for r in report.rules}
    assert "Trend" in categories
    assert "Momentum" in categories
    assert "Volume" in categories
