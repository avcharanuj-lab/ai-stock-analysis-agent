"""
Unit tests for Strategy Backtesting module.
"""

import pandas as pd
import pytest

from stock_agent.backtester import run_backtest
from stock_agent.indicators import add_all_indicators
from stock_agent.sample_data import generate_market_data


@pytest.fixture
def test_dataset():
    df = generate_market_data(symbol="MSFT", days=252, seed=789)
    return add_all_indicators(df)


def test_macd_backtest(test_dataset):
    res = run_backtest(
        df=test_dataset,
        strategy="MACD_TREND",
        initial_capital=10_000,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )

    assert res.initial_capital == 10_000
    assert res.final_equity > 0
    assert not res.equity_curve.empty
    assert "Portfolio_Value" in res.equity_curve.columns
    assert "Benchmark_Value" in res.equity_curve.columns
    assert 0.0 <= res.win_rate_pct <= 100.0


def test_rsi_reversion_backtest(test_dataset):
    res = run_backtest(
        df=test_dataset,
        strategy="RSI_REVERSION",
        initial_capital=5_000,
        stop_loss_pct=0.04,
        take_profit_pct=0.08,
    )

    assert res.initial_capital == 5_000
    assert isinstance(res.total_trades, int)
    assert res.winning_trades + res.losing_trades == res.total_trades


def test_backtest_with_empty_data():
    empty_df = pd.DataFrame()
    res = run_backtest(df=empty_df)
    assert res.total_trades == 0
    assert res.final_equity == 10_000.0
