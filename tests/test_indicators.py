"""
Unit tests for Technical Indicators Module.
"""

import numpy as np
import pandas as pd
import pytest

from stock_agent.indicators import (
    add_all_indicators,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    find_support_resistance,
)


@pytest.fixture
def sample_ohlcv():
    dates = pd.bdate_range(start="2023-01-01", periods=60)
    closes = 100 + np.sin(np.linspace(0, 3.14 * 4, 60)) * 10 + np.linspace(0, 15, 60)
    highs = closes + 2.0
    lows = closes - 2.0
    opens = closes - 0.5
    volumes = np.random.randint(100_000, 1_000_000, size=60)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_sma(sample_ohlcv):
    close = sample_ohlcv["Close"]
    sma20 = calculate_sma(close, 20)
    assert len(sma20) == len(close)
    # 20th item should equal mean of first 20 items
    expected = close.iloc[:20].mean()
    assert np.isclose(sma20.iloc[19], expected, atol=1e-4)


def test_ema(sample_ohlcv):
    close = sample_ohlcv["Close"]
    ema12 = calculate_ema(close, 12)
    assert len(ema12) == len(close)
    assert not ema12.isna().any()


def test_rsi_bounds(sample_ohlcv):
    close = sample_ohlcv["Close"]
    rsi = calculate_rsi(close, 14)
    assert len(rsi) == len(close)
    # RSI must be strictly bounded between 0 and 100
    assert (rsi >= 0.0).all()
    assert (rsi <= 100.0).all()
    assert not rsi.isna().any()


def test_macd(sample_ohlcv):
    close = sample_ohlcv["Close"]
    macd, signal, hist = calculate_macd(close, 12, 26, 9)
    assert len(macd) == len(close)
    assert len(signal) == len(close)
    assert len(hist) == len(close)
    # Histogram must equal MACD - Signal
    assert np.allclose(hist, macd - signal, atol=1e-4)


def test_bollinger_bands(sample_ohlcv):
    close = sample_ohlcv["Close"]
    mid, upper, lower, bw = calculate_bollinger_bands(close, 20, 2.0)
    assert len(mid) == len(close)
    # Upper band must be >= middle >= lower
    assert (upper >= mid).all()
    assert (mid >= lower).all()
    assert (bw >= 0).all()


def test_atr(sample_ohlcv):
    atr = calculate_atr(sample_ohlcv, 14)
    assert len(atr) == len(sample_ohlcv)
    assert (atr > 0).all()


def test_support_resistance(sample_ohlcv):
    sr = find_support_resistance(sample_ohlcv, window=20)
    assert "support" in sr
    assert "resistance" in sr
    assert "pivot" in sr
    assert sr["support"] <= sr["resistance"]


def test_add_all_indicators(sample_ohlcv):
    df = add_all_indicators(sample_ohlcv)
    expected_cols = [
        "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Middle", "BB_Upper", "BB_Lower", "BB_Bandwidth",
        "ATR_14", "Volume_SMA20", "OBV"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
