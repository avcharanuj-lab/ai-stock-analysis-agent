"""
Technical Indicators Calculation Engine.
Pure pandas/numpy implementations without external C-extension dependencies (such as TA-Lib).
Resilient against NaN values and edge cases.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculates Simple Moving Average (SMA)."""
    return series.rolling(window=period, min_periods=1).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates Relative Strength Index (RSI) using Wilder's smoothed moving average.
    Returns values between 0 and 100.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's exponential smoothing: alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Fill beginning values
    rsi = rsi.fillna(50.0)
    return rsi


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculates MACD Line, Signal Line, and MACD Histogram.
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)
    Hist = MACD - Signal
    """
    fast_ema = calculate_ema(series, fast_period)
    slow_ema = calculate_ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculates Bollinger Bands: Middle Band (SMA), Upper Band, Lower Band, and Bandwidth %.
    """
    middle = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std().fillna(0)
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    bandwidth = ((upper - lower) / middle.replace(0, np.nan)) * 100.0
    return middle, upper, lower, bandwidth.fillna(0)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates Average True Range (ATR) for volatility and stop-loss estimation.
    Requires High, Low, Close columns.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr.fillna(tr.mean() if not tr.empty else 1.0)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculates On-Balance Volume (OBV)."""
    close = df["Close"]
    volume = df["Volume"]
    direction = np.sign(close.diff().fillna(0))
    obv = (volume * direction).cumsum()
    return obv


def find_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Identifies dynamic short-term support and resistance levels
    based on local minima/maxima and rolling quantiles.
    """
    if len(df) < window:
        return {
            "support": float(df["Low"].min()),
            "resistance": float(df["High"].max()),
            "recent_range": (float(df["Low"].min()), float(df["High"].max())),
        }

    recent_slice = df.tail(window)
    resistance = float(recent_slice["High"].max())
    support = float(recent_slice["Low"].min())
    current_close = float(df["Close"].iloc[-1])

    # Secondary Fibonacci / Pivot estimates
    pivot = (resistance + support + current_close) / 3.0
    r1 = 2 * pivot - support
    s1 = 2 * pivot - resistance

    return {
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "pivot": round(pivot, 2),
        "r1": round(r1, 2),
        "s1": round(s1, 2),
    }


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends all technical indicators into the dataframe.
    Returns a copy of the dataframe with added indicator columns.
    """
    data = df.copy()
    close = data["Close"]

    # Moving Averages
    data["SMA_20"] = calculate_sma(close, 20)
    data["SMA_50"] = calculate_sma(close, 50)
    data["SMA_200"] = calculate_sma(close, 200)
    data["EMA_12"] = calculate_ema(close, 12)
    data["EMA_26"] = calculate_ema(close, 26)

    # RSI
    data["RSI_14"] = calculate_rsi(close, 14)

    # MACD
    macd, signal, hist = calculate_macd(close, 12, 26, 9)
    data["MACD"] = macd
    data["MACD_Signal"] = signal
    data["MACD_Hist"] = hist

    # Bollinger Bands
    bb_mid, bb_upper, bb_lower, bb_bw = calculate_bollinger_bands(close, 20, 2.0)
    data["BB_Middle"] = bb_mid
    data["BB_Upper"] = bb_upper
    data["BB_Lower"] = bb_lower
    data["BB_Bandwidth"] = bb_bw

    # ATR
    data["ATR_14"] = calculate_atr(data, 14)

    # Volume Indicators
    data["Volume_SMA20"] = data["Volume"].rolling(window=20, min_periods=1).mean()
    data["OBV"] = calculate_obv(data)

    return data
