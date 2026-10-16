"""
Sample and Offline Data Generator.
Provides realistic historical stock data for demo mode or offline execution
when network/API access is unavailable.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd


TICKER_PROFILES = {
    "AAPL": {
        "name": "Apple Inc.",
        "sector": "Technology",
        "base_price": 185.0,
        "drift": 0.12,
        "volatility": 0.22,
        "volume_mean": 55_000_000,
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "sector": "Technology",
        "base_price": 420.0,
        "drift": 0.15,
        "volatility": 0.20,
        "volume_mean": 22_000_000,
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "sector": "Semiconductors",
        "base_price": 125.0,
        "drift": 0.35,
        "volatility": 0.45,
        "volume_mean": 60_000_000,
    },
    "TSLA": {
        "name": "Tesla, Inc.",
        "sector": "Automotive / Clean Energy",
        "base_price": 240.0,
        "drift": 0.08,
        "volatility": 0.50,
        "volume_mean": 80_000_000,
    },
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "sector": "Index ETF",
        "base_price": 530.0,
        "drift": 0.10,
        "volatility": 0.14,
        "volume_mean": 70_000_000,
    },
}


def generate_market_data(
    symbol: str = "AAPL",
    start_date: str = None,
    end_date: str = None,
    days: int = 504,  # Approx 2 trading years
    seed: int = None,
) -> pd.DataFrame:
    """
    Generates realistic daily OHLCV dataframe using Geometric Brownian Motion
    with realistic intraday ranges and volume spikes.
    """
    symbol_upper = symbol.upper().strip()
    profile = TICKER_PROFILES.get(
        symbol_upper,
        {
            "name": f"{symbol_upper} Corp.",
            "sector": "General Equities",
            "base_price": 100.0,
            "drift": 0.10,
            "volatility": 0.25,
            "volume_mean": 30_000_000,
        },
    )

    if seed is None:
        # Deterministic seed based on symbol so repeated demo queries are consistent
        seed = sum(ord(c) for c in symbol_upper) * 42

    rng = np.random.default_rng(seed)

    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = pd.to_datetime(end_date)

    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        date_range = pd.bdate_range(start=start_dt, end=end_dt)
        n_days = len(date_range)
        if n_days < 30:
            n_days = 60
            date_range = pd.bdate_range(end=end_dt, periods=n_days)
    else:
        n_days = max(days, 60)
        date_range = pd.bdate_range(end=end_dt, periods=n_days)

    dt_step = 1.0 / 252.0
    mu = profile["drift"]
    sigma = profile["volatility"]

    # Daily log returns with random shocks
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt_step, sigma * np.sqrt(dt_step), size=len(date_range))
    
    # Occasional earnings/momentum jump
    jump_indices = rng.choice(len(date_range), size=max(1, len(date_range) // 40), replace=False)
    for idx in jump_indices:
        shocks[idx] += rng.normal(0.01, 0.035)

    price_path = profile["base_price"] * np.exp(np.cumsum(shocks))

    # Construct OHLC
    # Daily relative fluctuations
    intraday_vol = sigma * np.sqrt(dt_step) * 0.7
    open_pct = rng.normal(0, intraday_vol * 0.5, size=len(date_range))
    high_pct = np.abs(rng.normal(0, intraday_vol, size=len(date_range))) + 0.003
    low_pct = np.abs(rng.normal(0, intraday_vol, size=len(date_range))) + 0.003

    closes = price_path
    opens = closes * (1.0 + open_pct)
    highs = np.maximum(opens, closes) * (1.0 + high_pct)
    lows = np.minimum(opens, closes) * (1.0 - low_pct)

    # Volume with correlation to absolute price movement
    pct_changes = np.abs(np.diff(np.insert(closes, 0, closes[0]))) / closes
    volume_base = profile["volume_mean"]
    volumes = volume_base * (0.6 + 0.8 * rng.lognormal(0, 0.3, size=len(date_range))) * (1.0 + pct_changes * 10)
    volumes = np.round(volumes).astype(np.int64)

    df = pd.DataFrame(
        {
            "Open": np.round(opens, 2),
            "High": np.round(highs, 2),
            "Low": np.round(lows, 2),
            "Close": np.round(closes, 2),
            "Volume": volumes,
        },
        index=date_range,
    )
    df.index.name = "Date"
    return df
