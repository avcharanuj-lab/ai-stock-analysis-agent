"""
Data Provider Module.
Handles fetching historical market data via yfinance, with seamless fallback
to bundled realistic demo data when network requests fail or demo mode is active.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd

from stock_agent.sample_data import TICKER_PROFILES, generate_market_data


def clean_yfinance_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw yfinance dataframe:
    - Flattens MultiIndex columns if present
    - Ensures standard columns ['Open', 'High', 'Low', 'Close', 'Volume']
    - Converts index to timezone-naive datetime
    - Drops empty rows
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    # Handle multi-level columns if ticker was part of multi-index
    if isinstance(df.columns, pd.MultiIndex):
        # Often ('Close', 'AAPL') or ('AAPL', 'Close')
        try:
            df.columns = [col[0] if col[0] in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] else col[1] for col in df.columns]
        except Exception:
            df.columns = df.columns.get_level_values(0)

    # Standardize column names
    col_map = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=col_map)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            if col == 'Volume':
                df['Volume'] = 1_000_000
            elif 'Close' in df.columns:
                df[col] = df['Close']
            else:
                return pd.DataFrame()

    # Use standard 5 columns
    df = df[required_cols].dropna()

    # Clean index
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    return df.astype(float)


def get_stock_data(
    symbol: str = "AAPL",
    period: str = "1y",
    force_demo: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict, bool, str]:
    """
    Fetches stock data for a given ticker symbol.
    
    Returns:
        (df, info_dict, is_demo, status_message)
    """
    symbol_upper = symbol.strip().upper()
    if not symbol_upper:
        symbol_upper = "AAPL"

    status_msg = ""
    is_demo = force_demo
    data_df = pd.DataFrame()
    info_dict = {}

    # If demo mode is not forced, try yfinance
    if not force_demo:
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol_upper)
            if start_date and end_date:
                raw_df = ticker.history(start=start_date, end=end_date, auto_adjust=True, timeout=8)
            else:
                raw_df = ticker.history(period=period, auto_adjust=True, timeout=8)

            cleaned_df = clean_yfinance_dataframe(raw_df)
            if not cleaned_df.empty and len(cleaned_df) >= 10:
                data_df = cleaned_df
                # Safely attempt to extract company info
                try:
                    raw_info = ticker.fast_info
                    info_dict = {
                        "name": symbol_upper,
                        "currency": getattr(raw_info, "currency", "USD") or "USD",
                        "market_cap": getattr(raw_info, "market_cap", None),
                        "year_high": getattr(raw_info, "year_high", None),
                        "year_low": getattr(raw_info, "year_low", None),
                        "last_price": getattr(raw_info, "last_price", float(cleaned_df["Close"].iloc[-1])),
                    }
                except Exception:
                    info_dict = {
                        "name": symbol_upper,
                        "currency": "USD",
                        "last_price": float(cleaned_df["Close"].iloc[-1]),
                    }
                status_msg = f"Live Market Data connected via Yahoo Finance for {symbol_upper}."
                return data_df, info_dict, False, status_msg
            else:
                status_msg = f"No live price data returned for '{symbol_upper}'. Falling back to Demo Mode."
                is_demo = True
        except Exception as e:
            err_str = str(e)
            if "HTTP" in err_str or "timeout" in err_str.lower() or "connection" in err_str.lower():
                status_msg = f"Network unavailable or Yahoo Finance connection timed out. Falling back to Demo Mode."
            else:
                status_msg = f"Data provider notice: {err_str[:80]}. Falling back to Demo Mode."
            is_demo = True

    # Demo Mode or Fallback Execution
    days_lookup = {
        "1mo": 22,
        "3mo": 66,
        "6mo": 130,
        "1y": 252,
        "2y": 504,
        "5y": 1260,
    }
    days_to_gen = days_lookup.get(period, 252)

    data_df = generate_market_data(
        symbol=symbol_upper,
        start_date=start_date,
        end_date=end_date,
        days=days_to_gen,
    )

    profile = TICKER_PROFILES.get(
        symbol_upper,
        {
            "name": f"{symbol_upper} Corp (Demo)",
            "sector": "Demo Asset",
            "base_price": float(data_df["Close"].iloc[-1]),
        },
    )

    info_dict = {
        "name": profile.get("name", symbol_upper),
        "sector": profile.get("sector", "General Equities"),
        "currency": "USD",
        "last_price": float(data_df["Close"].iloc[-1]),
        "year_high": float(data_df["High"].max()),
        "year_low": float(data_df["Low"].min()),
        "market_cap": 2_500_000_000_000 if symbol_upper in ["AAPL", "MSFT", "NVDA"] else 25_000_000_000,
    }

    if not status_msg:
        status_msg = f"Demo Mode active: Using synthetic realistic market data for {symbol_upper}."

    return data_df, info_dict, is_demo, status_msg
