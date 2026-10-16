"""
Risk Assessment Engine.
Computes risk metrics including Annualized Volatility, Maximum Drawdown,
Value at Risk (VaR), Sharpe & Sortino ratios, and Beta against a benchmark.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


def compute_drawdown_series(prices: pd.Series) -> pd.Series:
    """Computes percentage drawdown from rolling peak."""
    running_max = prices.cummax()
    return (prices - running_max) / running_max.replace(0, np.nan)


def calculate_risk_metrics(
    price_series: pd.Series,
    benchmark_series: Optional[pd.Series] = None,
    risk_free_rate: float = 0.04,
) -> Dict:
    """
    Computes comprehensive risk metrics for a given price series.
    
    Parameters:
        price_series: Historical Close prices
        benchmark_series: Optional benchmark prices (e.g., SPY) for Beta calculation
        risk_free_rate: Annualized risk-free rate (e.g., 0.04 = 4%)
    """
    if len(price_series) < 10:
        return {
            "annualized_volatility": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "var_95_daily": 0.0,
            "cvar_95_daily": 0.0,
            "beta": 1.0,
            "risk_rating": "Insufficient Data",
            "risk_color": "gray",
        }

    daily_returns = price_series.pct_change().dropna()
    n_days = len(daily_returns)
    trading_days = 252

    # Annualized Volatility
    daily_vol = daily_returns.std()
    annualized_vol = float(daily_vol * np.sqrt(trading_days))

    # Drawdown Series & Max Drawdown
    cumulative_returns = (1.0 + daily_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown_series = (cumulative_returns - running_max) / running_max
    max_drawdown = float(drawdown_series.min())  # Negative number
    current_drawdown = float(drawdown_series.iloc[-1])

    # Annualized Return (CAGR)
    total_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1.0
    years = max(n_days / trading_days, 0.05)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1.0 else -0.99

    # Sharpe Ratio
    excess_return = cagr - risk_free_rate
    sharpe_ratio = float(excess_return / annualized_vol) if annualized_vol > 0 else 0.0

    # Downside Deviation & Sortino Ratio
    negative_returns = daily_returns[daily_returns < 0]
    downside_vol = float(negative_returns.std() * np.sqrt(trading_days)) if len(negative_returns) > 2 else annualized_vol
    sortino_ratio = float(excess_return / downside_vol) if downside_vol > 0 else 0.0

    # Historical Value at Risk (VaR 95% 1-day)
    var_95_daily = float(np.percentile(daily_returns, 5))
    cvar_95_daily = float(daily_returns[daily_returns <= var_95_daily].mean()) if len(daily_returns[daily_returns <= var_95_daily]) > 0 else var_95_daily

    # Beta calculation if benchmark provided
    beta = 1.0
    if benchmark_series is not None and len(benchmark_series) >= 10:
        bench_returns = benchmark_series.pct_change().dropna()
        combined = pd.concat([daily_returns, bench_returns], axis=1).dropna()
        if len(combined) > 10:
            cov = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])[0, 1]
            bench_var = np.var(combined.iloc[:, 1])
            if bench_var > 0:
                beta = float(cov / bench_var)

    # Risk Rating & Category
    vol_pct = annualized_vol * 100.0
    mdd_pct = abs(max_drawdown) * 100.0

    if vol_pct < 18.0 and mdd_pct < 15.0:
        risk_rating = "Low Risk / Defensive"
        risk_color = "#22c55e"  # Green
    elif vol_pct < 28.0 and mdd_pct < 28.0:
        risk_rating = "Moderate Risk / Standard Equity"
        risk_color = "#3b82f6"  # Blue
    elif vol_pct < 45.0 or mdd_pct < 42.0:
        risk_rating = "High Risk / Volatile Growth"
        risk_color = "#f59e0b"  # Amber
    else:
        risk_rating = "Very High / Speculative"
        risk_color = "#ef4444"  # Red

    return {
        "annualized_volatility": round(annualized_vol * 100, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "current_drawdown": round(current_drawdown * 100, 2),
        "cagr": round(cagr * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "var_95_daily": round(abs(var_95_daily) * 100, 2),
        "cvar_95_daily": round(abs(cvar_95_daily) * 100, 2),
        "beta": round(beta, 2),
        "risk_rating": risk_rating,
        "risk_color": risk_color,
        "drawdown_series": drawdown_series,
        "daily_returns": daily_returns,
    }
