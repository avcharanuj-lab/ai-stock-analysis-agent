"""
Strategy Backtesting Module.
Simulates rule-based historical trading strategies, accounting for trade entry/exit,
stop-loss, take-profit, transaction slippage/fees, and benchmark comparisons.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    shares: float
    pnl: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = "OPEN"
    holding_days: int = 0


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    benchmark_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_pct: float
    avg_holding_days: float
    equity_curve: pd.DataFrame
    trades: List[Trade] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    strategy: str = "MACD_TREND",  # RSI_REVERSION, MACD_TREND, SMA_CROSS, AGENT_SCORE
    initial_capital: float = 10_000.0,
    stop_loss_pct: float = 0.05,  # 5%
    take_profit_pct: float = 0.12,  # 12%
    fee_pct: float = 0.001,  # 0.1% per trade (slippage/fee)
) -> BacktestResult:
    """
    Executes a historical bar-by-bar backtest of the selected strategy.
    """
    if df.empty or len(df) < 20:
        empty_curve = pd.DataFrame(columns=["Portfolio_Value", "Benchmark_Value"])
        return BacktestResult(
            strategy_name=strategy,
            symbol="N/A",
            initial_capital=initial_capital,
            final_equity=initial_capital,
            total_return_pct=0.0,
            benchmark_return_pct=0.0,
            cagr_pct=0.0,
            max_drawdown_pct=0.0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_trade_pct=0.0,
            avg_holding_days=0.0,
            equity_curve=empty_curve,
        )

    data = df.copy()
    cash = float(initial_capital)
    shares = 0.0
    current_trade: Optional[Trade] = None
    trades: List[Trade] = []
    equity_records = []

    # Benchmark initial tracking
    first_close = float(data["Close"].iloc[0])
    benchmark_shares = (initial_capital * (1.0 - fee_pct)) / first_close

    for i in range(1, len(data)):
        date = data.index[i]
        bar = data.iloc[i]
        prev_bar = data.iloc[i - 1]
        close = float(bar["Close"])
        high = float(bar["High"])
        low = float(bar["Low"])

        # Check existing position for Stop-Loss or Take-Profit
        if current_trade is not None:
            entry_p = current_trade.entry_price
            holding_days = (date - current_trade.entry_date).days

            # Stop loss hit?
            sl_price = entry_p * (1.0 - stop_loss_pct)
            tp_price = entry_p * (1.0 + take_profit_pct)

            hit_sl = low <= sl_price
            hit_tp = high >= tp_price

            if hit_sl:
                exit_price = sl_price
                gross_proceeds = shares * exit_price
                net_proceeds = gross_proceeds * (1.0 - fee_pct)
                cash += net_proceeds
                pnl = net_proceeds - (shares * entry_p * (1.0 + fee_pct))
                ret = (exit_price / entry_p) - 1.0 - (2 * fee_pct)

                current_trade.exit_date = date
                current_trade.exit_price = round(exit_price, 2)
                current_trade.pnl = round(pnl, 2)
                current_trade.return_pct = round(ret * 100, 2)
                current_trade.exit_reason = "Stop-Loss Hit"
                current_trade.holding_days = max(holding_days, 1)
                trades.append(current_trade)

                current_trade = None
                shares = 0.0

            elif hit_tp:
                exit_price = tp_price
                gross_proceeds = shares * exit_price
                net_proceeds = gross_proceeds * (1.0 - fee_pct)
                cash += net_proceeds
                pnl = net_proceeds - (shares * entry_p * (1.0 + fee_pct))
                ret = (exit_price / entry_p) - 1.0 - (2 * fee_pct)

                current_trade.exit_date = date
                current_trade.exit_price = round(exit_price, 2)
                current_trade.pnl = round(pnl, 2)
                current_trade.return_pct = round(ret * 100, 2)
                current_trade.exit_reason = "Take-Profit Hit"
                current_trade.holding_days = max(holding_days, 1)
                trades.append(current_trade)

                current_trade = None
                shares = 0.0

        # Evaluate Strategy Entry / Exit Signals
        entry_signal = False
        exit_signal = False

        if strategy == "RSI_REVERSION":
            # Buy when RSI dips below 35 and turns up
            rsi = bar.get("RSI_14", 50)
            prev_rsi = prev_bar.get("RSI_14", 50)
            entry_signal = (prev_rsi <= 35) and (rsi > prev_rsi)
            exit_signal = rsi >= 65

        elif strategy == "MACD_TREND":
            # Buy on MACD bullish cross while above SMA 50
            macd = bar.get("MACD", 0)
            signal = bar.get("MACD_Signal", 0)
            prev_macd = prev_bar.get("MACD", 0)
            prev_signal = prev_bar.get("MACD_Signal", 0)
            sma50 = bar.get("SMA_50", close)

            entry_signal = (prev_macd <= prev_signal) and (macd > signal) and (close >= sma50)
            exit_signal = (prev_macd >= prev_signal) and (macd < signal)

        elif strategy == "SMA_CROSS":
            # Golden cross short/medium term: SMA 20 crosses SMA 50
            sma20 = bar.get("SMA_20", close)
            sma50 = bar.get("SMA_50", close)
            prev_sma20 = prev_bar.get("SMA_20", close)
            prev_sma50 = prev_bar.get("SMA_50", close)

            entry_signal = (prev_sma20 <= prev_sma50) and (sma20 > sma50)
            exit_signal = (prev_sma20 >= prev_sma50) and (sma20 < sma50)

        elif strategy == "AGENT_SCORE":
            # Heuristic multi-factor: RSI < 60, MACD Hist > 0, Price > SMA 20
            rsi = bar.get("RSI_14", 50)
            hist = bar.get("MACD_Hist", 0)
            sma20 = bar.get("SMA_20", close)
            entry_signal = (rsi < 62) and (hist > 0) and (close > sma20)
            exit_signal = (hist < 0) or (rsi > 72)

        # Process Signals
        if current_trade is not None and exit_signal:
            exit_price = close
            gross_proceeds = shares * exit_price
            net_proceeds = gross_proceeds * (1.0 - fee_pct)
            cash += net_proceeds
            pnl = net_proceeds - (shares * current_trade.entry_price * (1.0 + fee_pct))
            ret = (exit_price / current_trade.entry_price) - 1.0 - (2 * fee_pct)
            holding_days = (date - current_trade.entry_date).days

            current_trade.exit_date = date
            current_trade.exit_price = round(exit_price, 2)
            current_trade.pnl = round(pnl, 2)
            current_trade.return_pct = round(ret * 100, 2)
            current_trade.exit_reason = "Signal Exit"
            current_trade.holding_days = max(holding_days, 1)
            trades.append(current_trade)

            current_trade = None
            shares = 0.0

        elif current_trade is None and entry_signal and cash > 100:
            # Allocate available cash
            invest_amount = cash * 0.98  # Leave buffer
            entry_price = close
            shares = (invest_amount * (1.0 - fee_pct)) / entry_price
            cash -= invest_amount

            current_trade = Trade(
                entry_date=date,
                exit_date=None,
                entry_price=round(entry_price, 2),
                exit_price=None,
                shares=round(shares, 4),
            )

        # Record daily equity
        current_equity = cash + (shares * close)
        bench_equity = benchmark_shares * close
        equity_records.append({
            "Date": date,
            "Portfolio_Value": round(current_equity, 2),
            "Benchmark_Value": round(bench_equity, 2),
            "Cash": round(cash, 2),
            "Invested": round(shares * close, 2),
        })

    # Close any open trade at the last bar
    if current_trade is not None and len(data) > 0:
        last_date = data.index[-1]
        last_close = float(data["Close"].iloc[-1])
        gross_proceeds = shares * last_close
        net_proceeds = gross_proceeds * (1.0 - fee_pct)
        cash += net_proceeds
        pnl = net_proceeds - (shares * current_trade.entry_price * (1.0 + fee_pct))
        ret = (last_close / current_trade.entry_price) - 1.0 - (2 * fee_pct)
        holding_days = (last_date - current_trade.entry_date).days

        current_trade.exit_date = last_date
        current_trade.exit_price = round(last_close, 2)
        current_trade.pnl = round(pnl, 2)
        current_trade.return_pct = round(ret * 100, 2)
        current_trade.exit_reason = "End of Data"
        current_trade.holding_days = max(holding_days, 1)
        trades.append(current_trade)

    equity_df = pd.DataFrame(equity_records).set_index("Date") if equity_records else pd.DataFrame()

    final_equity = float(equity_df["Portfolio_Value"].iloc[-1]) if not equity_df.empty else initial_capital
    total_return_pct = ((final_equity / initial_capital) - 1.0) * 100.0

    final_bench = float(equity_df["Benchmark_Value"].iloc[-1]) if not equity_df.empty else initial_capital
    bench_return_pct = ((final_bench / initial_capital) - 1.0) * 100.0

    # Max Drawdown of Strategy
    if not equity_df.empty:
        port_series = equity_df["Portfolio_Value"]
        running_max = port_series.cummax()
        dds = (port_series - running_max) / running_max
        max_dd = float(dds.min()) * 100.0
    else:
        max_dd = 0.0

    # Trade Stats
    closed_trades = [t for t in trades if t.exit_price is not None]
    total_trades = len(closed_trades)
    winning_trades = len([t for t in closed_trades if t.pnl > 0])
    losing_trades = len([t for t in closed_trades if t.pnl <= 0])
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_gains = sum(t.pnl for t in closed_trades if t.pnl > 0)
    gross_losses = abs(sum(t.pnl for t in closed_trades if t.pnl < 0))
    profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else (99.0 if gross_gains > 0 else 0.0)

    avg_trade = (sum(t.return_pct for t in closed_trades) / total_trades) if total_trades > 0 else 0.0
    avg_hold = (sum(t.holding_days for t in closed_trades) / total_trades) if total_trades > 0 else 0.0

    # CAGR calculation
    total_days = max((data.index[-1] - data.index[0]).days, 1) if len(data) > 1 else 1
    years = total_days / 365.25
    cagr = float(((final_equity / initial_capital) ** (1.0 / max(years, 0.08)) - 1.0) * 100.0) if final_equity > 0 else -99.0

    return BacktestResult(
        strategy_name=strategy,
        symbol="TICKER",
        initial_capital=initial_capital,
        final_equity=round(final_equity, 2),
        total_return_pct=round(total_return_pct, 2),
        benchmark_return_pct=round(bench_return_pct, 2),
        cagr_pct=round(cagr, 2),
        max_drawdown_pct=round(max_dd, 2),
        win_rate_pct=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        avg_trade_pct=round(avg_trade, 2),
        avg_holding_days=round(avg_hold, 1),
        equity_curve=equity_df,
        trades=closed_trades,
    )
