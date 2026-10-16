"""
AI Market Analysis Agent.
Rule-based heuristic analysis engine that synthesizes multiple technical factors
into a transparent, audited market verdict with explicit rule checklists,
confidence ratings, and risk-controlled entry/exit parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from stock_agent.indicators import find_support_resistance


@dataclass
class RuleEvaluation:
    name: str
    category: str  # Trend, Momentum, Volatility, Volume
    condition: str
    status: str  # BULLISH, BEARISH, NEUTRAL
    weight: float
    explanation: str
    passed: bool


@dataclass
class AgentAnalysisReport:
    symbol: str
    current_price: float
    overall_verdict: str  # Strong Bullish, Moderate Bullish, Neutral / Watch, Moderate Bearish, Strong Bearish
    signal_type: str  # POTENTIAL_BUY_WATCH, NEUTRAL_HOLD, POTENTIAL_EXIT_WATCH
    sentiment_score: float  # Normalized -100 to +100
    confidence_score: float  # 0 to 100%
    trend_summary: str
    momentum_summary: str
    volatility_summary: str
    rules: List[RuleEvaluation] = field(default_factory=list)
    support_resistance: Dict[str, float] = field(default_factory=dict)
    trade_plan: Dict[str, Optional[float]] = field(default_factory=dict)
    executive_summary: str = ""
    disclaimer: str = (
        "EDUCATIONAL DISCLAIMER: Algorithmic indicators are backward-looking heuristics "
        "and do not predict future market outcomes or guarantee returns. "
        "Never risk capital based solely on automated rule outputs."
    )


def analyze_stock(df: pd.DataFrame, symbol: str = "TICKER") -> AgentAnalysisReport:
    """
    Executes a comprehensive, transparent rule-based analysis on the provided
    dataframe with indicators.
    """
    if df.empty or len(df) < 5:
        return AgentAnalysisReport(
            symbol=symbol,
            current_price=0.0,
            overall_verdict="Insufficient Data",
            signal_type="NEUTRAL_HOLD",
            sentiment_score=0.0,
            confidence_score=0.0,
            trend_summary="Not enough data points.",
            momentum_summary="Not enough data points.",
            volatility_summary="Not enough data points.",
            executive_summary="Data length is insufficient to compute technical analysis.",
        )

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) >= 2 else last_row

    price = float(last_row["Close"])
    rules: List[RuleEvaluation] = []
    bullish_points = 0.0
    bearish_points = 0.0
    max_possible = 0.0

    # -------------------------------------------------------------
    # 1. TREND RULES
    # -------------------------------------------------------------
    # Rule 1.1: Price vs SMA 50
    sma50 = last_row.get("SMA_50", np.nan)
    if not np.isnan(sma50) and sma50 > 0:
        max_possible += 15
        if price > sma50:
            bullish_points += 15
            rules.append(RuleEvaluation(
                name="Medium-Term Trend (SMA 50)",
                category="Trend",
                condition=f"Price ({price:.2f}) > SMA 50 ({sma50:.2f})",
                status="BULLISH",
                weight=15,
                explanation=f"Price is trading {((price/sma50)-1)*100:.1f}% above the 50-day moving average, signaling upward medium-term trend.",
                passed=True,
            ))
        else:
            bearish_points += 15
            rules.append(RuleEvaluation(
                name="Medium-Term Trend (SMA 50)",
                category="Trend",
                condition=f"Price ({price:.2f}) < SMA 50 ({sma50:.2f})",
                status="BEARISH",
                weight=-15,
                explanation=f"Price is trading {((sma50/price)-1)*100:.1f}% below the 50-day moving average, signaling downward medium-term trend.",
                passed=False,
            ))

    # Rule 1.2: Price vs SMA 200 & Golden Cross check
    sma200 = last_row.get("SMA_200", np.nan)
    if not np.isnan(sma200) and sma200 > 0:
        max_possible += 20
        is_above_200 = price > sma200
        has_golden_cross = not np.isnan(sma50) and sma50 > sma200
        if is_above_200 and has_golden_cross:
            bullish_points += 20
            rules.append(RuleEvaluation(
                name="Long-Term Trend & Golden Cross",
                category="Trend",
                condition=f"Price > SMA 200 and SMA 50 ({sma50:.2f}) > SMA 200 ({sma200:.2f})",
                status="BULLISH",
                weight=20,
                explanation="Classic Golden Cross regime: SMA 50 is above SMA 200 and current price holds above long-term baseline.",
                passed=True,
            ))
        elif not is_above_200 and not has_golden_cross:
            bearish_points += 20
            rules.append(RuleEvaluation(
                name="Long-Term Trend & Death Cross",
                category="Trend",
                condition=f"Price < SMA 200 and SMA 50 ({sma50:.2f}) < SMA 200 ({sma200:.2f})",
                status="BEARISH",
                weight=-20,
                explanation="Death Cross regime: SMA 50 is below SMA 200 and price trades below long-term baseline.",
                passed=False,
            ))
        else:
            # Transitional
            rules.append(RuleEvaluation(
                name="Long-Term Trend Alignment",
                category="Trend",
                condition="Price / SMA 200 mixed",
                status="NEUTRAL",
                weight=0,
                explanation="Long-term moving averages are in a transitional/consolidation alignment.",
                passed=True,
            ))

    # Rule 1.3: Short-Term Momentum vs SMA 20
    sma20 = last_row.get("SMA_20", np.nan)
    if not np.isnan(sma20) and sma20 > 0:
        max_possible += 10
        if price >= sma20:
            bullish_points += 10
            rules.append(RuleEvaluation(
                name="Short-Term Moving Average (SMA 20)",
                category="Trend",
                condition=f"Price ({price:.2f}) >= SMA 20 ({sma20:.2f})",
                status="BULLISH",
                weight=10,
                explanation="Price holds above the 20-day mean, maintaining short-term momentum.",
                passed=True,
            ))
        else:
            bearish_points += 10
            rules.append(RuleEvaluation(
                name="Short-Term Moving Average (SMA 20)",
                category="Trend",
                condition=f"Price ({price:.2f}) < SMA 20 ({sma20:.2f})",
                status="BEARISH",
                weight=-10,
                explanation="Price has breached below the 20-day moving average, showing short-term weakness.",
                passed=False,
            ))

    # -------------------------------------------------------------
    # 2. MOMENTUM RULES (RSI & MACD)
    # -------------------------------------------------------------
    # Rule 2.1: RSI (14)
    rsi = last_row.get("RSI_14", 50.0)
    max_possible += 20
    if rsi < 30:
        bullish_points += 15
        rules.append(RuleEvaluation(
            name="RSI Oversold Condition",
            category="Momentum",
            condition=f"RSI 14 ({rsi:.1f}) < 30",
            status="BULLISH",
            weight=15,
            explanation=f"RSI is oversold at {rsi:.1f}, indicating selling pressure may be overextended and ripe for mean-reversion bounce.",
            passed=True,
        ))
    elif rsi > 70:
        bearish_points += 15
        rules.append(RuleEvaluation(
            name="RSI Overbought Condition",
            category="Momentum",
            condition=f"RSI 14 ({rsi:.1f}) > 70",
            status="BEARISH",
            weight=-15,
            explanation=f"RSI is overbought at {rsi:.1f}, indicating short-term exhaustion and elevated pullback risk.",
            passed=False,
        ))
    elif 50 <= rsi <= 65:
        bullish_points += 12
        rules.append(RuleEvaluation(
            name="RSI Healthy Bullish Zone",
            category="Momentum",
            condition=f"50 <= RSI 14 ({rsi:.1f}) <= 65",
            status="BULLISH",
            weight=12,
            explanation=f"RSI is {rsi:.1f}, indicating healthy positive momentum without being overstretched.",
            passed=True,
        ))
    elif 35 <= rsi < 50:
        bearish_points += 8
        rules.append(RuleEvaluation(
            name="RSI Sub-50 Weakness",
            category="Momentum",
            condition=f"35 <= RSI 14 ({rsi:.1f}) < 50",
            status="BEARISH",
            weight=-8,
            explanation=f"RSI is {rsi:.1f}, reflecting subdued demand and slight downward pressure.",
            passed=False,
        ))
    else:
        rules.append(RuleEvaluation(
            name="RSI Neutral Range",
            category="Momentum",
            condition=f"RSI 14 ({rsi:.1f})",
            status="NEUTRAL",
            weight=0,
            explanation=f"RSI is {rsi:.1f}, hovering near midpoint.",
            passed=True,
        ))

    # Rule 2.2: MACD Histogram & Crossover
    macd = last_row.get("MACD", 0.0)
    signal = last_row.get("MACD_Signal", 0.0)
    hist = last_row.get("MACD_Hist", 0.0)
    prev_hist = prev_row.get("MACD_Hist", 0.0)
    max_possible += 20

    if hist > 0 and hist > prev_hist:
        bullish_points += 20
        rules.append(RuleEvaluation(
            name="MACD Bullish Expansion",
            category="Momentum",
            condition=f"MACD Hist ({hist:.3f}) > 0 and expanding",
            status="BULLISH",
            weight=20,
            explanation="MACD line is above signal line with widening positive histogram bars, signaling accelerating upward momentum.",
            passed=True,
        ))
    elif hist > 0 and hist <= prev_hist:
        bullish_points += 10
        rules.append(RuleEvaluation(
            name="MACD Bullish Deceleration",
            category="Momentum",
            condition=f"MACD Hist ({hist:.3f}) > 0 but contracting",
            status="NEUTRAL",
            weight=10,
            explanation="MACD remains positive but upward momentum is slowing down.",
            passed=True,
        ))
    elif hist < 0 and hist < prev_hist:
        bearish_points += 20
        rules.append(RuleEvaluation(
            name="MACD Bearish Expansion",
            category="Momentum",
            condition=f"MACD Hist ({hist:.3f}) < 0 and expanding downward",
            status="BEARISH",
            weight=-20,
            explanation="MACD line is below signal line with negative histogram bars widening, signaling growing downside momentum.",
            passed=False,
        ))
    else:
        bearish_points += 10
        rules.append(RuleEvaluation(
            name="MACD Bearish Deceleration",
            category="Momentum",
            condition=f"MACD Hist ({hist:.3f}) < 0 but contracting upward",
            status="NEUTRAL",
            weight=-10,
            explanation="MACD is below signal line, but downside momentum is decelerating.",
            passed=True,
        ))

    # -------------------------------------------------------------
    # 3. VOLATILITY & BOLLINGER BANDS
    # -------------------------------------------------------------
    bb_upper = last_row.get("BB_Upper", price * 1.05)
    bb_lower = last_row.get("BB_Lower", price * 0.95)
    bb_bw = last_row.get("BB_Bandwidth", 10.0)
    max_possible += 10

    if price >= bb_upper:
        rules.append(RuleEvaluation(
            name="Bollinger Upper Band Test",
            category="Volatility",
            condition=f"Price ({price:.2f}) >= Upper Band ({bb_upper:.2f})",
            status="NEUTRAL",
            weight=0,
            explanation="Price is testing the upper envelope. Can indicate powerful trend breakout or short-term exhaustion.",
            passed=True,
        ))
    elif price <= bb_lower:
        bullish_points += 8
        rules.append(RuleEvaluation(
            name="Bollinger Lower Band Support",
            category="Volatility",
            condition=f"Price ({price:.2f}) <= Lower Band ({bb_lower:.2f})",
            status="BULLISH",
            weight=8,
            explanation="Price tagged the lower band; statistically oversold on a 2-sigma basis.",
            passed=True,
        ))
    else:
        rules.append(RuleEvaluation(
            name="Bollinger Band Position",
            category="Volatility",
            condition="Price within 2-sigma channels",
            status="NEUTRAL",
            weight=0,
            explanation="Price fluctuates within normal 2-sigma volatility boundaries.",
            passed=True,
        ))

    # -------------------------------------------------------------
    # 4. VOLUME CONFIRMATION
    # -------------------------------------------------------------
    volume = last_row.get("Volume", 0)
    vol_sma20 = last_row.get("Volume_SMA20", volume)
    max_possible += 15

    if vol_sma20 > 0:
        vol_ratio = volume / vol_sma20
        if vol_ratio >= 1.25 and price >= prev_row.get("Close", price):
            bullish_points += 15
            rules.append(RuleEvaluation(
                name="High Volume Bullish Confirmation",
                category="Volume",
                condition=f"Volume ({volume:,.0f}) >= 1.25x 20-Day Avg",
                status="BULLISH",
                weight=15,
                explanation=f"Trading volume is {vol_ratio:.2f}x the 20-day average on an up-move, confirming strong institutional interest.",
                passed=True,
            ))
        elif vol_ratio >= 1.25 and price < prev_row.get("Close", price):
            bearish_points += 15
            rules.append(RuleEvaluation(
                name="High Volume Bearish Selling",
                category="Volume",
                condition=f"Volume ({volume:,.0f}) >= 1.25x 20-Day Avg",
                status="BEARISH",
                weight=-15,
                explanation=f"Trading volume is {vol_ratio:.2f}x average on a down-move, indicating active distribution.",
                passed=False,
            ))
        else:
            rules.append(RuleEvaluation(
                name="Volume Activity",
                category="Volume",
                condition=f"Volume ({volume:,.0f}) near normal average",
                status="NEUTRAL",
                weight=0,
                explanation=f"Volume is {vol_ratio:.2f}x average, indicating typical participation.",
                passed=True,
            ))

    # -------------------------------------------------------------
    # SCORE SYNTHESIS
    # -------------------------------------------------------------
    net_score = bullish_points - bearish_points
    if max_possible > 0:
        norm_score = float((net_score / max_possible) * 100.0)
    else:
        norm_score = 0.0

    # Verdict assignment
    if norm_score >= 50:
        verdict = "Strong Bullish"
        signal_type = "POTENTIAL_BUY_WATCH"
    elif norm_score >= 18:
        verdict = "Moderate Bullish"
        signal_type = "POTENTIAL_BUY_WATCH"
    elif norm_score <= -50:
        verdict = "Strong Bearish"
        signal_type = "POTENTIAL_EXIT_WATCH"
    elif norm_score <= -18:
        verdict = "Moderate Bearish"
        signal_type = "POTENTIAL_EXIT_WATCH"
    else:
        verdict = "Neutral / Consolidation"
        signal_type = "NEUTRAL_HOLD"

    # Confidence score calculation
    # High confidence occurs when indicators align in one direction rather than contradict
    total_active_weight = bullish_points + bearish_points
    agreement = abs(bullish_points - bearish_points) / max(total_active_weight, 1.0)
    data_density_factor = min(len(df) / 100.0, 1.0)
    confidence = float(min(max(agreement * 90.0 * data_density_factor + 10.0, 15.0), 95.0))

    # Support / Resistance & Risk-Adjusted Trade Plan
    sr = find_support_resistance(df)
    atr = float(last_row.get("ATR_14", price * 0.02))

    stop_loss = max(round(price - 1.5 * atr, 2), 0.01)
    take_profit = round(price + 2.5 * atr, 2)
    risk_reward = round((take_profit - price) / max(price - stop_loss, 0.01), 2)

    trade_plan = {
        "current_price": round(price, 2),
        "suggested_entry": round(price, 2) if "BULLISH" in verdict.upper() else None,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward_ratio": risk_reward,
        "atr_14": round(atr, 2),
        "support": sr["support"],
        "resistance": sr["resistance"],
    }

    trend_summary = (
        f"Price is {'above' if price > sma50 else 'below'} SMA 50. "
        f"{'Golden cross active' if sma50 > sma200 else 'Moving averages bearishly stacked'}."
    )
    momentum_summary = f"RSI is {rsi:.1f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}). MACD histogram is {hist:+.3f}."
    volatility_summary = f"14-Day ATR is ${atr:.2f} ({((atr/price)*100):.1f}% of price). BB Bandwidth is {bb_bw:.1f}%."

    summary_text = (
        f"The AI Agent registers a **{verdict}** bias (Score: {norm_score:+.0f}/100, Confidence: {confidence:.0f}%). "
        f"{trend_summary} {momentum_summary} "
        f"Key nearby support is identified at ${sr['support']} and resistance at ${sr['resistance']}."
    )

    return AgentAnalysisReport(
        symbol=symbol,
        current_price=round(price, 2),
        overall_verdict=verdict,
        signal_type=signal_type,
        sentiment_score=round(norm_score, 1),
        confidence_score=round(confidence, 1),
        trend_summary=trend_summary,
        momentum_summary=momentum_summary,
        volatility_summary=volatility_summary,
        rules=rules,
        support_resistance=sr,
        trade_plan=trade_plan,
        executive_summary=summary_text,
    )
