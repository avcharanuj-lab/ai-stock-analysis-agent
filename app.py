"""
AI Stock Market Analysis Agent
A professional, transparent technical analysis dashboard, rule-based AI reasoning agent,
risk assessment engine, and strategy backtester.
"""

from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from stock_agent.analysis_agent import analyze_stock
from stock_agent.backtester import run_backtest
from stock_agent.data_provider import get_stock_data
from stock_agent.indicators import add_all_indicators
from stock_agent.risk import calculate_risk_metrics

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & THEMING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Market Analysis Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished fintech aesthetic
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .badge-bullish {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-bearish {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #374151;
        color: #9ca3af;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #1e1b4b;
        border-left: 4px solid #6366f1;
        padding: 12px 18px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 20px;
        color: #e0e7ff;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.title("📈 Stock Agent Controls")

# Quick ticker selection
popular_tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "SPY", "AMZN", "GOOGL"]
default_ticker = "AAPL"

col_sym1, col_sym2 = st.sidebar.columns([3, 1])
with col_sym1:
    ticker_input = st.text_input("Enter Ticker Symbol", value=default_ticker).strip().upper()
with col_sym2:
    st.write("")
    st.write("")
    refresh_button = st.button("🔄", help="Refresh Data")

st.sidebar.caption("Quick Select:")
quick_cols = st.sidebar.columns(4)
for i, sym in enumerate(popular_tickers[:4]):
    if quick_cols[i].button(sym, use_container_width=True):
        ticker_input = sym
        st.rerun()

quick_cols_2 = st.sidebar.columns(3)
for i, sym in enumerate(popular_tickers[4:7]):
    if quick_cols_2[i].button(sym, use_container_width=True):
        ticker_input = sym
        st.rerun()

st.sidebar.divider()

# Time Period Selection
period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
}
selected_period_label = st.sidebar.selectbox("Historical Horizon", list(period_options.keys()), index=3)
selected_period = period_options[selected_period_label]

# Benchmark selection for Beta & comparison
benchmark_symbol = st.sidebar.selectbox("Market Benchmark", ["SPY", "QQQ", "DIA"], index=0)

# Offline / Demo Mode Toggle
st.sidebar.divider()
force_demo = st.sidebar.checkbox(
    "Force Demo Mode (Offline)",
    value=False,
    help="Runs entirely offline using realistic historical/synthetic market data without making live network requests.",
)

# Technical Indicators Visibility Toggles
st.sidebar.subheader("Chart Overlays")
show_sma20 = st.sidebar.checkbox("SMA 20 (Short-term)", value=True)
show_sma50 = st.sidebar.checkbox("SMA 50 (Medium-term)", value=True)
show_sma200 = st.sidebar.checkbox("SMA 200 (Long-term)", value=True)
show_bollinger = st.sidebar.checkbox("Bollinger Bands (20, 2σ)", value=True)
show_volume = st.sidebar.checkbox("Volume Subplot", value=True)
show_rsi = st.sidebar.checkbox("RSI (14) Subplot", value=True)
show_macd = st.sidebar.checkbox("MACD (12, 26, 9) Subplot", value=True)

st.sidebar.divider()
st.sidebar.info(
    "💡 **Notice**: This tool is an educational market analysis assistant. "
    "It never claims to predict future price changes or guarantee returns."
)

# -----------------------------------------------------------------------------
# 3. DATA FETCHING & PROCESSING
# -----------------------------------------------------------------------------
with st.spinner(f"Analyzing {ticker_input}..."):
    raw_df, info_dict, is_demo, status_msg = get_stock_data(
        symbol=ticker_input,
        period=selected_period,
        force_demo=force_demo,
    )

    # Fetch benchmark data for Beta calculation if needed
    bench_df, _, _, _ = get_stock_data(
        symbol=benchmark_symbol,
        period=selected_period,
        force_demo=force_demo,
    )

if raw_df.empty or len(raw_df) < 5:
    st.error(f"Could not load data for symbol '{ticker_input}'. Please verify the ticker or enable Demo Mode in the sidebar.")
    st.stop()

# Compute all technical indicators
df = add_all_indicators(raw_df)
latest_bar = df.iloc[-1]
prev_bar = df.iloc[-2] if len(df) >= 2 else latest_bar

curr_price = float(latest_bar["Close"])
price_diff = curr_price - float(prev_bar["Close"])
pct_diff = (price_diff / float(prev_bar["Close"])) * 100.0 if prev_bar["Close"] > 0 else 0.0

# -----------------------------------------------------------------------------
# 4. APP HEADER & METRIC SUMMARY CARDS
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    company_name = info_dict.get("name", ticker_input)
    st.title(f"{company_name} ({ticker_input})")
    currency = info_dict.get("currency", "USD")
    st.caption(f"Historical Price Analysis | Time Horizon: {selected_period_label} | Currency: {currency}")

with header_col2:
    if is_demo:
        st.warning(f"🟡 **Mode: Demo / Offline**\n\n{status_msg}", icon="⚠️")
    else:
        st.success(f"🟢 **Mode: Live Yahoo Finance**\n\n{status_msg}", icon="✅")

# Top KPI Metric Row
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    st.metric(
        label="Current Close",
        value=f"${curr_price:,.2f}",
        delta=f"{price_diff:+,.2f} ({pct_diff:+.2f}%)",
    )
with m_col2:
    day_high = float(latest_bar["High"])
    day_low = float(latest_bar["Low"])
    st.metric(label="Day Range", value=f"${day_low:,.2f} - ${day_high:,.2f}")
with m_col3:
    year_high = info_dict.get("year_high", float(df["High"].max()))
    year_low = info_dict.get("year_low", float(df["Low"].min()))
    st.metric(label="52-Week Range", value=f"${year_low:,.2f} - ${year_high:,.2f}")
with m_col4:
    volume = int(latest_bar["Volume"])
    vol_sma = int(latest_bar.get("Volume_SMA20", volume))
    st.metric(label="Volume / 20D Avg", value=f"{volume:,.0f}", delta=f"{((volume/vol_sma)-1)*100:+.1f}% vs Avg" if vol_sma > 0 else None)
with m_col5:
    rsi_val = float(latest_bar.get("RSI_14", 50.0))
    rsi_desc = "Oversold" if rsi_val < 30 else ("Overbought" if rsi_val > 70 else "Neutral")
    st.metric(label="RSI (14)", value=f"{rsi_val:.1f}", delta=rsi_desc, delta_color="off")

# -----------------------------------------------------------------------------
# 5. EXECUTE AI AGENT ANALYSIS & RISK COMPUTATIONS
# -----------------------------------------------------------------------------
agent_report = analyze_stock(df, symbol=ticker_input)

bench_series = bench_df["Close"] if not bench_df.empty else None
risk_metrics = calculate_risk_metrics(df["Close"], benchmark_series=bench_series)

# -----------------------------------------------------------------------------
# 6. TAB NAVIGATION
# -----------------------------------------------------------------------------
tab_charts, tab_agent, tab_risk, tab_backtest, tab_methodology = st.tabs([
    "📊 Technical Charts",
    "🤖 AI Agent Analysis & Signals",
    "🛡️ Risk & Volatility Assessment",
    "🧪 Strategy Backtester",
    "ℹ️ Methodology & Disclaimers",
])

# =============================================================================
# TAB 1: TECHNICAL CHARTS
# =============================================================================
with tab_charts:
    st.subheader("Interactive Price & Indicators Chart")

    # Determine subplot layout based on user toggles
    row_heights = [0.55]
    specs = [[{"secondary_y": False}]]
    num_rows = 1

    if show_volume:
        num_rows += 1
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])

    if show_rsi:
        num_rows += 1
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])

    if show_macd:
        num_rows += 1
        row_heights.append(0.15)
        specs.append([{"secondary_y": False}])

    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=specs,
    )

    # 1. Main Candlestick Chart (Row 1)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC Price",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    # Overlays: SMA 20, 50, 200
    if show_sma20:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_20"], line=dict(color="#38bdf8", width=1.5), name="SMA 20"),
            row=1,
            col=1,
        )
    if show_sma50:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_50"], line=dict(color="#f59e0b", width=1.5), name="SMA 50"),
            row=1,
            col=1,
        )
    if show_sma200:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_200"], line=dict(color="#a855f7", width=1.8, dash="dot"), name="SMA 200"),
            row=1,
            col=1,
        )

    # Bollinger Bands
    if show_bollinger:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Upper"], line=dict(color="rgba(148, 163, 184, 0.4)", width=1), name="Upper BB (2σ)", showlegend=False),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                line=dict(color="rgba(148, 163, 184, 0.4)", width=1),
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.05)",
                name="Bollinger Envelope",
            ),
            row=1,
            col=1,
        )

    current_row = 2

    # 2. Volume Subplot
    if show_volume:
        vol_colors = [
            "#22c55e" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef4444"
            for i in range(len(df))
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume", opacity=0.7),
            row=current_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Volume_SMA20"], line=dict(color="#fbbf24", width=1.2), name="20D Vol Avg"),
            row=current_row,
            col=1,
        )
        fig.update_yaxes(title_text="Volume", row=current_row, col=1)
        current_row += 1

    # 3. RSI Subplot
    if show_rsi:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI_14"], line=dict(color="#ec4899", width=1.8), name="RSI (14)"),
            row=current_row,
            col=1,
        )
        # Threshold lines 70 and 30
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(34, 197, 94, 0.6)", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
        current_row += 1

    # 4. MACD Subplot
    if show_macd:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#38bdf8", width=1.5), name="MACD"),
            row=current_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD_Signal"], line=dict(color="#f97316", width=1.5), name="Signal"),
            row=current_row,
            col=1,
        )
        hist_colors = [
            "#22c55e" if val >= 0 else "#ef4444"
            for val in df["MACD_Hist"]
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=hist_colors, name="Histogram"),
            row=current_row,
            col=1,
        )
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=750,
        margin=dict(l=40, r=40, t=20, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: AI AGENT ANALYSIS & SIGNALS
# =============================================================================
with tab_agent:
    st.subheader("Transparent AI Market Reasoning & Signal Audit")

    # High Level Agent Verdict Banner
    v_col1, v_col2, v_col3 = st.columns([2, 1, 1])

    with v_col1:
        verdict = agent_report.overall_verdict
        if "BULLISH" in verdict.upper():
            badge_html = f"<span class='badge-bullish'>🟢 {verdict.upper()}</span>"
        elif "BEARISH" in verdict.upper():
            badge_html = f"<span class='badge-bearish'>🔴 {verdict.upper()}</span>"
        else:
            badge_html = f"<span class='badge-neutral'>⚪ {verdict.upper()}</span>"

        st.markdown(
            f"""
            <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 18px 24px;">
                <div style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase;">Synthesized Agent Verdict</div>
                <div style="margin-top: 6px; margin-bottom: 10px;">{badge_html}</div>
                <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.5;">{agent_report.executive_summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with v_col2:
        st.metric(
            label="Agent Sentiment Score",
            value=f"{agent_report.sentiment_score:+.0f} / 100",
            delta="Bullish Bias" if agent_report.sentiment_score > 0 else ("Bearish Bias" if agent_report.sentiment_score < 0 else "Neutral"),
        )
        st.caption("Normalized composite score across trend, momentum, volatility, and volume indicators.")

    with v_col3:
        st.metric(
            label="Agent Confidence Rating",
            value=f"{agent_report.confidence_score:.0f}%",
            delta="Rule Alignment",
            delta_color="off",
        )
        st.caption("Reflects the degree of unanimity among active rules and historical data sample depth.")

    st.write("")

    # Trade Bracket & Key Levels (Entry, SL, TP)
    st.subheader("Risk-Controlled Potential Execution Brackets")
    plan = agent_report.trade_plan
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    with p_col1:
        entry_txt = f"${plan['suggested_entry']:.2f}" if plan.get("suggested_entry") else "Stand Aside / Watch"
        st.metric("Potential Trigger / Entry", entry_txt, help="Calculated based on current price if technical conditions align.")
    with p_col2:
        st.metric("Suggested Stop-Loss", f"${plan['stop_loss']:.2f}", delta=f"{((plan['stop_loss']/plan['current_price'])-1)*100:.1f}%", delta_color="inverse", help="Calculated via 1.5x 14-day ATR buffer to avoid market noise.")
    with p_col3:
        st.metric("Suggested Take-Profit", f"${plan['take_profit']:.2f}", delta=f"{((plan['take_profit']/plan['current_price'])-1)*100:+.1f}%", help="Calculated via 2.5x 14-day ATR expansion target.")
    with p_col4:
        st.metric("Target Risk / Reward", f"1 : {plan['risk_reward_ratio']:.2f}", help="Ratio of potential profit target against stop-loss downside.")

    st.write("")

    # Dynamic Support & Resistance Cards
    sr = agent_report.support_resistance
    st.markdown(
        f"""
        <div style="background-color: #1e293b; border-radius: 8px; padding: 12px 18px; margin-bottom: 20px;">
            <strong style="color: #38bdf8;">Key Technical Price Pivots:</strong> &nbsp;&nbsp;
            <span style="color: #94a3b8;">Support 1:</span> <b style="color: #22c55e;">${sr.get('support', 0):.2f}</b> &nbsp;|&nbsp; 
            <span style="color: #94a3b8;">Pivot Point:</span> <b>${sr.get('pivot', 0):.2f}</b> &nbsp;|&nbsp; 
            <span style="color: #94a3b8;">Resistance 1:</span> <b style="color: #ef4444;">${sr.get('resistance', 0):.2f}</b> &nbsp;|&nbsp; 
            <span style="color: #94a3b8;">14-Day ATR:</span> <b>${plan.get('atr_14', 0):.2f}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Transparent Rule-by-Rule Audit Checklist
    st.subheader("Audited Indicator Rule Breakdown")
    st.caption("Every signal output is generated from the following transparent, deterministic rule evaluations:")

    rule_rows = []
    for r in agent_report.rules:
        status_icon = "🟢 Passed" if r.passed and r.status == "BULLISH" else ("🔴 Active Bearish" if not r.passed or r.status == "BEARISH" else "⚪ Neutral")
        rule_rows.append({
            "Category": r.category,
            "Rule Name": r.name,
            "Condition Tested": r.condition,
            "Evaluation": status_icon,
            "Weight Impact": f"{r.weight:+0.0f} pts",
            "Agent Explanation": r.explanation,
        })

    rule_table_df = pd.DataFrame(rule_rows)
    st.dataframe(rule_table_df, use_container_width=True, hide_index=True)

    # Disclaimers
    st.markdown(
        f"""
        <div class="disclaimer-box">
            🛡️ <b>Transparent Rule Guarantee</b>: {agent_report.disclaimer}
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# TAB 3: RISK & VOLATILITY ASSESSMENT
# =============================================================================
with tab_risk:
    st.subheader("Risk & Volatility Profile")

    rk_col1, rk_col2, rk_col3, rk_col4 = st.columns(4)
    with rk_col1:
        st.metric(
            "Annualized Volatility",
            f"{risk_metrics['annualized_volatility']:.1f}%",
            help="Annualized standard deviation of daily logarithmic returns (scaled by sqrt(252)).",
        )
    with rk_col2:
        st.metric(
            "Maximum Drawdown (MDD)",
            f"{risk_metrics['max_drawdown']:.1f}%",
            delta=f"Current DD: {risk_metrics['current_drawdown']:.1f}%",
            delta_color="inverse",
            help="Largest peak-to-trough drop observed over the selected time window.",
        )
    with rk_col3:
        st.metric(
            "Daily VaR (95% Confidence)",
            f"-{risk_metrics['var_95_daily']:.2f}%",
            help="Maximum expected 1-day loss 95% of the time based on historical distribution.",
        )
    with rk_col4:
        st.metric(
            f"Beta vs {benchmark_symbol}",
            f"{risk_metrics['beta']:.2f}",
            help=f"Covariance relative to benchmark ETF ({benchmark_symbol}). Beta > 1.0 implies higher market sensitivity.",
        )

    # Secondary Risk KPIs
    rk_col5, rk_col6, rk_col7, rk_col8 = st.columns(4)
    with rk_col5:
        st.metric("Sharpe Ratio (Rf=4%)", f"{risk_metrics['sharpe_ratio']:.2f}")
    with rk_col6:
        st.metric("Sortino Ratio", f"{risk_metrics['sortino_ratio']:.2f}")
    with rk_col7:
        st.metric("Expected Shortfall (CVaR)", f"-{risk_metrics['cvar_95_daily']:.2f}%")
    with rk_col8:
        st.markdown(
            f"""
            <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px 14px; text-align: center;">
                <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Risk Categorization</div>
                <div style="color: {risk_metrics['risk_color']}; font-size: 1.15rem; font-weight: 700; margin-top: 4px;">
                    {risk_metrics['risk_rating']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Historical Underwater Drawdown Chart
    st.subheader("Historical Underwater Drawdown (%)")
    dd_series = risk_metrics.get("drawdown_series")
    if dd_series is not None and not dd_series.empty:
        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=dd_series.index,
                y=dd_series * 100.0,
                fill="tozeroy",
                line=dict(color="#ef4444", width=1.5),
                name="Drawdown %",
            )
        )
        fig_dd.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=40, r=40, t=10, b=30),
            yaxis_title="Decline from Peak (%)",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # Return Distribution Histogram
    st.subheader("Daily Returns Distribution")
    daily_rets = risk_metrics.get("daily_returns")
    if daily_rets is not None and not daily_rets.empty:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=daily_rets * 100.0,
                nbinsx=40,
                marker_color="#6366f1",
                opacity=0.8,
                name="Daily Returns %",
            )
        )
        # Add VaR cutoff line
        fig_hist.add_vline(
            x=-risk_metrics["var_95_daily"],
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"95% VaR (-{risk_metrics['var_95_daily']}%)",
        )
        fig_hist.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=40, r=40, t=10, b=30),
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# =============================================================================
# TAB 4: STRATEGY BACKTESTER
# =============================================================================
with tab_backtest:
    st.subheader("Rule-Based Strategy Backtesting Engine")
    st.caption("Simulate how deterministic rules would have performed historically on this asset.")

    bt_param_col1, bt_param_col2, bt_param_col3, bt_param_col4 = st.columns(4)

    with bt_param_col1:
        strategy_choice = st.selectbox(
            "Select Strategy",
            options=["MACD_TREND", "RSI_REVERSION", "SMA_CROSS", "AGENT_SCORE"],
            format_func=lambda x: {
                "MACD_TREND": "MACD Trend Following + SMA 50 Filter",
                "RSI_REVERSION": "RSI Mean Reversion (Oversold Dip-Buy)",
                "SMA_CROSS": "Moving Average Crossover (SMA 20/50)",
                "AGENT_SCORE": "Multi-Factor AI Agent Heuristic",
            }[x],
        )
    with bt_param_col2:
        initial_cap = st.number_input("Starting Capital ($)", value=10000, step=1000)
    with bt_param_col3:
        stop_loss_input = st.slider("Stop-Loss %", min_value=1.0, max_value=20.0, value=6.0, step=0.5) / 100.0
    with bt_param_col4:
        take_profit_input = st.slider("Take-Profit %", min_value=2.0, max_value=40.0, value=12.0, step=1.0) / 100.0

    run_bt_btn = st.button("🚀 Run Historical Backtest", type="primary")

    # Run backtest
    bt_res = run_backtest(
        df=df,
        strategy=strategy_choice,
        initial_capital=float(initial_cap),
        stop_loss_pct=stop_loss_input,
        take_profit_pct=take_profit_input,
        fee_pct=0.001,
    )

    # Performance Metrics
    bt_m1, bt_m2, bt_m3, bt_m4, bt_m5, bt_m6 = st.columns(6)
    with bt_m1:
        st.metric(
            "Strategy Return",
            f"{bt_res.total_return_pct:+.1f}%",
            delta=f"${bt_res.final_equity - bt_res.initial_capital:+,.0f}",
        )
    with bt_m2:
        st.metric(
            "Buy & Hold Return",
            f"{bt_res.benchmark_return_pct:+.1f}%",
            delta=f"Alpha: {bt_res.total_return_pct - bt_res.benchmark_return_pct:+.1f}%",
        )
    with bt_m3:
        st.metric("Win Rate", f"{bt_res.win_rate_pct:.1f}%", delta=f"{bt_res.winning_trades}W / {bt_res.losing_trades}L", delta_color="off")
    with bt_m4:
        st.metric("Profit Factor", f"{bt_res.profit_factor:.2f}", help="Gross Profits / Gross Losses. Above 1.5 indicates healthy strategy.")
    with bt_m5:
        st.metric("Strategy Max DD", f"{bt_res.max_drawdown_pct:.1f}%", delta_color="inverse")
    with bt_m6:
        st.metric("Total Trades", f"{bt_res.total_trades}", delta=f"Avg Hold: {bt_res.avg_holding_days:.0f}d", delta_color="off")

    st.write("")

    # Equity Curve Plotly Chart
    if not bt_res.equity_curve.empty:
        st.subheader("Cumulative Portfolio Equity Curve vs Buy & Hold")
        eq_fig = go.Figure()
        eq_fig.add_trace(
            go.Scatter(
                x=bt_res.equity_curve.index,
                y=bt_res.equity_curve["Portfolio_Value"],
                line=dict(color="#38bdf8", width=2.2),
                name=f"Strategy Equity (${bt_res.final_equity:,.0f})",
            )
        )
        eq_fig.add_trace(
            go.Scatter(
                x=bt_res.equity_curve.index,
                y=bt_res.equity_curve["Benchmark_Value"],
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
                name="Buy & Hold Benchmark",
            )
        )
        eq_fig.update_layout(
            template="plotly_dark",
            height=360,
            margin=dict(l=40, r=40, t=10, b=30),
            yaxis_title="Portfolio Value ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(eq_fig, use_container_width=True)

    # Detailed Trade Log
    if bt_res.trades:
        st.subheader("Trade Execution Log")
        trade_data = [
            {
                "Entry Date": t.entry_date.strftime("%Y-%m-%d"),
                "Exit Date": t.exit_date.strftime("%Y-%m-%d") if t.exit_date else "Open",
                "Entry Price": f"${t.entry_price:.2f}",
                "Exit Price": f"${t.exit_price:.2f}" if t.exit_price else "Open",
                "Shares": f"{t.shares:.2f}",
                "P&L ($)": f"${t.pnl:+,.2f}",
                "Return %": f"{t.return_pct:+.2f}%",
                "Holding Days": f"{t.holding_days} d",
                "Exit Trigger": t.exit_reason,
            }
            for t in bt_res.trades
        ]
        trade_df = pd.DataFrame(trade_data)
        st.dataframe(trade_df, use_container_width=True, hide_index=True)

        # CSV Download Button
        csv_trades = trade_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Trade Log as CSV",
            data=csv_trades,
            file_name=f"{ticker_input}_backtest_trades.csv",
            mime="text/csv",
        )
    else:
        st.info("No trades were triggered for this strategy across the selected time horizon.")

# =============================================================================
# TAB 5: METHODOLOGY & DISCLAIMERS
# =============================================================================
with tab_methodology:
    st.subheader("Mathematical Methodology & System Architecture")

    st.markdown(
        """
        ### 1. Technical Indicators Formulas
        - **Simple Moving Average (SMA)**:
          $$\\text{SMA}_k = \\frac{1}{k} \\sum_{i=0}^{k-1} P_{t-i}$$
        - **Exponential Moving Average (EMA)**:
          $$\\text{EMA}_t = P_t \\cdot \\alpha + \\text{EMA}_{t-1} \\cdot (1 - \\alpha), \\quad \\alpha = \\frac{2}{k+1}$$
        - **Relative Strength Index (RSI)**:
          $$\\text{RSI} = 100 - \\frac{100}{1 + \\text{RS}}, \\quad \\text{RS} = \\frac{\\text{Smoothed Gain}}{\\text{Smoothed Loss}}$$
        - **MACD (12, 26, 9)**:
          $$\\text{MACD Line} = \\text{EMA}_{12}(P) - \\text{EMA}_{26}(P)$$
          $$\\text{Signal Line} = \\text{EMA}_9(\\text{MACD Line})$$
          $$\\text{Histogram} = \\text{MACD Line} - \\text{Signal Line}$$
        - **Average True Range (ATR 14)**:
          $$\\text{TR} = \\max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$$
        - **Annualized Volatility ($\\sigma_{\\text{ann}}$)**:
          $$\\sigma_{\\text{ann}} = \\text{std}(R_t) \\times \\sqrt{252}$$
        - **Historical Value at Risk (VaR 95%)**:
          $$\\text{VaR}_{95} = -\\text{Percentile}(R, 5)$$

        ---

        ### 2. Multi-Factor AI Agent Reasoning
        The AI Agent synthesizes directional signals across four distinct market dimensions:
        1. **Trend Regime**: Evaluates price relation to the 20, 50, and 200-day moving averages and Golden/Death cross regimes.
        2. **Momentum & Divergence**: Evaluates Wilder's RSI (14) overbought/oversold boundaries and MACD histogram velocity.
        3. **Volatility Envelope**: Tracks 2-sigma Bollinger Band bandwidth expansion/contraction and ATR noise filters.
        4. **Institutional Volume Confirmation**: Compares current transaction volume against its 20-day moving average.

        ---

        ### 3. Strict Compliance & Financial Disclaimer
        > **LEGAL NOTICE**:
        > The AI Stock Market Analysis Agent is created strictly for **educational, analytical, and software demonstration purposes**.
        > - It **does not provide investment, financial, or tax advice**.
        > - It **never claims to predict future stock prices accurately or guarantee profits**.
        > - Financial markets are subject to significant volatility, unpredictable macro-economic shocks, and potential total loss of invested capital.
        > - Past backtested performance is strictly hypothetical and does not guarantee future results.
        > - Always perform independent research and consult licensed financial advisors before making investment decisions.
        """
    )

# -----------------------------------------------------------------------------
# 7. APP FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "AI Stock Market Analysis Agent v1.0.0 | Pure Python & Streamlit Architecture | "
    "Data provided via Yahoo Finance (where available) with deterministic offline fallback mode."
)
