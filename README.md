# 📈 AI Stock Market Analysis Agent

An institutional-grade, transparent **AI Stock Market Analysis Agent** built as an interactive Python Streamlit application. 

The platform combines technical indicator modeling, transparent rule-based AI reasoning, statistical risk assessment, and a historical strategy backtester. It is architected for zero-configuration resilience, featuring an automated **Demo Mode** with realistic historical datasets that operates without requiring API keys or internet connectivity.

---

## 🌟 Core Features

- **Dynamic Ticker Input & Auto-Profiles**:
  - Analyze any global ticker (e.g., `AAPL`, `MSFT`, `NVDA`, `TSLA`, `SPY`, `AMZN`, `GOOGL`).
  - Automatic lookup of 52-week price ranges, day highs/lows, and volume benchmarks.
- **Interactive Technical Candlestick Charts (Plotly)**:
  - Multi-panel interactive charting with customizable overlays.
  - Moving Averages: SMA 20, SMA 50, SMA 200, EMA 12, EMA 26.
  - Volatility Envelopes: Bollinger Bands (20 periods, 2 standard deviations).
  - Subplots for Volume (with 20-day moving average), RSI (14), and MACD (12, 26, 9) with histogram coloring.
- **Transparent Rule-Based AI Reasoning Agent**:
  - Synthesizes market signals across 4 dimensions: **Trend**, **Momentum**, **Volatility**, and **Institutional Volume**.
  - Generates clear verdict categories (*Strong Bullish, Moderate Bullish, Neutral / Consolidation, Moderate Bearish, Strong Bearish*).
  - **Rule Checklist Audit Table**: Every recommendation is explicitly explained with exact conditions tested, mathematical weights, and pass/fail status.
  - **Risk-Controlled Trade Plan**: Calculates dynamic support/resistance pivots, suggested entry triggers, volatility-based stop-loss ($1.5 \times \text{ATR}$), take-profit ($2.5 \times \text{ATR}$), and risk-to-reward ratios.
- **Institutional Risk & Volatility Assessment**:
  - Annualized Volatility ($\sigma \times \sqrt{252}$).
  - Maximum Drawdown (MDD %) and interactive underwater drawdown chart.
  - Historical Value at Risk (VaR 95% 1-day) and Expected Shortfall (CVaR).
  - Beta calculation against benchmark index ETFs (`SPY`, `QQQ`, `DIA`).
  - Sharpe and Sortino ratios.
- **Strategy Backtesting Engine**:
  - Historical bar-by-bar simulation of multiple rule-based strategies:
    1. *MACD Trend Following + SMA 50 Filter*
    2. *RSI Mean Reversion (Oversold Dip-Buy)*
    3. *Moving Average Crossover (SMA 20/50)*
    4. *Multi-Factor AI Agent Heuristic*
  - Accounts for stop-loss %, take-profit %, and configurable transaction fees/slippage.
  - Performance Metrics: Cumulative Return % vs Buy & Hold Benchmark %, Win Rate %, Profit Factor, Max Drawdown %, and Total Trades.
  - Cumulative Portfolio Equity Curve comparison chart.
  - Complete Trade Execution Log with CSV download capability.
- **Zero-Failure Demo Mode**:
  - Seamless fallback to bundled realistic Geometric Brownian Motion data if Yahoo Finance encounters network errors, timeouts, or rate limits.
  - Manual toggle to run 100% offline without network calls or API keys.

---

## 🏛️ Project Architecture

```
ai-stock-analysis-agent/
│
├── stock_agent/
│   ├── __init__.py               # Package metadata
│   ├── sample_data.py            # Offline market data generator & profiles
│   ├── data_provider.py          # Resilient yfinance ingestion + demo fallback
│   ├── indicators.py             # Vectorized technical indicator computations
│   ├── analysis_agent.py         # Transparent rule-based AI reasoning & audit engine
│   ├── risk.py                   # Risk metrics: Volatility, Drawdown, VaR, Sharpe, Beta
│   └── backtester.py             # Bar-by-bar backtester, trade simulation & equity curves
│
├── tests/
│   ├── __init__.py
│   ├── test_indicators.py        # Validates indicator mathematical correctness
│   ├── test_agent.py             # Validates AI agent verdicts & rule evaluations
│   ├── test_backtester.py        # Validates backtester P&L, fees, and metrics
│   └── test_data_provider.py     # Validates demo data schemas & risk outputs
│
├── app.py                        # Streamlit web application & user interface
├── requirements.txt              # Production dependency specifications
├── README.md                     # Documentation & usage manual
└── .gitignore                    # Git ignore file
```

---

## 🚀 Setup & Installation Instructions

### 1. Prerequisites
- Python 3.10+ (tested with Python 3.10, 3.11, 3.12, 3.15)
- Git (optional)

### 2. Create and Activate a Virtual Environment (Recommended)

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Test Suite
Ensure all mathematical calculations and backtesting rules pass:
```bash
pytest tests/ -v
```

### 5. Launch the Streamlit Application
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 💡 Using the Application

1. **Enter a Stock Ticker**: Enter any symbol (e.g., `AAPL`, `NVDA`, `MSFT`) in the sidebar or click a quick-select chip.
2. **Select Time Horizon**: Choose between 1 Month, 3 Months, 6 Months, 1 Year, 2 Years, or 5 Years.
3. **Explore the Tabs**:
   - **Technical Charts**: Pan and zoom the candlestick chart; toggle SMA, Bollinger Bands, RSI, and MACD overlays.
   - **AI Agent Analysis & Signals**: Review the synthesized verdict, confidence score, dynamic support/resistance, and audited rule checklist.
   - **Risk & Volatility**: Inspect annualized volatility, VaR (95%), Beta vs SPY, and underwater drawdown curves.
   - **Strategy Backtester**: Pick a strategy, tune Stop-Loss and Take-Profit sliders, click **Run Historical Backtest**, and analyze the resulting equity curve and trade log.
4. **Offline / Demo Mode**: Check the "Force Demo Mode (Offline)" box in the sidebar to run simulations without internet access or API dependencies.

---

## ⚖️ Financial & Legal Disclaimer

> [!IMPORTANT]
> **Strictly for Educational, Analytical, and Research Purposes**:
> - This software **never claims to predict stock prices or guarantee financial profits**.
> - Algorithmic and technical indicators are **backward-looking mathematical models** derived from historical data. Past performance—whether backtested or historical—is **no guarantee of future results**.
> - Financial markets carry inherent financial risks, including the potential loss of principal capital.
> - Nothing in this repository constitutes investment, financial, legal, or tax advice. Always conduct your own due diligence and consult with a licensed, certified financial advisor before committing real capital to trading.
