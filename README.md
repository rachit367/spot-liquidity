# ICT Trading Bot

A production-grade algorithmic trading bot implementing the **Inner Circle Trading (ICT)** methodology with **AI-powered signal intelligence** and **self-learning capability**. Supports paper and live trading on **Upstox** (Indian equities/F&O) and **Delta Exchange** (crypto derivatives), with a real-time web dashboard, walk-forward backtesting, and continuous model training.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Feature Overview](#feature-overview)
- [Setup & Configuration](#setup--configuration)
- [Dashboard Guide](#dashboard-guide)
  - [Top Bar](#top-bar)
  - [Stat Cards](#stat-cards)
  - [Equity Curve](#equity-curve)
  - [ICT Signal Panel](#ict-signal-panel)
  - [AI Signal Intelligence](#ai-signal-intelligence)
  - [Continuous Training](#continuous-training)
  - [Mistake Analysis](#mistake-analysis)
  - [Backtest Engine](#backtest-engine)
  - [Trade Log](#trade-log)
- [ICT Strategy Logic](#ict-strategy-logic)
- [ML & Self-Learning System](#ml--self-learning-system)
  - [32-Feature Pipeline](#32-feature-pipeline)
  - [AI Signal Gating](#ai-signal-gating)
  - [Training Data Flow](#training-data-flow)
  - [Dynamic Threshold](#dynamic-threshold)
  - [Drift Detection](#drift-detection)
  - [Ensemble Voting](#ensemble-voting)
- [Risk Management](#risk-management)
- [Command-Line Usage](#command-line-usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Broker Notes](#broker-notes)
- [Requirements](#requirements)

---

## Quick Start

```
Double-click  start.bat
Then open     http://localhost:8303
```

That's it. The script creates the venv, installs all dependencies, and launches the dashboard automatically.

---

## Feature Overview

| Feature | What It Does |
|---|---|
| **ICT Strategy** | Detects BOS/CHoCH, Liquidity Sweeps, Order Blocks, Fair Value Gaps — all 5 must align |
| **AI Signal Intelligence** | ML model scores every signal with a win/loss confidence % before execution |
| **Continuous Training** | Background loop fetches live data, simulates trades, and auto-retrains the model |
| **Ensemble Voting** | All classifiers (RF + XGBoost + LightGBM) vote — majority must agree |
| **Dynamic Threshold** | Auto-adjusts confidence threshold based on win/loss streaks (0.50–0.85) |
| **Drift Detection** | Tracks feature importances across retrains, alerts on market regime changes |
| **Mistake Analysis** | Identifies loss patterns by feature, time, direction, and streaks |
| **Backtesting** | Synthetic (parameterised) + real data (historical candles from broker) |
| **Trailing Stop-Loss** | Moves SL to breakeven after 1R profit, then trails at 0.5R behind best price |
| **Correlation Filter** | Blocks same-direction trades on correlated symbols (BTC+ETH, etc.) |
| **Paper Trading** | Full in-memory simulation with trailing stops & correlation checks |
| **Live Trading** | Real order placement with stop-loss via broker API |
| **Dual Brokers** | Upstox (NSE/F&O) + Delta Exchange (crypto perpetuals) |
| **Risk Management** | % risk per trade, daily loss limit, position sizing, correlation limits |
| **SQLite Backend** | Fast indexed trade storage, auto-migrates from CSV |
| **Unit Tests** | `pytest` suite covering strategy, features, dataset, and training |

---

## Setup & Configuration

### Installation

**1. Clone / download the project**

**2. Run the start script**
```bat
start.bat
```
This automatically:
- Creates `venv/` and activates it
- Installs all dependencies from `requirements.txt`
- Creates `.env` from `.env.example` if it doesn't exist
- Launches the dashboard at `http://localhost:8303`

**Linux/Mac alternative:**
```bash
source venv/bin/activate && python run_server.py
```

### Configuration (`.env`)

```env
# ── Trading mode ──────────────────────────────────────────────
TRADING_MODE=paper               # paper | live
DEFAULT_BROKER=upstox            # upstox | delta

# ── Risk management ───────────────────────────────────────────
RISK_PER_TRADE_PCT=1.0           # % of balance risked per trade
MAX_DAILY_LOSS_PCT=3.0           # circuit breaker — stops trading for the day
INITIAL_PAPER_BALANCE=100000.0   # starting balance for paper mode

# ── Upstox ────────────────────────────────────────────────────
UPSTOX_ENV=sandbox               # sandbox | live
UPSTOX_SANDBOX_API_KEY=
UPSTOX_SANDBOX_API_SECRET=
UPSTOX_SANDBOX_ACCESS_TOKEN=
UPSTOX_LIVE_API_KEY=
UPSTOX_LIVE_API_SECRET=
UPSTOX_LIVE_ACCESS_TOKEN=
UPSTOX_DATA_ACCESS_TOKEN=        # optional: separate read-only token

# ── Delta Exchange ────────────────────────────────────────────
DELTA_API_KEY=
DELTA_API_SECRET=
DELTA_BASE_URL=https://api.india.delta.exchange

# ── Self-Learning Training ────────────────────────────────────
TRAINING_DURATION_HOURS=2.0      # how long each training session runs
TRAINING_FETCH_INTERVAL_SEC=300  # seconds between OHLCV fetches
TRAINING_RETRAIN_EVERY=50        # retrain model every N simulated trades
TRAINING_SYMBOL=BTCUSD           # symbol to train on
TRAINING_INTERVAL=15m            # candle interval for training
TRAINING_FETCH_COUNT=500         # candles per fetch
```

> **Tip:** You can use the bot with Delta Exchange without any API keys for backtesting — historical candles are public. For paper trading, no keys are needed either.

---

## Dashboard Guide

The dashboard is the main interface for everything. Open `http://localhost:8303` after running `start.bat`.

### Top Bar

| Element | What It Does | How to Use |
|---|---|---|
| **PAPER / LIVE badge** | Shows current trading mode | Click to toggle between Paper and Live mode |
| **Broker dropdown** | Select which exchange to use | Choose "Delta Exchange" for crypto or "Upstox" for Indian markets |
| **Symbol input** | Instrument to trade | Type the symbol (e.g. `BTCUSD`, `ETHUSD`, `NSE_FO\|NIFTY25MARFUT`) |
| **▶ Scan Signal** | Run the ICT strategy on live data | Click to fetch candles from the broker and scan for an ICT setup |

**How to scan for a signal:**
1. Select your broker (Delta Exchange recommended for starting out)
2. Enter a symbol (e.g. `BTCUSD`)
3. Click **▶ Scan Signal**
4. The bot fetches live 15m candles, runs all 5 ICT checks, and if a signal is found, the ML model scores it

---

### Stat Cards

Five cards at the top show key metrics at a glance:

| Card | Source | What It Shows |
|---|---|---|
| **Balance** | Paper executor / broker API | Current account balance |
| **Total P&L** | Calculated from initial balance | Profit/loss in $ and % |
| **Open Positions** | Paper executor | Number of currently active trades |
| **Win Rate** | Last backtest result | Win percentage from the most recent backtest |
| **Max Drawdown** | Last backtest result | Maximum peak-to-trough equity decline |

---

### Equity Curve

A Chart.js line chart showing your paper trading balance over time. Updates after every trade execution. Useful for spotting drawdown periods visually.

---

### ICT Signal Panel

This panel shows the result of the last **▶ Scan Signal** press:

| Element | Description |
|---|---|
| **Direction badge** | 🟢 LONG or 🔴 SHORT (or grey "Scanning…" when idle) |
| **Entry / SL / TP** | Exact price levels for the trade |
| **ICT Checklist** | 5 confluence checks — each shows ✅ or ❌ |
| **AI Confidence bar** | ML model's win probability (only visible when a model is trained) |
| **Reason text** | Human-readable explanation of why the signal was generated |

**When a signal is found:**
- In **Paper mode**: The trade is automatically executed in the paper account
- In **Live mode**: The trade is placed as a real order on your broker, **but only if ML approves it** (confidence ≥ threshold)

---

### AI Signal Intelligence

This is the **ML model panel** — it adds an AI confidence layer on top of raw ICT signals.

| Element | What It Does |
|---|---|
| **Model** | Which classifier is currently active (random_forest / xgboost / lightgbm) |
| **Version** | Model version number (auto-incremented on each training) |
| **F1 Score** | Harmonic mean of precision & recall (higher = better) |
| **ROC-AUC** | Area under the ROC curve (0.5 = random, 1.0 = perfect) |
| **Trained On** | Number of samples the model was trained on |
| **Live Accuracy** | Real-time accuracy tracking of the model's predictions vs actual outcomes |
| **Confidence Threshold** | Slider (50%–90%) — signals below this confidence are blocked |
| **🔄 Train Model** | Button to trigger a one-shot retrain |

**How to train the model:**
1. Click **🔄 Train Model**
2. The bot generates synthetic trade data (ICT setups + simulated exits) and trains RF, XGBoost, and LightGBM
3. The best model (by F1 score) is promoted and saved
4. You'll see the metrics update (F1, AUC, etc.)

**How to adjust the confidence threshold:**
1. Drag the slider to your desired threshold
2. Click **Set**
3. Higher threshold = fewer trades but higher quality; lower = more trades but more risk

**Top Feature Importances** (side card): Shows which features influence the model's decisions most — e.g., "RSI 14" with 12% importance means RSI is the biggest factor.

---

### Continuous Training

This is the **self-learning engine** — a background loop that autonomously collects real market data, simulates trades, and improves the ML model over time.

| Setting | What It Controls | Recommended |
|---|---|---|
| **Duration** | How long the loop runs | Start with 30 min, increase to 2–8 hours |
| **Symbol** | What instrument to train on | `BTCUSD` for crypto |
| **Interval** | Candle timeframe | `15m` (matches scan interval) |
| **Fetch Every** | How often to pull fresh candles | 5 min (default) |
| **Retrain Every** | Auto-retrain after N simulated trades | 50 trades (default) |

**How to use it:**
1. Select your desired duration and settings
2. Click **▶ Start Training**
3. Watch the progress section update in real-time:
   - **Status**: Running / Idle
   - **Epoch**: How many data-fetch cycles have completed
   - **Windows Scanned**: Number of 100-candle windows checked
   - **Signals**: How many ICT setups were found
   - **Trades**: Number of simulated trade outcomes logged
   - **Retrains**: How many times the model was auto-retrained
   - **Model F1**: Latest model quality score
4. Click **■ Stop** at any time to end the loop early

**What happens behind the scenes:**
```
1. Fetch 500 OHLCV candles from Delta Exchange
2. Slide 100-candle windows across the data
3. Run ICT strategy on each window
4. Simulate entry/exit on forward bars
5. Extract 32 features + win/loss label
6. Append to training dataset
7. Auto-retrain model every 50 trades
8. Wait → fetch fresh data → repeat
```

---

### Mistake Analysis

The bot analyzes your losing trades to find patterns you can learn from.

| Analysis Type | What It Shows |
|---|---|
| **Feature analysis** | Which feature values correlate most with losses (e.g., "73% of losses when RSI > 0.7") |
| **Time analysis** | Which hours and days have the highest loss rates |
| **Direction analysis** | Long vs Short win rate comparison |
| **Streak analysis** | Longest losing streak statistics |

Click **↻ Refresh** to update the analysis. Requires at least 10 completed trades.

---

### Backtest Engine

Test the strategy on historical data before risking real money.

#### Synthetic Backtest (no API keys needed)

Generates artificial candlestick data with embedded ICT setups and simulates N trades.

| Parameter | What It Controls | Default |
|---|---|---|
| **Trials** | Number of independent test cases | 500 |
| **Win Rate %** | Simulated win probability | 58% |
| **R:R Ratio** | Minimum reward-to-risk ratio | 2.0 |
| **Risk % / trade** | Equity % risked per trade | 1.0% |

**How to use:**
1. Make sure the **Synthetic** tab is selected
2. Adjust parameters if desired
3. Click **▶ Run**
4. Results appear in the grid below: win rate, profit factor, max drawdown, Sharpe, etc.

#### Real Data Backtest

Fetches actual historical candles from your broker and walks the strategy through them.

| Parameter | What It Controls | Default |
|---|---|---|
| **Broker** | Which exchange to pull data from | Delta Exchange |
| **Symbol** | Instrument to backtest | BTCUSD |
| **Interval** | Candle timeframe | 4h |
| **R:R Ratio** | Minimum reward-to-risk | 2.0 |
| **Risk % / trade** | Equity % risked per trade | 1.0% |

**How to use:**
1. Switch to the **Real Data** tab
2. Select broker, symbol, and interval
3. Click **▶ Run**
4. The bot fetches ~500 historical candles, slides 100-bar windows through them, and reports real strategy performance

> **Note:** Delta Exchange historical data is public — no API key needed for backtesting.

#### Backtest Results Grid

| Metric | What It Means |
|---|---|
| **Win Rate** | % of trades that hit take-profit |
| **Profit Factor** | Gross profit ÷ gross loss (>1 = profitable) |
| **Expectancy** | Average P&L per trade in R multiples |
| **Max Drawdown** | Largest peak-to-trough equity decline |
| **Sharpe** | Risk-adjusted return (per-trade, not annualised) |
| **Recovery Factor** | Net profit ÷ max drawdown |
| **Signal Rate** | % of scanned windows that produced a valid ICT signal |
| **Break-even WR** | Minimum win rate needed to be profitable at this R:R |

---

### Trade Log

Shows the last 100 paper trades in a table format:

| Column | Description |
|---|---|
| **Order ID** | Unique trade identifier |
| **Dir** | LONG or SHORT |
| **Entry** | Entry price |
| **Exit** | Exit price (when trade closes) |
| **P&L** | Profit/loss in $ |
| **Qty** | Position size |
| **Result** | TP (win) / SL (loss) / timeout |

---

## ICT Strategy Logic

All **5 confluences must align** before a trade signal is generated:

```
1. Market Structure    BOS / CHoCH confirms trend direction (bullish or bearish)
                       Detected by comparing first vs last confirmed swing points

2. Liquidity Sweep     Price wicks through a prior swing high/low cluster
                       then closes back above/below it (stop hunt reversal)

3. Order Block         Last bearish candle before a bullish impulse (or vice versa)
                       Price must return to this origin zone after the sweep

4. Price in OB Zone    Current LTP must be inside the Order Block [low, high]

5. Fair Value Gap      3-candle imbalance: candle[i].high < candle[i+2].low
                       Confirms smart money involvement near the OB

  + R:R Gate           Signal is rejected unless reward ≥ 2× risk (default 2:1)
```

**Kill zones** (time-of-day filter):
- **Delta Exchange** (crypto): London Open (08:00–10:00 UTC) + NY Open (13:30–15:30 UTC)
- **Upstox** (Indian market): No time filter — trades any time during market hours

**Optional: Multi-Timeframe Confirmation**
- 15m signals → confirmed against 4h trend
- 1h signals → confirmed against 1d trend
- Prevents counter-trend entries that are more likely to fail

---

## ML & Self-Learning System

### 32-Feature Pipeline

Every trade signal is described by 32 numerical features extracted at entry time:

| Group | Features | What They Capture |
|---|---|---|
| **A — Technical** | RSI-14, EMA dist (20/50), ATR%, BB width, MACD hist | Momentum, volatility, trend |
| **B — Structure** | Trend slope, HH count, LL count, trend strength | Market structure quality |
| **C — ICT** | Sweep depth, OB width/ATR, price-in-OB, FVG size/ATR, kill zone | ICT setup geometry |
| **D — Volume** | Vol spike, vol trend, body ratio | Volume confirmation |
| **E — Time** | Hour sin/cos, day sin/cos | Cyclical time encoding |
| **F — Signal** | R:R ratio, risk % | Trade geometry quality |
| **G — Context** | Close vs OB mid, prev close change | Price context |
| **H — Order Flow** | VWAP dist, vol delta, relative vol, OBV slope, vol-price corr, vol concentration | Institutional activity |

### AI Signal Gating

Every trade signal passes through the ML model before execution:

1. **Feature extraction** — 32 features computed from OHLCV data + signal geometry
2. **Prediction** — model outputs win probability (0–100%)
3. **Confidence gate** — signal blocked if confidence < threshold (default 60%)
4. **Ensemble check** — all 3 classifiers vote; majority must agree (if ensemble is active)

In **Paper mode**, trades execute regardless of ML approval (the confidence is shown but doesn't block).  
In **Live mode**, trades are **blocked** if ML confidence is below the threshold.

### Training Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS TRAINING LOOP                      │
│                                                                 │
│  1. Fetch 500 OHLCV candles from Delta Exchange                 │
│  2. Slide 100-candle windows across the data                    │
│  3. Run ICT strategy on each window                             │
│  4. Simulate entry/exit on 50 forward candles                   │
│  5. Extract 32 features + win/loss label                        │
│  6. Append to training dataset (SQLite + CSV)                   │
│  7. Auto-retrain model every 50 trades                          │
│  8. Wait → fetch fresh data → repeat for hours                  │
│                                                                 │
│  Result: Model continuously improves from new market data       │
└─────────────────────────────────────────────────────────────────┘
```

**Cold-start:** When you have zero live trades, the bot generates synthetic data to bootstrap the model. As real trade data accumulates, it gradually replaces synthetic data.

### Dynamic Threshold

The confidence threshold auto-adjusts based on recent performance:
- **Losing streak** → threshold increases (becomes more selective)
- **Winning streak** → threshold decreases (takes more trades)
- Range: 0.50 – 0.85
- Can be toggled on/off via the dashboard or API

### Drift Detection

After each retrain, feature importances are recorded. If the top features shift significantly between model versions, it indicates a **market regime change** — the model is now relying on different signals. This is surfaced via the `/api/ml/drift` endpoint.

### Ensemble Voting

All three classifiers are trained simultaneously:
- **Random Forest** (200 trees, max_depth=6)
- **XGBoost** (200 trees, max_depth=4, lr=0.05)
- **LightGBM** (200 trees, max_depth=5, lr=0.05)

The best individual model (by F1) is used as the primary. Optionally, all three vote and the majority decision is used.

---

## Risk Management

| Control | How It Works | Default |
|---|---|---|
| **Position sizing** | `qty = (balance × risk%) / \|entry − SL\|` | 1% risk per trade |
| **Daily loss limit** | Trading halts for the day if cumulative losses exceed the limit | 3% of balance |
| **Trailing stop-loss** | After 1R profit, SL moves to breakeven; then trails at 0.5R behind best price | Active in paper mode |
| **Correlation filter** | Blocks same-direction trades on correlated pairs (BTC+ETH, etc.) | Active in paper mode |
| **ML confidence gate** | Blocks trades below the confidence threshold | Active in live mode |
| **R:R gate** | Rejects signals where reward < 2× risk | Always active |

---

## Command-Line Usage

### Single scan (CLI bot)
```bat
venv\Scripts\python.exe main.py --broker mock --mode paper
venv\Scripts\python.exe main.py --broker upstox --mode paper
venv\Scripts\python.exe main.py --broker delta --mode live
```

### Backtesting (CLI)
```bat
# Synthetic — 500 trials, 58% simulated win rate
venv\Scripts\python.exe backtest.py

# Sensitivity table across win rates
venv\Scripts\python.exe backtest.py --scenarios

# Real data — Delta Exchange (no API key needed)
venv\Scripts\python.exe backtest.py --broker delta --symbol BTCUSD --interval 4h
venv\Scripts\python.exe backtest.py --broker delta --symbol BTCUSD --interval 1h --step 5

# Real data — Upstox (requires live access token)
venv\Scripts\python.exe backtest.py --broker upstox --symbol "NSE_INDEX|Nifty 50" --interval 1d
```

---

## API Reference

### Core

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Bot status — mode (paper/live), broker, timestamp |
| `/api/balance` | GET | Account balance, P&L, initial balance |
| `/api/equity` | GET | Equity curve history (timestamps + values) |
| `/api/positions` | GET | Open positions list |
| `/api/trades` | GET | Trade history (last 100) |
| `/api/signal` | GET | Last scanned signal details |

### Actions

| Endpoint | Method | Params | Description |
|---|---|---|---|
| `/api/mode` | POST | `mode` (paper/live) | Switch trading mode |
| `/api/broker` | POST | `name` (upstox/delta) | Switch active broker |
| `/api/scan` | POST | `broker_name`, `symbol` | Run ICT scan + ML scoring + execute |

### Backtesting

| Endpoint | Method | Params | Description |
|---|---|---|---|
| `/api/backtest/synthetic` | POST | JSON body: trials, win_rate, rr, risk, seed | Run synthetic backtest |
| `/api/backtest/real` | POST | `broker_name`, `symbol`, `interval`, `rr`, `risk` | Run real-data backtest |
| `/api/backtest/last` | GET | — | Get last backtest results |

### ML & Training

| Endpoint | Method | Params | Description |
|---|---|---|---|
| `/api/ml/status` | GET | — | Model info: version, metrics, accuracy |
| `/api/ml/train` | POST | `n_synthetic` (default 500) | Force model retrain |
| `/api/ml/threshold` | POST | `value` (0.50–0.95) | Set confidence threshold |
| `/api/ml/versions` | GET | — | List all saved model versions |
| `/api/ml/outcome` | POST | `trade_id`, `result` (0/1), `pnl` | Record trade outcome for learning |
| `/api/ml/mistakes` | GET | `min_trades` (default 10) | Mistake analysis report |
| `/api/ml/drift` | GET | — | Feature importance drift report |
| `/api/ml/threshold/dynamic` | GET | — | Dynamic threshold status |
| `/api/ml/threshold/dynamic` | POST | `enabled`, `base` | Toggle dynamic threshold |
| `/api/training/start` | POST | `duration_hours`, `symbol`, `interval`, `fetch_interval_sec`, `retrain_every`, `broker_name` | Start continuous training loop |
| `/api/training/stop` | POST | — | Stop training loop |
| `/api/training/status` | GET | — | Training progress stats |

### Risk

| Endpoint | Method | Description |
|---|---|---|
| `/api/risk/correlation` | GET | Correlation group exposure |

### WebSocket

| Endpoint | Events |
|---|---|
| `/ws` | `snapshot`, `signal`, `balance`, `mode_changed`, `backtest_done`, `training_progress`, `training_complete` |

---

## Project Structure

```
spot-liquidity/
│
├── start.bat                  ← Double-click to start everything
├── run_server.py              ← Dashboard server launcher
├── main.py                    ← CLI bot (single scan)
├── backtest.py                ← Backtest CLI
├── requirements.txt
├── .env                       ← Your API keys (git-ignored)
├── .env.example               ← Template
│
├── frontend/
│   ├── index.html             ← Dashboard (served by FastAPI)
│   └── static/
│       ├── app.js             ← Dashboard logic + Chart.js
│       └── style.css          ← Dark trading theme
│
├── api/
│   └── server.py              ← FastAPI backend (REST + WebSocket)
│
├── strategy/
│   ├── __init__.py            ← Full ICT strategy (+ optional HTF)
│   └── multi_timeframe.py     ← Higher-TF trend confirmation
│
├── ml/
│   ├── __init__.py            ← ML module public API
│   ├── feature_engineering.py ← 32-feature extraction pipeline
│   ├── train_model.py         ← RandomForest + XGBoost + LightGBM
│   ├── inference.py           ← Signal scoring + dynamic threshold
│   ├── model.py               ← Model versioning, save/load
│   ├── dataset_builder.py     ← Build datasets (SQLite + CSV)
│   ├── database.py            ← SQLite backend (WAL mode)
│   ├── retrain_pipeline.py    ← Auto-retrain + drift recording
│   ├── training_loop.py       ← Multi-symbol self-learning engine
│   ├── mistake_analyzer.py    ← Loss pattern analysis
│   ├── ensemble.py            ← Multi-model ensemble voting
│   ├── dynamic_threshold.py   ← Auto-adjusting confidence threshold
│   └── drift_detector.py      ← Feature importance drift detection
│
├── brokers/
│   ├── base.py                ← Abstract broker + retry decorator
│   ├── upstox.py              ← Upstox V2 (sandbox + live modes)
│   ├── delta.py               ← Delta Exchange (HMAC-SHA256 auth)
│   └── mock.py                ← Synthetic broker for testing
│
├── execution/
│   ├── paper_executor.py      ← Paper trading (trailing SL + correlation check)
│   ├── live_executor.py       ← Real order placement
│   ├── risk_manager.py        ← Position sizing + daily loss limit
│   ├── trailing_stop.py       ← Dynamic trailing stop-loss manager
│   └── correlation_filter.py  ← Blocks correlated same-direction trades
│
├── backtesting/
│   ├── __init__.py            ← Synthetic walk-forward engine
│   └── real_data.py           ← Real historical data backtest
│
├── tests/
│   ├── conftest.py            ← Shared test fixtures
│   ├── test_strategy.py       ← ICT strategy tests
│   ├── test_feature_engineering.py
│   ├── test_dataset_builder.py
│   └── test_train_model.py    ← ML training tests
│
├── config/
│   └── settings.py            ← Pydantic settings (loads from .env)
│
├── models/                    ← Trained ML model versions (auto-created)
│
└── logs/
    ├── bot.log                ← Rotating application log
    ├── trades.jsonl           ← Structured trade log (JSONL)
    ├── trades_ml.csv          ← ML training dataset (CSV fallback)
    ├── trades.db              ← SQLite trade database (primary)
    └── drift_log.json         ← Feature importance history
```

---

## Broker Notes

### Upstox
- **Sandbox mode** (`UPSTOX_ENV=sandbox`): Paper API for order testing. No market data available in sandbox — use `--broker mock` for full strategy testing.
- **Live mode** (`UPSTOX_ENV=live`): Real orders on NSE/BSE. Requires valid live credentials.
- **Data token**: Set `UPSTOX_DATA_ACCESS_TOKEN` for a separate analytics/read-only token. Falls back to the main token if not set.

### Delta Exchange
- Historical candles are **public** (no API key needed for backtesting or training).
- Symbol format: `BTCUSD`, `ETHUSD`, etc.
- Live trading requires `DELTA_API_KEY` and `DELTA_API_SECRET`.

---

## Requirements

- Python 3.11+
- Windows (for `start.bat`) — on Linux/Mac run: `source venv/bin/activate && python run_server.py`
- Dependencies: FastAPI, uvicorn, scikit-learn, xgboost, lightgbm, pandas, numpy, Chart.js (CDN)
