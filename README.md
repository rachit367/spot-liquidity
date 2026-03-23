# ICT Trading Bot

A production-grade algorithmic trading bot implementing the **Inner Circle Trading (ICT)** methodology with **AI-powered signal intelligence** and **self-learning capability**. Supports paper and live trading on **Upstox** (Indian equities/F&O) and **Delta Exchange** (crypto derivatives), with a real-time web dashboard, walk-forward backtesting, and continuous model training.

---

## Quick Start

```
Double-click  start.bat
Then open     http://localhost:8303
```

That's it. The script creates the venv, installs all dependencies, and launches the dashboard automatically.

---

## Features

| Feature | Details |
|---|---|
| **ICT Strategy** | BOS/CHoCH · Liquidity Sweeps · Order Blocks · Fair Value Gaps |
| **Multi-Timeframe** | Optional higher-TF confirmation (15m→4h, 1h→1d) prevents counter-trend entries |
| **Ensemble Voting** | All classifiers (RF + XGBoost + LightGBM) vote — majority must agree |
| **Self-Learning Training** | Multi-symbol continuous OHLCV fetch → strategy simulation → auto-retrain |
| **Dynamic Threshold** | Auto-adjusts confidence threshold based on win/loss streaks (0.50–0.85) |
| **Drift Detection** | Tracks feature importances across retrains, alerts on market regime changes |
| **Trailing Stop-Loss** | Moves SL to breakeven after 1R profit, then trails at 0.5R behind best price |
| **Correlation Filter** | Blocks same-direction trades on correlated symbols (BTC+ETH, etc.) |
| **Mistake Analysis** | Identifies loss patterns by feature, time, direction, and streaks |
| **SQLite Backend** | Fast indexed trade storage, auto-migrates from CSV |
| **Dual Brokers** | Upstox (NSE/F&O) + Delta Exchange (crypto perps) |
| **Paper Trading** | Full in-memory simulation with trailing stops & correlation checks |
| **Live Trading** | Real order placement with stop-loss via broker API |
| **Backtesting** | Synthetic + real data walk-forward with `TimeSeriesSplit` validation |
| **Dashboard** | FastAPI + Chart.js — balance, equity curve, signals, trade log, training panel |
| **Unit Tests** | `pytest` test suite covering strategy, features, dataset, and training |
| **Risk Management** | % risk per trade, daily loss limit, position sizing, correlation limits |

---

## Setup

**1. Clone / download the project**

**2. Run the start script**
```bat
start.bat
```
This automatically:
- Creates `venv/`
- Installs all dependencies from `requirements.txt`
- Creates `.env` from `.env.example` if it doesn't exist
- Opens the dashboard at `http://localhost:8303`

**3. Configure your API keys in `.env`**

---

## Configuration (`.env`)

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

---

## Usage

### Dashboard (recommended)
```bat
start.bat                          # opens http://localhost:8303
```
- **Scan Signal** — runs the ICT strategy on live data, ML scores the signal
- **Backtest → Synthetic** — parameterised simulation (fast, no API needed)
- **Backtest → Real Data** — walk-forward on actual historical candles
- **Train Model** — one-shot ML training on accumulated trade data
- **Continuous Training** — start a learning loop that fetches data for hours and auto-retrains
- **Mistake Analysis** — see what features/times/directions cause losses

### Command-line bot (single scan)
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

## ML & Self-Learning System

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS TRAINING LOOP                      │
│                                                                 │
│  1. Fetch 500 OHLCV candles from Delta Exchange                 │
│  2. Slide 100-candle windows across the data                    │
│  3. Run ICT strategy on each window                             │
│  4. Simulate entry/exit on forward bars                         │
│  5. Extract 26 features + win/loss label                        │
│  6. Append to training dataset (trades_ml.csv)                  │
│  7. Auto-retrain model every 50 trades                          │
│  8. Wait → fetch fresh data → repeat for hours                  │
│                                                                 │
│  Result: Model continuously improves from new market data       │
└─────────────────────────────────────────────────────────────────┘
```

### AI Signal Gating

Every trade signal passes through the ML model before execution:

1. **Feature extraction** — 26 features (RSI, ATR, market structure, volume, time, OB quality, etc.)
2. **Prediction** — model outputs win probability (0–100%)
3. **Confidence gate** — signal blocked if confidence < threshold (default 60%)
4. **Adjusted R:R** — model suggests optimal risk/reward ratio

### Mistake Analysis

The bot analyzes losing trades to find patterns:
- Which **features** correlate most with losses (e.g. "73% of losses when RSI > 0.7")
- Which **hours** and **days** have the highest loss rates
- **Long vs Short** win rate comparison
- Losing **streak** statistics

---

## ICT Strategy Logic

All **5 confluences must align** before a trade signal is generated:

```
1. Market Structure   BOS / CHoCH confirms trend direction (bullish or bearish)
                      Detected by comparing first vs last confirmed swing points

2. Liquidity Sweep    Price wicks through a prior swing high/low cluster
                      then closes back above/below it (stop hunt reversal)

3. Order Block        Last bearish candle before a bullish impulse (or vice versa)
                      Price must return to this origin zone after the sweep

4. Price in OB Zone   Current LTP must be inside the Order Block [low, high]

5. Fair Value Gap     3-candle imbalance: candle[i].high < candle[i+2].low
                      Confirms smart money involvement near the OB

  + R:R Gate          Signal is rejected unless reward ≥ 2× risk (default 2:1)
```

**Kill zones** (time filter):
- Delta Exchange (crypto): London Open (08:00–10:00 UTC) + NY Open (13:30–15:30 UTC)
- Upstox / Indian market: **no time filter** — trades any time during market hours

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
│   ├── feature_engineering.py ← 26-feature extraction pipeline
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

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Bot status (mode, broker) |
| `/api/balance` | GET | Account balance |
| `/api/scan` | POST | Run ICT scan + ML scoring |
| `/api/backtest/synthetic` | POST | Run synthetic backtest |
| `/api/backtest/real` | POST | Run real-data backtest |
| `/api/ml/status` | GET | ML model status |
| `/api/ml/train` | POST | Force model retrain |
| `/api/ml/mistakes` | GET | Mistake analysis report |
| `/api/ml/drift` | GET | Feature importance drift report |
| `/api/ml/threshold/dynamic` | GET | Dynamic threshold status |
| `/api/ml/threshold/dynamic` | POST | Toggle dynamic threshold (enabled, base) |
| `/api/training/start` | POST | Start multi-symbol training loop |
| `/api/training/stop` | POST | Stop training loop |
| `/api/training/status` | GET | Training progress + current symbol |
| `/api/risk/correlation` | GET | Correlation group exposure |
| `/ws` | WebSocket | Real-time updates |

---

## Broker Notes

### Upstox
- **Sandbox mode** (`UPSTOX_ENV=sandbox`): Paper API for order testing. No market data available in sandbox — use `--broker mock` for full strategy testing.
- **Live mode** (`UPSTOX_ENV=live`): Real orders on NSE/BSE. Requires valid live credentials.
- **Data token**: Set `UPSTOX_DATA_ACCESS_TOKEN` if you have a separate analytics/read-only token. Falls back to the main token if not set.

### Delta Exchange
- Historical candles are **public** (no API key needed for backtesting).
- Symbol format: `BTCUSD`, `ETHUSD`, etc.
- Live trading requires `DELTA_API_KEY` and `DELTA_API_SECRET`.

---

## Requirements

- Python 3.11+
- Windows (for `start.bat`) — on Linux/Mac run: `source venv/bin/activate && python run_server.py`
