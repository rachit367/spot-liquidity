# ICT Trading Bot

A production-grade algorithmic trading bot implementing the **Inner Circle Trading (ICT)** methodology. Supports paper and live trading on **Upstox** (Indian equities/F&O) and **Delta Exchange** (crypto derivatives), with a real-time web dashboard and walk-forward backtesting engine.

---

## Quick Start

```
Double-click  start.bat
Then open     http://localhost:8001
```

That's it. The script creates the venv, installs all dependencies, and launches the dashboard automatically.

---

## Features

| Feature | Details |
|---|---|
| **ICT Strategy** | BOS/CHoCH · Liquidity Sweeps · Order Blocks · Fair Value Gaps |
| **Dual brokers** | Upstox (NSE/F&O) + Delta Exchange (crypto perps) |
| **Paper trading** | Full in-memory simulation with P&L tracking |
| **Live trading** | Real order placement with stop-loss via broker API |
| **Backtesting** | Synthetic (parameterised) + real historical data walk-forward |
| **Dashboard** | FastAPI + Chart.js — balance, equity curve, signals, trade log |
| **Risk management** | % risk per trade, daily loss limit, position sizing |
| **Upstox modes** | Sandbox (paper API) and Live with separate credentials |

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
- Opens the dashboard at `http://localhost:8001`

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

# Sandbox (for paper API testing — no real orders)
UPSTOX_SANDBOX_API_KEY=
UPSTOX_SANDBOX_API_SECRET=
UPSTOX_SANDBOX_ACCESS_TOKEN=

# Live (for real order placement)
UPSTOX_LIVE_API_KEY=
UPSTOX_LIVE_API_SECRET=
UPSTOX_LIVE_ACCESS_TOKEN=

# Optional: separate read-only token for market data / analytics
UPSTOX_DATA_ACCESS_TOKEN=

# ── Delta Exchange ────────────────────────────────────────────
DELTA_API_KEY=
DELTA_API_SECRET=
DELTA_BASE_URL=https://api.india.delta.exchange
```

---

## Usage

### Dashboard (recommended)
```bat
start.bat                          # opens http://localhost:8001
```
- **Scan Signal** button — runs the ICT strategy on live data
- **Backtest → Synthetic** — parameterised simulation (fast, no API needed)
- **Backtest → Real Data** — walk-forward on actual historical candles

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

## Backtest Results (synthetic, 500 trials)

| Metric | Value |
|---|---|
| Signal rate | 66% of windows |
| Win rate | 43.6% |
| Break-even win rate | 33.3% |
| Profit factor | 1.47 |
| Expectancy | +0.31R per trade |
| Max drawdown | 9.67% |
| Net return | +164.78% |

The 2:1 R:R means you only need to win 1 in 3 trades to break even. The strategy's selectivity (all 5 ICT conditions required) protects the edge.

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
│   └── __init__.py            ← Full ICT strategy implementation
│
├── brokers/
│   ├── base.py                ← Abstract broker + retry decorator
│   ├── upstox.py              ← Upstox V2 (sandbox + live modes)
│   ├── delta.py               ← Delta Exchange (HMAC-SHA256 auth)
│   └── mock.py                ← Synthetic broker for testing
│
├── execution/
│   ├── paper_executor.py      ← In-memory paper trading
│   ├── live_executor.py       ← Real order placement
│   └── risk_manager.py        ← Position sizing + daily loss limit
│
├── backtesting/
│   ├── __init__.py            ← Synthetic walk-forward engine
│   └── real_data.py           ← Real historical data backtest
│
├── config/
│   └── settings.py            ← Pydantic settings (loads from .env)
│
└── logs/
    ├── bot.log                ← Rotating application log
    └── trades.jsonl           ← Structured trade log (JSONL)
```

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
