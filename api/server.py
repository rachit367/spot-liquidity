"""
FastAPI backend for the ICT Trading Bot dashboard.

Start with:
    uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
or:
    python run_server.py
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backtesting import run_backtest, print_report
from backtesting.real_data import run_real_backtest
from brokers import get_broker
from config.settings import settings
from execution.paper_executor import PaperExecutor
from execution.risk_manager import RiskManager
from strategy import ICTStrategy

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title       = "ICT Trading Bot",
    description = "Real-time ICT strategy dashboard with backtesting",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR / "static")),
    name="static",
)


# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────

_executor = PaperExecutor(
    initial_balance = settings.initial_paper_balance,
    risk_pct        = settings.risk_per_trade_pct,
)
_equity_history: list[dict] = [
    {"t": datetime.now(timezone.utc).isoformat(), "v": settings.initial_paper_balance}
]
_last_signal: dict | None   = None
_last_backtest: dict | None = None
_ws_clients: list[WebSocket] = []


async def _broadcast(event: dict) -> None:
    """Send a JSON event to all connected WebSocket clients."""
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# REST API — status / balance / positions / trades
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    return {
        "mode":      settings.trading_mode,
        "broker":    settings.default_broker,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/balance")
async def api_balance():
    bal   = _executor.get_balance()
    init  = settings.initial_paper_balance
    return {
        "balance":  round(bal, 2),
        "initial":  round(init, 2),
        "pnl":      round(bal - init, 2),
        "pnl_pct":  round((bal - init) / init * 100, 2),
    }


@app.get("/api/equity")
async def api_equity():
    return {"history": _equity_history}


@app.get("/api/positions")
async def api_positions():
    return {"positions": _executor.get_positions()}


@app.get("/api/trades")
async def api_trades():
    return {"trades": _executor.get_trade_history()}


@app.get("/api/signal")
async def api_signal():
    return {"signal": _last_signal}


# ─────────────────────────────────────────────────────────────────────────────
# REST API — scan for signal
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/scan")
async def api_scan(
    broker_name: str = Query(default="mock"),
    symbol:      str = Query(default=""),
):
    global _last_signal

    if not symbol:
        symbol = "BTCUSD" if broker_name in ("delta", "mock") else "NSE_FO|NIFTY25MARFUT"

    try:
        broker = get_broker(broker_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    strategy = ICTStrategy(
        symbol      = symbol,
        interval    = "15m",
        swing_length = 5,
        ob_lookback  = 20,
        kill_zones   = ["london_open", "ny_open"] if broker_name == "delta" else [],
        rr_ratio     = 2.0,
    )

    signal = strategy.generate_signal(broker)
    if signal is None:
        _last_signal = None
        return {"signal": None, "message": "No ICT setup found on current data"}

    _last_signal = {
        "symbol":      signal.symbol,
        "direction":   signal.direction,
        "entry":       signal.entry,
        "stop_loss":   signal.stop_loss,
        "take_profit": signal.take_profit,
        "reason":      signal.reason,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    exec_result = _executor.execute(signal)

    # Update equity history
    bal = _executor.get_balance()
    _equity_history.append({"t": datetime.now(timezone.utc).isoformat(), "v": bal})

    await _broadcast({"event": "signal", "data": _last_signal})
    await _broadcast({"event": "balance", "data": {"balance": bal}})

    return {"signal": _last_signal, "execution": exec_result}


# ─────────────────────────────────────────────────────────────────────────────
# REST API — backtest (synthetic)
# ─────────────────────────────────────────────────────────────────────────────

class BacktestParams(BaseModel):
    trials:   int   = 500
    win_rate: float = 0.58
    rr:       float = 2.0
    risk:     float = 1.0
    seed:     int   = 42


@app.post("/api/backtest/synthetic")
async def api_backtest_synthetic(params: BacktestParams):
    global _last_backtest
    _last_backtest = run_backtest(
        n_trials  = params.trials,
        win_rate  = params.win_rate,
        rr_ratio  = params.rr,
        risk_pct  = params.risk,
        seed      = params.seed,
    )
    await _broadcast({"event": "backtest_done", "data": _last_backtest})
    return _last_backtest


# ─────────────────────────────────────────────────────────────────────────────
# REST API — backtest (real data)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/backtest/real")
async def api_backtest_real(
    broker_name: str   = Query(default="delta"),
    symbol:      str   = Query(default="BTCUSDT"),
    interval:    str   = Query(default="4h"),
    rr:          float = Query(default=2.0),
    risk:        float = Query(default=1.0),
):
    global _last_backtest
    result = run_real_backtest(
        broker_name     = broker_name,
        symbol          = symbol,
        interval        = interval,
        rr_ratio        = rr,
        risk_pct        = risk,
    )
    if "error" not in result:
        _last_backtest = result
        await _broadcast({"event": "backtest_done", "data": result})
    return result


@app.get("/api/backtest/last")
async def api_backtest_last():
    return {"results": _last_backtest}


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — live push updates
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        # Send initial snapshot
        bal = _executor.get_balance()
        init = settings.initial_paper_balance
        await ws.send_json({
            "event": "snapshot",
            "data": {
                "balance":  bal,
                "pnl":      round(bal - init, 2),
                "pnl_pct":  round((bal - init) / init * 100, 2),
                "signal":   _last_signal,
                "equity":   _equity_history[-20:],   # last 20 points
            },
        })
        while True:
            await asyncio.sleep(60)   # keep alive
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
