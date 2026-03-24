"""
FastAPI backend for the ICT Trading Bot dashboard.

Start with:
    uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
or:
    python run_server.py
"""

from __future__ import annotations

import asyncio
import json
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
from execution.live_executor import LiveExecutor
from execution.paper_executor import PaperExecutor
from execution.risk_manager import RiskManager
from strategy import ICTStrategy
import ml

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
LOGS_DIR     = Path(__file__).resolve().parent.parent / "logs"

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

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

_trading_mode:       str = settings.trading_mode    # "paper" | "live"
_active_broker_name: str = settings.default_broker  # "upstox" | "delta"

_paper_executor = PaperExecutor(
    initial_balance = settings.initial_paper_balance,
    risk_pct        = settings.risk_per_trade_pct,
)
_risk_mgr = RiskManager(
    max_daily_loss_pct = settings.max_daily_loss_pct,
    daily_reset_hour   = settings.daily_reset_hour,
)
_equity_history: list[dict] = [
    {"t": datetime.now(timezone.utc).isoformat(), "v": settings.initial_paper_balance}
]
_last_signal:   dict | None = None
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


def _default_symbol(broker_name: str) -> str:
    return "NSE_FO|NIFTY25MARFUT" if broker_name == "upstox" else "BTCUSD"


def _read_live_trades(limit: int = 100) -> list[dict]:
    """Read last ``limit`` placed-order entries from trades.jsonl."""
    path = LOGS_DIR / "trades.jsonl"
    if not path.exists():
        return []
    trades = []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        for line in reversed(lines[-limit * 2:]):
            try:
                entry = json.loads(line)
                if entry.get("event") == "order_placed":
                    trades.append({
                        "order_id":   entry.get("entry_order_id", "—"),
                        "direction":  entry.get("direction"),
                        "entry":      entry.get("entry_price"),
                        "exit_price": None,
                        "pnl":        None,
                        "quantity":   entry.get("quantity"),
                        "outcome":    entry.get("status", "placed"),
                        "symbol":     entry.get("symbol"),
                        "timestamp":  entry.get("timestamp"),
                    })
                    if len(trades) >= limit:
                        break
            except Exception:
                pass
    except Exception:
        pass
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# REST API — mode + broker control
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/mode")
async def api_set_mode(mode: str = Query(...)):
    global _trading_mode
    if mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")
    _trading_mode = mode
    logger.info("Trading mode switched to: %s", mode)
    await _broadcast({"event": "mode_changed", "data": {"mode": mode}})
    return {"mode": _trading_mode}


@app.post("/api/broker")
async def api_set_broker(name: str = Query(...)):
    global _active_broker_name
    if name not in ("upstox", "delta"):
        raise HTTPException(status_code=400, detail="broker must be 'upstox' or 'delta'")
    _active_broker_name = name
    return {"broker": _active_broker_name}


# ─────────────────────────────────────────────────────────────────────────────
# REST API — status / balance / positions / trades
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    return {
        "mode":      _trading_mode,
        "broker":    _active_broker_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/balance")
async def api_balance():
    if _trading_mode == "live":
        try:
            broker = get_broker(_active_broker_name)
            bal = broker.get_balance()
            return {
                "balance": round(bal, 2),
                "initial": round(bal, 2),
                "pnl":     0.0,
                "pnl_pct": 0.0,
                "mode":    "live",
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Live balance fetch failed: {exc}")

    bal  = _paper_executor.get_balance()
    init = settings.initial_paper_balance
    return {
        "balance": round(bal, 2),
        "initial": round(init, 2),
        "pnl":     round(bal - init, 2),
        "pnl_pct": round((bal - init) / init * 100, 2),
        "mode":    "paper",
    }


@app.get("/api/equity")
async def api_equity():
    return {"history": _equity_history}


@app.get("/api/positions")
async def api_positions():
    if _trading_mode == "live":
        # Live positions tracked on broker; not maintained in-memory
        return {"positions": [], "note": "Check broker for live positions"}
    return {"positions": _paper_executor.get_positions()}


@app.get("/api/trades")
async def api_trades():
    if _trading_mode == "live":
        return {"trades": _read_live_trades(), "mode": "live"}
    return {"trades": _paper_executor.get_trade_history(), "mode": "paper"}


@app.get("/api/signal")
async def api_signal():
    return {"signal": _last_signal}


# ─────────────────────────────────────────────────────────────────────────────
# REST API — scan for signal
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/scan")
async def api_scan(
    broker_name: str = Query(default=""),
    symbol:      str = Query(default=""),
):
    global _last_signal, _active_broker_name

    # Use active broker if none specified
    if not broker_name:
        broker_name = _active_broker_name
    else:
        _active_broker_name = broker_name

    if not symbol:
        symbol = _default_symbol(broker_name)

    try:
        broker = get_broker(broker_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    strategy = ICTStrategy(
        symbol       = symbol,
        interval     = "15m",
        swing_length = 5,
        ob_lookback  = 20,
        kill_zones   = ["london_open", "ny_open"] if broker_name == "delta" else [],
        rr_ratio     = 2.0,
    )

    try:
        signal = strategy.generate_signal(broker)
    except Exception as exc:
        logger.error("Signal scan error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Scan failed: {exc}")

    if signal is None:
        _last_signal = None
        return {"signal": None, "message": "No ICT setup found on current data"}

    # ── ML scoring ────────────────────────────────────────────────────────────
    ml_decision = None
    try:
        df_for_ml = getattr(strategy, "_last_df", None)
        if df_for_ml is not None:
            ml_decision = ml.score(df_for_ml, signal)
    except Exception as exc:
        logger.warning("ML scoring failed (non-fatal): %s", exc)

    ml_info = None
    if ml_decision is not None:
        ml_info = {
            "approved":     ml_decision.approved,
            "confidence":   round(ml_decision.confidence, 4),
            "threshold":    round(ml_decision.threshold, 2),
            "suggested_rr": ml_decision.suggested_rr,
            "model":        ml_decision.model_name,
            "top_features": ml_decision.top_features,
        }

    # In live mode: block trades the ML rejects
    if _trading_mode == "live" and ml_decision is not None and not ml_decision.approved:
        return {
            "signal":  {
                "symbol":      signal.symbol,
                "direction":   signal.direction,
                "entry":       signal.entry,
                "stop_loss":   signal.stop_loss,
                "take_profit": signal.take_profit,
                "reason":      signal.reason,
            },
            "ml":      ml_info,
            "message": f"Trade blocked by ML (confidence {ml_decision.confidence:.2f} < {ml_decision.threshold:.2f})",
            "mode":    _trading_mode,
        }

    _last_signal = {
        "symbol":      signal.symbol,
        "direction":   signal.direction,
        "entry":       signal.entry,
        "stop_loss":   signal.stop_loss,
        "take_profit": signal.take_profit,
        "reason":      signal.reason,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "ml":          ml_info,
    }

    # Execute based on current mode
    try:
        if _trading_mode == "live":
            executor = LiveExecutor(
                broker       = broker,
                risk_pct     = settings.risk_per_trade_pct,
                risk_manager = _risk_mgr,
            )
            exec_result = executor.execute(signal)
            try:
                bal = broker.get_balance()
            except Exception:
                bal = _equity_history[-1]["v"] if _equity_history else 0.0
        else:
            exec_result = _paper_executor.execute(signal)
            bal = _paper_executor.get_balance()
    except Exception as exc:
        logger.error("Execution error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Execution failed: {exc}")

    # Register pending trade for ML outcome tracking
    try:
        if ml_decision is not None and isinstance(exec_result, dict):
            trade_id = exec_result.get("order_id") or exec_result.get("trade_id") or f"scan_{datetime.now(timezone.utc).timestamp()}"
            ml.register_pending(
                trade_id   = str(trade_id),
                signal     = signal,
                features   = ml_decision.feature_vec,
                confidence = ml_decision.confidence,
                prediction = 1 if ml_decision.confidence >= ml_decision.threshold else 0,
                timestamp  = datetime.now(timezone.utc),
            )
    except Exception as exc:
        logger.debug("ML register_pending failed (non-fatal): %s", exc)

    _equity_history.append({"t": datetime.now(timezone.utc).isoformat(), "v": bal})

    await _broadcast({"event": "signal",  "data": _last_signal})
    await _broadcast({"event": "balance", "data": {"balance": bal, "mode": _trading_mode}})

    return {"signal": _last_signal, "execution": exec_result, "ml": ml_info, "mode": _trading_mode}


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
    try:
        _last_backtest = run_backtest(
            n_trials  = params.trials,
            win_rate  = params.win_rate,
            rr_ratio  = params.rr,
            risk_pct  = params.risk,
            seed      = params.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    await _broadcast({"event": "backtest_done", "data": _last_backtest})
    return _last_backtest


# ─────────────────────────────────────────────────────────────────────────────
# REST API — backtest (real data)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/backtest/real")
async def api_backtest_real(
    broker_name: str   = Query(default="delta"),
    symbol:      str   = Query(default="BTCUSD"),
    interval:    str   = Query(default="4h"),
    rr:          float = Query(default=2.0),
    risk:        float = Query(default=1.0),
):
    global _last_backtest
    result = run_real_backtest(
        broker_name = broker_name,
        symbol      = symbol,
        interval    = interval,
        rr_ratio    = rr,
        risk_pct    = risk,
    )
    if "error" not in result:
        _last_backtest = result
        await _broadcast({"event": "backtest_done", "data": result})
    return result


@app.get("/api/backtest/last")
async def api_backtest_last():
    return {"results": _last_backtest}


# ─────────────────────────────────────────────────────────────────────────────
# REST API — ML model management
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/ml/status")
async def api_ml_status():
    """Return current ML model info (version, metrics, performance tracker)."""
    try:
        return ml.get_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ml/train")
async def api_ml_train(n_synthetic: int = Query(default=500)):
    """Force a full model retrain and return the new metrics."""
    try:
        from ml.retrain_pipeline import force_retrain
        result = force_retrain(n_synthetic=n_synthetic, skip_improvement_check=True)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ml/threshold")
async def api_ml_threshold(value: float = Query(...)):
    """Update the ML confidence threshold (0.5 – 0.95)."""
    if not (0.5 <= value <= 0.95):
        raise HTTPException(status_code=400, detail="threshold must be between 0.50 and 0.95")
    ml.set_threshold(value)
    return {"threshold": value}


@app.get("/api/ml/versions")
async def api_ml_versions():
    """List all saved model versions."""
    from ml.model import list_versions
    return {"versions": list_versions()}


@app.post("/api/ml/outcome")
async def api_ml_outcome(
    trade_id: str  = Query(...),
    result:   int  = Query(...),
    pnl:      float = Query(default=0.0),
):
    """
    Record the outcome of a trade to feed the learning system.
    result: 1 = win, 0 = loss.
    """
    if result not in (0, 1):
        raise HTTPException(status_code=400, detail="result must be 0 or 1")
    ml.record_outcome(trade_id=trade_id, actual_result=result, pnl=pnl)
    return {"recorded": True, "trade_id": trade_id}


# ─────────────────────────────────────────────────────────────────────────────
# REST API — training loop (continuous OHLCV learning)
# ─────────────────────────────────────────────────────────────────────────────

_training_task: asyncio.Task | None = None


async def _training_progress_broadcaster(status: dict) -> None:
    """Broadcast training progress via WebSocket."""
    await _broadcast({"event": "training_progress", "data": status})


@app.post("/api/training/start")
async def api_training_start(
    duration_hours:       float = Query(default=settings.training_duration_hours),
    symbol:               str   = Query(default=settings.training_symbol),
    interval:             str   = Query(default=settings.training_interval),
    fetch_count:          int   = Query(default=settings.training_fetch_count),
    fetch_interval_sec:   int   = Query(default=settings.training_fetch_interval_sec),
    retrain_every:        int   = Query(default=settings.training_retrain_every),
    broker_name:          str   = Query(default="delta"),
    deep_history_candles: int   = Query(default=50_000),
):
    """Start the continuous training loop (fetches OHLCV and retrains)."""
    from ml.training_loop import get_training_loop

    loop = get_training_loop()

    # Set up async progress broadcasting
    def on_progress(status):
        try:
            asyncio.get_event_loop().create_task(
                _broadcast({"event": "training_progress", "data": status})
            )
        except RuntimeError:
            pass  # No event loop in thread

    def on_complete(status):
        try:
            asyncio.get_event_loop().create_task(
                _broadcast({"event": "training_complete", "data": status})
            )
        except RuntimeError:
            pass

    loop._on_progress = on_progress
    loop._on_complete = on_complete

    result = loop.start(
        duration_hours=duration_hours,
        symbols=symbol,
        interval=interval,
        fetch_count=fetch_count,
        fetch_interval_sec=fetch_interval_sec,
        retrain_every=retrain_every,
        broker_name=broker_name,
        deep_history_candles=deep_history_candles,
    )

    if "error" in result:
        raise HTTPException(status_code=409, detail=result["error"])

    return result


@app.post("/api/training/stop")
async def api_training_stop():
    """Stop the continuous training loop."""
    from ml.training_loop import get_training_loop
    loop = get_training_loop()
    result = loop.stop()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/training/status")
async def api_training_status():
    """Get current training loop progress."""
    from ml.training_loop import get_training_loop
    loop = get_training_loop()
    return loop.status()


# ─────────────────────────────────────────────────────────────────────────────
# REST API — mistake analysis
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/ml/mistakes")
async def api_ml_mistakes(min_trades: int = Query(default=10)):
    """Get mistake analysis report (learns from losing trades)."""
    try:
        from ml.mistake_analyzer import get_mistake_report
        return get_mistake_report(min_trades=min_trades)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# REST API — drift detection
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/ml/drift")
async def api_ml_drift():
    """Get feature importance drift report."""
    try:
        from ml.drift_detector import get_drift_report
        return get_drift_report()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# REST API — dynamic threshold
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/ml/threshold/dynamic")
async def api_dynamic_threshold_status():
    """Get dynamic threshold status."""
    from ml.dynamic_threshold import get_dynamic_threshold
    return get_dynamic_threshold().status()


@app.post("/api/ml/threshold/dynamic")
async def api_dynamic_threshold_toggle(
    enabled: bool = Query(default=True),
    base: float = Query(default=0.60),
):
    """Enable/disable dynamic threshold and set base value."""
    from ml.dynamic_threshold import get_dynamic_threshold
    dt = get_dynamic_threshold()
    dt.enabled = enabled
    dt.base = max(0.40, min(0.90, base))
    return {"updated": True, **dt.status()}


# ─────────────────────────────────────────────────────────────────────────────
# REST API — correlation exposure
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/risk/correlation")
async def api_correlation_exposure():
    """Get current correlation group exposure from the paper executor."""
    from execution.correlation_filter import CorrelationFilter
    cf = CorrelationFilter()
    positions = _paper_executor.get_positions()
    return {
        "exposure": cf.get_exposure(positions),
        "open_positions": len(positions),
        "groups": list(cf.groups.keys()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Auto-scan loop — background signal scanner
# ─────────────────────────────────────────────────────────────────────────────

_autoscan_task: asyncio.Task | None = None
_autoscan_state: dict = {
    "running": False,
    "interval_sec": 60,
    "started_at": "",
    "scans": 0,
    "signals_found": 0,
    "trades_executed": 0,
    "last_scan_at": "",
    "last_error": "",
}


async def _autoscan_loop(interval_sec: int, broker_name: str, symbol: str) -> None:
    """Background loop that periodically scans for ICT signals and auto-executes."""
    global _last_signal, _autoscan_state
    s = _autoscan_state
    s["running"] = True
    s["interval_sec"] = interval_sec
    s["started_at"] = datetime.now(timezone.utc).isoformat()
    s["scans"] = 0
    s["signals_found"] = 0
    s["trades_executed"] = 0
    s["last_error"] = ""

    logger.info("Auto-scan started: %s/%s every %ds", broker_name, symbol, interval_sec)

    try:
        broker = get_broker(broker_name)
    except Exception as exc:
        s["last_error"] = f"Broker init failed: {exc}"
        s["running"] = False
        return

    strategy = ICTStrategy(
        symbol=symbol,
        interval="15m",
        swing_length=5,
        ob_lookback=20,
        kill_zones=["london_open", "ny_open"] if broker_name == "delta" else [],
        rr_ratio=2.0,
    )

    while s["running"]:
        s["scans"] += 1
        s["last_scan_at"] = datetime.now(timezone.utc).isoformat()

        try:
            # ── Check daily loss limit ───────────────────────────────────
            bal = (_paper_executor.get_balance()
                   if _trading_mode == "paper"
                   else broker.get_balance())
            if not _risk_mgr.can_trade(bal):
                logger.info("Auto-scan: daily loss limit reached — pausing")
                await _broadcast({"event": "autoscan_tick", "data": {
                    **s, "message": "Daily loss limit reached — pausing",
                }})
                await asyncio.sleep(interval_sec)
                continue

            # ── Check for duplicate position on same symbol ──────────────
            if _trading_mode == "paper":
                open_pos = _paper_executor.get_positions()
                if any(p.get("symbol") == symbol for p in open_pos):
                    await _broadcast({"event": "autoscan_tick", "data": {
                        **s, "message": f"Position already open on {symbol}",
                    }})
                    await asyncio.sleep(interval_sec)
                    continue

            # ── Run ICT strategy ─────────────────────────────────────────
            signal = strategy.generate_signal(broker)

            if signal is None:
                await _broadcast({"event": "autoscan_tick", "data": {
                    **s, "message": "No setup found",
                }})
                await asyncio.sleep(interval_sec)
                continue

            s["signals_found"] += 1

            # ── ML scoring ───────────────────────────────────────────────
            ml_decision = None
            try:
                df_for_ml = getattr(strategy, "_last_df", None)
                if df_for_ml is not None:
                    ml_decision = ml.score(df_for_ml, signal)
            except Exception as exc:
                logger.warning("Auto-scan ML scoring failed: %s", exc)

            ml_info = None
            if ml_decision is not None:
                ml_info = {
                    "approved":     ml_decision.approved,
                    "confidence":   round(ml_decision.confidence, 4),
                    "threshold":    round(ml_decision.threshold, 2),
                    "suggested_rr": ml_decision.suggested_rr,
                    "model":        ml_decision.model_name,
                    "top_features": ml_decision.top_features,
                }

            # In live mode: block trades the ML rejects
            if _trading_mode == "live" and ml_decision is not None and not ml_decision.approved:
                logger.info(
                    "Auto-scan: ML blocked trade (%.2f < %.2f)",
                    ml_decision.confidence, ml_decision.threshold,
                )
                await _broadcast({"event": "autoscan_tick", "data": {
                    **s,
                    "message": f"Signal found but ML blocked ({ml_decision.confidence:.0%} < {ml_decision.threshold:.0%})",
                    "signal": {"direction": signal.direction, "entry": signal.entry},
                    "ml": ml_info,
                }})
                await asyncio.sleep(interval_sec)
                continue

            # ── Execute trade ────────────────────────────────────────────
            _last_signal = {
                "symbol":      signal.symbol,
                "direction":   signal.direction,
                "entry":       signal.entry,
                "stop_loss":   signal.stop_loss,
                "take_profit": signal.take_profit,
                "reason":      signal.reason,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "ml":          ml_info,
                "auto":        True,
            }

            try:
                exec_result = None
                if _trading_mode == "live":
                    executor = LiveExecutor(
                        broker=broker,
                        risk_pct=settings.risk_per_trade_pct,
                        risk_manager=_risk_mgr,
                    )
                    exec_result = executor.execute(signal)
                    try:
                        new_bal = broker.get_balance()
                    except Exception:
                        new_bal = _equity_history[-1]["v"] if _equity_history else 0.0
                else:
                    exec_result = _paper_executor.execute(signal)
                    new_bal = _paper_executor.get_balance()

                s["trades_executed"] += 1
                _equity_history.append({
                    "t": datetime.now(timezone.utc).isoformat(),
                    "v": new_bal,
                })

                await _broadcast({"event": "signal",  "data": _last_signal})
                await _broadcast({"event": "balance", "data": {"balance": new_bal, "mode": _trading_mode}})

                logger.info(
                    "Auto-scan: %s %s @ %.2f  [scan #%d]",
                    signal.direction.upper(), signal.symbol, signal.entry, s["scans"],
                )
            except Exception as exc:
                logger.error("Auto-scan execution error: %s", exc)
                s["last_error"] = str(exc)

            # Register for ML outcome tracking
            try:
                if ml_decision is not None and isinstance(exec_result, dict):
                    trade_id = exec_result.get("order_id") or exec_result.get("trade_id") or f"auto_{datetime.now(timezone.utc).timestamp()}"
                    ml.register_pending(
                        trade_id=str(trade_id),
                        signal=signal,
                        features=ml_decision.feature_vec,
                        confidence=ml_decision.confidence,
                        prediction=1 if ml_decision.confidence >= ml_decision.threshold else 0,
                        timestamp=datetime.now(timezone.utc),
                    )
            except Exception:
                pass

            await _broadcast({"event": "autoscan_tick", "data": {
                **s,
                "message": f"Trade executed: {signal.direction.upper()} @ {signal.entry}",
                "signal": _last_signal,
                "ml": ml_info,
            }})

        except Exception as exc:
            logger.error("Auto-scan error: %s", exc)
            s["last_error"] = str(exc)
            await _broadcast({"event": "autoscan_tick", "data": {
                **s, "message": f"Error: {exc}",
            }})

        await asyncio.sleep(interval_sec)

    s["running"] = False
    logger.info("Auto-scan stopped after %d scans", s["scans"])


@app.post("/api/autoscan/start")
async def api_autoscan_start(
    interval_sec: int = Query(default=60, ge=30, le=300),
    broker_name:  str = Query(default=""),
    symbol:       str = Query(default=""),
):
    """Start automatic signal scanning on a timer."""
    global _autoscan_task
    if _autoscan_state["running"]:
        raise HTTPException(status_code=409, detail="Auto-scan is already running")

    if not broker_name:
        broker_name = _active_broker_name
    if not symbol:
        symbol = _default_symbol(broker_name)

    _autoscan_task = asyncio.create_task(
        _autoscan_loop(interval_sec, broker_name, symbol)
    )
    return {"started": True, "interval_sec": interval_sec, "symbol": symbol}


@app.post("/api/autoscan/stop")
async def api_autoscan_stop():
    """Stop the auto-scan loop."""
    global _autoscan_task
    if not _autoscan_state["running"]:
        raise HTTPException(status_code=400, detail="Auto-scan is not running")

    _autoscan_state["running"] = False
    if _autoscan_task and not _autoscan_task.done():
        _autoscan_task.cancel()
    _autoscan_task = None
    return {"stopped": True}


@app.get("/api/autoscan/status")
async def api_autoscan_status():
    """Get auto-scan loop status."""
    return _autoscan_state


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — live push updates
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        bal  = _paper_executor.get_balance()
        init = settings.initial_paper_balance
        await ws.send_json({
            "event": "snapshot",
            "data": {
                "mode":    _trading_mode,
                "broker":  _active_broker_name,
                "balance": bal,
                "pnl":     round(bal - init, 2),
                "pnl_pct": round((bal - init) / init * 100, 2),
                "signal":  _last_signal,
                "equity":  _equity_history[-20:],
            },
        })
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
