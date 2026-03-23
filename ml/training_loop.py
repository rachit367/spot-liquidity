"""
Continuous self-learning training loop.

Fetches fresh OHLCV data from Delta Exchange in a background loop,
runs the ICT strategy on sliding windows, simulates exits, extracts
features + labels, and periodically retrains the ML model.

Usage
-----
    from ml.training_loop import TrainingLoop
    loop = TrainingLoop(on_progress=my_callback)
    loop.start(duration_hours=2.0, symbol="BTCUSD", interval="15m")
    loop.stop()
    loop.status()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtesting import _run_strategy_on_window, _simulate_exit
from brokers import get_broker
from config.settings import settings
from ml.dataset_builder import append_trade
from ml.feature_engineering import extract
from strategy import ICTStrategy, _prepare

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingState:
    """Mutable state shared between the training thread and the API."""
    running:           bool  = False
    started_at:        str   = ""
    stopped_at:        str   = ""
    duration_hours:    float = 0.0
    symbol:            str   = ""      # comma-separated list
    current_symbol:    str   = ""      # currently processing
    interval:          str   = ""
    # Progress counters
    epoch:             int   = 0        # how many fetch cycles
    total_windows:     int   = 0        # ICT windows scanned
    total_signals:     int   = 0        # valid signals found
    total_trades:      int   = 0        # trades simulated (win + loss)
    total_wins:        int   = 0
    total_losses:      int   = 0
    # Model info
    retrains_done:     int   = 0
    last_f1:           float = 0.0
    last_model_version: int  = 0
    # Error tracking
    last_error:        str   = ""
    consecutive_errors: int  = 0
    # Config
    retrain_every:     int   = 50
    fetch_interval_sec: int  = 300


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop class
# ─────────────────────────────────────────────────────────────────────────────

class TrainingLoop:
    """
    Background training loop that continuously fetches OHLCV data,
    simulates ICT trades, and retrains the ML model.

    Thread-safe start/stop with async-compatible progress callbacks.
    """

    def __init__(
        self,
        on_progress: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
    ) -> None:
        self.state = TrainingState()
        self._on_progress = on_progress    # called with state dict
        self._on_complete = on_complete    # called when loop finishes
        self._stop_event  = threading.Event()
        self._thread:     Optional[threading.Thread] = None
        self._trades_since_retrain = 0

    # ── Public API ────────────────────────────────────────────────────────

    def start(
        self,
        duration_hours:     float = 2.0,
        symbols:            str   = "BTCUSD",
        interval:           str   = "15m",
        fetch_count:        int   = 500,
        fetch_interval_sec: int   = 300,
        retrain_every:      int   = 50,
        broker_name:        str   = "delta",
    ) -> dict:
        """Start the training loop in a background thread.

        Parameters
        ----------
        symbols : Comma-separated list of symbols (e.g. "BTCUSD,ETHUSD,SOLUSD")
        """
        if self.state.running:
            return {"error": "Training loop is already running", "status": self.status()}

        self._stop_event.clear()
        self.state = TrainingState(
            running           = True,
            started_at        = datetime.now(timezone.utc).isoformat(),
            duration_hours    = duration_hours,
            symbol            = symbols,
            interval          = interval,
            retrain_every     = retrain_every,
            fetch_interval_sec = fetch_interval_sec,
        )
        self._trades_since_retrain = 0

        self._thread = threading.Thread(
            target=self._run_loop,
            args=(duration_hours, symbols, interval, fetch_count,
                  fetch_interval_sec, retrain_every, broker_name),
            daemon=True,
            name="training-loop",
        )
        self._thread.start()

        logger.info(
            "Training loop started: symbols=%s interval=%s duration=%.1fh",
            symbols, interval, duration_hours,
        )
        return {"started": True, "status": self.status()}

    def stop(self) -> dict:
        """Signal the training loop to stop."""
        if not self.state.running:
            return {"error": "Training loop is not running"}

        self._stop_event.set()
        logger.info("Training loop stop requested")
        return {"stopping": True, "status": self.status()}

    def status(self) -> dict:
        """Return a snapshot of the training loop state."""
        s = self.state
        elapsed = 0.0
        if s.started_at and s.running:
            started = datetime.fromisoformat(s.started_at)
            elapsed = (datetime.now(timezone.utc) - started).total_seconds() / 3600.0

        return {
            "running":            s.running,
            "started_at":         s.started_at,
            "stopped_at":         s.stopped_at,
            "duration_hours":     s.duration_hours,
            "elapsed_hours":      round(elapsed, 2),
            "symbols":            s.symbol,
            "current_symbol":     s.current_symbol,
            "interval":           s.interval,
            "epoch":              s.epoch,
            "total_windows":      s.total_windows,
            "total_signals":      s.total_signals,
            "total_trades":       s.total_trades,
            "total_wins":         s.total_wins,
            "total_losses":       s.total_losses,
            "win_rate":           round(s.total_wins / max(s.total_trades, 1) * 100, 1),
            "retrains_done":      s.retrains_done,
            "last_f1":            round(s.last_f1, 4),
            "last_model_version": s.last_model_version,
            "last_error":         s.last_error,
            "consecutive_errors": s.consecutive_errors,
            "retrain_every":      s.retrain_every,
            "fetch_interval_sec": s.fetch_interval_sec,
        }

    # ── Internal loop ─────────────────────────────────────────────────────

    def _run_loop(
        self,
        duration_hours:     float,
        symbols:            str,
        interval:           str,
        fetch_count:        int,
        fetch_interval_sec: int,
        retrain_every:      int,
        broker_name:        str,
    ) -> None:
        """Main training loop, runs in a background thread."""
        end_time = time.monotonic() + (duration_hours * 3600)
        s = self.state

        try:
            broker = get_broker(broker_name)
        except Exception as exc:
            s.last_error = f"Failed to initialize broker: {exc}"
            s.running = False
            logger.error("Training loop broker init failed: %s", exc)
            return

        # Parse symbol list for multi-symbol training
        symbol_list = [sym.strip() for sym in symbols.split(",") if sym.strip()]
        if not symbol_list:
            symbol_list = ["BTCUSD"]

        kill_zones = [] if broker_name == "upstox" else ["london_open", "ny_open"]

        # Create strategy per symbol
        strategies = {}
        for sym in symbol_list:
            strategies[sym] = ICTStrategy(
                sym, interval=interval, ob_lookback=20,
                kill_zones=kill_zones, swing_length=5, rr_ratio=2.0,
            )

        logger.info(
            "Training loop running (%.1fh, %d symbols, fetch every %ds)",
            duration_hours, len(symbol_list), fetch_interval_sec,
        )

        symbol_idx = 0

        while not self._stop_event.is_set():
            # Check duration
            if time.monotonic() >= end_time:
                logger.info("Training loop duration reached")
                break

            # Round-robin symbol selection
            current_symbol = symbol_list[symbol_idx % len(symbol_list)]
            symbol_idx += 1
            s.current_symbol = current_symbol
            s.epoch += 1

            strategy = strategies[current_symbol]

            # ── Fetch OHLCV ──────────────────────────────────────────────
            try:
                df_raw = broker.get_ohlc(current_symbol, interval, count=fetch_count)
                s.consecutive_errors = 0
                s.last_error = ""
            except Exception as exc:
                s.consecutive_errors += 1
                s.last_error = str(exc)
                logger.warning(
                    "OHLCV fetch failed (attempt %d): %s",
                    s.consecutive_errors, exc,
                )
                if s.consecutive_errors >= 10:
                    logger.error("Too many consecutive errors — stopping training")
                    break
                # Wait and retry
                self._stop_event.wait(min(fetch_interval_sec, 30))
                continue

            if df_raw is None or len(df_raw) < 150:
                logger.warning("Not enough data (%d candles) — skipping epoch",
                               len(df_raw) if df_raw is not None else 0)
                self._stop_event.wait(fetch_interval_sec)
                continue

            # ── Process data ─────────────────────────────────────────────
            df = _prepare(df_raw.copy().reset_index(drop=True))
            new_trades = self._process_ohlcv(df, strategy)

            # ── Maybe retrain ────────────────────────────────────────────
            self._trades_since_retrain += new_trades
            if self._trades_since_retrain >= retrain_every:
                self._do_retrain()

            # ── Progress callback ────────────────────────────────────────
            if self._on_progress:
                try:
                    self._on_progress(self.status())
                except Exception:
                    pass

            # ── Wait before next fetch ───────────────────────────────────
            self._stop_event.wait(fetch_interval_sec)

        # ── Finished ─────────────────────────────────────────────────────
        # Final retrain if there are pending unseen trades
        if self._trades_since_retrain > 0:
            self._do_retrain()

        s.running = False
        s.stopped_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Training loop finished: %d epochs, %d trades, %d retrains, f1=%.3f",
            s.epoch, s.total_trades, s.retrains_done, s.last_f1,
        )

        if self._on_complete:
            try:
                self._on_complete(self.status())
            except Exception:
                pass

    def _process_ohlcv(self, df: pd.DataFrame, strategy: ICTStrategy) -> int:
        """
        Slide a 100-candle window across the DataFrame, run ICT strategy,
        simulate exits, extract features, and append to trades_ml.csv.

        Returns the number of new trades generated.
        """
        s = self.state
        window = 100
        forward = 50
        step = 10
        new_trades = 0

        for start in range(0, len(df) - window - forward + 1, step):
            if self._stop_event.is_set():
                break

            end     = start + window
            fwd_end = min(end + forward, len(df))

            setup_df   = df.iloc[start:end].copy().reset_index(drop=True)
            forward_df = df.iloc[end:fwd_end].copy().reset_index(drop=True)

            if len(forward_df) < 2:
                continue

            s.total_windows += 1

            signal = _run_strategy_on_window(setup_df, rr_ratio=2.0, strategy=strategy)
            if signal is None:
                continue

            s.total_signals += 1

            # Simulate exit
            exit_price, outcome = _simulate_exit(signal, forward_df, max_candles=forward)

            # Determine result
            is_win = 1 if outcome == "tp" else 0
            risk_amount = abs(signal.entry - signal.stop_loss)

            if signal.direction == "long":
                pnl = (exit_price - signal.entry)
            else:
                pnl = (signal.entry - exit_price)

            pnl_r = (pnl / risk_amount) if risk_amount > 0 else 0.0

            # Extract features
            try:
                features = extract(setup_df, signal)
            except Exception:
                continue

            # Append to trades_ml.csv
            trade_id = f"train_{s.epoch}_{s.total_trades}_{int(time.time())}"
            record = {
                "trade_id":         trade_id,
                "timestamp":        datetime.now(timezone.utc).isoformat(),
                "symbol":           signal.symbol,
                "direction":        signal.direction,
                "entry_price":      signal.entry,
                "exit_price":       exit_price,
                "sl":               signal.stop_loss,
                "tp":               signal.take_profit,
                "quantity":         1,
                "pnl":              round(pnl, 4),
                "pnl_pct":          round(pnl / signal.entry * 100, 4) if signal.entry else 0,
                "rr_achieved":      round(pnl_r, 3),
                "result":           is_win,
                "strategy":         "ICT_training",
                "duration_bars":    min(forward, len(forward_df)),
                "model_confidence": None,
                "model_prediction": None,
                "features":         features,
            }
            append_trade(record)

            s.total_trades += 1
            if is_win:
                s.total_wins += 1
            else:
                s.total_losses += 1
            new_trades += 1

        return new_trades

    def _do_retrain(self) -> None:
        """Trigger a model retrain and update state."""
        s = self.state
        try:
            from ml.retrain_pipeline import force_retrain
            result = force_retrain(n_synthetic=200, skip_improvement_check=False)
            s.retrains_done += 1
            self._trades_since_retrain = 0

            if result.get("promoted"):
                s.last_f1 = result.get("new_f1", 0.0)
                s.last_model_version = result.get("version", 0)
                logger.info(
                    "Training retrain #%d: promoted v%d f1=%.3f",
                    s.retrains_done, s.last_model_version, s.last_f1,
                )
            else:
                logger.info(
                    "Training retrain #%d: not promoted (f1=%.3f < prev=%.3f)",
                    s.retrains_done,
                    result.get("new_f1", 0),
                    result.get("prev_f1", 0),
                )
        except Exception as exc:
            logger.error("Training retrain failed: %s", exc)
            s.last_error = f"Retrain failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_training_loop: Optional[TrainingLoop] = None


def get_training_loop() -> TrainingLoop:
    """Return the global TrainingLoop singleton."""
    global _training_loop
    if _training_loop is None:
        _training_loop = TrainingLoop()
    return _training_loop
