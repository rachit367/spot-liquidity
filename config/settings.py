"""
Centralized configuration and logging setup.

Virtual Environment Setup:
    python -m venv venv
    # Windows:  venv\\Scripts\\activate
    # Linux/Mac: source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env   # then fill in your API keys
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Trading mode
    trading_mode: Literal["paper", "live"] = "paper"
    default_broker: Literal["upstox", "delta"] = "upstox"

    # Risk management
    risk_per_trade_pct: float = Field(default=1.0, ge=0.01, le=10.0)
    max_daily_loss_pct: float = Field(default=3.0, ge=0.1, le=50.0)
    daily_reset_hour: int = Field(default=9, ge=0, le=23)
    initial_paper_balance: float = Field(default=100000.0, gt=0)

    # Upstox environment: "sandbox" uses paper-trading API, "live" uses real API
    upstox_env: Literal["sandbox", "live"] = "sandbox"

    # Upstox — Live credentials
    upstox_live_api_key: str = ""
    upstox_live_api_secret: str = ""
    upstox_live_access_token: str = ""

    # Upstox — Sandbox credentials (separate key/token from Upstox developer portal)
    upstox_sandbox_api_key: str = ""
    upstox_sandbox_api_secret: str = ""
    upstox_sandbox_access_token: str = ""

    # Upstox — dedicated read-only token for market data / analytics API.
    # When set, all data-fetching calls (OHLC, LTP, balance) use this token
    # instead of the main order-placement token.  Leave blank to reuse the
    # active environment's access token for data calls as well.
    upstox_data_access_token: str = ""

    # Delta Exchange
    delta_api_key: str = ""
    delta_api_secret: str = ""
    delta_base_url: str = "https://api.india.delta.exchange"

    # Training loop
    training_duration_hours: float = Field(default=2.0, ge=0.1, le=48.0)
    training_fetch_interval_sec: int = Field(default=300, ge=10, le=3600)
    training_retrain_every: int = Field(default=50, ge=5, le=500)
    training_symbol: str = "BTCUSD"
    training_interval: str = "15m"
    training_fetch_count: int = Field(default=500, ge=100, le=2000)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _JsonTradeFormatter(logging.Formatter):
    """Emits one JSON object per line for the trade log."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
        }
        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["message"] = record.getMessage()
        return json.dumps(payload, default=str)


def _setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(logging.DEBUG)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)

    # Rotating file
    file_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "bot.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)

    # Structured trade log (JSONL)
    trade_logger = logging.getLogger("trades")
    trade_logger.propagate = False
    trade_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "trades.jsonl",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    trade_handler.setFormatter(_JsonTradeFormatter())
    trade_logger.addHandler(trade_handler)


_setup_logging()
settings = Settings()
