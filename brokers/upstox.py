"""Upstox broker integration (V2 API).

Upstox supports two environments:
  - sandbox : order placement  → https://api-sandbox.upstox.com/v2
              market data      → https://api.upstox.com/v2  (sandbox has no market data)
  - live    : order placement  → https://api-hft.upstox.com/v2
              market data      → https://api.upstox.com/v2

Market data (OHLC, LTP, balance) always comes from the live data URL because
the sandbox API only simulates order placement — it does not serve quotes.

Set UPSTOX_ENV=sandbox or UPSTOX_ENV=live in .env to switch environments.

Interval mapping (Upstox V2 historical-candle):
  "1m"  → "1minute"   "30m" → "30minute"
  "1d"  → "day"       "1w"  → "week"      "1mo" → "month"
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

from brokers.base import BaseBroker, BrokerAPIError, retry
from config.settings import settings

logger = logging.getLogger(__name__)

# Order placement URLs
_LIVE_ORDER_BASE    = "https://api-hft.upstox.com/v2"
_SANDBOX_ORDER_BASE = "https://api-sandbox.upstox.com/v2"

# Market data always comes from the live data API (sandbox has no quotes)
_DATA_BASE = "https://api.upstox.com/v2"

# Maps generic interval strings to Upstox historical-candle interval names
_INTERVAL_MAP = {
    "1m":  "1minute",
    "30m": "30minute",
    "1d":  "day",
    "1w":  "week",
    "1mo": "month",
    # pass-through if already in Upstox format
    "1minute":  "1minute",
    "30minute": "30minute",
    "day":      "day",
}

# Maps generic side/order_type to Upstox enums
SIDE_MAP = {"buy": "BUY", "sell": "SELL"}
ORDER_TYPE_MAP = {
    "market": "MARKET",
    "limit":  "LIMIT",
    "sl":     "SL",
    "sl-m":   "SL-M",
}


class UpstoxBroker(BaseBroker):

    def __init__(self) -> None:
        env = settings.upstox_env.lower()  # "sandbox" or "live"

        if env == "sandbox":
            access_token     = settings.upstox_sandbox_access_token
            self._order_base = _SANDBOX_ORDER_BASE
            self._data_base  = _SANDBOX_ORDER_BASE   # sandbox token only valid on sandbox domain
            logger.info("UpstoxBroker initialised in SANDBOX mode")
        else:
            access_token     = settings.upstox_live_access_token
            self._order_base = _LIVE_ORDER_BASE
            self._data_base  = _DATA_BASE
            logger.info("UpstoxBroker initialised in LIVE mode")

        self._env = env

        # Order-placement session (uses main access token)
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {access_token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        })

        # Data / analytics session — uses the dedicated data token when
        # UPSTOX_DATA_ACCESS_TOKEN is set, otherwise reuses access_token.
        data_token = settings.upstox_data_access_token or access_token
        self._data_session = requests.Session()
        self._data_session.headers.update({
            "Authorization": f"Bearer {data_token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        })
        if data_token != access_token:
            logger.info("UpstoxBroker: using separate data/analytics token")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check(self, resp: requests.Response, url: str) -> dict:
        if not resp.ok:
            raise BrokerAPIError(resp.status_code, resp.text, url)
        data = resp.json()
        if data.get("status") == "error":
            raise BrokerAPIError(resp.status_code, str(data), url)
        return data

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    @retry()
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "market",
        quantity: int = 1,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> str:
        url = f"{self._order_base}/order/place"
        payload = {
            "instrument_token": symbol,
            "transaction_type": SIDE_MAP[side.lower()],
            "order_type": ORDER_TYPE_MAP[order_type.lower()],
            "quantity": quantity,
            "product": "I",       # intraday
            "validity": "DAY",
            "disclosed_quantity": 0,
            "is_amo": False,
        }
        if price is not None:
            payload["price"] = price
        if stop_price is not None:
            payload["trigger_price"] = stop_price

        resp = self._session.post(url, json=payload)
        data = self._check(resp, url)
        order_id = data.get("data", {}).get("order_id", "")
        logger.info("Upstox order placed: %s (side=%s qty=%d)", order_id, side, quantity)
        return order_id

    @retry()
    def get_price(self, symbol: str) -> float:
        url = f"{self._data_base}/market-quote/ltp"
        resp = self._data_session.get(url, params={"instrument_key": symbol})
        data = self._check(resp, url)
        # Response: {"data": {"<symbol>": {"last_price": 22450.5, ...}}}
        quotes = data.get("data", {})
        for key, quote in quotes.items():
            return float(quote["last_price"])
        raise BrokerAPIError(200, f"No LTP data for {symbol}", url)

    @retry()
    def get_ohlc(self, symbol: str, interval: str = "1d", count: int = 100) -> pd.DataFrame:
        """
        Fetch historical OHLC candles using the Upstox historical-candle endpoint.

        Uses intraday endpoint for minute intervals (returns today's candles),
        and the dated endpoint for daily/weekly/monthly candles.

        Upstox candle response format:
            [timestamp, open, high, low, close, volume, open_interest]
        """
        upstox_interval = _INTERVAL_MAP.get(interval, interval)
        is_intraday = upstox_interval in ("1minute", "30minute")

        # Instrument key contains '|' which must be encoded in URL paths
        encoded_symbol = symbol.replace("|", "%7C")

        if is_intraday:
            # Intraday: returns all candles for today's session
            url = f"{self._data_base}/historical-candle/intraday/{encoded_symbol}/{upstox_interval}"
            resp = self._data_session.get(url)
        else:
            # Daily / weekly / monthly: fetch `count` candles back from today
            to_date   = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=count * 2)).strftime("%Y-%m-%d")
            url = f"{self._data_base}/historical-candle/{encoded_symbol}/{upstox_interval}/{to_date}/{from_date}"
            resp = self._data_session.get(url)

        data = self._check(resp, url)
        candles = data.get("data", {}).get("candles", [])

        if not candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df["open"]   = pd.to_numeric(df["open"],   errors="coerce")
        df["high"]   = pd.to_numeric(df["high"],   errors="coerce")
        df["low"]    = pd.to_numeric(df["low"],    errors="coerce")
        df["close"]  = pd.to_numeric(df["close"],  errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna().sort_values("timestamp").reset_index(drop=True)

        return df.tail(count).reset_index(drop=True)

    @retry()
    def get_balance(self) -> float:
        url = f"{self._data_base}/user/get-funds-and-margin"
        resp = self._data_session.get(url)
        data = self._check(resp, url)
        # Equity segment available margin
        equity = data.get("data", {}).get("equity", {})
        return float(equity.get("available_margin", 0))

    @retry()
    def get_order_status(self, order_id: str) -> dict:
        url = f"{self._data_base}/order/details"
        resp = self._session.get(url, params={"order_id": order_id})
        data = self._check(resp, url)
        return data.get("data", {})

    @retry()
    def cancel_order(self, order_id: str) -> bool:
        url = f"{self._order_base}/order/cancel"
        resp = self._session.delete(url, params={"order_id": order_id})
        self._check(resp, url)
        logger.info("Upstox order cancelled: %s", order_id)
        return True
