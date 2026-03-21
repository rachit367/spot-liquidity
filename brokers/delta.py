"""Delta Exchange broker integration (V2 API with HMAC-SHA256 auth)."""

from __future__ import annotations

import hashlib
import hmac
import json as _json
import logging
import time

import pandas as pd
import requests

from brokers.base import BaseBroker, BrokerAPIError, retry
from config.settings import settings

logger = logging.getLogger(__name__)

SIDE_MAP = {"buy": "buy", "sell": "sell"}
ORDER_TYPE_MAP = {
    "market": "market_order",
    "limit": "limit_order",
    "stop": "stop_market_order",
}


class DeltaBroker(BaseBroker):

    def __init__(self) -> None:
        self._api_key = settings.delta_api_key
        self._api_secret = settings.delta_api_secret
        self._base_url = settings.delta_base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # HMAC signing
    # ------------------------------------------------------------------

    def _sign_headers(
        self,
        method: str,
        path: str,
        query_string: str = "",
        body: str = "",
    ) -> dict:
        """Generate auth headers. Must be called immediately before request."""
        timestamp = str(int(time.time()))
        message = method.upper() + timestamp + path + query_string + body
        signature = hmac.new(
            self._api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "api-key": self._api_key,
            "timestamp": timestamp,
            "signature": signature,
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self._base_url}/v2{path}"

    def _check(self, resp: requests.Response, url: str) -> dict:
        if not resp.ok:
            raise BrokerAPIError(resp.status_code, resp.text, url)
        data = resp.json()
        if not data.get("success", True):
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
        path = "/orders"
        payload: dict = {
            "product_symbol": symbol,
            "size": quantity,
            "side": SIDE_MAP[side.lower()],
            "order_type": ORDER_TYPE_MAP.get(order_type.lower(), "market_order"),
            "time_in_force": "gtc",
        }
        if price is not None:
            payload["limit_price"] = str(price)
        if stop_price is not None:
            payload["stop_price"] = str(stop_price)
            if order_type.lower() not in ("stop",):
                payload["order_type"] = "stop_market_order"

        body = _json.dumps(payload, separators=(",", ":"))
        url = self._url(path)
        headers = self._sign_headers("POST", path, body=body)
        resp = self._session.post(url, data=body, headers=headers)
        data = self._check(resp, url)

        order_id = str(data.get("result", {}).get("id", ""))
        logger.info(
            "Delta order placed: %s (side=%s qty=%d)", order_id, side, quantity
        )
        return order_id

    @retry()
    def get_price(self, symbol: str) -> float:
        path = f"/tickers/{symbol}"
        url = self._url(path)
        resp = self._session.get(url)
        data = self._check(resp, url)
        result = data.get("result", {})
        return float(result.get("mark_price") or result.get("close", 0))

    @retry()
    def get_ohlc(
        self, symbol: str, interval: str = "1d", count: int = 30
    ) -> pd.DataFrame:
        path = "/history/candles"
        now = int(time.time())
        # Map interval strings to seconds for start time calculation
        interval_seconds = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800,
        }
        period = interval_seconds.get(interval, 86400)
        start = now - (count * period)

        url = self._url(path)
        params = {
            "resolution": interval,
            "symbol": symbol,
            "start": start,
            "end": now,
        }
        resp = self._session.get(url, params=params)
        data = self._check(resp, url)
        candles = data.get("result", [])

        rows = []
        for c in candles:
            rows.append({
                "timestamp": c.get("time", c.get("t", "")),
                "open": float(c.get("open", c.get("o", 0))),
                "high": float(c.get("high", c.get("h", 0))),
                "low": float(c.get("low", c.get("l", 0))),
                "close": float(c.get("close", c.get("c", 0))),
                "volume": float(c.get("volume", c.get("v", 0))),
            })
        return pd.DataFrame(rows)

    @retry()
    def get_balance(self) -> float:
        path = "/wallet/balances"
        url = self._url(path)
        headers = self._sign_headers("GET", path)
        resp = self._session.get(url, headers=headers)
        data = self._check(resp, url)
        balances = data.get("result", [])
        total = 0.0
        for b in balances:
            total += float(b.get("available_balance", 0))
        return total

    @retry()
    def get_order_status(self, order_id: str) -> dict:
        path = f"/orders/{order_id}"
        url = self._url(path)
        headers = self._sign_headers("GET", path)
        resp = self._session.get(url, headers=headers)
        data = self._check(resp, url)
        return data.get("result", {})

    @retry()
    def cancel_order(self, order_id: str) -> bool:
        path = f"/orders/{order_id}"
        url = self._url(path)
        headers = self._sign_headers("DELETE", path)
        resp = self._session.delete(url, headers=headers)
        self._check(resp, url)
        logger.info("Delta order cancelled: %s", order_id)
        return True
