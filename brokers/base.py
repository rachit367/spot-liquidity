"""Abstract broker interface, retry decorator, and custom exceptions."""

from __future__ import annotations

import functools
import logging
import time
from abc import ABC, abstractmethod

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BrokerAPIError(Exception):
    """Raised when a broker API returns a non-2xx response."""

    def __init__(self, status_code: int, body: str, url: str = ""):
        self.status_code = status_code
        self.body = body
        self.url = url
        super().__init__(f"[{status_code}] {url} — {body[:300]}")


class InsufficientBalanceError(Exception):
    """Raised when account balance is too low to place a trade."""


class RiskLimitExceeded(Exception):
    """Raised when daily loss limit has been breached."""


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry(
    max_retries: int = 3,
    backoff_base: float = 1.0,
    retryable_exceptions: tuple = (requests.RequestException,),
    retryable_status_codes: tuple = (429, 500, 502, 503, 504),
):
    """Retry with exponential backoff for transient failures."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    result = fn(*args, **kwargs)
                    return result
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        wait = backoff_base * (2 ** attempt)
                        logger.warning(
                            "Retry %d/%d for %s — %s (waiting %.1fs)",
                            attempt + 1, max_retries, fn.__name__, exc, wait,
                        )
                        time.sleep(wait)
                except BrokerAPIError as exc:
                    if exc.status_code in retryable_status_codes and attempt < max_retries - 1:
                        last_exc = exc
                        wait = backoff_base * (2 ** attempt)
                        logger.warning(
                            "Retry %d/%d for %s — HTTP %d (waiting %.1fs)",
                            attempt + 1, max_retries, fn.__name__,
                            exc.status_code, wait,
                        )
                        time.sleep(wait)
                    else:
                        raise
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Abstract base broker
# ---------------------------------------------------------------------------

class BaseBroker(ABC):
    """
    Unified broker interface.  All broker implementations must subclass this
    and implement every abstract method.
    """

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: int,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> str:
        """Place an order. Returns the broker's order ID."""

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Return the last traded price for *symbol*."""

    @abstractmethod
    def get_ohlc(
        self, symbol: str, interval: str, count: int
    ) -> pd.DataFrame:
        """
        Return recent OHLC candles as a DataFrame with columns:
        [open, high, low, close, volume, timestamp].
        """

    @abstractmethod
    def get_balance(self) -> float:
        """Return available account balance (in base currency)."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict:
        """Return order details as a dict."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
