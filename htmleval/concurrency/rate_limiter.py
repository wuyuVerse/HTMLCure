"""
Token-bucket rate limiter and circuit breaker for browser launch control.
"""

from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """
    Async token-bucket rate limiter.

    rate:  tokens/second  (e.g. 2.0 = max 2 browser launches/sec)
    burst: max tokens that can accumulate (allows short bursts)
    """

    def __init__(self, rate: float = 2.0, burst: int = 1):
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            await asyncio.sleep(1.0 / self._rate)

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._burst, self._tokens + (now - self._last_refill) * self._rate)
        self._last_refill = now


class CircuitBreaker:
    """
    Simple circuit breaker: CLOSED → OPEN → HALF_OPEN.

    After `fail_threshold` consecutive failures, opens for `reset_timeout` seconds.
    One probe is allowed after timeout; success closes, failure re-opens.
    """

    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

    def __init__(self, fail_threshold: int = 5, reset_timeout: float = 60.0):
        self._fail_threshold = fail_threshold
        self._reset_timeout  = reset_timeout
        self._state          = self.CLOSED
        self._fail_count     = 0
        self._last_fail_time = 0.0
        self._lock           = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    async def __aenter__(self):
        async with self._lock:
            if self._state == self.OPEN:
                if time.monotonic() - self._last_fail_time > self._reset_timeout:
                    self._state = self.HALF_OPEN
                else:
                    raise CircuitBreakerOpen(f"circuit open, retry after {self._reset_timeout}s")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock:
            if exc_type is None:
                self._fail_count = 0
                self._state = self.CLOSED
            else:
                self._fail_count += 1
                self._last_fail_time = time.monotonic()
                if self._fail_count >= self._fail_threshold:
                    self._state = self.OPEN
        return False


class CircuitBreakerOpen(Exception):
    """Raised when a call is made through an open circuit breaker."""
