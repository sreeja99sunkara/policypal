"""In-memory per-client rate limiter for POST /api/query.

One process, one instance: counts live in this process's memory and reset on
restart. Behind several replicas each would keep its own counts (use Redis or
the platform's limiter then).
"""
import math
import threading
import time
from collections import deque
from typing import Callable, Optional

from app.core import settings

MINUTE = 60.0
DAY = 24 * 60 * 60.0


class RateLimited(Exception):
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


class RateLimiter:
    """Rolling-window limits: at most `per_minute` hits in any 60 s and
    `per_day` hits in any 24 h per key. A limit of 0 disables that window."""

    def __init__(self, per_minute: int, per_day: int, clock: Callable[[], float] = time.monotonic):
        self.per_minute = per_minute
        self.per_day = per_day
        self._clock = clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def hit(self, key: str) -> None:
        """Record one request for `key`, or raise RateLimited without recording it."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= DAY:
                hits.popleft()

            if self.per_day and len(hits) >= self.per_day:
                wait = DAY - (now - hits[0])
                raise RateLimited(
                    f"Daily limit reached: {self.per_day} questions per day. "
                    f"Please try again in about {_human(wait)}.",
                    retry_after=math.ceil(wait),
                )
            if self.per_minute:
                recent = [t for t in hits if now - t < MINUTE]
                if len(recent) >= self.per_minute:
                    wait = MINUTE - (now - recent[0])
                    raise RateLimited(
                        f"Too many questions: the limit is {self.per_minute} per minute. "
                        f"Please wait {math.ceil(wait)} seconds and try again.",
                        retry_after=math.ceil(wait),
                    )
            hits.append(now)

            self._calls += 1
            if self._calls % 1000 == 0:  # drop keys idle for a day so memory stays bounded
                for k in [k for k, d in self._hits.items() if not d or now - d[-1] >= DAY]:
                    del self._hits[k]


def _human(seconds: float) -> str:
    if seconds >= 3600:
        return f"{math.ceil(seconds / 3600)} hours"
    return f"{max(1, math.ceil(seconds / 60))} minutes"


def client_ip(x_forwarded_for: Optional[str], peer: Optional[str]) -> str:
    """Client address for rate limiting.

    Honours X-Forwarded-For: the right-most entry is used, because that is the
    one appended by the proxy directly in front of the app; entries further left
    come from the client and can be forged to dodge the limit. Falls back to the
    TCP peer address.
    """
    if x_forwarded_for:
        parts = [p.strip() for p in x_forwarded_for.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return peer or "unknown"


_limiter: Optional[RateLimiter] = None


def get_query_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(settings.rate_limit_per_minute(), settings.rate_limit_per_day())
    return _limiter


def reset_query_limiter() -> None:
    """Forget all counts and re-read the limits from the environment (tests)."""
    global _limiter
    _limiter = None
