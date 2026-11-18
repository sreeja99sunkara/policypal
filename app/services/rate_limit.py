"""In-memory guards for POST /api/query: a per-client rate limiter, a global
daily query cap, and a concurrency limit.

One process, one instance: counts live in this process's memory and reset on
restart. Behind several replicas each would keep its own counts (use Redis or
the platform's limiter then).
"""
import math
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
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

    def check(self, key: str) -> None:
        """Raise RateLimited if `key` is over a limit; records nothing."""
        self._check_or_hit(key, record=False)

    def hit(self, key: str) -> None:
        """Record one request for `key`, or raise RateLimited without recording it."""
        self._check_or_hit(key, record=True)

    def unhit(self, key: str) -> None:
        """Remove the most recent hit for `key` (a request rejected after hit())."""
        with self._lock:
            hits = self._hits.get(key)
            if hits:
                hits.pop()

    def _check_or_hit(self, key: str, record: bool) -> None:
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
            if not record:
                return
            hits.append(now)

            self._calls += 1
            if self._calls % 1000 == 0:  # drop keys idle for a day so memory stays bounded
                for k in [k for k, d in self._hits.items() if not d or now - d[-1] >= DAY]:
                    del self._hits[k]


class DailyCap:
    """At most `cap` recorded queries per UTC calendar day, across all clients.
    A cap of 0 disables it."""

    def __init__(self, cap: int, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.cap = cap
        self._now = now
        self._day = None
        self._count = 0
        self._lock = threading.Lock()

    def _roll(self, now: datetime) -> None:
        if now.date() != self._day:
            self._day, self._count = now.date(), 0

    def check(self, record: bool = False) -> None:
        """Raise RateLimited if today's cap is used up; record one query if asked."""
        if not self.cap:
            return
        now = self._now()
        with self._lock:
            self._roll(now)
            if self._count >= self.cap:
                midnight = datetime.combine(
                    now.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
                )
                wait = (midnight - now).total_seconds()
                raise RateLimited(
                    f"This demo has used its {self.cap} questions for today. "
                    f"It resets at 00:00 UTC, in about {_human(wait)}.",
                    retry_after=math.ceil(wait),
                )
            if record:
                self._count += 1

    def hit(self) -> None:
        self.check(record=True)

    @property
    def used_today(self) -> int:
        with self._lock:
            self._roll(self._now())
            return self._count


class Busy(Exception):
    pass


class QuerySlots:
    """At most `size` queries run at once; others wait up to `wait_seconds`."""

    def __init__(self, size: int, wait_seconds: float):
        self.size = size
        self.wait_seconds = wait_seconds
        self._sem = threading.BoundedSemaphore(size)

    def acquire(self) -> None:
        if not self._sem.acquire(timeout=self.wait_seconds):
            raise Busy(
                "PolicyPal is busy answering other questions right now. "
                "Please try again in a moment."
            )

    def release(self) -> None:
        self._sem.release()


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
_daily_cap: Optional[DailyCap] = None
_slots: Optional[QuerySlots] = None


def get_query_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(settings.rate_limit_per_minute(), settings.rate_limit_per_day())
    return _limiter


def get_daily_cap() -> DailyCap:
    global _daily_cap
    if _daily_cap is None:
        _daily_cap = DailyCap(settings.demo_daily_query_cap())
    return _daily_cap


def get_query_slots() -> QuerySlots:
    global _slots
    if _slots is None:
        _slots = QuerySlots(settings.max_concurrent_queries(), settings.query_wait_seconds())
    return _slots


def reset_query_limiter() -> None:
    """Forget all counts and re-read every limit from the environment (tests)."""
    global _limiter, _daily_cap, _slots
    _limiter = _daily_cap = _slots = None
