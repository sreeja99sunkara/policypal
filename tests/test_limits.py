"""Rate limit, question length, demo-mode upload block. No LLM, no models:
the chain is replaced with a stub and the lifespan is not run."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import query as query_router
from app.services import rate_limit
from app.services.rate_limit import RateLimited, RateLimiter, client_ip


class StubChain:
    def __init__(self):
        self.calls = 0

    def invoke(self, question):
        self.calls += 1
        return {"answer": "Stub answer (Page 3).", "sources": [], "docs": []}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    monkeypatch.setenv("RATE_LIMIT_PER_DAY", "5")
    monkeypatch.setenv("MAX_QUESTION_CHARS", "500")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    rate_limit.reset_query_limiter()
    monkeypatch.setattr(query_router, "has_documents", lambda: True)
    stub = StubChain()
    app.dependency_overrides[query_router.get_chain] = lambda: stub
    # no `with`: the lifespan (model loading) does not run
    yield TestClient(app), stub
    app.dependency_overrides.clear()
    rate_limit.reset_query_limiter()


def ask(c, question="What is the co-payment?", ip=None):
    headers = {"X-Forwarded-For": ip} if ip else {}
    return c.post("/api/query", json={"question": question}, headers=headers)


# ---- limiter unit tests (fake clock) ----

class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_minute_window_blocks_then_recovers():
    clock = Clock()
    lim = RateLimiter(per_minute=2, per_day=100, clock=clock)
    lim.hit("a"); lim.hit("a")
    with pytest.raises(RateLimited) as e:
        lim.hit("a")
    assert e.value.retry_after == 60
    assert "2 per minute" in e.value.message
    clock.t += 61
    lim.hit("a")  # window rolled over


def test_day_window_and_rejections_not_counted():
    clock = Clock()
    lim = RateLimiter(per_minute=0, per_day=3, clock=clock)
    for _ in range(3):
        lim.hit("a"); clock.t += 1
    for _ in range(5):  # rejected attempts must not extend the block
        with pytest.raises(RateLimited) as e:
            lim.hit("a")
    assert "3 questions per day" in e.value.message
    clock.t = 1000.0 + 24 * 3600  # first hit expires
    lim.hit("a")


def test_keys_are_independent():
    lim = RateLimiter(per_minute=1, per_day=10, clock=Clock())
    lim.hit("a")
    lim.hit("b")
    with pytest.raises(RateLimited):
        lim.hit("a")


def test_client_ip_uses_rightmost_forwarded_entry():
    assert client_ip("6.6.6.6, 203.0.113.7", "10.0.0.1") == "203.0.113.7"
    assert client_ip("203.0.113.7", "10.0.0.1") == "203.0.113.7"
    assert client_ip(None, "10.0.0.1") == "10.0.0.1"
    assert client_ip(" , ", "10.0.0.1") == "10.0.0.1"
    assert client_ip(None, None) == "unknown"


# ---- endpoint tests ----

def test_query_rate_limited_with_friendly_429(client):
    c, stub = client
    for _ in range(3):
        assert ask(c, ip="203.0.113.7").status_code == 200
    r = ask(c, ip="203.0.113.7")
    assert r.status_code == 429
    assert "3 per minute" in r.json()["detail"]
    assert int(r.headers["Retry-After"]) > 0
    assert stub.calls == 3  # the blocked request never reached the chain
    # another client (by X-Forwarded-For) is unaffected
    assert ask(c, ip="198.51.100.9").status_code == 200


def test_spoofed_leftmost_forwarded_entry_does_not_reset_limit(client):
    c, _ = client
    for i in range(3):
        assert ask(c, ip=f"6.6.6.{i}, 203.0.113.7").status_code == 200
    assert ask(c, ip="6.6.6.99, 203.0.113.7").status_code == 429


def test_question_too_long_is_422_with_message(client):
    c, stub = client
    r = ask(c, question="x" * 501)
    assert r.status_code == 422
    assert "501 characters" in r.json()["detail"]
    assert "500" in r.json()["detail"]
    assert ask(c, question="x" * 500).status_code == 200
    assert stub.calls == 1


def test_too_long_question_does_not_use_quota(client):
    c, _ = client
    for _ in range(5):
        assert ask(c, question="x" * 600, ip="203.0.113.7").status_code == 422
    assert ask(c, ip="203.0.113.7").status_code == 200


def test_upload_refused_in_demo_mode(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("DEMO_MODE", "true")
    r = c.post("/api/upload", files={"file": ("p.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 403
    assert "disabled in this demo" in r.json()["detail"]
    assert c.get("/health").json() == {"status": "ok", "demo_mode": True}


def test_upload_not_blocked_when_demo_mode_off(client):
    c, _ = client
    # not a real upload: a non-PDF name fails validation, proving the 403 gate is off
    r = c.post("/api/upload", files={"file": ("p.txt", b"x", "text/plain")})
    assert r.status_code == 400
    assert c.get("/health").json()["demo_mode"] is False


# ---- global daily cap ----

from datetime import datetime, timedelta, timezone
import threading
import time as _time

from app.core import settings
from app.services.rate_limit import DailyCap, QuerySlots, Busy


def test_daily_cap_blocks_and_resets_at_utc_midnight():
    now = [datetime(2026, 11, 19, 21, 30, tzinfo=timezone.utc)]
    cap = DailyCap(2, now=lambda: now[0])
    cap.hit(); cap.hit()
    with pytest.raises(RateLimited) as e:
        cap.hit()
    assert "used its 2 questions for today" in e.value.message
    assert "00:00 UTC" in e.value.message and "3 hours" in e.value.message
    assert e.value.retry_after == int(2.5 * 3600)
    assert cap.used_today == 2  # the rejected attempt was not counted
    now[0] += timedelta(hours=3)  # next UTC day
    cap.hit()
    assert cap.used_today == 1


def test_daily_cap_defaults(monkeypatch):
    monkeypatch.delenv("DEMO_DAILY_QUERY_CAP", raising=False)
    monkeypatch.setenv("DEMO_MODE", "true")
    assert settings.demo_daily_query_cap() == 150
    monkeypatch.setenv("DEMO_MODE", "false")
    assert settings.demo_daily_query_cap() == 0  # off outside the demo unless set
    monkeypatch.setenv("DEMO_DAILY_QUERY_CAP", "7")
    assert settings.demo_daily_query_cap() == 7
    monkeypatch.delenv("MAX_CONCURRENT_QUERIES", raising=False)
    monkeypatch.delenv("QUERY_WAIT_SECONDS", raising=False)
    assert settings.max_concurrent_queries() == 1
    assert settings.query_wait_seconds() == 30.0


def test_daily_cap_endpoint_and_rejections_not_counted(client, monkeypatch):
    c, stub = client
    monkeypatch.setenv("DEMO_DAILY_QUERY_CAP", "2")
    rate_limit.reset_query_limiter()
    assert ask(c, question="x" * 600, ip="1.1.1.1").status_code == 422  # not counted
    assert ask(c, ip="1.1.1.1").status_code == 200
    assert ask(c, ip="2.2.2.2").status_code == 200
    r = ask(c, ip="3.3.3.3")
    assert r.status_code == 429
    assert "used its 2 questions for today" in r.json()["detail"]
    assert int(r.headers["Retry-After"]) > 0
    assert stub.calls == 2
    assert rate_limit.get_daily_cap().used_today == 2


def test_cap_rejection_does_not_use_client_quota():
    lim = RateLimiter(per_minute=1, per_day=10, clock=Clock())
    lim.hit("a")
    lim.unhit("a")  # what the router does when the daily cap rejects after hit()
    lim.hit("a")


# ---- concurrency guard ----

class BlockingChain(StubChain):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def invoke(self, question):
        self.started.set()
        self.release.wait(10)
        return super().invoke(question)


def test_busy_returns_503_and_is_not_counted(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("MAX_CONCURRENT_QUERIES", "1")
    monkeypatch.setenv("QUERY_WAIT_SECONDS", "0.3")
    monkeypatch.setenv("DEMO_DAILY_QUERY_CAP", "2")
    rate_limit.reset_query_limiter()
    chain = BlockingChain()
    app.dependency_overrides[query_router.get_chain] = lambda: chain

    first = {}
    t = threading.Thread(target=lambda: first.setdefault("r", ask(c, ip="1.1.1.1")))
    t.start()
    assert chain.started.wait(10)
    started = _time.monotonic()
    r = ask(c, ip="2.2.2.2")
    waited = _time.monotonic() - started
    assert r.status_code == 503
    assert "busy answering other questions" in r.json()["detail"]
    assert r.headers["Retry-After"] == "10"
    assert 0.25 <= waited < 5  # it waited for a slot before giving up
    chain.release.set()
    t.join(10)
    assert first["r"].status_code == 200
    # the 503 used neither the cap nor 2.2.2.2's quota
    assert rate_limit.get_daily_cap().used_today == 1
    assert ask(c, ip="2.2.2.2").status_code == 200


def test_waiting_request_runs_when_slot_frees(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("MAX_CONCURRENT_QUERIES", "1")
    monkeypatch.setenv("QUERY_WAIT_SECONDS", "5")
    rate_limit.reset_query_limiter()
    chain = BlockingChain()
    app.dependency_overrides[query_router.get_chain] = lambda: chain
    first = {}
    t = threading.Thread(target=lambda: first.setdefault("r", ask(c, ip="1.1.1.1")))
    t.start()
    assert chain.started.wait(10)
    threading.Timer(0.3, chain.release.set).start()
    r = ask(c, ip="2.2.2.2")  # waits about 0.3 s, then runs
    t.join(10)
    assert r.status_code == 200 and first["r"].status_code == 200
    assert chain.calls == 2


def test_query_slots_unit():
    slots = QuerySlots(2, wait_seconds=0.05)
    slots.acquire(); slots.acquire()
    with pytest.raises(Busy):
        slots.acquire()
    slots.release()
    slots.acquire()
