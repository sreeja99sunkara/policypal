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
