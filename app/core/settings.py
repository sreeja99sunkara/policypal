"""Deployment settings read from environment variables.

Read on each call (not cached at import) so tests and restarts pick up changes.
"""
import os

_TRUE = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in _TRUE


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def demo_mode() -> bool:
    """DEMO_MODE=true: fixed policy, uploads refused."""
    return env_flag("DEMO_MODE", False)


def default_policy_pdf() -> str | None:
    """PDF to index on startup when the vector store is empty (DEFAULT_POLICY_PDF)."""
    path = os.getenv("DEFAULT_POLICY_PDF", "").strip()
    return path or None


def max_question_chars() -> int:
    return env_int("MAX_QUESTION_CHARS", 500)


def rate_limit_per_minute() -> int:
    """Queries per client IP per rolling minute; 0 disables the minute limit."""
    return env_int("RATE_LIMIT_PER_MINUTE", 10)


def rate_limit_per_day() -> int:
    """Queries per client IP per rolling 24 hours; 0 disables the daily limit."""
    return env_int("RATE_LIMIT_PER_DAY", 100)


def demo_daily_query_cap() -> int:
    """Total answered queries per UTC day across all clients (protects the Groq
    key). Default 150 in DEMO_MODE, off otherwise; set 0 to disable."""
    return env_int("DEMO_DAILY_QUERY_CAP", 150 if demo_mode() else 0)


def max_concurrent_queries() -> int:
    """Queries answered at the same time; extra requests wait for a free slot."""
    return max(1, env_int("MAX_CONCURRENT_QUERIES", 1))


def query_wait_seconds() -> float:
    """How long a request waits for a free slot before getting a 503."""
    try:
        return max(0.0, float(os.getenv("QUERY_WAIT_SECONDS") or 30))
    except ValueError:
        return 30.0
