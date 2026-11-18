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
