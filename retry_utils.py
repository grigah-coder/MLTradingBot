"""Retry / backoff helpers for flaky network calls and API rate limits.

Provides :func:`call_with_retry` and the :func:`retry` decorator. Both apply
**exponential backoff with full jitter** and understand the error types that
matter for this bot:

* network-level failures (``ConnectionError``/``Timeout``/requests errors),
* Alpaca ``APIError`` with a retryable HTTP status (429 rate-limit, 5xx), and
* Alpaca's internal ``RetryException``.

Permanent errors (e.g. 401/403/422) are *not* retried - they are re-raised
immediately so bugs surface fast instead of being masked.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)

# --- Optional Alpaca imports (kept soft so the module stays importable) -----
try:  # pragma: no cover - depends on install
    from alpaca.common.exceptions import APIError, RetryException # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    APIError = None  # type: ignore
    RetryException = None  # type: ignore

try:  # pragma: no cover
    import requests # pyright: ignore[reportMissingModuleSource]

    _REQUESTS_ERRORS: Tuple[Type[BaseException], ...] = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )
except Exception:  # pragma: no cover
    _REQUESTS_ERRORS = ()

# HTTP statuses worth retrying (rate limit + transient server errors).
RATE_LIMIT_CODES = frozenset({429, 503})
RETRYABLE_HTTP_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

_NETWORK_ERRORS: Tuple[Type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
) + _REQUESTS_ERRORS


def _status_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    if response is not None:
        rc = getattr(response, "status_code", None)
        if isinstance(rc, int):
            return rc
    return None


def is_rate_limit(exc: BaseException) -> bool:
    """True when ``exc`` looks like an API rate-limit response."""
    if _status_code(exc) in RATE_LIMIT_CODES:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "too many requests" in text


def is_retryable(exc: BaseException) -> bool:
    """Decide whether an exception is worth retrying.

    Uses duck-typing on a ``status_code``/``response.status_code`` attribute so
    it works with Alpaca's ``APIError`` as well as httpx/requests HTTP errors.
    """
    if RetryException is not None and isinstance(exc, RetryException):
        return True

    # Any exception exposing an HTTP status code is classified by that code.
    code = _status_code(exc)
    if code is not None:
        return code in RETRYABLE_HTTP_CODES

    # Alpaca APIError without a resolvable status: retry only if transient.
    if APIError is not None and isinstance(exc, APIError):
        return is_rate_limit(exc) or "timeout" in str(exc).lower()

    if isinstance(exc, _NETWORK_ERRORS):
        return True

    return False


def call_with_retry(
    func: Callable,
    *args,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    description: Optional[str] = None,
    retry_on: Optional[Callable[[BaseException], bool]] = None,
    **kwargs,
):
    """Call ``func`` retrying transient failures with exponential backoff.

    Re-raises the last exception once attempts are exhausted (or immediately
    for non-retryable errors).
    """
    attempts = max(1, int(attempts))
    predicate = retry_on or is_retryable
    desc = description or getattr(func, "__name__", "operation")
    last_exc: Optional[BaseException] = None

    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - we classify then re-raise
            if not predicate(exc):
                logger.debug("%s raised non-retryable %r", desc, exc)
                raise
            last_exc = exc
            if attempt >= attempts:
                break

            delay = min(max_delay, base_delay * (backoff_factor ** (attempt - 1)))
            if is_rate_limit(exc):
                # Respect rate limits more generously than plain net errors.
                delay = min(max_delay, delay * 2)
            if jitter:
                delay *= 0.5 + random.random()  # full jitter

            logger.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.2fs",
                desc,
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)

    logger.error("%s failed after %d attempts: %s", desc, attempts, last_exc)
    assert last_exc is not None
    raise last_exc


def retry(
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    description: Optional[str] = None,
    retry_on: Optional[Callable[[BaseException], bool]] = None,
):
    """Decorator form of :func:`call_with_retry`."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return call_with_retry(
                func,
                *args,
                attempts=attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                jitter=jitter,
                description=description or func.__name__,
                retry_on=retry_on,
                **kwargs,
            )

        return wrapper

    return decorator
