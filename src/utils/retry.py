"""
Retry utilities for handling transient failures.

Provides decorators and context managers for retrying operations
with exponential backoff, jitter, and circuit breaker patterns.

Usage:
    from utils.retry import retry, retry_with_backoff, CircuitBreaker

    @retry(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
    def fetch_data():
        ...

    @retry_with_backoff(base_delay=1.0, max_delay=60.0)
    def api_call():
        ...

    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    with breaker:
        risky_operation()
"""

import functools
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: Tuple[float, float] = (0.5, 1.5)
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
    on_retry: Optional[Callable[[Exception, int], None]] = None


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
    jitter_range: Tuple[float, float]
) -> float:
    """Calculate delay with exponential backoff and optional jitter."""
    delay = min(base_delay * (exponential_base ** attempt), max_delay)

    if jitter:
        jitter_mult = random.uniform(*jitter_range)
        delay *= jitter_mult

    return delay


def retry(
    max_attempts: int = 3,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """
    Simple retry decorator with fixed delay.

    Args:
        max_attempts: Maximum number of attempts
        exceptions: Tuple of exception types to catch
        on_retry: Callback called on each retry (exception, attempt_number)

    Example:
        @retry(max_attempts=3, exceptions=(ConnectionError,))
        def fetch():
            return requests.get(url)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}"
                        )
                        time.sleep(1.0)  # Fixed 1 second delay
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


def retry_with_backoff(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
) -> Callable:
    """
    Retry decorator with exponential backoff and jitter.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Base for exponential growth
        jitter: Whether to add random jitter
        exceptions: Tuple of exception types to catch
        on_retry: Callback (exception, attempt, delay)

    Example:
        @retry_with_backoff(max_attempts=5, base_delay=2.0)
        def api_call():
            return external_api.fetch()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        delay = calculate_delay(
                            attempt, base_delay, max_delay,
                            exponential_base, jitter, (0.5, 1.5)
                        )

                        if on_retry:
                            on_retry(e, attempt + 1, delay)

                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for failing fast on repeated errors.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing if service recovered

    Example:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        @breaker
        def external_call():
            return api.fetch()

        # Or as context manager:
        with breaker:
            api.fetch()
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 1

    # Internal state
    _failures: int = field(default=0, init=False)
    _last_failure_time: Optional[datetime] = field(default=None, init=False)
    _state: str = field(default='CLOSED', init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> str:
        """Get current circuit state, checking for recovery."""
        if self._state == 'OPEN' and self._last_failure_time:
            if datetime.now() - self._last_failure_time > timedelta(seconds=self.recovery_timeout):
                self._state = 'HALF_OPEN'
                self._half_open_calls = 0
        return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == 'HALF_OPEN':
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._reset()
        elif self._state == 'CLOSED':
            # Gradual recovery
            self._failures = max(0, self._failures - 1)

    def record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        self._failures += 1
        self._last_failure_time = datetime.now()

        if self._state == 'HALF_OPEN':
            self._state = 'OPEN'
            logger.warning(f"Circuit breaker reopened after failure: {exception}")
        elif self._failures >= self.failure_threshold:
            self._state = 'OPEN'
            logger.warning(
                f"Circuit breaker opened after {self._failures} failures: {exception}"
            )

    def _reset(self) -> None:
        """Reset to closed state."""
        self._failures = 0
        self._last_failure_time = None
        self._state = 'CLOSED'
        self._half_open_calls = 0
        logger.info("Circuit breaker reset to CLOSED")

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper

    def __enter__(self) -> 'CircuitBreaker':
        """Context manager entry."""
        if self.state == 'OPEN':
            raise CircuitBreakerOpen(
                f"Circuit breaker is OPEN, {self.failure_threshold} failures recorded"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val)
        return False  # Don't suppress exceptions

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.state == 'OPEN':
            raise CircuitBreakerOpen(
                f"Circuit breaker is OPEN, failing fast"
            )

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


# Convenience instances for common use cases
api_retry = retry_with_backoff(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    exceptions=(ConnectionError, TimeoutError, OSError)
)

yfinance_retry = retry_with_backoff(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    exceptions=(Exception,)  # yfinance can raise various exceptions
)
