"""
Enhanced circuit breaker implementation for IB API operations
"""

import asyncio
import functools
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, reject calls
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class ErrorCategory(Enum):
    """Error categorization for smart handling"""

    TRANSIENT = "TRANSIENT"  # Temporary errors (retry)
    PERMANENT = "PERMANENT"  # Permanent errors (don't retry)
    CONNECTION = "CONNECTION"  # Connection errors (reconnect)
    RATE_LIMIT = "RATE_LIMIT"  # Rate limiting (backoff)
    UNKNOWN = "UNKNOWN"  # Unknown errors


# IB error codes categorization
IB_ERROR_CATEGORIES = {
    # Transient errors - can retry
    ErrorCategory.TRANSIENT: {
        1100,  # Connectivity between IB and TWS has been lost
        1101,  # Connectivity between IB and TWS has been restored - data maintained
        1102,  # Connectivity between IB and TWS has been restored - data lost
        2103,  # Market data farm connection is broken
        2104,  # Market data farm connection is OK
        2105,  # HMDS data farm connection is broken
        2106,  # HMDS data farm connection is OK
        2107,  # HMDS data farm connection is inactive but should be available upon demand
        2108,  # Market data farm connection is inactive but should be available upon demand
    },
    # Connection errors - need reconnection
    ErrorCategory.CONNECTION: {
        502,  # Couldn't connect to TWS
        503,  # The TWS is out of date and must be upgraded
        504,  # Not connected
        1100,  # Connectivity lost
    },
    # Rate limit errors - need backoff
    ErrorCategory.RATE_LIMIT: {
        420,  # Pacing violation
        523,  # Too many requests
    },
    # Permanent errors - don't retry
    ErrorCategory.PERMANENT: {
        200,  # No security definition found
        201,  # Order rejected
        202,  # Order cancelled
        321,  # Error validating request
        322,  # Invalid contract
        323,  # Invalid order type
    },
}


def categorize_error(error_code: int, error_message: str) -> ErrorCategory:
    """Categorize error based on code and message"""
    for category, codes in IB_ERROR_CATEGORIES.items():
        if error_code in codes:
            return category

    # Analyze error message if code not found
    error_lower = error_message.lower()
    if any(term in error_lower for term in ["connection", "disconnect", "not connected"]):
        return ErrorCategory.CONNECTION
    elif any(term in error_lower for term in ["rate", "pacing", "too many"]):
        return ErrorCategory.RATE_LIMIT
    elif any(term in error_lower for term in ["invalid", "rejected", "not found"]):
        return ErrorCategory.PERMANENT

    return ErrorCategory.UNKNOWN


class CircuitBreakerStats:
    """Track circuit breaker statistics"""

    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        self.error_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self.last_failure_time: Optional[datetime] = None
        self.circuit_opened_count = 0

    def record_success(self):
        self.total_calls += 1
        self.successful_calls += 1

    def record_failure(self, category: ErrorCategory):
        self.total_calls += 1
        self.failed_calls += 1
        self.error_counts[category] += 1
        self.last_failure_time = datetime.utcnow()

    def record_rejection(self):
        self.rejected_calls += 1

    def get_success_rate(self) -> float:
        if self.total_calls == 0:
            return 100.0
        return (self.successful_calls / self.total_calls) * 100


class EnhancedCircuitBreaker:
    """Enhanced circuit breaker with error categorization and smart recovery"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        excluded_exceptions: Optional[Set[type]] = None,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions or set()

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self.consecutive_successes = 0

        self.stats = CircuitBreakerStats()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if asyncio.iscoroutinefunction(func):
            return asyncio.create_task(self._async_call(func, *args, **kwargs))
        else:
            return self._sync_call(func, *args, **kwargs)

    async def _async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.stats.record_rejection()
                raise CircuitOpenError(f"Circuit breaker is OPEN (will retry after {self._time_until_retry()}s)")

        if self.state == CircuitState.HALF_OPEN and self.half_open_calls >= self.half_open_max_calls:
            self.stats.record_rejection()
            raise CircuitOpenError("Circuit breaker is HALF_OPEN (max calls reached)")

        try:
            # Execute function
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1

            result = await func(*args, **kwargs)

            # Record success
            self._on_success()
            return result

        except Exception as e:
            # Check if exception should be excluded
            if type(e) in self.excluded_exceptions:
                raise

            # Categorize and handle error
            self._on_failure(e)
            raise

    def _sync_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with circuit breaker protection"""
        # Similar to async but without await
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.stats.record_rejection()
                raise CircuitOpenError(f"Circuit breaker is OPEN (will retry after {self._time_until_retry()}s)")

        if self.state == CircuitState.HALF_OPEN and self.half_open_calls >= self.half_open_max_calls:
            self.stats.record_rejection()
            raise CircuitOpenError("Circuit breaker is HALF_OPEN (max calls reached)")

        try:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1

            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            if type(e) in self.excluded_exceptions:
                raise
            self._on_failure(e)
            raise

    def _on_success(self):
        """Handle successful call"""
        self.stats.record_success()
        self.consecutive_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            if self.consecutive_successes >= self.half_open_max_calls:
                self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self, exception: Exception):
        """Handle failed call"""
        # Categorize error
        category = ErrorCategory.UNKNOWN
        if hasattr(exception, "code"):
            category = categorize_error(exception.code, str(exception))

        self.stats.record_failure(category)
        self.consecutive_successes = 0
        self.last_failure_time = datetime.utcnow()

        # Don't count permanent errors toward circuit opening
        if category != ErrorCategory.PERMANENT:
            self.failure_count += 1

            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()

        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()

    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.stats.circuit_opened_count += 1
        logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.failure_count = 0
        logger.info("Circuit breaker transitioned to HALF_OPEN")

    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        logger.info("Circuit breaker closed after successful recovery")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.utcnow() - self.last_failure_time).total_seconds() > self.recovery_timeout

    def _time_until_retry(self) -> int:
        """Calculate seconds until retry attempt"""
        if self.last_failure_time is None:
            return 0
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return max(0, int(self.recovery_timeout - elapsed))

    def get_status(self) -> dict:
        """Get circuit breaker status"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "consecutive_successes": self.consecutive_successes,
            "stats": {
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "rejected_calls": self.stats.rejected_calls,
                "success_rate": f"{self.stats.get_success_rate():.1f}%",
                "circuit_opened_count": self.stats.circuit_opened_count,
                "error_breakdown": dict(self.stats.error_counts),
            },
        }

    def get_stats(self) -> dict:
        """Get circuit breaker statistics in test-compatible format"""
        return {
            "state": self.state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "rejected_calls": self.stats.rejected_calls,
            "success_rate": self.stats.successful_calls / max(1, self.stats.total_calls),
            "error_categories": dict(self.stats.error_counts),
        }

    def reset(self):
        """Reset circuit breaker to initial state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.consecutive_successes = 0
        self.stats = CircuitBreakerStats()

    def __enter__(self):
        """Enter context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        if exc_type is not None:
            # An exception occurred
            self._on_failure(exc_val)
            return False  # Don't suppress the exception
        else:
            # No exception
            self._on_success()
            return False


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""

    pass


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    half_open_max_calls: int = 3,
    excluded_exceptions: Optional[Set[type]] = None,
):
    """Decorator to apply circuit breaker to functions"""
    breaker = EnhancedCircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        half_open_max_calls=half_open_max_calls,
        excluded_exceptions=excluded_exceptions,
    )

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker._async_call(func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return breaker._sync_call(func, *args, **kwargs)

        # Attach breaker for status access
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper.circuit_breaker = breaker
        return wrapper

    return decorator
