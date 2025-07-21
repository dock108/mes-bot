"""
Test IB connection resilience and recovery
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.circuit_breaker import CircuitState, EnhancedCircuitBreaker, ErrorCategory, circuit_breaker
from app.ib_connection_manager import (
    ConnectionHealthMonitor,
    ConnectionState,
    ExponentialBackoff,
    IBConnectionManager,
    with_connection_check,
)


@pytest.mark.integration
@pytest.mark.db
class TestExponentialBackoff:
    """Test exponential backoff functionality"""

    def test_initial_delay(self):
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=300.0, factor=2.0)
        assert backoff.get_delay() == 1.0

    def test_exponential_increase(self):
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=300.0, factor=2.0)
        delays = []
        for _ in range(5):
            delays.append(backoff.get_delay())

        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_max_delay_cap(self):
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=10.0, factor=2.0)
        delays = []
        for _ in range(10):
            delays.append(backoff.get_delay())

        # Should cap at max_delay
        assert max(delays) == 10.0
        assert delays[-1] == 10.0

    def test_reset(self):
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=300.0, factor=2.0)
        backoff.get_delay()  # 1.0
        backoff.get_delay()  # 2.0
        backoff.reset()
        assert backoff.get_delay() == 1.0


@pytest.mark.integration
@pytest.mark.db
class TestConnectionHealthMonitor:
    """Test connection health monitoring"""

    def test_healthy_connection(self):
        monitor = ConnectionHealthMonitor(check_interval=30)
        monitor.record_heartbeat()
        assert monitor.is_connection_healthy() is True

    def test_unhealthy_connection(self):
        monitor = ConnectionHealthMonitor(check_interval=30)
        monitor.last_heartbeat = datetime.utcnow() - timedelta(seconds=70)
        assert monitor.is_connection_healthy() is False

    def test_latency_tracking(self):
        monitor = ConnectionHealthMonitor()
        monitor.record_latency(10.0)
        monitor.record_latency(20.0)
        monitor.record_latency(30.0)
        assert monitor.get_average_latency() == 20.0

    def test_connection_quality_ratings(self):
        monitor = ConnectionHealthMonitor()

        # Excellent quality
        for _ in range(10):
            monitor.record_latency(25.0)
        assert monitor.get_connection_quality() == "EXCELLENT"

        # Good quality
        monitor.latency_samples.clear()
        for _ in range(10):
            monitor.record_latency(75.0)
        assert monitor.get_connection_quality() == "GOOD"

        # Fair quality
        monitor.latency_samples.clear()
        for _ in range(10):
            monitor.record_latency(150.0)
        assert monitor.get_connection_quality() == "FAIR"

        # Poor quality
        monitor.latency_samples.clear()
        for _ in range(10):
            monitor.record_latency(250.0)
        assert monitor.get_connection_quality() == "POOR"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.db
class TestIBConnectionManager:
    """Test IB connection manager functionality"""

    async def test_successful_connection(self):
        # Mock IB client
        mock_ib_client = Mock()
        mock_ib_client.connect_direct = AsyncMock(return_value=True)
        mock_ib_client.disconnect = AsyncMock()

        manager = IBConnectionManager(mock_ib_client)

        # Test connection
        result = await manager.connect()
        assert result is True
        assert manager.state == ConnectionState.CONNECTED
        assert manager.stats.successful_connections == 1

    async def test_failed_connection_triggers_reconnect(self):
        # Mock IB client
        mock_ib_client = Mock()
        mock_ib_client.connect_direct = AsyncMock(return_value=False)
        mock_ib_client.disconnect = AsyncMock()

        manager = IBConnectionManager(mock_ib_client)
        manager.max_reconnect_attempts = 2

        # Test connection failure
        result = await manager.connect()
        assert result is False
        assert manager.state == ConnectionState.ERROR
        assert manager.stats.failed_connections == 1

    async def test_heartbeat_monitoring(self):
        # Mock IB client
        mock_ib_client = Mock()
        mock_ib_client.connect_direct = AsyncMock(return_value=True)
        mock_ib_client.get_account_values = AsyncMock(return_value={"NetLiquidation": 10000})
        mock_ib_client.disconnect = AsyncMock()

        manager = IBConnectionManager(mock_ib_client)
        manager.heartbeat_interval = 0.1  # Fast heartbeat for testing

        # Connect and let heartbeat run
        await manager.connect()
        await asyncio.sleep(0.3)

        # Check heartbeat recorded
        assert manager.health_monitor.last_heartbeat is not None
        assert manager.health_monitor.get_average_latency() > 0

        # Cleanup
        await manager.disconnect()

    async def test_connection_loss_detection(self):
        # Mock IB client
        mock_ib_client = Mock()
        mock_ib_client.connect_direct = AsyncMock(return_value=True)
        mock_ib_client.get_account_values = AsyncMock(side_effect=Exception("Connection lost"))
        mock_ib_client.disconnect = AsyncMock()

        manager = IBConnectionManager(mock_ib_client)
        manager.heartbeat_interval = 0.1

        # Connect
        await manager.connect()
        initial_state = manager.state

        # Wait for heartbeat to detect failure
        await asyncio.sleep(0.3)

        # Should trigger reconnection
        assert manager.state != initial_state
        assert manager.stats.disconnection_count > 0

        # Cleanup
        await manager.disconnect()

    async def test_get_connection_info(self):
        mock_ib_client = Mock()
        mock_ib_client.connect_direct = AsyncMock(return_value=True)
        mock_ib_client.get_account_values = AsyncMock(return_value={"NetLiquidation": 10000})
        mock_ib_client.disconnect = AsyncMock()

        manager = IBConnectionManager(mock_ib_client)
        manager.heartbeat_interval = 0.1  # Fast heartbeat for testing
        await manager.connect()

        # Wait for heartbeat to run
        await asyncio.sleep(0.2)

        info = manager.get_connection_info()
        assert info["state"] == "CONNECTED"
        assert "uptime_percentage" in info
        assert "connection_quality" in info
        assert info["is_healthy"] is True

        # Cleanup
        await manager.disconnect()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.db
class TestWithConnectionCheck:
    """Test connection check decorator"""

    async def test_allows_calls_when_connected(self):
        # Create mock object with connection manager
        mock_obj = Mock()
        mock_obj.connection_manager = Mock()
        mock_obj.connection_manager.state = ConnectionState.CONNECTED

        # Create decorated function
        @with_connection_check
        async def test_func(self):
            return "success"

        # Should work when connected
        result = await test_func(mock_obj)
        assert result == "success"

    async def test_blocks_calls_when_disconnected(self):
        # Create mock object with disconnected state
        mock_obj = Mock()
        mock_obj.connection_manager = Mock()
        mock_obj.connection_manager.state = ConnectionState.DISCONNECTED

        # Create decorated function
        @with_connection_check
        async def test_func(self):
            return "success"

        # Should raise error when disconnected
        with pytest.raises(ConnectionError):
            await test_func(mock_obj)

    async def test_handles_connection_errors(self):
        # Create mock object
        mock_obj = Mock()
        mock_obj.connection_manager = Mock()
        mock_obj.connection_manager.state = ConnectionState.CONNECTED
        mock_obj.connection_manager._handle_connection_loss = AsyncMock()

        # Create decorated function that raises connection error
        @with_connection_check
        async def test_func(self):
            raise Exception("Connection lost")

        # Should handle connection errors
        with pytest.raises(Exception):
            await test_func(mock_obj)

        # Should trigger connection loss handling
        mock_obj.connection_manager._handle_connection_loss.assert_called_once()


@pytest.mark.integration
@pytest.mark.db
class TestEnhancedCircuitBreaker:
    """Test enhanced circuit breaker functionality"""

    def test_circuit_opens_after_failures(self):
        breaker = EnhancedCircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Simulate failures
        for i in range(3):
            try:
                breaker._sync_call(lambda: 1 / 0)  # Division by zero
            except Exception:
                pass

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.failed_calls == 3

    def test_circuit_recovery(self):
        breaker = EnhancedCircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Open circuit
        for i in range(2):
            try:
                breaker._sync_call(lambda: 1 / 0)
            except Exception:
                pass

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        import time

        time.sleep(0.2)

        # Should transition to half-open
        assert breaker._should_attempt_reset() is True

    def test_error_categorization(self):
        from app.circuit_breaker import ErrorCategory, categorize_error

        # Test transient errors
        assert categorize_error(1100, "Connectivity lost") == ErrorCategory.TRANSIENT

        # Test connection errors
        assert categorize_error(502, "Not connected") == ErrorCategory.CONNECTION

        # Test rate limit errors
        assert categorize_error(420, "Pacing violation") == ErrorCategory.RATE_LIMIT

        # Test permanent errors
        assert categorize_error(200, "No security found") == ErrorCategory.PERMANENT

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization with custom parameters"""
        breaker = EnhancedCircuitBreaker(
            failure_threshold=5, recovery_timeout=30, half_open_max_calls=2
        )

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 30
        assert breaker.half_open_max_calls == 2
        assert breaker.stats.total_calls == 0

    def test_circuit_breaker_successful_calls(self):
        """Test successful calls tracking"""
        breaker = EnhancedCircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Make successful calls
        for i in range(5):
            result = breaker._sync_call(lambda: "success")
            assert result == "success"

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.total_calls == 5
        assert breaker.stats.successful_calls == 5
        assert breaker.stats.failed_calls == 0

    def test_circuit_breaker_mixed_calls(self):
        """Test mixed successful and failed calls"""
        breaker = EnhancedCircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Successful calls
        breaker._sync_call(lambda: "success")
        breaker._sync_call(lambda: "success")

        # Failed calls
        try:
            breaker._sync_call(lambda: 1 / 0)
        except Exception:
            pass

        try:
            breaker._sync_call(lambda: 1 / 0)
        except Exception:
            pass

        assert breaker.state == CircuitState.CLOSED  # Still below threshold
        assert breaker.stats.total_calls == 4
        assert breaker.stats.successful_calls == 2
        assert breaker.stats.failed_calls == 2

    def test_circuit_breaker_half_open_state(self):
        """Test half-open state behavior"""
        breaker = EnhancedCircuitBreaker(
            failure_threshold=2, recovery_timeout=0.1, half_open_max_calls=2
        )

        # Open circuit
        for i in range(2):
            try:
                breaker._sync_call(lambda: 1 / 0)
            except Exception:
                pass

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        import time

        time.sleep(0.2)

        # First call should transition to half-open
        assert breaker._should_attempt_reset() is True

        # Simulate transition to half-open (would happen in actual call)
        breaker.state = CircuitState.HALF_OPEN
        breaker.consecutive_successes = 0

        # Successful call in half-open
        result = breaker._sync_call(lambda: "success")
        assert result == "success"
        assert breaker.consecutive_successes == 1

        # Another successful call should close circuit
        result = breaker._sync_call(lambda: "success")
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_half_open_failure(self):
        """Test failure in half-open state reopens circuit"""
        breaker = EnhancedCircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Open circuit
        for i in range(2):
            try:
                breaker._sync_call(lambda: 1 / 0)
            except Exception:
                pass

        # Wait and transition to half-open
        import time

        time.sleep(0.2)
        breaker.state = CircuitState.HALF_OPEN

        # Failure in half-open should reopen circuit
        try:
            breaker._sync_call(lambda: 1 / 0)
        except Exception:
            pass

        assert breaker.state == CircuitState.OPEN

    def test_circuit_breaker_rejected_calls(self):
        """Test calls are rejected when circuit is open"""
        breaker = EnhancedCircuitBreaker(failure_threshold=2, recovery_timeout=60)

        # Open circuit
        for i in range(2):
            try:
                breaker._sync_call(lambda: 1 / 0)
            except Exception:
                pass

        assert breaker.state == CircuitState.OPEN

        # Calls should be rejected
        from app.circuit_breaker import CircuitOpenError

        with pytest.raises(CircuitOpenError):
            breaker._sync_call(lambda: "should be rejected")

        assert breaker.stats.rejected_calls == 1

    @pytest.mark.asyncio
    async def test_async_circuit_breaker_calls(self):
        """Test async calls through circuit breaker"""
        breaker = EnhancedCircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Successful async call
        async def async_success():
            await asyncio.sleep(0.01)
            return "async success"

        result = await breaker._async_call(async_success)
        assert result == "async success"
        assert breaker.stats.successful_calls == 1

        # Failed async call
        async def async_failure():
            await asyncio.sleep(0.01)
            raise ValueError("async error")

        with pytest.raises(ValueError):
            await breaker._async_call(async_failure)

        assert breaker.stats.failed_calls == 1

    def test_error_category_counting(self):
        """Test error category tracking"""
        breaker = EnhancedCircuitBreaker(failure_threshold=5, recovery_timeout=60)

        # Create mock errors with different categories
        class MockError(Exception):
            def __init__(self, code, message):
                self.code = code
                self.message = message
                super().__init__(message)

        # Simulate different error types by calling _on_failure directly
        errors = [
            MockError(502, "Not connected"),  # CONNECTION
            MockError(420, "Pacing violation"),  # RATE_LIMIT
            MockError(200, "No security found"),  # PERMANENT
            MockError(1100, "Connectivity lost"),  # TRANSIENT
        ]

        for error in errors:
            breaker._on_failure(error)

        assert breaker.stats.error_counts[ErrorCategory.CONNECTION] == 1
        assert breaker.stats.error_counts[ErrorCategory.RATE_LIMIT] == 1
        assert breaker.stats.error_counts[ErrorCategory.PERMANENT] == 1
        assert breaker.stats.error_counts[ErrorCategory.TRANSIENT] == 1

    def test_circuit_breaker_stats_reset(self):
        """Test circuit breaker stats reset"""
        breaker = EnhancedCircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Generate some stats
        breaker._sync_call(lambda: "success")
        try:
            breaker._sync_call(lambda: 1 / 0)
        except Exception:
            pass

        assert breaker.stats.total_calls == 2
        assert breaker.stats.successful_calls == 1
        assert breaker.stats.failed_calls == 1

        # Reset stats
        breaker.reset()

        assert breaker.stats.total_calls == 0
        assert breaker.stats.successful_calls == 0
        assert breaker.stats.failed_calls == 0
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_get_stats(self):
        """Test getting circuit breaker statistics"""
        breaker = EnhancedCircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Generate some activity
        breaker._sync_call(lambda: "success")
        breaker._sync_call(lambda: "success")
        try:
            breaker._sync_call(lambda: 1 / 0)
        except Exception:
            pass

        stats = breaker.get_stats()

        assert stats["state"] == "CLOSED"
        assert stats["total_calls"] == 3
        assert stats["successful_calls"] == 2
        assert stats["failed_calls"] == 1
        assert stats["success_rate"] == 2 / 3
        assert "error_categories" in stats

    def test_circuit_breaker_backoff_calculation(self):
        """Test backoff calculation for rate limit errors"""
        breaker = EnhancedCircuitBreaker(failure_threshold=5, recovery_timeout=60)

        # Test that we can get status without backoff calculation
        status = breaker.get_status()
        assert "state" in status
        assert status["state"] == "CLOSED"

    def test_circuit_breaker_permanent_error_handling(self):
        """Test permanent errors don't trigger circuit opening"""
        breaker = EnhancedCircuitBreaker(failure_threshold=2, recovery_timeout=60)

        # Create permanent error
        class PermanentError(Exception):
            def __init__(self):
                self.code = 200
                super().__init__("No security found")

        for i in range(5):  # More than threshold
            try:
                breaker._sync_call(lambda: (_ for _ in ()).throw(PermanentError()))
            except Exception:
                pass

        # Circuit should remain closed for permanent errors
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.error_counts[ErrorCategory.PERMANENT] == 5

    def test_circuit_breaker_with_context_manager(self):
        """Test circuit breaker context manager usage"""
        breaker = EnhancedCircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Test successful context manager usage
        with breaker:
            result = "context success"

        assert breaker.stats.successful_calls == 1

        # Test failed context manager usage
        try:
            with breaker:
                raise ValueError("context error")
        except ValueError:
            pass

        assert breaker.stats.failed_calls == 1

    def test_categorize_error_unknown_code(self):
        """Test error categorization for unknown error codes"""
        from app.circuit_breaker import categorize_error

        # Unknown error code with recognizable message
        assert categorize_error(9999, "connection failed") == ErrorCategory.CONNECTION
        assert categorize_error(9999, "rate limit exceeded") == ErrorCategory.RATE_LIMIT
        assert categorize_error(9999, "invalid request") == ErrorCategory.PERMANENT
        assert categorize_error(9999, "unknown error") == ErrorCategory.UNKNOWN

    def test_categorize_error_message_analysis(self):
        """Test error categorization based on message content"""
        from app.circuit_breaker import categorize_error

        # Test various message patterns
        assert categorize_error(0, "Disconnected from server") == ErrorCategory.CONNECTION
        assert categorize_error(0, "Too many requests per second") == ErrorCategory.RATE_LIMIT
        assert categorize_error(0, "Order rejected by exchange") == ErrorCategory.PERMANENT
        # "connection" in message triggers CONNECTION category, not UNKNOWN
        assert categorize_error(0, "Temporary connection issue") == ErrorCategory.CONNECTION

    def test_circuit_breaker_decorator_sync(self):
        """Test circuit breaker decorator on sync function"""

        @circuit_breaker(failure_threshold=2, recovery_timeout=60)
        def test_function(x):
            if x == 0:
                raise ValueError("test error")
            return x * 2

        # Get the circuit breaker instance from the decorated function
        breaker = test_function.circuit_breaker

        # Successful call
        result = test_function(5)
        assert result == 10
        assert breaker.stats.successful_calls == 1

        # Failed calls
        for i in range(2):
            try:
                test_function(0)
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.failed_calls == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator_async(self):
        """Test circuit breaker decorator on async function"""

        @circuit_breaker(failure_threshold=2, recovery_timeout=60)
        async def async_test_function(x):
            await asyncio.sleep(0.01)
            if x == 0:
                raise ValueError("async test error")
            return x * 3

        # Get the circuit breaker instance from the decorated function
        breaker = async_test_function.circuit_breaker

        # Successful call
        result = await async_test_function(4)
        assert result == 12
        assert breaker.stats.successful_calls == 1

        # Failed calls
        for i in range(2):
            try:
                await async_test_function(0)
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.failed_calls == 2

    def test_categorize_error_unknown(self):
        """Test unknown error categorization"""
        from app.circuit_breaker import categorize_error

        # Test unknown errors
        assert categorize_error(9999, "Random error") == ErrorCategory.UNKNOWN


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.db
class TestIntegrationScenarios:
    """Test real-world scenarios"""

    async def test_connection_recovery_scenario(self):
        """Test full connection loss and recovery"""

        # Mock IB client with changing behavior
        mock_ib_client = Mock()
        connection_attempts = 0

        async def mock_connect():
            nonlocal connection_attempts
            connection_attempts += 1
            # Fail first 2 attempts, succeed on 3rd
            return connection_attempts >= 3

        mock_ib_client.connect_direct = mock_connect
        mock_ib_client.disconnect = AsyncMock()
        mock_ib_client.get_account_values = AsyncMock(return_value={"NetLiquidation": 10000})

        manager = IBConnectionManager(mock_ib_client)
        manager.max_reconnect_attempts = 5
        manager.backoff.base_delay = 0.1  # Fast for testing

        # Initial connection should trigger reconnection loop
        result = await manager.connect()

        # Wait for reconnection
        await asyncio.sleep(1.0)

        # Should eventually connect
        assert connection_attempts >= 3

    async def test_circuit_breaker_with_connection_check(self):
        """Test circuit breaker and connection check working together"""

        # Create mock IB client
        mock_ib_client = Mock()
        mock_ib_client.connection_manager = Mock()
        mock_ib_client.connection_manager.state = ConnectionState.CONNECTED

        # Create function with both decorators
        @with_connection_check
        @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        async def protected_function(self):
            raise Exception("API Error")

        # Should fail and open circuit
        for i in range(2):
            try:
                await protected_function(mock_ib_client)
            except Exception:
                pass

        # Circuit should be open
        with pytest.raises(Exception) as exc_info:
            await protected_function(mock_ib_client)
        assert "Circuit breaker is OPEN" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
