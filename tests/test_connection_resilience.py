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

        # Test unknown errors
        assert categorize_error(9999, "Random error") == ErrorCategory.UNKNOWN


@pytest.mark.asyncio
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
