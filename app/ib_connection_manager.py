"""
Enhanced IB connection management with resilience and auto-reconnection
"""

import asyncio
import functools
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

from app.config import config

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration"""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


class ConnectionStats:
    """Track connection statistics"""

    def __init__(self):
        self.connection_attempts = 0
        self.successful_connections = 0
        self.failed_connections = 0
        self.total_downtime_seconds = 0
        self.last_connected_time: Optional[datetime] = None
        self.last_disconnected_time: Optional[datetime] = None
        self.disconnection_count = 0

    def record_connection_attempt(self):
        self.connection_attempts += 1

    def record_successful_connection(self):
        self.successful_connections += 1
        self.last_connected_time = datetime.utcnow()

    def record_failed_connection(self):
        self.failed_connections += 1

    def record_disconnection(self):
        self.disconnection_count += 1
        self.last_disconnected_time = datetime.utcnow()

    def get_uptime_percentage(self) -> float:
        """Calculate uptime percentage"""
        if self.last_connected_time is None:
            return 0.0

        total_time = (datetime.utcnow() - self.last_connected_time).total_seconds()
        if total_time == 0:
            return 0.0

        uptime = (total_time - self.total_downtime_seconds) / total_time * 100
        return max(0.0, min(100.0, uptime))


class ExponentialBackoff:
    """Exponential backoff for reconnection attempts"""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 300.0, factor: float = 2.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.factor = factor
        self.attempt = 0

    def get_delay(self) -> float:
        """Get next delay in seconds"""
        delay = min(self.base_delay * (self.factor**self.attempt), self.max_delay)
        self.attempt += 1
        return delay

    def reset(self):
        """Reset backoff counter"""
        self.attempt = 0


class ConnectionHealthMonitor:
    """Monitor connection health and quality"""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.last_heartbeat: Optional[datetime] = None
        self.missed_heartbeats = 0
        self.latency_samples = []
        self.max_latency_samples = 100

    def record_heartbeat(self):
        """Record successful heartbeat"""
        self.last_heartbeat = datetime.utcnow()
        self.missed_heartbeats = 0

    def record_latency(self, latency_ms: float):
        """Record API call latency"""
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > self.max_latency_samples:
            self.latency_samples.pop(0)

    def is_connection_healthy(self) -> bool:
        """Check if connection is healthy"""
        if self.last_heartbeat is None:
            return False

        time_since_heartbeat = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat < self.check_interval * 2

    def get_average_latency(self) -> float:
        """Get average latency in milliseconds"""
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)

    def get_connection_quality(self) -> str:
        """Get connection quality assessment"""
        avg_latency = self.get_average_latency()

        if avg_latency < 50:
            return "EXCELLENT"
        elif avg_latency < 100:
            return "GOOD"
        elif avg_latency < 200:
            return "FAIR"
        else:
            return "POOR"


class IBConnectionManager:
    """Enhanced IB connection manager with resilience"""

    def __init__(self, ib_client):
        self.ib_client = ib_client
        self.state = ConnectionState.DISCONNECTED
        self.backoff = ExponentialBackoff()
        self.health_monitor = ConnectionHealthMonitor()
        self.stats = ConnectionStats()

        # Configuration
        self.max_reconnect_attempts = getattr(config.ib, "max_reconnect_attempts", 10)
        self.reconnect_delay_base = getattr(config.ib, "reconnect_delay_base", 1.0)
        self.reconnect_delay_max = getattr(config.ib, "reconnect_delay_max", 300.0)
        self.heartbeat_interval = getattr(config.ib, "heartbeat_interval", 30)

        # Callbacks
        self.on_connected_callback: Optional[Callable] = None
        self.on_disconnected_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None

        # Tasks
        self.reconnect_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Connect with automatic retry and resilience"""
        if self.state == ConnectionState.CONNECTED:
            logger.info("Already connected to IB")
            return True

        self.state = ConnectionState.CONNECTING
        self.stats.record_connection_attempt()

        try:
            # Attempt connection using direct method to avoid recursion
            success = await self.ib_client.connect_direct()

            if success:
                self.state = ConnectionState.CONNECTED
                self.stats.record_successful_connection()
                self.backoff.reset()

                # Start health monitoring
                self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                # Call connected callback
                if self.on_connected_callback:
                    await self.on_connected_callback()

                logger.info("Successfully connected to IB with resilience enabled")
                return True
            else:
                self.state = ConnectionState.ERROR
                self.stats.record_failed_connection()

                # Start reconnection attempts
                await self._start_reconnection()
                return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.state = ConnectionState.ERROR
            self.stats.record_failed_connection()

            # Start reconnection attempts
            await self._start_reconnection()
            return False

    async def disconnect(self):
        """Disconnect and cleanup"""
        self.state = ConnectionState.DISCONNECTED

        # Cancel tasks
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()

        if self.reconnect_task and not self.reconnect_task.done():
            self.reconnect_task.cancel()

        # Disconnect IB client
        await self.ib_client.disconnect()

        # Call disconnected callback
        if self.on_disconnected_callback:
            await self.on_disconnected_callback()

    async def _start_reconnection(self):
        """Start automatic reconnection process"""
        if self.reconnect_task and not self.reconnect_task.done():
            return  # Already reconnecting

        self.reconnect_task = asyncio.create_task(self._reconnection_loop())

    async def _reconnection_loop(self):
        """Reconnection loop with exponential backoff"""
        self.state = ConnectionState.RECONNECTING
        reconnect_attempts = 0

        while reconnect_attempts < self.max_reconnect_attempts:
            delay = self.backoff.get_delay()
            logger.info(
                f"Attempting reconnection in {delay:.1f} seconds "
                f"(attempt {reconnect_attempts + 1}/{self.max_reconnect_attempts})"
            )

            await asyncio.sleep(delay)

            try:
                # Use connect_direct to avoid recursion
                success = await self.ib_client.connect_direct()
                if success:
                    logger.info("Reconnection successful")
                    return

            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")

            reconnect_attempts += 1

        logger.error(f"Failed to reconnect after {self.max_reconnect_attempts} attempts")
        self.state = ConnectionState.ERROR

        # Call error callback
        if self.on_error_callback:
            await self.on_error_callback("Max reconnection attempts exceeded")

    async def _heartbeat_loop(self):
        """Heartbeat monitoring loop"""
        while self.state == ConnectionState.CONNECTED:
            try:
                # Perform heartbeat check
                start_time = datetime.utcnow()

                # Simple check - get account values
                account_values = await self.ib_client.get_account_values()

                if account_values:
                    # Calculate latency
                    latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    self.health_monitor.record_latency(latency_ms)
                    self.health_monitor.record_heartbeat()

                    logger.debug(f"Heartbeat successful (latency: {latency_ms:.1f}ms)")
                else:
                    logger.warning("Heartbeat check returned no data")
                    self.health_monitor.missed_heartbeats += 1

                    if self.health_monitor.missed_heartbeats > 3:
                        logger.error("Multiple heartbeats missed, initiating reconnection")
                        await self._handle_connection_loss()
                        return

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await self._handle_connection_loss()
                return

            await asyncio.sleep(self.heartbeat_interval)

    async def _handle_connection_loss(self):
        """Handle detected connection loss"""
        if self.state != ConnectionState.CONNECTED:
            return  # Already handling

        logger.warning("Connection loss detected")
        self.state = ConnectionState.DISCONNECTED
        self.stats.record_disconnection()

        # Disconnect and attempt reconnection
        await self.ib_client.disconnect()
        await self._start_reconnection()

    def get_connection_info(self) -> dict:
        """Get connection information and statistics"""
        return {
            "state": self.state.value,
            "uptime_percentage": self.stats.get_uptime_percentage(),
            "connection_quality": self.health_monitor.get_connection_quality(),
            "average_latency_ms": self.health_monitor.get_average_latency(),
            "total_connections": self.stats.successful_connections,
            "total_disconnections": self.stats.disconnection_count,
            "last_connected": (
                self.stats.last_connected_time.isoformat()
                if self.stats.last_connected_time
                else None
            ),
            "is_healthy": self.health_monitor.is_connection_healthy(),
        }


def with_connection_check(func):
    """Decorator to check connection before executing IB operations"""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not hasattr(self, "connection_manager"):
            # Fallback to original behavior if no connection manager
            return await func(self, *args, **kwargs)

        if self.connection_manager.state != ConnectionState.CONNECTED:
            raise ConnectionError(
                f"IB not connected (state: {self.connection_manager.state.value})"
            )

        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            # Check if it's a connection-related error
            error_str = str(e).lower()
            if any(term in error_str for term in ["disconnect", "connection", "not connected"]):
                logger.warning(f"Connection error detected in {func.__name__}: {e}")
                await self.connection_manager._handle_connection_loss()
            raise

    return wrapper
