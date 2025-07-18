"""
Tests for structured logging service with correlation IDs
"""

import json
import logging
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from app.logging_service import (
    CorrelationIdFilter,
    CorrelationIdManager,
    StructuredFormatter,
    StructuredLogger,
    get_logger,
    setup_structured_logging,
    with_correlation_id,
)


class TestCorrelationIdManager:
    """Test correlation ID management"""

    def test_generate_id(self):
        """Test correlation ID generation"""
        id1 = CorrelationIdManager.generate_id()
        id2 = CorrelationIdManager.generate_id()

        assert id1 != id2
        assert len(id1) == 36  # UUID4 length
        assert "-" in id1

    def test_set_and_get_id(self):
        """Test setting and getting correlation ID"""
        test_id = "test-correlation-id"

        # Initially should be None
        assert CorrelationIdManager.get_current_id() is None

        # Set and verify
        CorrelationIdManager.set_id(test_id)
        assert CorrelationIdManager.get_current_id() == test_id

        # Clear and verify
        CorrelationIdManager.clear_id()
        assert CorrelationIdManager.get_current_id() is None

    def test_nested_ids(self):
        """Test nested correlation ID contexts"""
        outer_id = "outer-id"
        inner_id = "inner-id"

        CorrelationIdManager.set_id(outer_id)
        assert CorrelationIdManager.get_current_id() == outer_id

        CorrelationIdManager.set_id(inner_id)
        assert CorrelationIdManager.get_current_id() == inner_id

        CorrelationIdManager.clear_id()
        assert CorrelationIdManager.get_current_id() is None


class TestStructuredFormatter:
    """Test structured JSON formatter"""

    def test_basic_formatting(self):
        """Test basic log formatting"""
        formatter = StructuredFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-id"

        # Format the record
        formatted = formatter.format(record)

        # Parse JSON
        log_data = json.loads(formatted)

        # Verify structure
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["correlation_id"] == "test-id"
        assert log_data["module"] == "test"
        assert log_data["line"] == 10
        assert "timestamp" in log_data

    def test_extra_fields(self):
        """Test extra fields in log record"""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-id"
        record.trade_signal = "BUY"
        record.market_price = 4200.0

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        # Verify extra fields
        assert "extra" in log_data
        assert log_data["extra"]["trade_signal"] == "BUY"
        assert log_data["extra"]["market_price"] == 4200.0

    def test_exception_formatting(self):
        """Test exception formatting"""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )
            record.correlation_id = "test-id"

            formatted = formatter.format(record)
            log_data = json.loads(formatted)

            # Verify exception info
            assert "exception" in log_data
            assert "ValueError" in log_data["exception"]
            assert "Test exception" in log_data["exception"]


class TestCorrelationIdFilter:
    """Test correlation ID filter"""

    def test_filter_adds_correlation_id(self):
        """Test that filter adds correlation ID to records"""
        filter_instance = CorrelationIdFilter()

        # Set correlation ID
        CorrelationIdManager.set_id("test-correlation-id")

        # Create record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        # Apply filter
        filter_instance.filter(record)

        # Verify correlation ID was added
        assert record.correlation_id == "test-correlation-id"

        # Clean up
        CorrelationIdManager.clear_id()

    def test_filter_handles_no_correlation_id(self):
        """Test filter when no correlation ID is set"""
        filter_instance = CorrelationIdFilter()

        # Ensure no correlation ID is set
        CorrelationIdManager.clear_id()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        filter_instance.filter(record)

        # Should default to "unknown"
        assert record.correlation_id == "unknown"


class TestStructuredLogger:
    """Test structured logger"""

    def test_structured_logger_creation(self):
        """Test creating structured logger"""
        logger = StructuredLogger("test.logger")
        assert logger.logger.name == "test.logger"

        # Verify filter is added
        assert any(isinstance(f, CorrelationIdFilter) for f in logger.logger.filters)

    def test_structured_logging_methods(self):
        """Test structured logging methods"""
        logger = StructuredLogger("test.logger")

        # Mock the underlying logger
        logger.logger.log = MagicMock()

        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        # Verify calls
        assert logger.logger.log.call_count == 5

        # Check log levels
        calls = logger.logger.log.call_args_list
        assert calls[0][0][0] == logging.DEBUG
        assert calls[1][0][0] == logging.INFO
        assert calls[2][0][0] == logging.WARNING
        assert calls[3][0][0] == logging.ERROR
        assert calls[4][0][0] == logging.CRITICAL

    def test_trade_decision_logging(self):
        """Test trade decision logging"""
        logger = StructuredLogger("test.logger")
        logger.logger.log = MagicMock()

        logger.trade_decision("Trade signal generated", "BUY", 0.85, price=4200.0)

        # Verify call
        call_args = logger.logger.log.call_args
        assert call_args[0][0] == logging.INFO
        assert call_args[0][1] == "Trade signal generated"
        assert call_args[1]["extra"]["trade_signal"] == "BUY"
        assert call_args[1]["extra"]["trade_confidence"] == 0.85
        assert call_args[1]["extra"]["price"] == 4200.0

    def test_ml_prediction_logging(self):
        """Test ML prediction logging"""
        logger = StructuredLogger("test.logger")
        logger.logger.log = MagicMock()

        logger.ml_prediction("ML prediction generated", "random_forest", 0.72)

        # Verify call
        call_args = logger.logger.log.call_args
        assert call_args[1]["extra"]["ml_model"] == "random_forest"
        assert call_args[1]["extra"]["ml_prediction"] == 0.72

    def test_notification_logging(self):
        """Test notification logging"""
        logger = StructuredLogger("test.logger")
        logger.logger.log = MagicMock()

        logger.notification_sent("Notification sent", "email", True)

        # Verify call
        call_args = logger.logger.log.call_args
        assert call_args[1]["extra"]["notification_channel"] == "email"
        assert call_args[1]["extra"]["notification_success"] == True


class TestCorrelationIdDecorator:
    """Test correlation ID decorator"""

    def test_decorator_with_auto_id(self):
        """Test decorator with auto-generated ID"""

        @with_correlation_id()
        def test_function():
            return CorrelationIdManager.get_current_id()

        # Function should generate and use correlation ID
        result = test_function()
        assert result is not None
        assert len(result) == 36  # UUID4 length

        # Should be cleared after function
        assert CorrelationIdManager.get_current_id() is None

    def test_decorator_with_specific_id(self):
        """Test decorator with specific ID"""
        test_id = "specific-test-id"

        @with_correlation_id(test_id)
        def test_function():
            return CorrelationIdManager.get_current_id()

        result = test_function()
        assert result == test_id

        # Should be cleared after function
        assert CorrelationIdManager.get_current_id() is None

    def test_decorator_preserves_nested_ids(self):
        """Test decorator preserves nested correlation IDs"""
        outer_id = "outer-id"
        inner_id = "inner-id"

        @with_correlation_id(inner_id)
        def inner_function():
            return CorrelationIdManager.get_current_id()

        # Set outer ID
        CorrelationIdManager.set_id(outer_id)

        # Call inner function
        result = inner_function()
        assert result == inner_id

        # Should restore outer ID
        assert CorrelationIdManager.get_current_id() == outer_id

        # Clean up
        CorrelationIdManager.clear_id()

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test decorator with async functions"""

        @with_correlation_id()
        async def async_test_function():
            return CorrelationIdManager.get_current_id()

        result = await async_test_function()
        assert result is not None
        assert len(result) == 36

        # Should be cleared after function
        assert CorrelationIdManager.get_current_id() is None


class TestGetLogger:
    """Test get_logger function"""

    def test_get_logger_returns_structured_logger(self):
        """Test that get_logger returns StructuredLogger"""
        logger = get_logger("test.logger")
        assert isinstance(logger, StructuredLogger)
        assert logger.logger.name == "test.logger"

    def test_get_logger_different_names(self):
        """Test getting loggers with different names"""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")

        assert logger1.logger.name == "logger1"
        assert logger2.logger.name == "logger2"
        assert logger1 != logger2


class TestIntegration:
    """Integration tests"""

    def test_end_to_end_logging(self):
        """Test complete logging flow"""
        # Setup
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        handler.addFilter(CorrelationIdFilter())

        # Create logger
        logger = get_logger("integration.test")
        logger.logger.handlers = [handler]
        logger.logger.setLevel(logging.INFO)

        # Set correlation ID
        CorrelationIdManager.set_id("integration-test-id")

        # Log message
        logger.trade_decision("Integration test", "BUY", 0.9, price=4200.0)

        # Get output
        output = stream.getvalue()
        log_data = json.loads(output)

        # Verify
        assert log_data["correlation_id"] == "integration-test-id"
        assert log_data["message"] == "Integration test"
        assert log_data["extra"]["trade_signal"] == "BUY"
        assert log_data["extra"]["trade_confidence"] == 0.9
        assert log_data["extra"]["price"] == 4200.0

        # Clean up
        CorrelationIdManager.clear_id()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
