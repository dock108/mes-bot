"""
Structured logging service with correlation IDs for enhanced debugging and monitoring
"""

import json
import logging
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional, Union

from app.config import config

# Context variable for correlation ID
correlation_id_context: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class CorrelationIdFilter(logging.Filter):
    """Filter to inject correlation ID into log records"""

    def filter(self, record):
        correlation_id = correlation_id_context.get()
        if correlation_id:
            record.correlation_id = correlation_id
        else:
            record.correlation_id = "unknown"
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        """Format log record as JSON"""
        # Get basic log info
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "unknown"),
            "thread_id": record.thread,
            "thread_name": record.threadName,
            "process_id": record.process,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "correlation_id",
            ]:
                extra_fields[key] = value

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry)


class StructuredLogger:
    """Enhanced logger with correlation ID support and structured logging"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        """Setup logger with correlation ID filter"""
        # Add correlation ID filter if not already present
        if not any(isinstance(f, CorrelationIdFilter) for f in self.logger.filters):
            self.logger.addFilter(CorrelationIdFilter())

    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log message with additional context"""
        # Extract structured fields from kwargs
        extra = {}
        for key, value in list(kwargs.items()):
            if key.startswith(
                ("trade_", "market_", "ml_", "notification_", "decision_", "startup_")
            ):
                extra[key] = value
                del kwargs[key]
            elif key not in ["exc_info", "extra", "stack_info"]:
                # Any other non-standard logging kwargs become extra fields
                extra[key] = value
                del kwargs[key]

        if extra:
            kwargs["extra"] = extra

        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message"""
        self._log_with_context(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message"""
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message"""
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)

    def trade_decision(self, msg: str, signal: str, confidence: float, **kwargs):
        """Log trading decision with structured data"""
        self.info(msg, trade_signal=signal, trade_confidence=confidence, **kwargs)

    def market_data(self, msg: str, price: float, volume: float, **kwargs):
        """Log market data with structured fields"""
        self.debug(msg, market_price=price, market_volume=volume, **kwargs)

    def ml_prediction(self, msg: str, model_name: str, prediction: float, **kwargs):
        """Log ML prediction with structured data"""
        self.info(msg, ml_model=model_name, ml_prediction=prediction, **kwargs)

    def notification_sent(self, msg: str, channel: str, success: bool, **kwargs):
        """Log notification event"""
        self.info(msg, notification_channel=channel, notification_success=success, **kwargs)


class CorrelationIdManager:
    """Manages correlation IDs for request/operation tracking"""

    @staticmethod
    def generate_id() -> str:
        """Generate a new correlation ID"""
        return str(uuid.uuid4())

    @staticmethod
    def get_current_id() -> Optional[str]:
        """Get current correlation ID from context"""
        return correlation_id_context.get()

    @staticmethod
    def set_id(correlation_id: str):
        """Set correlation ID in current context"""
        correlation_id_context.set(correlation_id)

    @staticmethod
    def clear_id():
        """Clear correlation ID from context"""
        correlation_id_context.set(None)


def with_correlation_id(correlation_id: Optional[str] = None):
    """Decorator to set correlation ID for a function"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate new ID if not provided
            if correlation_id is None:
                new_id = CorrelationIdManager.generate_id()
            else:
                new_id = correlation_id

            # Set correlation ID in context
            old_id = CorrelationIdManager.get_current_id()
            CorrelationIdManager.set_id(new_id)

            try:
                return func(*args, **kwargs)
            finally:
                # Restore previous correlation ID
                if old_id:
                    CorrelationIdManager.set_id(old_id)
                else:
                    CorrelationIdManager.clear_id()

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate new ID if not provided
            if correlation_id is None:
                new_id = CorrelationIdManager.generate_id()
            else:
                new_id = correlation_id

            # Set correlation ID in context
            old_id = CorrelationIdManager.get_current_id()
            CorrelationIdManager.set_id(new_id)

            try:
                return await func(*args, **kwargs)
            finally:
                # Restore previous correlation ID
                if old_id:
                    CorrelationIdManager.set_id(old_id)
                else:
                    CorrelationIdManager.clear_id()

        # Return async wrapper for async functions
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:
            return async_wrapper
        else:
            return wrapper

    return decorator


def setup_structured_logging():
    """Setup structured logging for the application"""
    # Get root logger
    root_logger = logging.getLogger()

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatters
    structured_formatter = StructuredFormatter()
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s"
    )

    # Create file handler with structured logging
    file_handler = logging.FileHandler(f"{config.logging.log_dir}/{config.logging.bot_log_file}")
    file_handler.setFormatter(structured_formatter)
    file_handler.addFilter(CorrelationIdFilter())

    # Create console handler with readable format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(CorrelationIdFilter())

    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Set log level
    root_logger.setLevel(getattr(logging, config.logging.level))

    # Create error handler
    error_handler = logging.FileHandler(f"{config.logging.log_dir}/{config.logging.error_log_file}")
    error_handler.setFormatter(structured_formatter)
    error_handler.addFilter(CorrelationIdFilter())
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)


# Initialize structured logging if not already done
_logging_initialized = False


def initialize_logging():
    """Initialize structured logging (call once at startup)"""
    global _logging_initialized
    if not _logging_initialized:
        setup_structured_logging()
        _logging_initialized = True


# Auto-initialize on import
initialize_logging()
