"""
Tests for Notification System
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.monitoring import AlertThresholds, TradingMonitor
from app.notification_service import (
    NotificationChannel,
    NotificationConfig,
    NotificationLevel,
    NotificationMessage,
    NotificationService,
    send_error_alert,
    send_performance_alert,
    send_system_alert,
    send_trade_alert,
)


class TestNotificationConfig:
    """Test notification configuration"""

    def test_default_config(self):
        """Test default notification configuration"""
        config = NotificationConfig()

        assert config.smtp_host == "smtp.gmail.com"
        assert config.smtp_port == 587
        assert config.rate_limit_minutes == 5
        assert config.max_notifications_per_period == 10
        assert config.channel_config[NotificationLevel.CRITICAL] == [
            NotificationChannel.CONSOLE,
            NotificationChannel.EMAIL,
            NotificationChannel.SMS,
            NotificationChannel.SLACK,
        ]

    def test_email_list_property(self):
        """Test email list property"""
        config = NotificationConfig(to_emails=["test1@example.com", "test2@example.com"])

        assert len(config.to_emails) == 2
        assert "test1@example.com" in config.to_emails
        assert "test2@example.com" in config.to_emails


class TestNotificationService:
    """Test notification service"""

    @pytest.fixture
    def mock_config(self):
        """Mock notification configuration"""
        return NotificationConfig(
            smtp_user="test@example.com",
            smtp_password="password",
            to_emails=["recipient@example.com"],
            twilio_account_sid="test_sid",
            twilio_auth_token="test_token",
            twilio_from_number="+1234567890",
            twilio_to_numbers=["+0987654321"],
            slack_webhook_url="https://hooks.slack.com/test",
            webhook_url="https://example.com/webhook",
        )

    @pytest.fixture
    def notification_service(self, mock_config):
        """Create notification service with mock config"""
        return NotificationService(mock_config)

    @pytest.fixture
    def test_notification(self):
        """Create test notification"""
        return NotificationMessage(
            level=NotificationLevel.WARNING,
            title="Test Alert",
            message="This is a test notification",
            timestamp=datetime.utcnow(),
            context={"test": "data"},
        )

    def test_initialization(self, notification_service):
        """Test service initialization"""
        assert notification_service.config is not None
        assert notification_service.enabled_channels is not None
        assert NotificationChannel.CONSOLE in notification_service.enabled_channels
        assert NotificationChannel.EMAIL in notification_service.enabled_channels
        assert NotificationChannel.SMS in notification_service.enabled_channels
        assert NotificationChannel.SLACK in notification_service.enabled_channels
        assert NotificationChannel.WEBHOOK in notification_service.enabled_channels

    def test_enabled_channels_detection(self):
        """Test channel detection logic"""
        # No configuration
        empty_config = NotificationConfig()
        service = NotificationService(empty_config)
        assert service.enabled_channels == {NotificationChannel.CONSOLE}

        # Email only
        email_config = NotificationConfig(
            smtp_user="test@example.com",
            smtp_password="password",
            to_emails=["recipient@example.com"],
        )
        service = NotificationService(email_config)
        assert NotificationChannel.EMAIL in service.enabled_channels
        assert NotificationChannel.CONSOLE in service.enabled_channels

    @pytest.mark.asyncio
    async def test_console_notification(self, notification_service, test_notification):
        """Test console notification"""
        with patch("builtins.print") as mock_print:
            result = await notification_service._send_console_notification(test_notification)

            assert result is True
            assert mock_print.called

            # Check that the message was printed
            call_args = mock_print.call_args_list
            assert len(call_args) >= 2  # At least title and message

    @pytest.mark.asyncio
    async def test_email_notification(self, notification_service, test_notification):
        """Test email notification"""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            result = await notification_service._send_email_notification(test_notification)

            assert result is True
            assert mock_server.starttls.called
            assert mock_server.login.called
            assert mock_server.send_message.called

    @pytest.mark.asyncio
    async def test_sms_notification(self, notification_service, test_notification):
        """Test SMS notification"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_post.return_value = mock_response

            result = await notification_service._send_sms_notification(test_notification)

            assert result is True
            assert mock_post.called

            # Check request details
            call_args = mock_post.call_args
            assert "twilio" in call_args[0][0]  # URL contains twilio
            assert call_args[1]["data"]["From"] == "+1234567890"
            assert call_args[1]["data"]["To"] == "+0987654321"

    @pytest.mark.asyncio
    async def test_slack_notification(self, notification_service, test_notification):
        """Test Slack notification"""
        # Use patch.object to mock the entire aiohttp module
        with patch("app.notification_service.aiohttp.ClientSession") as mock_session_class:
            # Create a proper async context manager mock
            mock_response = Mock()
            mock_response.status = 200

            # Create async context manager for response
            mock_response_context = AsyncMock()
            mock_response_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response_context.__aexit__ = AsyncMock(return_value=False)

            # Create session mock
            mock_session = Mock()
            mock_session.post = Mock(return_value=mock_response_context)

            # Create async context manager for session
            mock_session_context = AsyncMock()
            mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_context.__aexit__ = AsyncMock(return_value=False)

            mock_session_class.return_value = mock_session_context

            result = await notification_service._send_slack_notification(test_notification)

            assert result is True

    @pytest.mark.asyncio
    async def test_webhook_notification(self, notification_service, test_notification):
        """Test webhook notification"""
        # Use patch.object to mock the entire aiohttp module
        with patch("app.notification_service.aiohttp.ClientSession") as mock_session_class:
            # Create a proper async context manager mock
            mock_response = Mock()
            mock_response.status = 200

            # Create async context manager for response
            mock_response_context = AsyncMock()
            mock_response_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response_context.__aexit__ = AsyncMock(return_value=False)

            # Create session mock
            mock_session = Mock()
            mock_session.post = Mock(return_value=mock_response_context)

            # Create async context manager for session
            mock_session_context = AsyncMock()
            mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_context.__aexit__ = AsyncMock(return_value=False)

            mock_session_class.return_value = mock_session_context

            result = await notification_service._send_webhook_notification(test_notification)

            assert result is True

    @pytest.mark.asyncio
    async def test_rate_limiting(self, notification_service, test_notification):
        """Test rate limiting functionality"""
        # Send multiple notifications quickly
        for i in range(15):  # Exceed the limit of 10
            await notification_service.send_notification(test_notification)

        # Check that some notifications were rate limited
        assert len(notification_service.notification_history) <= 10

    @pytest.mark.asyncio
    async def test_notification_history(self, notification_service, test_notification):
        """Test notification history tracking"""
        initial_count = len(notification_service.notification_history)

        await notification_service.send_notification(test_notification)

        assert len(notification_service.notification_history) == initial_count + 1

        # Check history entry
        last_entry = notification_service.notification_history[-1]
        assert last_entry["level"] == "warning"
        assert last_entry["title"] == "Test Alert"
        assert last_entry["success"] is True

    def test_email_template_selection(self, notification_service):
        """Test email template selection"""
        trade_notification = NotificationMessage(
            level=NotificationLevel.INFO,
            title="Trade Executed",
            message="Trade message",
            timestamp=datetime.utcnow(),
        )

        error_notification = NotificationMessage(
            level=NotificationLevel.ERROR,
            title="System Error",
            message="Error message",
            timestamp=datetime.utcnow(),
        )

        assert notification_service._determine_email_template(trade_notification) == "trade_alert"
        assert notification_service._determine_email_template(error_notification) == "error_alert"

    def test_notification_stats(self, notification_service):
        """Test notification statistics"""
        # Add some mock history
        notification_service.notification_history = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "info",
                "title": "Test 1",
                "channels": ["console"],
                "success": True,
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "warning",
                "title": "Test 2",
                "channels": ["console", "email"],
                "success": True,
            },
        ]

        stats = notification_service.get_notification_stats()

        assert stats["total_notifications"] == 2
        assert stats["level_counts"]["info"] == 1
        assert stats["level_counts"]["warning"] == 1
        assert "console" in stats["enabled_channels"]


class TestNotificationHelpers:
    """Test notification helper functions"""

    @pytest.mark.asyncio
    async def test_send_trade_alert(self):
        """Test trade alert helper"""
        with patch("app.notification_service.config") as mock_config:
            mock_service = AsyncMock()
            mock_config.notification_service = mock_service

            await send_trade_alert(
                title="Trade Alert",
                message="Trade message",
                level=NotificationLevel.INFO,
                context={"trade_id": 123},
            )

            assert mock_service.send_notification.called
            call_args = mock_service.send_notification.call_args[0][0]
            assert call_args.title == "Trade Alert"
            assert call_args.level == NotificationLevel.INFO
            assert call_args.context["trade_id"] == 123

    @pytest.mark.asyncio
    async def test_send_error_alert(self):
        """Test error alert helper"""
        with patch("app.notification_service.config") as mock_config:
            mock_service = AsyncMock()
            mock_config.notification_service = mock_service

            test_error = ValueError("Test error")

            await send_error_alert(
                title="Error Alert",
                message="Error message",
                error=test_error,
                context={"component": "test"},
            )

            assert mock_service.send_notification.called
            call_args = mock_service.send_notification.call_args[0][0]
            assert call_args.title == "Error Alert"
            assert call_args.level == NotificationLevel.ERROR
            assert call_args.context["error_type"] == "ValueError"
            assert call_args.context["error_message"] == "Test error"

    @pytest.mark.asyncio
    async def test_send_performance_alert(self):
        """Test performance alert helper"""
        with patch("app.notification_service.config") as mock_config:
            mock_service = AsyncMock()
            mock_config.notification_service = mock_service

            metrics = {"win_rate": 0.6, "total_trades": 10}

            await send_performance_alert(
                title="Performance Alert",
                message="Performance message",
                metrics=metrics,
                level=NotificationLevel.WARNING,
            )

            assert mock_service.send_notification.called
            call_args = mock_service.send_notification.call_args[0][0]
            assert call_args.title == "Performance Alert"
            assert call_args.level == NotificationLevel.WARNING
            assert call_args.context["metrics"] == metrics

    @pytest.mark.asyncio
    async def test_send_system_alert(self):
        """Test system alert helper"""
        with patch("app.notification_service.config") as mock_config:
            mock_service = AsyncMock()
            mock_config.notification_service = mock_service

            await send_system_alert(
                title="System Alert",
                message="System message",
                level=NotificationLevel.CRITICAL,
                context={"system": "trading"},
            )

            assert mock_service.send_notification.called
            call_args = mock_service.send_notification.call_args[0][0]
            assert call_args.title == "System Alert"
            assert call_args.level == NotificationLevel.CRITICAL
            assert call_args.context["system"] == "trading"


class TestTradingMonitor:
    """Test trading monitor"""

    @pytest.fixture
    def mock_database_url(self):
        """Mock database URL"""
        return "sqlite:///:memory:"

    @pytest.fixture
    def trading_monitor(self, mock_database_url):
        """Create trading monitor"""
        return TradingMonitor(mock_database_url)

    def test_initialization(self, trading_monitor):
        """Test monitor initialization"""
        assert trading_monitor.session_maker is not None
        assert trading_monitor.thresholds is not None
        assert trading_monitor.monitoring_active is True
        assert trading_monitor.error_count == 0

    def test_alert_thresholds(self):
        """Test alert thresholds"""
        thresholds = AlertThresholds()

        assert thresholds.max_daily_loss_pct == 0.15
        assert thresholds.consecutive_loss_limit == 5
        assert thresholds.max_errors_per_hour == 10
        assert thresholds.model_accuracy_threshold == 0.4

    @pytest.mark.asyncio
    async def test_record_error(self, trading_monitor):
        """Test error recording"""
        with patch("app.monitoring.send_error_alert") as mock_send:
            test_error = ValueError("Test error")

            await trading_monitor.record_error(test_error, {"component": "test"})

            assert trading_monitor.error_count == 1
            assert mock_send.called

            call_args = mock_send.call_args
            assert call_args[1]["error"] == test_error

    @pytest.mark.asyncio
    async def test_record_trade_opened(self, trading_monitor):
        """Test trade opening record"""
        with patch("app.monitoring.send_trade_alert") as mock_send:
            from app.models import Trade

            trade = Trade(
                id=1,
                underlying_price_at_entry=5000.0,
                call_strike=5025.0,
                put_strike=4975.0,
                total_premium=15.0,
                implied_move=25.0,
            )

            await trading_monitor.record_trade_opened(trade)

            assert mock_send.called
            call_args = mock_send.call_args
            assert "New Trade Opened" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_record_trade_closed(self, trading_monitor):
        """Test trade closing record"""
        with patch("app.monitoring.send_trade_alert") as mock_send:
            from app.models import Trade

            trade = Trade(
                id=1,
                entry_time=datetime.utcnow(),
                exit_time=datetime.utcnow(),
                total_premium=15.0,
                realized_pnl=5.0,
            )

            await trading_monitor.record_trade_closed(trade)

            assert mock_send.called
            call_args = mock_send.call_args
            assert "Trade Closed" in call_args[1]["title"]

    def test_consecutive_losses_count(self, trading_monitor):
        """Test consecutive losses counting"""
        from app.models import Trade

        # Create mock trades (most recent first)
        trades = [
            Trade(entry_time=datetime.utcnow(), realized_pnl=-10.0),  # Loss
            Trade(entry_time=datetime.utcnow() - timedelta(hours=1), realized_pnl=-5.0),  # Loss
            Trade(entry_time=datetime.utcnow() - timedelta(hours=2), realized_pnl=8.0),  # Win
            Trade(entry_time=datetime.utcnow() - timedelta(hours=3), realized_pnl=-3.0),  # Loss
        ]

        consecutive = trading_monitor._count_consecutive_losses(trades)
        assert consecutive == 2  # Two most recent losses

    def test_monitoring_status(self, trading_monitor):
        """Test monitoring status"""
        trading_monitor.error_count = 5

        status = trading_monitor.get_monitoring_status()

        assert status["monitoring_active"] is True
        assert status["error_count_last_hour"] == 5
        assert "thresholds" in status
        assert status["thresholds"]["max_daily_loss_pct"] == 0.15

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, trading_monitor):
        """Test stopping monitoring"""
        with patch("app.monitoring.send_system_alert") as mock_send:
            await trading_monitor.stop_monitoring()

            assert trading_monitor.monitoring_active is False
            assert mock_send.called

            call_args = mock_send.call_args
            assert "Trading System Stopped" in call_args[1]["title"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
