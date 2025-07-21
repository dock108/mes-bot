"""
Comprehensive tests for the monitoring system
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models import DecisionHistory, MarketData, Trade
from app.monitoring import (
    AlertThresholds,
    AlertType,
    TradingMonitor,
    get_monitor,
    initialize_monitor,
)


@pytest.mark.unit
class TestAlertThresholds:
    """Test AlertThresholds configuration"""

    def test_default_thresholds(self):
        """Test default threshold values"""
        thresholds = AlertThresholds()

        assert thresholds.max_daily_loss_pct == 0.15
        assert thresholds.max_drawdown_pct == 0.25
        assert thresholds.consecutive_loss_limit == 5
        assert thresholds.low_win_rate_threshold == 0.2
        assert thresholds.max_errors_per_hour == 10
        assert thresholds.connection_timeout_minutes == 5
        assert thresholds.model_accuracy_threshold == 0.4
        assert thresholds.high_volatility_threshold == 0.5
        assert thresholds.unusual_volume_multiplier == 3.0

    def test_custom_thresholds(self):
        """Test custom threshold configuration"""
        thresholds = AlertThresholds(
            max_daily_loss_pct=0.10,
            consecutive_loss_limit=3,
            max_errors_per_hour=5,
        )

        assert thresholds.max_daily_loss_pct == 0.10
        assert thresholds.consecutive_loss_limit == 3
        assert thresholds.max_errors_per_hour == 5
        # Other values should remain default
        assert thresholds.max_drawdown_pct == 0.25


@pytest.mark.integration
@pytest.mark.db
class TestTradingMonitor:
    """Test TradingMonitor functionality"""

    @pytest.fixture
    def monitor(self, test_db_url):
        """Create test monitoring instance"""
        return TradingMonitor(test_db_url)

    @pytest.fixture
    def mock_session(self):
        """Create mock database session"""
        session = Mock()
        session.query.return_value = session
        session.filter.return_value = session
        session.order_by.return_value = session
        session.limit.return_value = session
        session.all.return_value = []
        session.count.return_value = 0
        session.first.return_value = None
        return session

    def test_monitor_initialization(self, monitor):
        """Test monitor initializes correctly"""
        assert monitor is not None
        assert isinstance(monitor.thresholds, AlertThresholds)
        assert monitor.error_count == 0
        assert monitor.monitoring_active is True
        assert isinstance(monitor.daily_stats, dict)
        assert isinstance(monitor.performance_history, list)

    def test_custom_thresholds(self, test_db_url):
        """Test monitor with custom thresholds"""
        custom_thresholds = AlertThresholds(max_daily_loss_pct=0.10)
        monitor = TradingMonitor(test_db_url)
        monitor.thresholds = custom_thresholds

        assert monitor.thresholds.max_daily_loss_pct == 0.10

    @pytest.mark.asyncio
    async def test_record_error(self, monitor):
        """Test error recording and alerting"""
        test_error = ValueError("Test error")
        test_context = {"component": "test"}

        with patch("app.monitoring.send_error_alert") as mock_alert:
            await monitor.record_error(test_error, test_context)

            assert monitor.error_count == 1
            mock_alert.assert_called_once()
            call_args = mock_alert.call_args
            assert call_args[1]["title"] == "System Error Detected"
            assert "Test error" in call_args[1]["message"]
            assert call_args[1]["error"] == test_error
            assert call_args[1]["context"] == test_context

    @pytest.mark.asyncio
    async def test_record_trade_opened(self, monitor):
        """Test trade opening notification"""
        test_trade = Mock(spec=Trade)
        test_trade.id = 123
        test_trade.call_strike = 4100
        test_trade.put_strike = 3900
        test_trade.total_premium = 150.0
        test_trade.underlying_price_at_entry = 4000.0
        test_trade.implied_move = 50.0

        with patch("app.monitoring.send_trade_alert") as mock_alert:
            await monitor.record_trade_opened(test_trade)

            mock_alert.assert_called_once()
            call_args = mock_alert.call_args
            assert call_args[1]["title"] == "New Trade Opened"
            assert "4100C/3900P" in call_args[1]["message"]
            assert "$150.00" in call_args[1]["message"]

    @pytest.mark.asyncio
    async def test_record_trade_closed_profit(self, monitor):
        """Test trade closing notification for profitable trade"""
        test_trade = Mock(spec=Trade)
        test_trade.id = 123
        test_trade.realized_pnl = 75.0
        test_trade.total_premium = 150.0
        test_trade.entry_time = datetime.utcnow() - timedelta(hours=2)
        test_trade.exit_time = datetime.utcnow()

        with patch("app.monitoring.send_trade_alert") as mock_alert:
            await monitor.record_trade_closed(test_trade)

            mock_alert.assert_called_once()
            call_args = mock_alert.call_args
            assert call_args[1]["title"] == "Trade Closed"
            assert "$75.00" in call_args[1]["message"]

    @pytest.mark.asyncio
    async def test_record_trade_closed_loss(self, monitor):
        """Test trade closing notification for losing trade"""
        test_trade = Mock(spec=Trade)
        test_trade.id = 123
        test_trade.realized_pnl = -25.0
        test_trade.total_premium = 150.0
        test_trade.entry_time = datetime.utcnow() - timedelta(hours=1)
        test_trade.exit_time = datetime.utcnow()

        with patch("app.monitoring.send_trade_alert") as mock_alert:
            await monitor.record_trade_closed(test_trade)

            mock_alert.assert_called_once()
            call_args = mock_alert.call_args
            assert call_args[1]["title"] == "Trade Closed"
            assert "$-25.00" in call_args[1]["message"]

    def test_count_consecutive_losses_empty(self, monitor):
        """Test consecutive loss counting with no trades"""
        result = monitor._count_consecutive_losses([])
        assert result == 0

    def test_count_consecutive_losses_no_losses(self, monitor):
        """Test consecutive loss counting with no losses"""
        trades = []
        for i in range(3):
            trade = Mock(spec=Trade)
            trade.realized_pnl = 50.0  # All profitable
            trade.entry_time = datetime.utcnow() - timedelta(hours=i)
            trades.append(trade)

        result = monitor._count_consecutive_losses(trades)
        assert result == 0

    def test_count_consecutive_losses_with_losses(self, monitor):
        """Test consecutive loss counting with actual losses"""
        trades = []
        # Most recent trade (loss)
        trade1 = Mock(spec=Trade)
        trade1.realized_pnl = -25.0
        trade1.entry_time = datetime.utcnow()
        trades.append(trade1)

        # Second most recent (loss)
        trade2 = Mock(spec=Trade)
        trade2.realized_pnl = -15.0
        trade2.entry_time = datetime.utcnow() - timedelta(hours=1)
        trades.append(trade2)

        # Third most recent (profit, breaks streak)
        trade3 = Mock(spec=Trade)
        trade3.realized_pnl = 30.0
        trade3.entry_time = datetime.utcnow() - timedelta(hours=2)
        trades.append(trade3)

        result = monitor._count_consecutive_losses(trades)
        assert result == 2

    def test_get_monitoring_status(self, monitor):
        """Test monitoring status retrieval"""
        monitor.error_count = 5

        status = monitor.get_monitoring_status()

        assert status["monitoring_active"] is True
        assert status["error_count_last_hour"] == 5
        assert "last_error_reset" in status
        assert "thresholds" in status
        assert status["thresholds"]["max_daily_loss_pct"] == 0.15

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, monitor):
        """Test monitoring shutdown"""
        with patch("app.monitoring.send_system_alert") as mock_alert:
            await monitor.stop_monitoring()

            assert monitor.monitoring_active is False
            mock_alert.assert_called_once()
            call_args = mock_alert.call_args
            assert call_args[1]["title"] == "Trading System Stopped"

    @pytest.mark.asyncio
    async def test_check_performance_alerts_high_loss(self, monitor, mock_session):
        """Test performance alert for high daily loss"""
        # Mock the session_maker directly on the monitor instance
        monitor.session_maker = Mock(return_value=mock_session)

        # Use a real config-like object or set the attribute directly
        with patch("app.monitoring.config") as mock_config:
            mock_config.trading.start_cash = 10000

            # Create losing trades
            losing_trades = []
            for i in range(3):
                trade = Mock(spec=Trade)
                trade.realized_pnl = -600.0  # Total -1800, 18% of 10k (exceeds 15% threshold)
                trade.status = "CLOSED"
                trade.date = datetime.utcnow().date()
                losing_trades.append(trade)

            mock_session.query.return_value.filter.return_value.all.return_value = losing_trades

            with patch("app.monitoring.send_performance_alert") as mock_alert:
                await monitor._check_performance_alerts()

                # The function should send an alert for high daily loss
                # The function should send an alert for high daily loss
                assert (
                    mock_alert.call_count > 0
                ), f"Expected performance alert to be called but it wasn't. Mock: {mock_alert}"
                call_args = mock_alert.call_args
                assert "High Daily Loss Alert" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_check_performance_alerts_low_win_rate(self, monitor, mock_session):
        """Test performance alert for low win rate"""
        monitor.session_maker = Mock(return_value=mock_session)

        # Create trades with low win rate (1 win, 5 losses = 6 total > 5 required)
        trades = []
        # 1 winning trade
        win_trade = Mock(spec=Trade)
        win_trade.realized_pnl = 100.0
        win_trade.status = "CLOSED"
        win_trade.date = datetime.utcnow().date()
        win_trade.entry_time = datetime.utcnow()  # Add entry_time for sorting
        trades.append(win_trade)

        # 5 losing trades (to ensure we have > 5 trades total)
        for i in range(5):
            trade = Mock(spec=Trade)
            trade.realized_pnl = -50.0
            trade.status = "CLOSED"
            trade.date = datetime.utcnow().date()
            trade.entry_time = datetime.utcnow() - timedelta(hours=i)  # Add entry_time for sorting
            trades.append(trade)

        mock_session.query.return_value.filter.return_value.all.return_value = trades

        with patch("app.monitoring.config") as mock_config:
            mock_config.trading.start_cash = 10000

            with patch("app.monitoring.send_performance_alert") as mock_alert:
                await monitor._check_performance_alerts()

                mock_alert.assert_called()
                call_args = mock_alert.call_args
                assert "Low Win Rate Alert" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_check_performance_alerts_consecutive_losses(self, monitor, mock_session):
        """Test performance alert for consecutive losses"""
        monitor.session_maker = Mock(return_value=mock_session)

        # Create 5 consecutive losing trades
        losing_trades = []
        for i in range(5):
            trade = Mock(spec=Trade)
            trade.realized_pnl = -100.0
            trade.status = "CLOSED"
            trade.date = datetime.utcnow().date()
            trade.entry_time = datetime.utcnow() - timedelta(hours=i)
            losing_trades.append(trade)

        mock_session.query.return_value.filter.return_value.all.return_value = losing_trades

        with patch("app.monitoring.config") as mock_config:
            mock_config.trading.start_cash = 10000

            with patch("app.monitoring.send_performance_alert") as mock_alert:
                await monitor._check_performance_alerts()

                mock_alert.assert_called()
            call_args = mock_alert.call_args
            assert "Consecutive Losses Alert" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_check_system_health_error_rate(self, monitor):
        """Test system health check for high error rate"""
        monitor.error_count = 15  # Above threshold of 10

        with patch("app.monitoring.send_system_alert") as mock_alert:
            await monitor._check_system_health()

            mock_alert.assert_called()
            call_args = mock_alert.call_args
            assert "High Error Rate Alert" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_check_ml_model_performance_low_accuracy(self, monitor, mock_session):
        """Test ML model performance check for low accuracy"""
        monitor.session_maker = Mock(return_value=mock_session)

        # Create ML decisions with low accuracy (2 correct out of 10)
        decisions = []
        for i in range(10):
            decision = Mock(spec=DecisionHistory)
            decision.confidence = 0.7  # Predicts positive
            decision.actual_outcome = 1.0 if i < 2 else -1.0  # Only first 2 are correct
            decision.timestamp = datetime.utcnow() - timedelta(hours=i)
            decisions.append(decision)

        mock_session.query.return_value.filter.return_value.all.return_value = decisions

        with patch("app.monitoring.send_system_alert") as mock_alert:
            await monitor._check_ml_model_performance()

            mock_alert.assert_called()
            call_args = mock_alert.call_args
            assert "Low ML Model Accuracy" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_check_market_conditions_high_volatility(self, monitor, mock_session):
        """Test market condition check for high volatility"""
        monitor.session_maker = Mock(return_value=mock_session)

        # Create market data with high volatility
        market_data = []
        data = Mock(spec=MarketData)
        data.atm_iv = 0.6  # Above threshold of 0.5
        data.vix_level = 35.0
        data.volume = 1000
        data.timestamp = datetime.utcnow()
        market_data.append(data)

        # Add more data points for average calculation
        for i in range(9):
            data = Mock(spec=MarketData)
            data.atm_iv = 0.3
            data.vix_level = 20.0
            data.volume = 800
            data.timestamp = datetime.utcnow() - timedelta(minutes=i + 1)
            market_data.append(data)

        mock_query_chain = (
            mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value
        )
        mock_query_chain.all.return_value = market_data

        with patch("app.monitoring.send_system_alert") as mock_alert:
            await monitor._check_market_conditions()

            mock_alert.assert_called()
            call_args = mock_alert.call_args
            assert "High Market Volatility" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_check_market_conditions_unusual_volume(self, monitor, mock_session):
        """Test market condition check for unusual volume"""
        monitor.session_maker = Mock(return_value=mock_session)

        # Create market data with unusual volume
        market_data = []
        # Current data with high volume (needs to be much higher to trigger alert)
        current_data = Mock(spec=MarketData)
        current_data.atm_iv = 0.3
        current_data.vix_level = 20.0
        current_data.volume = 4000  # 4x normal to account for average including this value
        current_data.timestamp = datetime.utcnow()
        market_data.append(current_data)

        # Historical data with normal volume
        for i in range(9):
            data = Mock(spec=MarketData)
            data.atm_iv = 0.3
            data.vix_level = 20.0
            data.volume = 1000  # Normal volume
            data.timestamp = datetime.utcnow() - timedelta(minutes=i + 1)
            market_data.append(data)

        mock_query_chain = (
            mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value
        )
        mock_query_chain.all.return_value = market_data

        with patch("app.monitoring.send_system_alert") as mock_alert:
            await monitor._check_market_conditions()

            mock_alert.assert_called()
            call_args = mock_alert.call_args
            assert "Unusual Market Volume" in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_send_daily_summary(self, monitor, mock_session):
        """Test daily summary generation"""
        monitor.session_maker = Mock(return_value=mock_session)

        # Create sample trades
        trades = []
        # 2 winning trades
        for i in range(2):
            trade = Mock(spec=Trade)
            trade.realized_pnl = 100.0
            trade.status = "CLOSED"
            trade.date = datetime.utcnow().date()
            trades.append(trade)

        # 1 losing trade
        trade = Mock(spec=Trade)
        trade.realized_pnl = -50.0
        trade.status = "CLOSED"
        trade.date = datetime.utcnow().date()
        trades.append(trade)

        # 1 open trade
        trade = Mock(spec=Trade)
        trade.realized_pnl = None
        trade.status = "OPEN"
        trade.date = datetime.utcnow().date()
        trades.append(trade)

        mock_session.query.return_value.filter.return_value.all.return_value = trades
        mock_session.query.return_value.filter.return_value.count.return_value = 5  # ML decisions

        with patch("app.monitoring.config") as mock_config:
            mock_config.trading.daily_loss_limit = 1000

            with patch("app.monitoring.send_performance_alert") as mock_alert:
                await monitor._send_daily_summary()

                mock_alert.assert_called_once()
                call_args = mock_alert.call_args
                assert call_args[1]["title"] == "Daily Trading Summary"
                assert "$150.00" in call_args[1]["message"]  # Total PnL
                assert "66.7%" in call_args[1]["message"]  # Win rate (2/3)

    @pytest.mark.asyncio
    async def test_error_handling_in_monitoring_loops(self, monitor):
        """Test error handling in monitoring loops"""
        # Test that monitoring loops handle exceptions gracefully
        with patch.object(
            monitor, "_check_performance_alerts", side_effect=Exception("Test error")
        ):
            with patch("app.monitoring.send_error_alert") as mock_alert:
                # This should not raise an exception
                try:
                    await asyncio.wait_for(monitor._monitor_performance(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass  # Expected due to infinite loop

                # Should have sent error alert
                mock_alert.assert_called()


@pytest.mark.integration
@pytest.mark.db
class TestGlobalMonitorFunctions:
    """Test global monitor initialization and access"""

    def test_initialize_monitor(self, test_db_url):
        """Test global monitor initialization"""
        monitor_instance = initialize_monitor(test_db_url)

        assert monitor_instance is not None
        assert isinstance(monitor_instance, TradingMonitor)

        # Should be accessible via get_monitor
        global_monitor = get_monitor()
        assert global_monitor is monitor_instance

    def test_get_monitor_before_initialization(self):
        """Test get_monitor returns None before initialization"""
        # Reset global monitor
        import app.monitoring

        app.monitoring.monitor = None

        result = get_monitor()
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
