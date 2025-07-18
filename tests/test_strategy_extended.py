"""Extended tests for strategy.py to improve coverage"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from app.strategy import LottoGridStrategy
from app.ib_client import IBClient
from app.risk_manager import RiskManager
from app.models import Trade


class TestLottoGridStrategyExtended:
    """Extended test cases for uncovered strategy methods"""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance with mocked dependencies"""
        mock_ib_client = Mock(spec=IBClient)
        mock_risk_manager = Mock(spec=RiskManager)
        mock_db_url = "sqlite:///./data/test.db"
        
        strategy = LottoGridStrategy(mock_ib_client, mock_risk_manager, mock_db_url)
        return strategy

    @pytest.mark.asyncio
    async def test_initialize_daily_session_ib_error(self, strategy):
        """Test daily initialization when IB client fails"""
        # Mock IB client to raise error
        strategy.ib_client.get_mes_contract = AsyncMock(side_effect=Exception("IB connection error"))
        
        result = await strategy.initialize_daily_session()
        
        assert result is False
        assert strategy.daily_high is None
        assert strategy.daily_low is None

    @pytest.mark.asyncio
    async def test_initialize_daily_session_no_price(self, strategy):
        """Test daily initialization when no price available"""
        strategy.ib_client.get_mes_contract = AsyncMock(return_value=Mock())
        strategy.ib_client.get_current_price = AsyncMock(return_value=None)
        
        result = await strategy.initialize_daily_session()
        
        assert result is False

    @pytest.mark.asyncio
    async def test_place_strangle_trade_complete(self, strategy):
        """Test complete strangle trade placement"""
        # Setup
        strategy.implied_move = 50
        strategy.underlying_price = 4250
        
        # Mock order placement
        mock_order = Mock(orderId=123)
        strategy.ib_client.place_strangle_order = AsyncMock(return_value=mock_order)
        
        # Mock risk check
        strategy.risk_manager.can_open_new_trade = Mock(return_value=(True, "OK"))
        
        # Execute
        result = await strategy.place_strangle_trade(4300, 4200, 5.0)
        
        assert result is True
        strategy.ib_client.place_strangle_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_strangle_trade_risk_rejection(self, strategy):
        """Test strangle trade rejected by risk manager"""
        strategy.risk_manager.can_open_new_trade = Mock(return_value=(False, "Max trades reached"))
        
        result = await strategy.place_strangle_trade(4300, 4200, 5.0)
        
        assert result is False
        strategy.ib_client.place_strangle_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_open_positions_complete(self, strategy):
        """Test updating open positions with exits"""
        # Create mock trade
        mock_trade = Mock(spec=Trade)
        mock_trade.id = 1
        mock_trade.call_strike = 4300
        mock_trade.put_strike = 4200
        mock_trade.total_premium = 10.0
        mock_trade.entry_time = datetime.utcnow() - timedelta(hours=1)
        mock_trade.status = "OPEN"
        
        # Mock database session
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.all.return_value = [mock_trade]
        strategy.session_maker = Mock(return_value=mock_session)
        
        # Mock IB client responses
        strategy.ib_client.get_option_prices = AsyncMock(return_value=(40.0, 2.0))  # Hit profit target
        strategy.ib_client.close_position = AsyncMock(return_value=True)
        
        # Mock risk manager
        strategy.risk_manager.update_daily_summary = Mock()
        
        # Execute
        await strategy.update_open_positions()
        
        # Verify
        assert mock_trade.status == "CLOSED"
        assert mock_trade.exit_time is not None
        strategy.ib_client.close_position.assert_called()
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_update_open_positions_expiry(self, strategy):
        """Test position update at expiry"""
        # Create expired trade
        mock_trade = Mock(spec=Trade)
        mock_trade.id = 1
        mock_trade.entry_time = datetime.utcnow() - timedelta(hours=8)  # Past expiry
        mock_trade.status = "OPEN"
        
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.all.return_value = [mock_trade]
        strategy.session_maker = Mock(return_value=mock_session)
        
        strategy.ib_client.close_position = AsyncMock(return_value=True)
        strategy.risk_manager.update_daily_summary = Mock()
        
        await strategy.update_open_positions()
        
        assert mock_trade.status == "EXPIRED"
        strategy.ib_client.close_position.assert_called()

    @pytest.mark.asyncio
    async def test_update_open_positions_error_handling(self, strategy):
        """Test error handling in position updates"""
        mock_trade = Mock(spec=Trade)
        mock_trade.id = 1
        mock_trade.status = "OPEN"
        
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.all.return_value = [mock_trade]
        strategy.session_maker = Mock(return_value=mock_session)
        
        # Mock IB client to raise error
        strategy.ib_client.get_option_prices = AsyncMock(side_effect=Exception("API error"))
        
        # Should not raise, just log error
        await strategy.update_open_positions()
        
        # Trade should remain open
        assert mock_trade.status == "OPEN"

    @pytest.mark.asyncio
    async def test_flatten_all_positions(self, strategy):
        """Test flattening all positions"""
        # Mock open trades
        mock_trades = [
            Mock(spec=Trade, id=1, ib_order_id=101, status="OPEN"),
            Mock(spec=Trade, id=2, ib_order_id=102, status="OPEN"),
        ]
        
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.all.return_value = mock_trades
        strategy.session_maker = Mock(return_value=mock_session)
        
        strategy.ib_client.close_position = AsyncMock(return_value=True)
        strategy.risk_manager.update_daily_summary = Mock()
        
        await strategy.flatten_all_positions()
        
        # All trades should be closed
        for trade in mock_trades:
            assert trade.status == "FLATTENED"
        assert strategy.ib_client.close_position.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_trading_cycle_all_conditions(self, strategy):
        """Test complete trading cycle with all conditions"""
        # Setup strategy state
        strategy.underlying_price = 4250
        strategy.implied_move = 50
        strategy.last_trade_time = datetime.utcnow() - timedelta(hours=1)
        
        # Mock price update
        strategy.ib_client.get_mes_contract = AsyncMock(return_value=Mock())
        strategy.ib_client.get_current_price = AsyncMock(return_value=4255)
        
        # Mock should place trade conditions
        strategy.ib_client.is_market_hours = Mock(return_value=True)
        strategy.risk_manager.should_halt_trading = Mock(return_value=False)
        
        # Mock trade placement
        with patch.object(strategy, 'should_place_trade', return_value=(True, "Good conditions")):
            with patch.object(strategy, 'place_strangle_trade', new_callable=AsyncMock) as mock_place:
                mock_place.return_value = True
                
                await strategy.execute_trading_cycle()
                
                # Should have placed a trade
                mock_place.assert_called_once()

    def test_get_strategy_status_complete(self, strategy):
        """Test complete strategy status"""
        # Setup strategy state
        strategy.underlying_price = 4250
        strategy.implied_move = 50
        strategy.daily_high = 4260
        strategy.daily_low = 4240
        strategy.last_trade_time = datetime.utcnow()
        strategy.session_start_time = datetime.utcnow() - timedelta(hours=2)
        
        # Add price history
        strategy.price_history = [
            (datetime.utcnow() - timedelta(minutes=10), 4245),
            (datetime.utcnow() - timedelta(minutes=5), 4250),
            (datetime.utcnow(), 4255),
        ]
        
        # Mock open trades
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.count.return_value = 3
        strategy.session_maker = Mock(return_value=mock_session)
        
        status = strategy.get_strategy_status()
        
        assert status["underlying_price"] == 4250
        assert status["implied_move"] == 50
        assert status["daily_high"] == 4260
        assert status["daily_low"] == 4240
        assert status["open_trades"] == 3
        assert status["last_trade_time"] is not None
        assert "session_duration" in status
        mock_session.close.assert_called()

    def test_calculate_realized_range_edge_cases(self, strategy):
        """Test realized range calculation edge cases"""
        # Empty history
        strategy.price_history = []
        assert strategy.calculate_realized_range(60) == 0
        
        # Single price point
        strategy.price_history = [(datetime.utcnow(), 4250)]
        assert strategy.calculate_realized_range(60) == 0
        
        # All prices the same
        now = datetime.utcnow()
        strategy.price_history = [
            (now - timedelta(minutes=10), 4250),
            (now - timedelta(minutes=5), 4250),
            (now, 4250),
        ]
        assert strategy.calculate_realized_range(60) == 0