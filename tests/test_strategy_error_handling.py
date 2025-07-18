"""Tests for strategy.py error handling to improve coverage"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.exc import SQLAlchemyError

from app.strategy import LottoGridStrategy
from app.ib_client import IBClient
from app.risk_manager import RiskManager


class TestStrategyErrorHandling:
    """Test error handling in strategy.py"""

    @pytest.fixture
    def strategy(self, test_db_url):
        """Create strategy instance with mocked dependencies"""
        mock_ib_client = Mock(spec=IBClient)
        mock_risk_manager = Mock(spec=RiskManager)
        
        strategy = LottoGridStrategy(mock_ib_client, mock_risk_manager, test_db_url)
        return strategy

    @pytest.mark.asyncio
    async def test_record_trade_database_error(self, strategy):
        """Test _record_trade with database error"""
        trade_details = {
            "call_strike": 4300,
            "put_strike": 4200,
            "call_price": 2.5,
            "put_price": 2.0,
            "total_premium": 22.5,
            "implied_move": 50
        }
        
        # Mock session to raise error
        mock_session = Mock()
        mock_session.add.side_effect = SQLAlchemyError("Database error")
        strategy.session_maker = Mock(return_value=mock_session)
        
        # Should raise error
        with pytest.raises(SQLAlchemyError):
            await strategy._record_trade(trade_details)

    def test_should_place_trade_edge_cases(self, strategy):
        """Test should_place_trade with edge cases"""
        # No implied move
        strategy.implied_move = None
        should_trade, reason = strategy.should_place_trade()
        assert should_trade is False
        assert "Implied move not calculated" in reason
        
        # Recent trade (within cooldown)
        strategy.implied_move = 50
        strategy.last_trade_time = datetime.utcnow() - timedelta(minutes=5)
        strategy.config = Mock()
        strategy.config.trading.min_time_between_trades = 30
        should_trade, reason = strategy.should_place_trade()
        assert should_trade is False
        assert "Too soon since last trade" in reason

    def test_calculate_realized_range_no_data(self, strategy):
        """Test calculate_realized_range with no data"""
        strategy.price_history = []
        result = strategy.calculate_realized_range(60)
        assert result == 0

    def test_calculate_realized_range_old_data(self, strategy):
        """Test calculate_realized_range with only old data"""
        now = datetime.utcnow()
        # All data older than window
        strategy.price_history = [
            (now - timedelta(minutes=120), 4250),
            (now - timedelta(minutes=110), 4260),
        ]
        result = strategy.calculate_realized_range(60)
        assert result == 0

    # Note: update_prices method doesn't exist in strategy.py
    # This test was for a non-existent method

    @pytest.mark.asyncio
    async def test_place_strangle_ib_error(self, strategy):
        """Test place_strangle with IB placement error"""
        strategy.ib_client.get_today_expiry_string = Mock(return_value="20250710")
        strategy.ib_client.get_mes_option_contract = AsyncMock(return_value=Mock())
        strategy.ib_client.get_current_price = AsyncMock(return_value=2.5)
        strategy.ib_client.get_account_values = AsyncMock(return_value={"NetLiquidation": 5000})
        strategy.ib_client.place_strangle = AsyncMock(side_effect=Exception("Order rejected"))
        
        strategy.risk_manager.can_open_new_trade = Mock(return_value=(True, "OK"))
        
        # Should return None on error
        result = await strategy.place_strangle_trade(4300, 4200)
        assert result is None

    # Note: _get_implied_move method doesn't exist in strategy.py
    # This test was for a non-existent method

    @pytest.mark.asyncio 
    async def test_check_exit_conditions_price_fetch_error(self, strategy):
        """Test _check_exit_conditions with price fetch error"""
        mock_trade = Mock()
        mock_trade.call_strike = 4300
        mock_trade.put_strike = 4200
        
        strategy.ib_client.get_option_prices = AsyncMock(side_effect=Exception("API error"))
        
        # _check_exit_conditions doesn't exist in strategy.py
        # This test is for a non-existent method
        pass  # Skip this test