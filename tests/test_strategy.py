"""
Tests for the trading strategy logic
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from app.ib_client import IBClient
from app.risk_manager import RiskManager
from app.strategy import LottoGridStrategy


class TestLottoGridStrategy:
    """Test cases for LottoGridStrategy"""

    @pytest.fixture
    def mock_ib_client(self):
        """Mock IB client"""
        client = Mock(spec=IBClient)
        client.get_current_price = AsyncMock(return_value=4200.0)
        client.get_atm_straddle_price = AsyncMock(return_value=(15.0, 12.0, 27.0))
        client.place_strangle = AsyncMock(return_value={"total_premium": 20.0})
        client.is_market_hours = Mock(return_value=True)
        client.get_today_expiry_string = Mock(return_value="20241213")
        return client

    @pytest.fixture
    def mock_risk_manager(self):
        """Mock risk manager"""
        risk_mgr = Mock(spec=RiskManager)
        risk_mgr.can_open_new_trade = Mock(return_value=(True, "Trade approved"))
        risk_mgr.set_daily_start_equity = Mock()
        return risk_mgr

    @pytest.fixture
    def strategy(self, mock_ib_client, mock_risk_manager):
        """Create strategy instance with mocks"""
        return LottoGridStrategy(mock_ib_client, mock_risk_manager, "sqlite:///:memory:")

    @pytest.mark.asyncio
    async def test_initialize_daily_session(self, strategy, mock_ib_client):
        """Test daily session initialization"""
        mock_ib_client.get_mes_contract = AsyncMock(return_value=Mock())
        mock_ib_client.get_account_values = AsyncMock(return_value={"NetLiquidation": 5000.0})

        result = await strategy.initialize_daily_session()

        assert result is True
        assert strategy.underlying_price == 4200.0
        assert strategy.implied_move == 27.0
        assert strategy.daily_high == 4200.0
        assert strategy.daily_low == 4200.0

    def test_calculate_strike_levels(self, strategy):
        """Test strike level calculation"""
        strategy.implied_move = 20.0

        strike_pairs = strategy.calculate_strike_levels(4200.0)

        assert len(strike_pairs) == 2

        # First level (1.25x)
        call_strike_1, put_strike_1 = strike_pairs[0]
        assert call_strike_1 == 4225.0  # 4200 + 25 (rounded)
        assert put_strike_1 == 4175.0  # 4200 - 25 (rounded)

        # Second level (1.5x)
        call_strike_2, put_strike_2 = strike_pairs[1]
        assert call_strike_2 == 4225.0  # 4200 + 30 (rounded to 4225)
        assert put_strike_2 == 4175.0  # 4200 - 30 (rounded to 4175)

    def test_round_to_strike(self, strategy):
        """Test strike rounding functionality"""
        assert strategy._round_to_strike(4213.7) == 4225.0
        assert strategy._round_to_strike(4187.2) == 4175.0
        assert strategy._round_to_strike(4200.0) == 4200.0

    def test_update_price_history(self, strategy):
        """Test price history tracking"""
        strategy.update_price_history(4200.0)
        strategy.update_price_history(4210.0)
        strategy.update_price_history(4195.0)

        assert len(strategy.price_history) == 3
        assert strategy.daily_high == 4210.0
        assert strategy.daily_low == 4195.0

    def test_calculate_realized_range(self, strategy):
        """Test realized range calculation"""
        # Add some price history
        now = datetime.utcnow()
        strategy.price_history = [
            (now - timedelta(minutes=90), 4200.0),
            (now - timedelta(minutes=59), 4190.0),  # Changed to 59 to be clearly within window
            (now - timedelta(minutes=30), 4205.0),
            (now, 4195.0),
        ]

        # Calculate range for last 60 minutes
        realized_range = strategy.calculate_realized_range(60)

        # Should include prices from last 60 minutes (4190, 4205, 4195)
        assert realized_range == 15.0  # 4205 - 4190

    def test_should_place_trade_conditions(self, strategy):
        """Test trade placement conditions"""
        strategy.implied_move = 20.0
        strategy.last_trade_time = None

        # Mock price history for low volatility
        now = datetime.utcnow()
        strategy.price_history = [
            (now - timedelta(minutes=60), 4200.0),
            (now, 4205.0),  # Only 5 point range
        ]

        should_trade, reason = strategy.should_place_trade()
        assert should_trade is True
        assert "Conditions met" in reason

    def test_should_place_trade_high_volatility(self, strategy):
        """Test trade rejection due to high volatility"""
        strategy.implied_move = 20.0
        strategy.last_trade_time = None

        # Mock price history for high volatility
        # With implied_move=20, threshold=20*0.67=13.4
        # Use 50-point range to be clearly above threshold
        now = datetime.utcnow()
        strategy.price_history = [
            (now - timedelta(minutes=59), 4200.0),  # Within 60m window
            (now - timedelta(minutes=30), 4250.0),  # 50-point range
            (now, 4220.0),
        ]

        should_trade, reason = strategy.should_place_trade()
        assert should_trade is False
        assert "Realized range" in reason

    def test_should_place_trade_time_restriction(self, strategy):
        """Test trade rejection due to time restrictions"""
        strategy.implied_move = 20.0
        strategy.last_trade_time = datetime.utcnow() - timedelta(minutes=15)  # Recent trade

        should_trade, reason = strategy.should_place_trade()
        assert should_trade is False
        assert "Too soon" in reason

    @pytest.mark.asyncio
    async def test_place_strangle_trade(self, strategy, mock_ib_client, mock_risk_manager):
        """Test strangle placement"""
        # Setup mocks
        mock_ib_client.get_account_values = AsyncMock(return_value={"NetLiquidation": 5000.0})
        mock_ib_client.get_mes_option_contract = AsyncMock(return_value=Mock())
        mock_ib_client.get_current_price = AsyncMock(side_effect=[2.5, 3.0])  # Call and put prices

        # Mock order IDs as integers for database compatibility
        mock_call_trade = Mock()
        mock_call_trade.order.orderId = 12345
        mock_put_trade = Mock()
        mock_put_trade.order.orderId = 12346

        strangle_result = {
            "call_strike": 4225.0,
            "put_strike": 4175.0,
            "call_price": 2.5,
            "put_price": 3.0,
            "total_premium": 27.5,
            "call_trades": [mock_call_trade],
            "put_trades": [mock_put_trade],
        }
        mock_ib_client.place_strangle = AsyncMock(return_value=strangle_result)

        strategy.underlying_price = 4200.0
        strategy.implied_move = 20.0

        result = await strategy.place_strangle_trade(4225.0, 4175.0)

        assert result is not None
        assert "trade_record" in result
        assert "strangle_result" in result

    @pytest.mark.asyncio
    async def test_execute_trading_cycle(self, strategy, mock_ib_client):
        """Test complete trading cycle"""
        # Setup for successful trade
        mock_ib_client.get_mes_contract = AsyncMock(return_value=Mock())
        strategy.underlying_price = 4200.0
        strategy.implied_move = 20.0
        strategy.last_trade_time = None

        # Mock low volatility condition
        now = datetime.utcnow()
        strategy.price_history = [(now - timedelta(minutes=60), 4200.0), (now, 4205.0)]

        # Mock successful strangle placement
        strategy.place_strangle_trade = AsyncMock(return_value={"success": True})

        result = await strategy.execute_trading_cycle()

        assert result is True
        strategy.place_strangle_trade.assert_called_once()

    def test_get_strategy_status(self, strategy):
        """Test strategy status reporting"""
        strategy.underlying_price = 4200.0
        strategy.implied_move = 20.0
        strategy.daily_high = 4210.0
        strategy.daily_low = 4190.0

        status = strategy.get_strategy_status()

        assert status["underlying_price"] == 4200.0
        assert status["implied_move"] == 20.0
        assert status["daily_range"] == 20.0
        assert "timestamp" in status
