"""Extended tests for strategy.py to improve coverage"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.ib_client import IBClient
from app.models import Trade
from app.risk_manager import RiskManager
from app.strategy import LottoGridStrategy


class TestLottoGridStrategyExtended:
    """Extended test cases for uncovered strategy methods"""

    @pytest.fixture
    def strategy(self, test_db_url):
        """Create strategy instance with mocked dependencies"""
        mock_ib_client = Mock(spec=IBClient)
        mock_risk_manager = Mock(spec=RiskManager)

        strategy = LottoGridStrategy(mock_ib_client, mock_risk_manager, test_db_url)
        return strategy

    @pytest.mark.asyncio
    async def test_initialize_daily_session_ib_error(self, strategy):
        """Test daily initialization when IB client fails"""
        # Mock IB client to raise error
        strategy.ib_client.get_mes_contract = AsyncMock(
            side_effect=Exception("IB connection error")
        )

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

        # Mock IB client methods
        strategy.ib_client.get_today_expiry_string = Mock(return_value="20250710")
        strategy.ib_client.get_mes_option_contract = AsyncMock(return_value=Mock())
        strategy.ib_client.get_current_price = AsyncMock(
            side_effect=[2.5, 2.0]
        )  # Call and put prices
        strategy.ib_client.get_account_values = AsyncMock(return_value={"NetLiquidation": 5000})
        strategy.ib_client.place_strangle = AsyncMock(
            return_value={
                "call_strike": 4300,
                "put_strike": 4200,
                "call_price": 2.5,
                "put_price": 2.0,
                "total_premium": 22.5,
                "call_trades": [Mock(order=Mock(orderId=123))],
                "put_trades": [Mock(order=Mock(orderId=124))],
            }
        )

        # Mock risk check
        strategy.risk_manager.can_open_new_trade = Mock(return_value=(True, "OK"))

        # Mock session for recording trade
        mock_session = Mock()
        strategy.session_maker = Mock(return_value=mock_session)

        # Execute (only 2 parameters)
        result = await strategy.place_strangle_trade(4300, 4200)

        assert result is not None
        strategy.ib_client.place_strangle.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_strangle_trade_risk_rejection(self, strategy):
        """Test strangle trade rejected by risk manager"""
        # Setup mocks
        strategy.ib_client.get_today_expiry_string = Mock(return_value="20250710")
        strategy.ib_client.get_mes_option_contract = AsyncMock(return_value=Mock())
        strategy.ib_client.get_current_price = AsyncMock(side_effect=[2.5, 2.0])
        strategy.ib_client.get_account_values = AsyncMock(return_value={"NetLiquidation": 5000})
        strategy.risk_manager.can_open_new_trade = Mock(return_value=(False, "Max trades reached"))

        result = await strategy.place_strangle_trade(4300, 4200)

        assert result is None
        strategy.ib_client.place_strangle.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_open_positions_complete(self, strategy):
        """Test updating open positions with exits"""
        # Create mock trade
        mock_trade = Mock(spec=Trade)
        mock_trade.id = 1
        mock_trade.call_strike = 4300
        mock_trade.put_strike = 4200
        mock_trade.call_premium = 2.0
        mock_trade.put_premium = 2.0
        mock_trade.total_premium = 20.0  # (2+2) * 5
        mock_trade.entry_time = datetime.utcnow() - timedelta(hours=1)
        mock_trade.status = "OPEN"

        # Mock database session
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_trade]
        strategy.session_maker = Mock(return_value=mock_session)

        # Mock IB client responses
        strategy.ib_client.get_today_expiry_string = Mock(return_value="20250710")
        strategy.ib_client.get_mes_option_contract = AsyncMock(return_value=Mock())
        strategy.ib_client.get_current_price = AsyncMock(
            side_effect=[8.0, 2.0]
        )  # Call hit 4x target

        # Execute
        await strategy.update_open_positions()

        # Verify P&L was calculated
        assert mock_trade.unrealized_pnl == (8.0 - 2.0 + 2.0 - 2.0) * 5  # 30.0
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_update_open_positions_expiry(self, strategy):
        """Test position update at expiry"""
        # Create trade with all required attributes
        mock_trade = Mock(spec=Trade)
        mock_trade.id = 1
        mock_trade.call_strike = 4300
        mock_trade.put_strike = 4200
        mock_trade.call_premium = 2.0
        mock_trade.put_premium = 2.0
        mock_trade.entry_time = datetime.utcnow() - timedelta(hours=8)  # Past expiry
        mock_trade.status = "OPEN"

        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_trade]
        strategy.session_maker = Mock(return_value=mock_session)

        # Mock IB client
        strategy.ib_client.get_today_expiry_string = Mock(return_value="20250710")
        strategy.ib_client.get_mes_option_contract = AsyncMock(return_value=Mock())
        strategy.ib_client.get_current_price = AsyncMock(side_effect=[1.0, 1.0])  # Current prices

        await strategy.update_open_positions()

        # Should calculate P&L
        assert mock_trade.unrealized_pnl is not None
        mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_update_open_positions_error_handling(self, strategy):
        """Test error handling in position updates"""
        mock_trade = Mock(spec=Trade)
        mock_trade.id = 1
        mock_trade.call_strike = 4300
        mock_trade.put_strike = 4200
        mock_trade.call_premium = 2.0
        mock_trade.put_premium = 2.0
        mock_trade.status = "OPEN"

        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_trade]
        strategy.session_maker = Mock(return_value=mock_session)

        # Mock query to raise error in main method
        mock_session.query.side_effect = Exception("Database error")

        # Should not raise, just log error
        await strategy.update_open_positions()

        # Session should rollback on database error
        mock_session.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_flatten_all_positions(self, strategy):
        """Test flattening all positions"""
        # Mock open trades with all required attributes
        mock_trades = [
            Mock(
                spec=Trade,
                id=1,
                status="OPEN",
                call_tp_order_id=201,
                put_tp_order_id=202,
                call_status="OPEN",
                put_status="OPEN",
                call_strike=4300,
                put_strike=4200,
                unrealized_pnl=-10,
                total_premium=20,
            ),
            Mock(
                spec=Trade,
                id=2,
                status="OPEN",
                call_tp_order_id=203,
                put_tp_order_id=204,
                call_status="OPEN",
                put_status="OPEN",
                call_strike=4350,
                put_strike=4150,
                unrealized_pnl=5,
                total_premium=20,
            ),
        ]

        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.all.return_value = mock_trades
        strategy.session_maker = Mock(return_value=mock_session)

        # Mock IB client methods
        strategy.ib_client.get_today_expiry_string = Mock(return_value="20250710")
        strategy.ib_client.cancel_order = AsyncMock(return_value=True)
        strategy.ib_client.get_mes_option_contract = AsyncMock(return_value=Mock())
        strategy.ib_client.close_position_at_market = AsyncMock(return_value=Mock())

        result = await strategy.flatten_all_positions()

        assert result is True
        # All trades should be marked as expired
        for trade in mock_trades:
            assert trade.status == "EXPIRED"
            assert trade.exit_time is not None
        assert strategy.ib_client.cancel_order.call_count == 4  # 2 trades * 2 TP orders
        mock_session.commit.assert_called()

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
        with patch.object(strategy, "should_place_trade", return_value=(True, "Good conditions")):
            with patch.object(
                strategy, "place_strangle_trade", new_callable=AsyncMock
            ) as mock_place:
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

        status = strategy.get_strategy_status()

        assert status["underlying_price"] == 4250
        assert status["implied_move"] == 50
        assert status["daily_high"] == 4260
        assert status["daily_low"] == 4240
        assert status["daily_range"] == 20  # 4260 - 4240
        assert status["last_trade_time"] is not None
        assert status["session_start"] is not None
        assert "realized_range_60m" in status
        assert status["price_history_length"] == 3

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
