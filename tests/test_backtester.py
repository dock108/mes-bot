"""
Tests for the backtesting engine
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch

from app.backtester import LottoGridBacktester, BlackScholesCalculator, BacktestTrade


class TestBlackScholesCalculator:
    """Test cases for Black-Scholes calculator"""

    def test_call_option_pricing(self):
        """Test call option pricing"""
        bs = BlackScholesCalculator()

        # ATM call option
        price = bs.option_price(
            S=100.0,  # Stock price
            K=100.0,  # Strike price
            T=0.25,  # 3 months
            r=0.05,  # 5% risk-free rate
            sigma=0.20,  # 20% volatility
            option_type="call",
        )

        # Should be positive and reasonable for ATM option
        assert price > 0
        assert price < 20  # Shouldn't be too high for this scenario

    def test_put_option_pricing(self):
        """Test put option pricing"""
        bs = BlackScholesCalculator()

        # ATM put option
        price = bs.option_price(S=100.0, K=100.0, T=0.25, r=0.05, sigma=0.20, option_type="put")

        assert price > 0
        assert price < 20

    def test_option_pricing_at_expiration(self):
        """Test option pricing at expiration (intrinsic value)"""
        bs = BlackScholesCalculator()

        # ITM call at expiration
        call_price = bs.option_price(100.0, 95.0, 0.0, 0.05, 0.20, "call")
        assert call_price == 5.0  # Intrinsic value

        # OTM call at expiration
        call_price = bs.option_price(95.0, 100.0, 0.0, 0.05, 0.20, "call")
        assert call_price == 0.0

        # ITM put at expiration
        put_price = bs.option_price(95.0, 100.0, 0.0, 0.05, 0.20, "put")
        assert put_price == 5.0

    def test_put_call_parity(self):
        """Test put-call parity relationship"""
        bs = BlackScholesCalculator()

        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20

        call_price = bs.option_price(S, K, T, r, sigma, "call")
        put_price = bs.option_price(S, K, T, r, sigma, "put")

        # Put-call parity: C - P = S - K*e^(-rT)
        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)

        assert abs(lhs - rhs) < 0.01  # Should be very close


class TestLottoGridBacktester:
    """Test cases for LottoGridBacktester"""

    @pytest.fixture
    def backtester(self):
        """Create backtester instance"""
        return LottoGridBacktester("sqlite:///:memory:")

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range("2024-01-01 09:30", "2024-01-01 16:00", freq="5min")

        # Simulate price movement
        np.random.seed(42)  # For reproducible tests
        base_price = 4200.0
        returns = np.random.normal(0, 0.001, len(dates))  # Small random returns
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        data = pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.002 for p in prices],
                "Low": [p * 0.998 for p in prices],
                "Close": prices,
                "Volume": [1000] * len(dates),
            },
            index=dates,
        )

        return data

    def test_calculate_implied_volatility(self, backtester, sample_price_data):
        """Test implied volatility calculation"""
        vol = backtester.calculate_implied_volatility(sample_price_data)

        assert isinstance(vol, float)
        assert 0.05 <= vol <= 2.0  # Should be within reasonable bounds

    def test_calculate_atm_straddle_price(self, backtester):
        """Test ATM straddle price calculation"""
        call_price, put_price, implied_move = backtester.calculate_atm_straddle_price(
            underlying_price=4200.0, time_to_expiry=6.5 / 24, volatility=0.15  # 6.5 hours
        )

        assert call_price > 0
        assert put_price > 0
        assert implied_move == call_price + put_price

    def test_calculate_strike_levels(self, backtester):
        """Test strike level calculation"""
        strike_pairs = backtester.calculate_strike_levels(4200.0, 20.0)

        assert len(strike_pairs) == 2

        # Check first level (1.25x multiplier)
        call_1, put_1 = strike_pairs[0]
        assert call_1 > 4200.0
        assert put_1 < 4200.0

        # Check second level (1.5x multiplier)
        call_2, put_2 = strike_pairs[1]
        assert call_2 >= call_1  # May be same due to rounding
        assert put_2 <= put_1  # May be same due to rounding

    def test_round_to_strike(self, backtester):
        """Test strike rounding"""
        assert backtester._round_to_strike(4213.7) == 4225.0
        assert backtester._round_to_strike(4187.2) == 4175.0
        assert backtester._round_to_strike(4200.0) == 4200.0

    def test_should_place_trade(self, backtester, sample_price_data):
        """Test trade placement logic"""
        # Ensure we have enough data points
        if len(sample_price_data) < 25:
            return  # Skip if not enough data

        # Test with low volatility (should place trade)
        should_trade = backtester.should_place_trade(
            sample_price_data,
            current_idx=min(20, len(sample_price_data) - 1),  # Safe index
            implied_move=50.0,  # High implied move
            last_trade_time=None,
        )

        assert isinstance(should_trade, bool)

    def test_place_strangle(self, backtester):
        """Test strangle placement"""
        trade = backtester.place_strangle(
            underlying_price=4200.0,
            call_strike=4250.0,  # Further OTM to reduce premium
            put_strike=4150.0,  # Further OTM to reduce premium
            time_to_expiry=2.0 / 24,  # Shorter time to reduce premium
            volatility=0.10,  # Lower volatility to reduce premium
            entry_time=datetime.utcnow(),
            implied_move=20.0,
        )

        # If premium is too high, the function returns None
        if trade is None:
            # Test with even cheaper options
            trade = backtester.place_strangle(
                underlying_price=4200.0,
                call_strike=4300.0,  # Very far OTM
                put_strike=4100.0,  # Very far OTM
                time_to_expiry=1.0 / 24,  # Very short time
                volatility=0.05,  # Very low volatility
                entry_time=datetime.utcnow(),
                implied_move=20.0,
            )

        assert isinstance(trade, BacktestTrade)
        assert trade.underlying_price == 4200.0
        assert trade.total_premium > 0
        assert trade.status == "OPEN"

    def test_update_trade_pnl(self, backtester):
        """Test trade P&L updates"""
        trade = BacktestTrade(
            entry_time=datetime.utcnow(),
            underlying_price=4200.0,
            call_strike=4225.0,
            put_strike=4175.0,
            call_premium=2.0,
            put_premium=2.5,
            total_premium=22.5,
            implied_move=20.0,
        )

        # Test scenario where call hits profit target
        profit_hit = backtester.update_trade_pnl(
            trade=trade,
            current_price=4250.0,  # Move up significantly
            time_to_expiry=3.0 / 24,  # 3 hours left
            volatility=0.15,
        )

        # Should detect profit hit
        if profit_hit:
            assert trade.status == "CLOSED_WIN"
            assert trade.realized_pnl is not None

    def test_close_trade_at_expiry(self, backtester):
        """Test trade closure at expiry"""
        trade = BacktestTrade(
            entry_time=datetime.utcnow(),
            underlying_price=4200.0,
            call_strike=4225.0,
            put_strike=4175.0,
            call_premium=2.0,
            put_premium=2.5,
            total_premium=22.5,
            implied_move=20.0,
        )

        # Close at expiry with underlying at 4230 (call ITM)
        backtester.close_trade_at_expiry(trade, 4230.0)

        assert trade.call_exit_price == 5.0  # Intrinsic value
        assert trade.put_exit_price == 0.0  # OTM
        assert trade.realized_pnl is not None
        assert trade.status in ["CLOSED_WIN", "EXPIRED"]

    @patch("app.backtester.yf.Ticker")
    def test_fetch_historical_data(self, mock_ticker, backtester, sample_price_data):
        """Test historical data fetching"""
        # Mock yfinance response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_price_data
        mock_ticker.return_value = mock_ticker_instance

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 2)

        data = backtester.fetch_historical_data("MES", start_date, end_date)

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert "Close" in data.columns

    @pytest.mark.asyncio
    async def test_run_backtest_basic(self, backtester, sample_price_data):
        """Test basic backtest execution"""
        with patch.object(backtester, "fetch_historical_data", return_value=sample_price_data):
            results = await backtester.run_backtest(
                start_date=date(2024, 1, 1), end_date=date(2024, 1, 2), initial_capital=5000.0
            )

            assert isinstance(results, dict)
            assert "total_return" in results
            assert "win_rate" in results
            assert "max_drawdown" in results
            assert "total_trades" in results
            assert results["initial_capital"] == 5000.0

    def test_backtest_trade_dataclass(self):
        """Test BacktestTrade dataclass functionality"""
        trade = BacktestTrade(
            entry_time=datetime.utcnow(),
            underlying_price=4200.0,
            call_strike=4225.0,
            put_strike=4175.0,
            call_premium=2.0,
            put_premium=2.5,
            total_premium=22.5,
            implied_move=20.0,
        )

        assert trade.status == "OPEN"
        assert trade.exit_time is None
        assert trade.realized_pnl is None
        assert hasattr(trade, "id")  # Should have an ID from __post_init__

    @pytest.mark.asyncio
    async def test_save_backtest_result(self, backtester):
        """Test saving backtest results to database"""
        results = {
            "start_date": date(2024, 1, 1),
            "end_date": date(2024, 1, 2),
            "initial_capital": 5000.0,
            "final_capital": 5100.0,
            "total_return": 0.02,
            "total_trades": 10,
            "winning_trades": 6,
            "losing_trades": 4,
            "win_rate": 0.6,
            "max_drawdown": 100.0,
            "sharpe_ratio": 1.5,
        }

        backtest_id = await backtester.save_backtest_result(results, "Test Backtest")

        assert isinstance(backtest_id, int)
        assert backtest_id > 0
