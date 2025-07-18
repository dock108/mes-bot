"""Integration tests for backtester with new data sources"""

import pytest
import asyncio
from datetime import date, timedelta
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np

from app.backtester import LottoGridBacktester
from app.data_providers.vix_provider import VIXProvider


class TestBacktesterIntegration:
    """Test backtester with VIX integration"""

    @pytest.fixture
    def backtester(self, test_db_url):
        """Create backtester instance"""
        return LottoGridBacktester(test_db_url)

    @pytest.fixture
    def mock_price_data(self):
        """Create mock price data"""
        dates = pd.date_range(start="2025-07-10 09:30", end="2025-07-10 16:00", freq="5min")
        data = pd.DataFrame({
            "Open": np.random.uniform(4200, 4300, len(dates)),
            "High": np.random.uniform(4210, 4310, len(dates)),
            "Low": np.random.uniform(4190, 4290, len(dates)),
            "Close": np.random.uniform(4200, 4300, len(dates)),
            "Volume": np.random.uniform(1000, 5000, len(dates))
        }, index=dates)
        return data

    def test_calculate_implied_volatility_with_vix(self, backtester, mock_price_data):
        """Test volatility calculation with VIX data"""
        # Mock VIX provider
        mock_vix_provider = Mock(spec=VIXProvider)
        mock_vix_provider.get_vix_value.return_value = 15.0
        backtester.vix_provider = mock_vix_provider

        # Calculate volatility with VIX
        volatility = backtester.calculate_implied_volatility(
            mock_price_data, 
            trading_date=date(2025, 7, 10)
        )

        # Should blend realized and VIX volatility
        assert 0.05 <= volatility <= 2.0
        mock_vix_provider.get_vix_value.assert_called_once_with(date(2025, 7, 10))

    def test_calculate_implied_volatility_without_vix(self, backtester, mock_price_data):
        """Test volatility calculation without VIX data"""
        # No VIX provider
        backtester.vix_provider = None

        # Calculate volatility without VIX
        volatility = backtester.calculate_implied_volatility(mock_price_data)

        # Should use only realized volatility
        assert 0.05 <= volatility <= 2.0

    def test_calculate_implied_volatility_vix_error(self, backtester, mock_price_data):
        """Test volatility calculation when VIX fetch fails"""
        # Mock VIX provider that raises error
        mock_vix_provider = Mock(spec=VIXProvider)
        mock_vix_provider.get_vix_value.side_effect = Exception("VIX API error")
        backtester.vix_provider = mock_vix_provider

        # Should fall back to realized volatility
        volatility = backtester.calculate_implied_volatility(
            mock_price_data,
            trading_date=date(2025, 7, 10)
        )

        assert 0.05 <= volatility <= 2.0

    @pytest.mark.asyncio
    @patch("app.backtester.VIXProvider")
    async def test_run_backtest_with_vix_integration(self, mock_vix_class, backtester):
        """Test full backtest run with VIX integration"""
        # Mock VIX provider
        mock_vix_instance = Mock()
        mock_vix_instance.get_vix_value.return_value = 18.0
        mock_vix_class.return_value = mock_vix_instance

        # Mock fetch_historical_data to return test data
        with patch.object(backtester, 'fetch_historical_data') as mock_fetch:
            # Create test data
            dates = pd.date_range(start="2025-07-10 09:30", end="2025-07-10 16:00", freq="5min")
            mock_data = pd.DataFrame({
                "Open": [4250] * len(dates),
                "High": [4260] * len(dates),
                "Low": [4240] * len(dates),
                "Close": [4255] * len(dates),
                "Volume": [1000] * len(dates)
            }, index=dates)
            mock_fetch.return_value = mock_data

            # Run backtest (await async method)
            results = await backtester.run_backtest(
                start_date=date(2025, 7, 10),
                end_date=date(2025, 7, 10),
                initial_capital=5000
            )

            # Verify VIX provider was initialized
            mock_vix_class.assert_called_once()
            
            # Verify results structure
            assert "total_trades" in results
            assert "final_capital" in results
            assert "total_return" in results

    @pytest.mark.asyncio
    async def test_run_backtest_without_vix_provider(self, backtester):
        """Test backtest continues without VIX when provider fails"""
        # Mock VIXProvider to raise error on init
        with patch("app.backtester.VIXProvider", side_effect=Exception("No API key")):
            # Mock fetch_historical_data
            with patch.object(backtester, 'fetch_historical_data') as mock_fetch:
                dates = pd.date_range(start="2025-07-10 09:30", end="2025-07-10 16:00", freq="5min")
                mock_data = pd.DataFrame({
                    "Open": [4250] * len(dates),
                    "High": [4260] * len(dates),
                    "Low": [4240] * len(dates),
                    "Close": [4255] * len(dates),
                    "Volume": [1000] * len(dates)
                }, index=dates)
                mock_fetch.return_value = mock_data

                # Should still run without VIX (await async method)
                results = await backtester.run_backtest(
                    start_date=date(2025, 7, 10),
                    end_date=date(2025, 7, 10),
                    initial_capital=5000
                )

                assert backtester.vix_provider is None
                assert "total_trades" in results

    @patch("yfinance.Ticker")
    def test_fetch_historical_data_with_mes_fallback(self, mock_ticker, backtester):
        """Test data fetching tries MES=F first, then falls back"""
        # First call (MES=F) returns empty
        mock_ticker_mes = Mock()
        mock_ticker_mes.history.return_value = pd.DataFrame()
        
        # Second call (ES=F) returns data
        dates = pd.date_range(start="2025-07-10", end="2025-07-10", periods=10)
        mock_data = pd.DataFrame({
            "Open": [4250] * 10,
            "High": [4260] * 10,
            "Low": [4240] * 10,
            "Close": [4255] * 10,
            "Volume": [1000] * 10
        }, index=dates)
        mock_ticker_es = Mock()
        mock_ticker_es.history.return_value = mock_data
        
        # Configure mock to return different tickers
        mock_ticker.side_effect = [mock_ticker_mes, mock_ticker_es]
        
        # Fetch data
        data = backtester.fetch_historical_data(
            "MES",
            date(2025, 7, 10),
            date(2025, 7, 10)
        )
        
        # Should have tried MES=F first, then ES=F
        assert mock_ticker.call_count == 2
        assert mock_ticker.call_args_list[0][0][0] == "MES=F"
        assert mock_ticker.call_args_list[1][0][0] == "ES=F"
        assert len(data) == 10

    @patch("yfinance.Ticker")
    def test_fetch_historical_data_all_fail(self, mock_ticker, backtester):
        """Test error when all data sources fail"""
        # All tickers return empty data
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        # Should raise error after trying all sources
        with pytest.raises(ValueError, match="Unable to fetch data from MES=F, ES=F, or SPY"):
            backtester.fetch_historical_data(
                "MES",
                date(2025, 7, 10),
                date(2025, 7, 10)
            )
        
        # Should have tried all three tickers
        assert mock_ticker.call_count == 3