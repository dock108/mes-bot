"""Extended tests for config.py to improve coverage"""

import os
from datetime import time
from unittest.mock import patch

import pytest

from app.config import Config, IBConfig, MarketHours, TradingConfig


class TestConfigProperties:
    """Test configuration properties and edge cases"""

    def test_ib_config_is_paper_trading(self):
        """Test is_paper_trading property"""
        # Paper trading port
        ib_config = IBConfig(port=7497)
        assert ib_config.is_paper_trading is True

        # Live trading port
        ib_config = IBConfig(port=7496)
        assert ib_config.is_paper_trading is False

    def test_trading_config_is_live_trading(self):
        """Test is_live_trading property"""
        # Live mode
        trading_config = TradingConfig(trade_mode="live")
        assert trading_config.is_live_trading is True

        # Paper mode
        trading_config = TradingConfig(trade_mode="paper")
        assert trading_config.is_live_trading is False

        # Case insensitive
        trading_config = TradingConfig(trade_mode="LIVE")
        assert trading_config.is_live_trading is True

    def test_config_validation_errors(self):
        """Test config validation with various error conditions"""
        config = Config()

        # Test with invalid max drawdown
        config.trading.max_drawdown = 10000  # Greater than start_cash
        config.trading.start_cash = 5000

        with pytest.raises(ValueError, match="Max drawdown should be less than starting cash"):
            config.validate()

        # Reset for next test
        config.trading.max_drawdown = 750

        # Test with invalid premium
        config.trading.max_premium_per_strangle = 0
        with pytest.raises(ValueError, match="Max premium per strangle must be positive"):
            config.validate()

        # Reset for next test
        config.trading.max_premium_per_strangle = 25

        # Test with invalid profit target
        config.trading.profit_target_multiplier = 0.5
        with pytest.raises(ValueError, match="Profit target multiplier must be > 1"):
            config.validate()

    def test_config_validation_live_trading_requirements(self):
        """Test validation requirements for live trading"""
        config = Config()
        config.trading.trade_mode = "live"
        config.ib.username = ""
        config.ib.password = ""

        with pytest.raises(ValueError, match="IB username and password required for live trading"):
            config.validate()

    def test_config_repr(self):
        """Test config string representation"""
        config = Config()
        config.trading.trade_mode = "paper"
        config.ib.port = 7497

        repr_str = repr(config)
        assert "<Config(mode=paper, paper=True)>" == repr_str

        # Test with live mode
        config.trading.trade_mode = "live"
        config.ib.port = 7496
        repr_str = repr(config)
        assert "<Config(mode=live, paper=False)>" == repr_str

    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_config_directory_creation(self, mock_makedirs, mock_exists):
        """Test config creates necessary directories"""
        mock_exists.return_value = False

        config = Config()
        config.validate()

        # Should create both data and log directories
        assert mock_makedirs.call_count == 2

    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_config_directory_creation_failure(self, mock_makedirs, mock_exists):
        """Test config handles directory creation failures"""
        mock_exists.return_value = False
        mock_makedirs.side_effect = OSError("Permission denied")

        config = Config()

        with pytest.raises(ValueError, match="Cannot create"):
            config.validate()
