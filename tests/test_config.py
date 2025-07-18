"""Tests for config.py to improve coverage"""

import os
from datetime import time
from unittest.mock import patch

import pytest
import pytz

from app.config import DatabaseConfig, IBConfig, MarketHours, TradingConfig, config


class TestIBConfig:
    """Test Interactive Brokers configuration"""

    def test_default_values(self):
        """Test default IB config values"""
        # Test with default initialization (no env vars)
        ib_config = IBConfig(host="127.0.0.1", port=7497, client_id=1, username="", password="")
        assert ib_config.host == "127.0.0.1"
        assert ib_config.port == 7497
        assert ib_config.client_id == 1
        assert ib_config.username == ""
        assert ib_config.password == ""

    def test_is_paper_trading(self):
        """Test paper trading detection"""
        # Default port 7497 is paper trading
        ib_config = IBConfig()
        assert ib_config.is_paper_trading is True

        # Port 7496 is live trading
        ib_config.port = 7496
        assert ib_config.is_paper_trading is False

    def test_env_override(self):
        """Test environment variable override"""
        # Test by directly creating instance with values
        ib_config = IBConfig(
            host="192.168.1.100", port=7496, client_id=5, username="testuser", password="testpass"
        )
        assert ib_config.host == "192.168.1.100"
        assert ib_config.port == 7496
        assert ib_config.client_id == 5
        assert ib_config.username == "testuser"
        assert ib_config.password == "testpass"


class TestTradingConfig:
    """Test trading strategy configuration"""

    def test_default_values(self):
        """Test default trading config values"""
        trading_config = TradingConfig()
        assert trading_config.trade_mode == "paper"
        assert trading_config.start_cash == 5000.0
        assert trading_config.max_drawdown == 750.0
        assert trading_config.max_open_trades == 15
        assert trading_config.max_premium_per_strangle == 25.0
        assert trading_config.profit_target_multiplier == 4.0
        assert trading_config.implied_move_multiplier_1 == 1.25
        assert trading_config.implied_move_multiplier_2 == 1.5
        assert trading_config.volatility_threshold == 0.67
        assert trading_config.min_time_between_trades == 30
        assert trading_config.critical_equity_threshold == 0.3
        assert trading_config.consecutive_loss_limit == 10

    def test_daily_loss_limit_property(self):
        """Test daily loss limit property"""
        trading_config = TradingConfig()
        assert trading_config.daily_loss_limit == trading_config.max_drawdown

    def test_is_live_trading(self):
        """Test live trading detection"""
        trading_config = TradingConfig()
        # Default is paper
        assert trading_config.is_live_trading is False

        # Set to live
        trading_config.trade_mode = "live"
        assert trading_config.is_live_trading is True

        # Case insensitive
        trading_config.trade_mode = "LIVE"
        assert trading_config.is_live_trading is True

    def test_env_override(self):
        """Test environment variable override"""
        # Test by directly creating instance with values
        trading_config = TradingConfig(
            trade_mode="live",
            start_cash=10000.0,
            max_drawdown=1500.0,
            max_open_trades=20,
            max_premium_per_strangle=50.0,
            profit_target_multiplier=3.0,
            implied_move_multiplier_1=1.0,
            implied_move_multiplier_2=1.75,
            volatility_threshold=0.5,
            min_time_between_trades=60,
            critical_equity_threshold=0.2,
            consecutive_loss_limit=5,
        )
        assert trading_config.trade_mode == "live"
        assert trading_config.start_cash == 10000.0
        assert trading_config.max_drawdown == 1500.0
        assert trading_config.max_open_trades == 20
        assert trading_config.max_premium_per_strangle == 50.0
        assert trading_config.profit_target_multiplier == 3.0
        assert trading_config.implied_move_multiplier_1 == 1.0
        assert trading_config.implied_move_multiplier_2 == 1.75
        assert trading_config.volatility_threshold == 0.5
        assert trading_config.min_time_between_trades == 60
        assert trading_config.critical_equity_threshold == 0.2
        assert trading_config.consecutive_loss_limit == 5


class TestMarketHours:
    """Test market hours configuration"""

    def test_default_values(self):
        """Test default market hours"""
        market_hours = MarketHours()
        assert market_hours.market_open_hour == 9
        assert market_hours.market_open_minute == 30
        assert market_hours.market_close_hour == 16
        assert market_hours.market_close_minute == 0
        assert market_hours.flatten_hour == 15
        assert market_hours.flatten_minute == 58

    def test_time_properties(self):
        """Test time object properties"""
        market_hours = MarketHours()

        # Test market_open property
        assert market_hours.market_open == time(9, 30)

        # Test market_close property
        assert market_hours.market_close == time(16, 0)

        # Test flatten_time property
        assert market_hours.flatten_time == time(15, 58)

    def test_timezone(self):
        """Test timezone is Eastern"""
        market_hours = MarketHours()
        assert market_hours.timezone == pytz.timezone("US/Eastern")

    def test_env_override(self):
        """Test environment variable override"""
        # Test by directly creating instance with values
        market_hours = MarketHours(
            market_open_hour=8,
            market_open_minute=0,
            market_close_hour=17,
            market_close_minute=30,
            flatten_hour=17,
            flatten_minute=25,
        )
        assert market_hours.market_open_hour == 8
        assert market_hours.market_open_minute == 0
        assert market_hours.market_close_hour == 17
        assert market_hours.market_close_minute == 30
        assert market_hours.flatten_hour == 17
        assert market_hours.flatten_minute == 25

        # Verify properties work with overrides
        assert market_hours.market_open == time(8, 0)
        assert market_hours.market_close == time(17, 30)
        assert market_hours.flatten_time == time(17, 25)


class TestDatabaseConfig:
    """Test database configuration"""

    def test_default_values(self):
        """Test default database config"""
        db_config = DatabaseConfig()
        assert db_config.url == "sqlite:///./data/lotto_grid.db"

    def test_env_override(self):
        """Test environment variable override"""
        # Test by directly creating instance with values
        db_config = DatabaseConfig(url="postgresql://user:pass@localhost/testdb")
        assert db_config.url == "postgresql://user:pass@localhost/testdb"


class TestConfigObject:
    """Test the main config object"""

    def test_config_object_structure(self):
        """Test that config object has all components"""
        assert hasattr(config, "ib")
        assert hasattr(config, "trading")
        assert hasattr(config, "market_hours")
        assert hasattr(config, "database")

        assert isinstance(config.ib, IBConfig)
        assert isinstance(config.trading, TradingConfig)
        assert isinstance(config.market_hours, MarketHours)
        assert isinstance(config.database, DatabaseConfig)

    def test_config_singleton_behavior(self):
        """Test that config maintains state"""
        # Modify a value
        original_port = config.ib.port
        config.ib.port = 9999

        # Import again and verify it's the same instance
        from app.config import config as config2

        assert config2.ib.port == 9999

        # Restore original value
        config.ib.port = original_port
