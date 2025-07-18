"""Tests for configuration module"""

import os
import pytest
from unittest.mock import patch
from datetime import time

from app.config import TradingConfig, MarketHours, Config, config


class TestTradingConfig:
    """Test trading configuration"""

    def test_default_values(self):
        """Test default configuration values"""
        cfg = TradingConfig()
        
        assert cfg.max_open_trades == 15
        assert cfg.max_premium_per_strangle == 25.0
        assert cfg.profit_target_multiplier == 4.0
        assert cfg.implied_move_multiplier_1 == 1.25
        assert cfg.implied_move_multiplier_2 == 1.5
        assert cfg.volatility_threshold == 0.67
        assert cfg.min_time_between_trades == 30
        assert cfg.daily_loss_limit == 750.0
        assert cfg.critical_equity_threshold == 0.3
        assert cfg.consecutive_loss_limit == 10

    def test_env_override(self):
        """Test environment variable overrides"""
        # Since environment variables are read at module import time,
        # we need to test the mechanism differently
        import importlib
        
        with patch.dict(os.environ, {
            "MAX_OPEN_TRADES": "20",
            "MAX_PREMIUM_PER_STRANGLE": "30.0",
            "PROFIT_TARGET_MULTIPLIER": "5.0"
        }):
            # Reload the config module to pick up new env vars
            from app import config as config_module
            importlib.reload(config_module)
            
            cfg = config_module.TradingConfig()
            
            assert cfg.max_open_trades == 20
            assert cfg.max_premium_per_strangle == 30.0
            assert cfg.profit_target_multiplier == 5.0
            
            # Reload again to restore original state
            importlib.reload(config_module)


class TestMarketHours:
    """Test market hours configuration"""

    def test_default_hours(self):
        """Test default market hours"""
        hours = MarketHours()
        
        assert hours.market_open == time(9, 30)
        assert hours.market_close == time(16, 0)
        assert hours.flatten_time == time(15, 58)

    def test_env_override(self):
        """Test market hours from environment"""
        import importlib
        
        with patch.dict(os.environ, {
            "MARKET_OPEN_HOUR": "8",
            "MARKET_OPEN_MINUTE": "0",
            "MARKET_CLOSE_HOUR": "15",
            "MARKET_CLOSE_MINUTE": "0",
            "FLATTEN_HOUR": "14",
            "FLATTEN_MINUTE": "55"
        }):
            # Reload the config module to pick up new env vars
            from app import config as config_module
            importlib.reload(config_module)
            
            hours = config_module.MarketHours()
            
            assert hours.market_open == time(8, 0)
            assert hours.market_close == time(15, 0)
            assert hours.flatten_time == time(14, 55)
            
            # Reload again to restore original state
            importlib.reload(config_module)


class TestConfig:
    """Test overall settings"""

    def test_config_structure(self):
        """Test config has all components"""
        cfg = Config()
        
        assert hasattr(cfg, 'trading')
        assert hasattr(cfg, 'market_hours')
        assert hasattr(cfg, 'ib')
        assert hasattr(cfg, 'database')
        assert hasattr(cfg, 'ui')
        assert hasattr(cfg, 'logging')

    def test_default_config(self):
        """Test default config values"""
        cfg = Config()
        
        assert cfg.ib.host == "127.0.0.1"
        assert cfg.ib.port == 7497
        assert cfg.ib.client_id == 1
        assert cfg.database.url == "sqlite:///./data/lotto_grid.db"
        assert cfg.logging.level == "INFO"

    def test_config_env_override(self):
        """Test config environment overrides"""
        import importlib
        
        with patch.dict(os.environ, {
            "IB_GATEWAY_HOST": "192.168.1.100",
            "IB_GATEWAY_PORT": "7496",
            "LOG_LEVEL": "DEBUG"
        }):
            # Reload the config module to pick up new env vars
            from app import config as config_module
            importlib.reload(config_module)
            
            cfg = config_module.Config()
            
            assert cfg.ib.host == "192.168.1.100"
            assert cfg.ib.port == 7496
            assert cfg.logging.level == "DEBUG"
            
            # Reload again to restore original state
            importlib.reload(config_module)


class TestConfigSingleton:
    """Test config singleton behavior"""

    def test_config_is_config_instance(self):
        """Test that config is an instance of Config"""
        assert isinstance(config, Config)
        assert hasattr(config, 'trading')
        assert hasattr(config, 'market_hours')
        assert hasattr(config, 'ib')

    def test_config_values(self):
        """Test accessing config values"""
        # Should be able to access nested configs
        assert config.trading.max_open_trades > 0
        assert config.market_hours.market_open_hour >= 0
        
        # Should have IB settings
        assert config.ib.port > 0