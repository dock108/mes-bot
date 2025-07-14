"""
Configuration management for the MES 0DTE Lotto-Grid Options Bot
"""

import os
from dataclasses import dataclass
from typing import Optional

import pytz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class IBConfig:
    """Interactive Brokers configuration"""

    host: str = os.getenv("IB_GATEWAY_HOST", "127.0.0.1")
    port: int = int(os.getenv("IB_GATEWAY_PORT", "7497"))
    client_id: int = int(os.getenv("IB_CLIENT_ID", "1"))
    username: str = os.getenv("IB_USERNAME", "")
    password: str = os.getenv("IB_PASSWORD", "")

    @property
    def is_paper_trading(self) -> bool:
        """Check if using paper trading port"""
        return self.port == 7497


@dataclass
class TradingConfig:
    """Trading strategy configuration"""

    trade_mode: str = os.getenv("TRADE_MODE", "paper")
    start_cash: float = float(os.getenv("START_CASH", "5000"))
    max_drawdown: float = float(os.getenv("MAX_DRAW", "750"))
    max_open_trades: int = int(os.getenv("MAX_OPEN_TRADES", "15"))
    max_premium_per_strangle: float = float(os.getenv("MAX_PREMIUM_PER_STRANGLE", "25"))
    profit_target_multiplier: float = float(os.getenv("PROFIT_TARGET_MULTIPLIER", "4"))

    # Strategy parameters
    implied_move_multiplier_1: float = float(os.getenv("IMPLIED_MOVE_MULTIPLIER_1", "1.25"))
    implied_move_multiplier_2: float = float(os.getenv("IMPLIED_MOVE_MULTIPLIER_2", "1.5"))
    volatility_threshold: float = float(os.getenv("VOLATILITY_THRESHOLD", "0.67"))
    min_time_between_trades: int = int(os.getenv("MIN_TIME_BETWEEN_TRADES", "30"))

    @property
    def is_live_trading(self) -> bool:
        """Check if live trading is enabled"""
        return self.trade_mode.lower() == "live"


@dataclass
class MarketHours:
    """Market hours configuration (ET timezone)"""

    timezone = pytz.timezone("US/Eastern")

    market_open_hour: int = int(os.getenv("MARKET_OPEN_HOUR", "9"))
    market_open_minute: int = int(os.getenv("MARKET_OPEN_MINUTE", "30"))
    market_close_hour: int = int(os.getenv("MARKET_CLOSE_HOUR", "16"))
    market_close_minute: int = int(os.getenv("MARKET_CLOSE_MINUTE", "0"))
    flatten_hour: int = int(os.getenv("FLATTEN_HOUR", "15"))
    flatten_minute: int = int(os.getenv("FLATTEN_MINUTE", "58"))


@dataclass
class DatabaseConfig:
    """Database configuration"""

    url: str = os.getenv("DATABASE_URL", "sqlite:///./data/lotto_grid.db")


@dataclass
class UIConfig:
    """User interface configuration"""

    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    refresh_interval: int = int(os.getenv("REFRESH_INTERVAL", "30"))


@dataclass
class LoggingConfig:
    """Logging configuration"""

    level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = "./logs"
    bot_log_file: str = "bot_run.log"
    error_log_file: str = "bot_errors.log"
    ib_log_file: str = "ib_messages.log"


class Config:
    """Main configuration class"""

    def __init__(self):
        self.ib = IBConfig()
        self.trading = TradingConfig()
        self.market_hours = MarketHours()
        self.database = DatabaseConfig()
        self.ui = UIConfig()
        self.logging = LoggingConfig()

    def validate(self) -> bool:
        """Validate configuration"""
        errors = []

        # Check required IB credentials for live trading
        if self.trading.is_live_trading:
            if not self.ib.username or not self.ib.password:
                errors.append("IB username and password required for live trading")

        # Check risk parameters
        if self.trading.max_drawdown >= self.trading.start_cash:
            errors.append("Max drawdown should be less than starting cash")

        if self.trading.max_premium_per_strangle <= 0:
            errors.append("Max premium per strangle must be positive")

        if self.trading.profit_target_multiplier <= 1:
            errors.append("Profit target multiplier must be > 1")

        # Check database path
        db_dir = os.path.dirname(self.database.url.replace("sqlite:///", ""))
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create database directory: {e}")

        # Check log directory
        if not os.path.exists(self.logging.log_dir):
            try:
                os.makedirs(self.logging.log_dir, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create log directory: {e}")

        if errors:
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            )

        return True

    def __repr__(self):
        return f"<Config(mode={self.trading.trade_mode}, paper={self.ib.is_paper_trading})>"


# Global configuration instance
config = Config()
