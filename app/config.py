"""
Configuration management for the MES 0DTE Lotto-Grid Options Bot
"""

import os
from dataclasses import dataclass
from datetime import time
from typing import List, Optional

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

    # Contract management
    mes_contract_month: Optional[str] = os.getenv("MES_CONTRACT_MONTH", None)  # None = auto-detect
    contract_rollover_days: int = int(
        os.getenv("CONTRACT_ROLLOVER_DAYS", "3")
    )  # Days before expiry to roll

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

    # Risk management parameters
    critical_equity_threshold: float = float(os.getenv("CRITICAL_EQUITY_THRESHOLD", "0.3"))
    consecutive_loss_limit: int = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", "10"))

    # Multi-instrument configuration
    active_instruments: List[str] = None  # Set from environment or use default
    primary_instrument: str = os.getenv("PRIMARY_INSTRUMENT", "MES")

    def __post_init__(self):
        """Post-init processing for instrument configuration"""
        if self.active_instruments is None:
            # Parse from environment variable or use default
            env_instruments = os.getenv("ACTIVE_INSTRUMENTS", "MES")
            self.active_instruments = [s.strip() for s in env_instruments.split(",")]

    @property
    def daily_loss_limit(self) -> float:
        """Daily loss limit (same as max drawdown for compatibility)"""
        return self.max_drawdown

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

    @property
    def market_open(self) -> time:
        """Get market open time as a time object"""
        return time(self.market_open_hour, self.market_open_minute)

    @property
    def market_close(self) -> time:
        """Get market close time as a time object"""
        return time(self.market_close_hour, self.market_close_minute)

    @property
    def flatten_time(self) -> time:
        """Get flatten time as a time object"""
        return time(self.flatten_hour, self.flatten_minute)


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

    # Structured logging configuration
    structured_logging: bool = os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"
    json_format: bool = os.getenv("JSON_LOG_FORMAT", "true").lower() == "true"
    correlation_ids: bool = os.getenv("CORRELATION_IDS", "true").lower() == "true"


@dataclass
class MLConfig:
    """Machine Learning configuration"""

    training_lookback_days: int = int(os.getenv("ML_TRAINING_LOOKBACK_DAYS", "30"))
    min_training_samples: int = int(os.getenv("ML_MIN_TRAINING_SAMPLES", "100"))
    retrain_interval_hours: int = int(os.getenv("ML_RETRAIN_INTERVAL_HOURS", "24"))
    model_confidence_threshold: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.6"))
    ensemble_weight_ml: float = float(os.getenv("ML_ENSEMBLE_WEIGHT", "0.3"))
    ensemble_weight_rules: float = float(os.getenv("RULES_ENSEMBLE_WEIGHT", "0.7"))


@dataclass
class DataConfig:
    """Data storage configuration"""

    cache_dir: str = os.getenv("DATA_CACHE_DIR", "./data/cache")


@dataclass
class NotificationConfig:
    """Notification system configuration"""

    # Enable notifications
    enabled: bool = os.getenv("NOTIFICATIONS_ENABLED", "true").lower() == "true"

    # Email settings
    email_enabled: bool = os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "true").lower() == "true"
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    from_email: str = os.getenv("FROM_EMAIL", "")
    to_emails: str = os.getenv("TO_EMAILS", "")  # Comma-separated list

    # SMS settings (Twilio)
    sms_enabled: bool = os.getenv("SMS_NOTIFICATIONS_ENABLED", "false").lower() == "true"
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_from_number: str = os.getenv("TWILIO_FROM_NUMBER", "")
    twilio_to_numbers: str = os.getenv("TWILIO_TO_NUMBERS", "")  # Comma-separated list

    # Slack settings
    slack_enabled: bool = os.getenv("SLACK_NOTIFICATIONS_ENABLED", "false").lower() == "true"
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    slack_channel: str = os.getenv("SLACK_CHANNEL", "#trading-alerts")

    # Webhook settings
    webhook_enabled: bool = os.getenv("WEBHOOK_NOTIFICATIONS_ENABLED", "false").lower() == "true"
    webhook_url: str = os.getenv("WEBHOOK_URL", "")

    # Rate limiting
    rate_limit_minutes: int = int(os.getenv("NOTIFICATION_RATE_LIMIT_MINUTES", "5"))
    max_notifications_per_period: int = int(os.getenv("MAX_NOTIFICATIONS_PER_PERIOD", "10"))

    @property
    def to_emails_list(self) -> List[str]:
        """Get to_emails as a list"""
        return [email.strip() for email in self.to_emails.split(",") if email.strip()]

    @property
    def to_numbers_list(self) -> List[str]:
        """Get to_numbers as a list"""
        return [number.strip() for number in self.twilio_to_numbers.split(",") if number.strip()]


class Config:
    """Main configuration class"""

    def __init__(self):
        self.ib = IBConfig()
        self.trading = TradingConfig()
        self.market_hours = MarketHours()
        self.database = DatabaseConfig()
        self.ui = UIConfig()
        self.logging = LoggingConfig()
        self.ml = MLConfig()
        self.data = DataConfig()
        self.notifications = NotificationConfig()

        # Initialize notification service if enabled
        self.notification_service = None
        if self.notifications.enabled:
            self._initialize_notification_service()

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

    def _initialize_notification_service(self):
        """Initialize notification service"""
        try:
            from app.notification_service import NotificationConfig as NotifConfig
            from app.notification_service import NotificationService

            # Create notification service configuration
            notif_config = NotifConfig(
                smtp_host=self.notifications.smtp_host,
                smtp_port=self.notifications.smtp_port,
                smtp_user=self.notifications.smtp_user,
                smtp_password=self.notifications.smtp_password,
                from_email=self.notifications.from_email,
                to_emails=self.notifications.to_emails_list,
                twilio_account_sid=self.notifications.twilio_account_sid,
                twilio_auth_token=self.notifications.twilio_auth_token,
                twilio_from_number=self.notifications.twilio_from_number,
                twilio_to_numbers=self.notifications.to_numbers_list,
                slack_webhook_url=self.notifications.slack_webhook_url,
                slack_channel=self.notifications.slack_channel,
                webhook_url=self.notifications.webhook_url,
                rate_limit_minutes=self.notifications.rate_limit_minutes,
                max_notifications_per_period=self.notifications.max_notifications_per_period,
            )

            self.notification_service = NotificationService(notif_config)

        except Exception as e:
            print(f"Warning: Could not initialize notification service: {e}")
            self.notification_service = None

    def __repr__(self):
        return f"<Config(mode={self.trading.trade_mode}, paper={self.ib.is_paper_trading})>"


# Global configuration instance
config = Config()
