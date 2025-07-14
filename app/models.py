"""
Database models for the MES 0DTE Lotto-Grid Options Bot
"""
from datetime import datetime, date
from decimal import Decimal
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Date,
    Float,
    Boolean,
    Text,
    ForeignKey,
    create_engine,
    JSON,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, backref
from sqlalchemy.sql import func

Base = declarative_base()


class Trade(Base):
    """Individual strangle trade record"""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, default=date.today)
    entry_time = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Underlying info
    underlying_symbol = Column(String(10), nullable=False, default="MES")
    underlying_price_at_entry = Column(Float, nullable=False)
    implied_move = Column(Float, nullable=False)

    # Strike and premium info
    call_strike = Column(Float, nullable=False)
    put_strike = Column(Float, nullable=False)
    call_premium = Column(Float, nullable=False)
    put_premium = Column(Float, nullable=False)
    total_premium = Column(Float, nullable=False)

    # Exit info
    exit_time = Column(DateTime, nullable=True)
    call_exit_price = Column(Float, nullable=True)
    put_exit_price = Column(Float, nullable=True)

    # P&L
    realized_pnl = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)

    # Status tracking
    status = Column(
        String(20), nullable=False, default="OPEN"
    )  # OPEN, CLOSED_WIN, CLOSED_LOSS, EXPIRED
    call_status = Column(String(20), nullable=False, default="OPEN")  # OPEN, CLOSED_PROFIT, EXPIRED
    put_status = Column(String(20), nullable=False, default="OPEN")  # OPEN, CLOSED_PROFIT, EXPIRED

    # IB order IDs for tracking
    call_order_id = Column(Integer, nullable=True)
    put_order_id = Column(Integer, nullable=True)
    call_tp_order_id = Column(Integer, nullable=True)  # take profit order
    put_tp_order_id = Column(Integer, nullable=True)  # take profit order

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Trade(id={self.id}, date={self.date}, status={self.status}, pnl={self.realized_pnl})>"


class DailySummary(Base):
    """Daily trading summary"""

    __tablename__ = "daily_summaries"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, unique=True)

    # Trade counts
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)

    # P&L
    gross_profit = Column(Float, nullable=False, default=0.0)
    gross_loss = Column(Float, nullable=False, default=0.0)
    net_pnl = Column(Float, nullable=False, default=0.0)

    # Risk metrics
    max_drawdown = Column(Float, nullable=False, default=0.0)
    max_concurrent_trades = Column(Integer, nullable=False, default=0)

    # Market data
    opening_price = Column(Float, nullable=True)
    closing_price = Column(Float, nullable=True)
    high_price = Column(Float, nullable=True)
    low_price = Column(Float, nullable=True)
    implied_move = Column(Float, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<DailySummary(date={self.date}, trades={self.total_trades}, pnl={self.net_pnl})>"


class BacktestResult(Base):
    """Backtest execution results"""

    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)

    # Parameters
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    initial_capital = Column(Float, nullable=False)

    # Strategy parameters
    max_trades = Column(Integer, nullable=False)
    profit_target = Column(Float, nullable=False)
    implied_move_mult_1 = Column(Float, nullable=False)
    implied_move_mult_2 = Column(Float, nullable=False)
    volatility_threshold = Column(Float, nullable=False)

    # Results
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=False, default=0.0)

    final_capital = Column(Float, nullable=False, default=0.0)
    total_return = Column(Float, nullable=False, default=0.0)
    max_drawdown = Column(Float, nullable=False, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)

    # Execution metadata
    execution_time = Column(Float, nullable=False)  # seconds

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<BacktestResult(name={self.name}, return={self.total_return:.2%}, win_rate={self.win_rate:.2%})>"


class SystemLog(Base):
    """System events and errors log"""

    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    level = Column(String(10), nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    module = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)  # JSON or additional info

    def __repr__(self):
        return f"<SystemLog(level={self.level}, module={self.module}, time={self.timestamp})>"


class MarketData(Base):
    """Historical market data for ML training"""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Basic market data
    underlying_price = Column(Float, nullable=False)
    bid_price = Column(Float, nullable=False)
    ask_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False, default=0.0)

    # Option data
    atm_iv = Column(Float, nullable=False)
    implied_move = Column(Float, nullable=True)

    # External data
    vix_level = Column(Float, nullable=True)

    def __repr__(self):
        return f"<MarketData(timestamp={self.timestamp}, price={self.underlying_price})>"


class MarketFeatures(Base):
    """Calculated market features for ML models"""

    __tablename__ = "market_features"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Volatility features
    realized_vol_15m = Column(Float, nullable=False, default=0.0)
    realized_vol_30m = Column(Float, nullable=False, default=0.0)
    realized_vol_60m = Column(Float, nullable=False, default=0.0)
    realized_vol_2h = Column(Float, nullable=False, default=0.0)
    realized_vol_daily = Column(Float, nullable=False, default=0.0)

    # Implied volatility features
    atm_iv = Column(Float, nullable=False, default=0.0)
    iv_rank = Column(Float, nullable=False, default=50.0)
    iv_percentile = Column(Float, nullable=False, default=50.0)
    iv_skew = Column(Float, nullable=False, default=0.0)
    iv_term_structure = Column(Float, nullable=False, default=0.0)

    # Technical indicators
    rsi_15m = Column(Float, nullable=False, default=50.0)
    rsi_30m = Column(Float, nullable=False, default=50.0)
    macd_signal = Column(Float, nullable=False, default=0.0)
    macd_histogram = Column(Float, nullable=False, default=0.0)
    bb_position = Column(Float, nullable=False, default=0.5)
    bb_squeeze = Column(Float, nullable=False, default=0.0)

    # Price action features
    price_momentum_15m = Column(Float, nullable=False, default=0.0)
    price_momentum_30m = Column(Float, nullable=False, default=0.0)
    price_momentum_60m = Column(Float, nullable=False, default=0.0)
    support_resistance_strength = Column(Float, nullable=False, default=0.0)
    mean_reversion_signal = Column(Float, nullable=False, default=0.0)

    # Market microstructure
    bid_ask_spread = Column(Float, nullable=False, default=0.0)
    option_volume_ratio = Column(Float, nullable=False, default=0.0)
    put_call_ratio = Column(Float, nullable=False, default=1.0)
    gamma_exposure = Column(Float, nullable=False, default=0.0)

    # Market regime indicators
    vix_level = Column(Float, nullable=False, default=20.0)
    vix_term_structure = Column(Float, nullable=False, default=0.0)
    market_correlation = Column(Float, nullable=False, default=0.0)
    volume_profile = Column(Float, nullable=False, default=1.0)

    # Time-based features
    time_of_day = Column(Float, nullable=False)
    day_of_week = Column(Float, nullable=False)
    time_to_expiry = Column(Float, nullable=False)
    days_since_last_trade = Column(Float, nullable=False, default=0.0)

    # Performance features
    win_rate_recent = Column(Float, nullable=False, default=0.25)
    profit_factor_recent = Column(Float, nullable=False, default=1.0)
    sharpe_ratio_recent = Column(Float, nullable=False, default=0.0)

    def __repr__(self):
        return f"<MarketFeatures(timestamp={self.timestamp}, iv_rank={self.iv_rank})>"


class DecisionHistory(Base):
    """Historical trading decisions for analysis and ML training"""

    __tablename__ = "decision_history"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Decision details
    action = Column(String(20), nullable=False)  # ENTER, EXIT, HOLD
    confidence = Column(Float, nullable=False)

    # Market context
    underlying_price = Column(Float, nullable=False)
    implied_move = Column(Float, nullable=False)
    features_id = Column(Integer, ForeignKey("market_features.id"), nullable=True)

    # Decision reasoning
    reasoning = Column(JSON, nullable=True)  # List of reasons
    model_predictions = Column(JSON, nullable=True)  # Dict of model predictions

    # Strike selection (if applicable)
    suggested_call_strike = Column(Float, nullable=True)
    suggested_put_strike = Column(Float, nullable=True)
    position_size_multiplier = Column(Float, nullable=False, default=1.0)
    profit_target_multiplier = Column(Float, nullable=False, default=4.0)

    # Outcome tracking
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    actual_outcome = Column(Float, nullable=True)  # Actual P&L if trade was taken
    outcome_recorded_at = Column(DateTime, nullable=True)

    # Relationships
    features = relationship("MarketFeatures", backref="decisions")
    trade = relationship("Trade", backref=backref("decision", uselist=False))

    def __repr__(self):
        return f"<DecisionHistory(action={self.action}, confidence={self.confidence:.2f})>"


class MLModelMetadata(Base):
    """Metadata for ML models"""

    __tablename__ = "ml_model_metadata"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False, unique=True)
    model_type = Column(String(50), nullable=False)  # 'entry', 'exit', 'strike_selection'
    version = Column(String(20), nullable=False, default="1.0.0")

    # Training metadata
    trained_on = Column(DateTime, nullable=True)
    training_start_date = Column(Date, nullable=True)
    training_end_date = Column(Date, nullable=True)
    training_samples = Column(Integer, nullable=False, default=0)

    # Model parameters
    hyperparameters = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)

    # Performance metrics
    validation_accuracy = Column(Float, nullable=True)
    validation_precision = Column(Float, nullable=True)
    validation_recall = Column(Float, nullable=True)
    validation_f1 = Column(Float, nullable=True)

    # Model file info
    model_file_path = Column(String(255), nullable=True)
    model_file_hash = Column(String(64), nullable=True)

    # Status
    is_active = Column(Boolean, nullable=False, default=False)
    is_production = Column(Boolean, nullable=False, default=False)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<MLModelMetadata(name={self.model_name}, version={self.version})>"


class MLPrediction(Base):
    """Individual ML model predictions for tracking"""

    __tablename__ = "ml_predictions"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Model info
    model_id = Column(Integer, ForeignKey("ml_model_metadata.id"), nullable=False)
    model_name = Column(String(100), nullable=False)

    # Prediction details
    prediction_type = Column(String(20), nullable=False)  # 'entry', 'exit', 'strike'
    prediction_value = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)

    # Input features
    features_id = Column(Integer, ForeignKey("market_features.id"), nullable=True)
    input_features = Column(JSON, nullable=True)  # Feature values used

    # Context
    decision_id = Column(Integer, ForeignKey("decision_history.id"), nullable=True)
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)

    # Outcome tracking
    actual_outcome = Column(Float, nullable=True)
    prediction_error = Column(Float, nullable=True)
    outcome_recorded_at = Column(DateTime, nullable=True)

    # Relationships
    model = relationship("MLModelMetadata", backref="predictions")
    features = relationship("MarketFeatures", backref="predictions")
    decision = relationship("DecisionHistory", backref="ml_predictions")
    trade = relationship("Trade", backref="ml_predictions")

    def __repr__(self):
        return f"<MLPrediction(model={self.model_name}, value={self.prediction_value:.3f})>"


class PerformanceMetrics(Base):
    """Aggregated performance metrics over time"""

    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # 'daily', 'weekly', 'monthly'

    # Trading performance
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=False, default=0.0)
    avg_win = Column(Float, nullable=False, default=0.0)
    avg_loss = Column(Float, nullable=False, default=0.0)
    profit_factor = Column(Float, nullable=False, default=0.0)

    # Risk metrics
    max_drawdown = Column(Float, nullable=False, default=0.0)
    sharpe_ratio = Column(Float, nullable=False, default=0.0)
    sortino_ratio = Column(Float, nullable=False, default=0.0)

    # ML model performance
    model_accuracy = Column(Float, nullable=True)
    prediction_accuracy = Column(Float, nullable=True)
    feature_drift_score = Column(Float, nullable=True)

    # Market conditions
    avg_iv_rank = Column(Float, nullable=True)
    avg_vix_level = Column(Float, nullable=True)
    market_regime = Column(String(20), nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<PerformanceMetrics(date={self.date}, type={self.metric_type})>"


def create_database(database_url: str):
    """Create database and tables"""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def get_session_maker(database_url: str):
    """Get SQLAlchemy session maker"""
    engine = create_database(database_url)
    return sessionmaker(bind=engine)
