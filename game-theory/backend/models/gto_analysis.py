from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.connection import Base
import enum

class MarketType(enum.Enum):
    PREDICTION_MARKET = "prediction_market"
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"

class AnalysisStatus(enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    ERROR = "error"

class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    market_type = Column(Enum(MarketType), nullable=False)
    market_id = Column(String, nullable=False)  # External market ID
    source = Column(String, nullable=False)  # kalshi, polymarket, binance, etc.
    symbol = Column(String, nullable=True)  # For crypto: BTC, ETH, etc.
    question = Column(Text, nullable=True)  # For prediction markets
    
    # Market data
    current_price = Column(Float, nullable=True)
    market_probability = Column(Float, nullable=True)  # For binary outcomes
    volume_24h = Column(Float, nullable=True)
    liquidity = Column(Float, nullable=True)
    
    # Raw data from external APIs
    raw_data = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    gto_analyses = relationship("GTOAnalysis", back_populates="market_data")

class GTOAnalysis(Base):
    __tablename__ = "gto_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    market_data_id = Column(Integer, ForeignKey("market_data.id"), nullable=False)
    
    # User inputs
    user_probability = Column(Float, nullable=False)  # User's belief about outcome
    bankroll = Column(Float, nullable=False)
    risk_tolerance = Column(Float, default=1.0)  # Kelly fraction multiplier
    
    # GTO calculations
    edge = Column(Float, nullable=True)  # User advantage over market
    kelly_fraction = Column(Float, nullable=True)  # Optimal bet size as fraction of bankroll
    recommended_stake = Column(Float, nullable=True)  # Dollar amount to bet
    expected_value = Column(Float, nullable=True)  # Expected profit/loss
    
    # Risk metrics
    probability_of_loss = Column(Float, nullable=True)
    max_loss = Column(Float, nullable=True)
    kelly_growth_rate = Column(Float, nullable=True)
    
    # Analysis metadata
    analysis_status = Column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING)
    ai_explanation = Column(Text, nullable=True)  # GPT-generated explanation
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="gto_analyses")
    market_data = relationship("MarketData", back_populates="gto_analyses")

class UserPortfolio(Base):
    __tablename__ = "user_portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Portfolio settings
    total_bankroll = Column(Float, nullable=False)
    available_bankroll = Column(Float, nullable=False)
    risk_preference = Column(Float, default=0.5)  # 0 = very conservative, 1 = aggressive
    
    # Portfolio metrics
    total_positions = Column(Integer, default=0)
    total_at_risk = Column(Float, default=0.0)
    portfolio_expected_value = Column(Float, default=0.0)
    portfolio_variance = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    positions = relationship("PortfolioPosition", back_populates="portfolio")

class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("user_portfolios.id"), nullable=False)
    analysis_id = Column(Integer, ForeignKey("gto_analyses.id"), nullable=False)
    
    # Position details
    stake_amount = Column(Float, nullable=False)
    entry_probability = Column(Float, nullable=False)  # Market probability when entered
    position_type = Column(String, nullable=False)  # "long", "short"
    
    # Position status
    is_active = Column(Boolean, default=True)
    is_closed = Column(Boolean, default=False)
    realized_pnl = Column(Float, nullable=True)
    
    # Timestamps
    opened_at = Column(DateTime(timezone=True), server_default=func.now())
    closed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    portfolio = relationship("UserPortfolio", back_populates="positions")
    analysis = relationship("GTOAnalysis")

class MarketAlert(Base):
    __tablename__ = "market_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    market_data_id = Column(Integer, ForeignKey("market_data.id"), nullable=False)
    
    # Alert conditions
    alert_type = Column(String, nullable=False)  # "edge_threshold", "price_target", etc.
    threshold_value = Column(Float, nullable=False)
    condition = Column(String, nullable=False)  # "greater_than", "less_than", "equals"
    
    # Alert settings
    is_active = Column(Boolean, default=True)
    notification_method = Column(String, default="email")  # "email", "push", "sms"
    
    # Alert status
    last_triggered = Column(DateTime(timezone=True), nullable=True)
    trigger_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User")
    market_data = relationship("MarketData")