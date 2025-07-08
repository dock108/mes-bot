from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class MarketTypeEnum(str, Enum):
    PREDICTION_MARKET = "prediction_market"
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"

class AnalysisStatusEnum(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    ERROR = "error"

class RiskLevelEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# Request Schemas
class GTOAnalysisRequest(BaseModel):
    market_id: str
    source: str  # polymarket, kalshi, binance, etc.
    user_probability: float
    bankroll: float
    risk_tolerance: Optional[float] = 1.0
    payout_ratio: Optional[float] = 1.0
    
    @validator('user_probability')
    def validate_probability(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Probability must be between 0 and 1')
        return v
    
    @validator('bankroll')
    def validate_bankroll(cls, v):
        if v <= 0:
            raise ValueError('Bankroll must be positive')
        return v
    
    @validator('risk_tolerance')
    def validate_risk_tolerance(cls, v):
        if v is not None and not 0 < v <= 1:
            raise ValueError('Risk tolerance must be between 0 and 1')
        return v

class MarketDataRequest(BaseModel):
    market_type: MarketTypeEnum
    symbols: Optional[List[str]] = None
    limit: Optional[int] = 20
    source: Optional[str] = None

class PortfolioRequest(BaseModel):
    total_bankroll: float
    risk_preference: Optional[float] = 0.5
    
    @validator('total_bankroll')
    def validate_bankroll(cls, v):
        if v <= 0:
            raise ValueError('Total bankroll must be positive')
        return v
    
    @validator('risk_preference')
    def validate_risk_preference(cls, v):
        if v is not None and not 0 <= v <= 1:
            raise ValueError('Risk preference must be between 0 and 1')
        return v

class AlertRequest(BaseModel):
    market_id: str
    source: str
    alert_type: str  # "edge_threshold", "price_target"
    threshold_value: float
    condition: str  # "greater_than", "less_than"
    notification_method: Optional[str] = "email"

# Response Schemas
class MarketDataResponse(BaseModel):
    market_id: str
    source: str
    market_type: str
    symbol: Optional[str] = None
    question: Optional[str] = None
    current_price: Optional[float] = None
    market_probability: Optional[float] = None
    volume_24h: Optional[float] = None
    liquidity: Optional[float] = None
    last_updated: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class GTOAnalysisResponse(BaseModel):
    id: int
    market_data: MarketDataResponse
    user_probability: float
    bankroll: float
    
    # GTO calculations
    edge: Optional[float] = None
    kelly_fraction: Optional[float] = None
    recommended_stake: Optional[float] = None
    expected_value: Optional[float] = None
    
    # Risk metrics
    probability_of_loss: Optional[float] = None
    max_loss: Optional[float] = None
    kelly_growth_rate: Optional[float] = None
    
    # Analysis metadata
    analysis_status: AnalysisStatusEnum
    ai_explanation: Optional[str] = None
    confidence_score: Optional[float] = None
    risk_assessment: Optional[RiskLevelEnum] = None
    should_bet: Optional[bool] = None
    
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class MonteCarloResponse(BaseModel):
    final_bankroll_mean: float
    final_bankroll_std: float
    probability_of_profit: float
    max_drawdown: float
    worst_case: float
    best_case: float
    percentile_5: float
    percentile_95: float

class PortfolioResponse(BaseModel):
    id: int
    total_bankroll: float
    available_bankroll: float
    risk_preference: float
    total_positions: int
    total_at_risk: float
    portfolio_expected_value: float
    portfolio_variance: float
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class PositionResponse(BaseModel):
    id: int
    stake_amount: float
    entry_probability: float
    position_type: str
    is_active: bool
    is_closed: bool
    realized_pnl: Optional[float] = None
    opened_at: datetime
    closed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class AlertResponse(BaseModel):
    id: int
    market_data: MarketDataResponse
    alert_type: str
    threshold_value: float
    condition: str
    is_active: bool
    notification_method: str
    last_triggered: Optional[datetime] = None
    trigger_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class MarketListResponse(BaseModel):
    markets: List[MarketDataResponse]
    total_count: int

class AnalysisListResponse(BaseModel):
    analyses: List[GTOAnalysisResponse]
    total_count: int

class PortfolioPositionsResponse(BaseModel):
    portfolio: PortfolioResponse
    positions: List[PositionResponse]
    total_positions: int

# Quick Analysis Response (for immediate feedback)
class QuickAnalysisResponse(BaseModel):
    edge: float
    kelly_fraction: float
    recommended_stake: float
    expected_value: float
    confidence_score: float
    risk_assessment: RiskLevelEnum
    should_bet: bool
    explanation_summary: str

# Chat Schemas
class ChatMessageRequest(BaseModel):
    message: str
    analysis_id: Optional[int] = None
    session_id: Optional[str] = None
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > 1000:
            raise ValueError('Message too long (max 1000 characters)')
        return v.strip()

class ChatMessageResponse(BaseModel):
    response: str
    session_id: str
    suggestions: List[str]
    analysis_referenced: bool
    timestamp: str
    model_used: Optional[str] = None

class ChatHistoryRequest(BaseModel):
    session_id: str

class ChatHistoryMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    analysis_id: Optional[int] = None

class ChatHistoryResponse(BaseModel):
    messages: List[ChatHistoryMessage]
    session_id: str
    total_messages: int

class ChatSessionRequest(BaseModel):
    action: str  # "clear" or "list"

class ChatSessionResponse(BaseModel):
    success: bool
    message: str
    active_sessions: Optional[List[str]] = None