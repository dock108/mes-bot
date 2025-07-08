from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime
from utils.validators import (
    asset_validator,
    action_validator,
    time_horizon_validator,
    quantity_validator,
    context_validator
)

class SimulationRequest(BaseModel):
    asset: str
    action: str
    quantity: Optional[str] = None
    time_horizon: Optional[str] = None
    context: Optional[str] = None
    
    # Validators
    _validate_asset = validator('asset', allow_reuse=True)(asset_validator)
    _validate_action = validator('action', allow_reuse=True)(action_validator)
    _validate_time_horizon = validator('time_horizon', allow_reuse=True)(time_horizon_validator)
    _validate_quantity = validator('quantity', allow_reuse=True)(quantity_validator)
    _validate_context = validator('context', allow_reuse=True)(context_validator)

class SimulationResponse(BaseModel):
    analysis: str
    model_used: str
    user_tier: str
    usage: dict
    disclaimer: str

class UsageResponse(BaseModel):
    user_tier: str
    usage: dict
    tier_limits: dict
    can_simulate: bool

class SimulationHistoryItem(BaseModel):
    id: int
    asset: str
    action: str
    quantity: Optional[str]
    time_horizon: Optional[str]
    context: Optional[str]
    created_at: datetime
    ai_response: str
    
    class Config:
        from_attributes = True

class SimulationHistoryResponse(BaseModel):
    simulations: list[SimulationHistoryItem]
    total_count: int