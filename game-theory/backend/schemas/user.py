from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import datetime
from models.user import SubscriptionTier
from utils.validators import email_validator, password_validator

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str
    
    # Validators
    _validate_email = validator('email', allow_reuse=True)(email_validator)
    _validate_password = validator('password', allow_reuse=True)(password_validator)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    subscription_tier: SubscriptionTier
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None