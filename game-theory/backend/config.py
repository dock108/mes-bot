from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./trading_sim.db"
    
    # OpenAI
    openai_api_key: Optional[str] = None
    
    # Stripe
    stripe_secret_key: Optional[str] = None
    stripe_publishable_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    
    # JWT
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # App settings
    environment: str = "development"
    debug: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()