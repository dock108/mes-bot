from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.connection import Base
import enum

class SubscriptionTier(enum.Enum):
    FREE = "free"
    BASE = "base"
    PREMIUM = "premium"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    subscription_tier = Column(Enum(SubscriptionTier), default=SubscriptionTier.FREE)
    stripe_customer_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    simulation_queries = relationship("SimulationQuery", back_populates="user")
    gto_analyses = relationship("GTOAnalysis", back_populates="user")
    portfolios = relationship("UserPortfolio", back_populates="user")