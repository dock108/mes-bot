from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database.connection import Base

class SimulationQuery(Base):
    __tablename__ = "simulation_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    asset = Column(String, nullable=False)
    action = Column(String, nullable=False)  # buy/sell
    quantity = Column(String, nullable=True)
    time_horizon = Column(String, nullable=True)
    context = Column(Text, nullable=True)
    ai_response = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    user = relationship("User", back_populates="simulation_queries")

# Relationship will be added in user.py to avoid circular imports