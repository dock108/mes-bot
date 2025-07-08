from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta
from models.user import User, SubscriptionTier
from models.usage import SimulationQuery
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class UsageService:
    # Define usage limits per tier
    TIER_LIMITS = {
        SubscriptionTier.FREE: {
            "daily_limit": 3,
            "monthly_limit": 10,
            "max_tokens": 300
        },
        SubscriptionTier.BASE: {
            "daily_limit": 20,
            "monthly_limit": 100,
            "max_tokens": 500
        },
        SubscriptionTier.PREMIUM: {
            "daily_limit": 100,
            "monthly_limit": 500,
            "max_tokens": 800
        }
    }
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_usage_today(self, user_id: int) -> int:
        """Get the number of simulations the user has run today."""
        today = datetime.utcnow().date()
        count = self.db.query(SimulationQuery).filter(
            and_(
                SimulationQuery.user_id == user_id,
                func.date(SimulationQuery.created_at) == today
            )
        ).count()
        return count
    
    def get_user_usage_this_month(self, user_id: int) -> int:
        """Get the number of simulations the user has run this month."""
        now = datetime.utcnow()
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        count = self.db.query(SimulationQuery).filter(
            and_(
                SimulationQuery.user_id == user_id,
                SimulationQuery.created_at >= start_of_month
            )
        ).count()
        return count
    
    def check_usage_limits(self, user: User) -> Dict[str, Any]:
        """Check if user can run another simulation based on their tier limits."""
        tier_config = self.TIER_LIMITS.get(user.subscription_tier)
        if not tier_config:
            return {"allowed": False, "reason": "Invalid subscription tier"}
        
        daily_usage = self.get_user_usage_today(user.id)
        monthly_usage = self.get_user_usage_this_month(user.id)
        
        # Check daily limit
        if daily_usage >= tier_config["daily_limit"]:
            return {
                "allowed": False,
                "reason": f"Daily limit reached ({daily_usage}/{tier_config['daily_limit']})",
                "usage": {
                    "daily": daily_usage,
                    "monthly": monthly_usage,
                    "daily_limit": tier_config["daily_limit"],
                    "monthly_limit": tier_config["monthly_limit"]
                }
            }
        
        # Check monthly limit
        if monthly_usage >= tier_config["monthly_limit"]:
            return {
                "allowed": False,
                "reason": f"Monthly limit reached ({monthly_usage}/{tier_config['monthly_limit']})",
                "usage": {
                    "daily": daily_usage,
                    "monthly": monthly_usage,
                    "daily_limit": tier_config["daily_limit"],
                    "monthly_limit": tier_config["monthly_limit"]
                }
            }
        
        # User is within limits
        return {
            "allowed": True,
            "usage": {
                "daily": daily_usage,
                "monthly": monthly_usage,
                "daily_limit": tier_config["daily_limit"],
                "monthly_limit": tier_config["monthly_limit"]
            }
        }
    
    def record_simulation(self, user_id: int, simulation_data: Dict[str, Any], ai_response: str) -> SimulationQuery:
        """Record a simulation in the database."""
        simulation = SimulationQuery(
            user_id=user_id,
            asset=simulation_data.get("asset", ""),
            action=simulation_data.get("action", ""),
            quantity=simulation_data.get("quantity", ""),
            time_horizon=simulation_data.get("time_horizon", ""),
            context=simulation_data.get("context", ""),
            ai_response=ai_response
        )
        
        self.db.add(simulation)
        self.db.commit()
        self.db.refresh(simulation)
        
        logger.info(f"Recorded simulation for user {user_id}: {simulation.id}")
        return simulation
    
    def get_user_simulations(self, user_id: int, limit: int = 10) -> list:
        """Get recent simulations for a user."""
        simulations = self.db.query(SimulationQuery).filter(
            SimulationQuery.user_id == user_id
        ).order_by(SimulationQuery.created_at.desc()).limit(limit).all()
        
        return simulations
    
    def get_tier_limits(self, tier: SubscriptionTier) -> Dict[str, Any]:
        """Get the limits for a specific tier."""
        return self.TIER_LIMITS.get(tier, {})