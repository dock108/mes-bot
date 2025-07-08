from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database.connection import get_db
from models.user import User
from api.auth import get_current_user
from services.openai_service import openai_service
from services.usage_service import UsageService
from schemas.simulation import (
    SimulationRequest,
    SimulationResponse,
    UsageResponse,
    SimulationHistoryResponse
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Simulation"])

@router.post("/test-openai")
async def test_openai_connection():
    """Test OpenAI API connectivity."""
    result = openai_service.test_connection()
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"OpenAI API test failed: {result.get('error', 'Unknown error')}"
        )
    
    return {
        "message": "OpenAI API is working correctly",
        "ai_response": result["response"],
        "model": result["model"],
        "usage": result["usage"]
    }

@router.post("/simulate", response_model=SimulationResponse)
async def create_simulation(
    request: SimulationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate trading simulation analysis."""
    
    logger.info(f"Simulation request from user {current_user.id}: {request.asset} {request.action}")
    
    # Initialize usage service
    usage_service = UsageService(db)
    
    # Check usage limits
    usage_check = usage_service.check_usage_limits(current_user)
    if not usage_check["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "message": usage_check["reason"],
                "usage": usage_check["usage"]
            }
        )
    
    # Create prompt from user input
    prompt_parts = [
        f"Trading Scenario Analysis Request:",
        f"Asset: {request.asset}",
        f"Action: {request.action.title()}"
    ]
    
    if request.quantity:
        prompt_parts.append(f"Quantity/Amount: {request.quantity}")
    
    if request.time_horizon:
        prompt_parts.append(f"Time Horizon: {request.time_horizon}")
    
    if request.context:
        prompt_parts.append(f"Additional Context: {request.context}")
    
    prompt_parts.append("\nPlease provide an educational analysis of this trading scenario.")
    
    prompt = "\n".join(prompt_parts)
    
    # Generate analysis using OpenAI
    result = openai_service.generate_trading_analysis(
        prompt=prompt,
        user_tier=current_user.subscription_tier.value
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Analysis generation failed: {result.get('error', 'Unknown error')}"
        )
    
    # Record the simulation in the database
    simulation_data = {
        "asset": request.asset,
        "action": request.action,
        "quantity": request.quantity,
        "time_horizon": request.time_horizon,
        "context": request.context
    }
    
    usage_service.record_simulation(
        user_id=current_user.id,
        simulation_data=simulation_data,
        ai_response=result["analysis"]
    )
    
    # Get updated usage info
    updated_usage = usage_service.check_usage_limits(current_user)
    
    return {
        "analysis": result["analysis"],
        "model_used": result["model"],
        "user_tier": result["user_tier"],
        "usage": updated_usage["usage"],
        "disclaimer": "This analysis is for educational purposes only and is not financial advice. All trading involves risk."
    }

@router.get("/usage", response_model=UsageResponse)
async def get_user_usage(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's usage statistics."""
    usage_service = UsageService(db)
    
    usage_check = usage_service.check_usage_limits(current_user)
    tier_limits = usage_service.get_tier_limits(current_user.subscription_tier)
    
    return {
        "user_tier": current_user.subscription_tier.value,
        "usage": usage_check["usage"],
        "tier_limits": tier_limits,
        "can_simulate": usage_check["allowed"]
    }

@router.get("/history", response_model=SimulationHistoryResponse)
async def get_simulation_history(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's simulation history."""
    usage_service = UsageService(db)
    simulations = usage_service.get_user_simulations(current_user.id, limit)
    
    return {
        "simulations": [
            {
                "id": sim.id,
                "asset": sim.asset,
                "action": sim.action,
                "quantity": sim.quantity,
                "time_horizon": sim.time_horizon,
                "context": sim.context,
                "created_at": sim.created_at,
                "ai_response": sim.ai_response[:200] + "..." if len(sim.ai_response) > 200 else sim.ai_response
            }
            for sim in simulations
        ],
        "total_count": len(simulations)
    }