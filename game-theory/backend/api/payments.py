from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from database.connection import get_db
from models.user import User, SubscriptionTier
from api.auth import get_current_user
from services.stripe_service import stripe_service
from pydantic import BaseModel
from typing import Optional
import stripe
import logging
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/payments", tags=["Payments"])

class CreateCheckoutRequest(BaseModel):
    tier: str  # "base" or "premium"

class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str

@router.post("/create-checkout", response_model=CheckoutResponse)
async def create_checkout_session(
    request: CreateCheckoutRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a Stripe checkout session for subscription upgrade."""
    
    # Validate tier
    try:
        tier = SubscriptionTier(request.tier)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid subscription tier"
        )
    
    if tier == SubscriptionTier.FREE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot upgrade to free tier"
        )
    
    # Check if user is already on this tier or higher
    if current_user.subscription_tier == tier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"You are already on the {tier.value} tier"
        )
    
    # Create checkout session
    success_url = "http://localhost:3000/subscription/success"
    cancel_url = "http://localhost:3000/subscription/cancel"
    
    result = stripe_service.create_checkout_session(
        user=current_user,
        tier=tier,
        success_url=success_url,
        cancel_url=cancel_url
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to create checkout session: {result['error']}"
        )
    
    # Update user's stripe_customer_id if it was created
    if not current_user.stripe_customer_id and "customer_id" in result:
        current_user.stripe_customer_id = result["customer_id"]
        db.commit()
    
    return CheckoutResponse(
        checkout_url=result["checkout_url"],
        session_id=result["session_id"]
    )

@router.post("/create-portal-session")
async def create_customer_portal_session(
    current_user: User = Depends(get_current_user)
):
    """Create a Stripe customer portal session for subscription management."""
    
    if not current_user.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active subscription found"
        )
    
    return_url = "http://localhost:3000/account"
    
    result = stripe_service.create_customer_portal_session(
        user=current_user,
        return_url=return_url
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to create portal session: {result['error']}"
        )
    
    return {"portal_url": result["portal_url"]}

@router.get("/subscription-status")
async def get_subscription_status(
    current_user: User = Depends(get_current_user)
):
    """Get current user's subscription status."""
    
    return {
        "current_tier": current_user.subscription_tier.value,
        "has_stripe_customer": bool(current_user.stripe_customer_id),
        "available_tiers": {
            "free": {
                "name": "Free",
                "price": 0,
                "daily_limit": 3,
                "monthly_limit": 10
            },
            "base": {
                "name": "Base",
                "price": 5,
                "daily_limit": 20,
                "monthly_limit": 100
            },
            "premium": {
                "name": "Premium", 
                "price": 20,
                "daily_limit": 100,
                "monthly_limit": 500
            }
        }
    }

@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    db: Session = Depends(get_db)
):
    """Handle Stripe webhook events."""
    
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    if not settings.stripe_webhook_secret:
        logger.error("Stripe webhook secret not configured")
        raise HTTPException(status_code=400, detail="Webhook secret not configured")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.stripe_webhook_secret
        )
    except ValueError:
        logger.error("Invalid payload in webhook")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        logger.error("Invalid signature in webhook")
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle the event
    success = stripe_service.handle_webhook_event(event, db)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process webhook")
    
    return {"status": "success"}