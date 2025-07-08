import stripe
from typing import Dict, Any, Optional
import logging
from config import settings
from models.user import User, SubscriptionTier
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Configure Stripe
stripe.api_key = settings.stripe_secret_key

class StripeService:
    # Define subscription price IDs (these would be created in Stripe dashboard)
    PRICE_IDS = {
        SubscriptionTier.BASE: "price_base_monthly",  # Replace with actual Stripe price ID
        SubscriptionTier.PREMIUM: "price_premium_monthly"  # Replace with actual Stripe price ID
    }
    
    def __init__(self):
        if not settings.stripe_secret_key:
            logger.warning("Stripe secret key not configured")
    
    def create_customer(self, user: User) -> Optional[str]:
        """Create a Stripe customer for the user."""
        if not stripe.api_key:
            logger.error("Stripe not configured")
            return None
            
        try:
            customer = stripe.Customer.create(
                email=user.email,
                metadata={'user_id': str(user.id)}
            )
            logger.info(f"Created Stripe customer {customer.id} for user {user.id}")
            return customer.id
        except Exception as e:
            logger.error(f"Failed to create Stripe customer: {str(e)}")
            return None
    
    def create_checkout_session(self, user: User, tier: SubscriptionTier, success_url: str, cancel_url: str) -> Optional[Dict[str, Any]]:
        """Create a Stripe checkout session for subscription."""
        if not stripe.api_key:
            return {"error": "Stripe not configured"}
            
        if tier not in self.PRICE_IDS:
            return {"error": "Invalid subscription tier"}
        
        try:
            # Ensure user has a Stripe customer ID
            if not user.stripe_customer_id:
                customer_id = self.create_customer(user)
                if not customer_id:
                    return {"error": "Failed to create customer"}
            else:
                customer_id = user.stripe_customer_id
            
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=['card'],
                line_items=[{
                    'price': self.PRICE_IDS[tier],
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    'user_id': str(user.id),
                    'tier': tier.value
                }
            )
            
            return {
                "checkout_url": session.url,
                "session_id": session.id
            }
            
        except Exception as e:
            logger.error(f"Failed to create checkout session: {str(e)}")
            return {"error": str(e)}
    
    def create_customer_portal_session(self, user: User, return_url: str) -> Optional[Dict[str, Any]]:
        """Create a Stripe customer portal session for subscription management."""
        if not stripe.api_key:
            return {"error": "Stripe not configured"}
            
        if not user.stripe_customer_id:
            return {"error": "User has no Stripe customer"}
        
        try:
            session = stripe.billing_portal.Session.create(
                customer=user.stripe_customer_id,
                return_url=return_url,
            )
            
            return {"portal_url": session.url}
            
        except Exception as e:
            logger.error(f"Failed to create portal session: {str(e)}")
            return {"error": str(e)}
    
    def handle_webhook_event(self, event: Dict[str, Any], db: Session) -> bool:
        """Handle Stripe webhook events."""
        try:
            event_type = event['type']
            
            if event_type == 'checkout.session.completed':
                return self._handle_checkout_completed(event, db)
            elif event_type == 'customer.subscription.updated':
                return self._handle_subscription_updated(event, db)
            elif event_type == 'customer.subscription.deleted':
                return self._handle_subscription_cancelled(event, db)
            elif event_type == 'invoice.payment_failed':
                return self._handle_payment_failed(event, db)
            else:
                logger.info(f"Unhandled webhook event: {event_type}")
                return True
                
        except Exception as e:
            logger.error(f"Error handling webhook: {str(e)}")
            return False
    
    def _handle_checkout_completed(self, event: Dict[str, Any], db: Session) -> bool:
        """Handle successful checkout completion."""
        session = event['data']['object']
        user_id = session['metadata'].get('user_id')
        tier = session['metadata'].get('tier')
        
        if not user_id or not tier:
            logger.error("Missing user_id or tier in checkout session metadata")
            return False
        
        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user:
            logger.error(f"User {user_id} not found")
            return False
        
        # Update user's subscription tier
        try:
            user.subscription_tier = SubscriptionTier(tier)
            user.stripe_customer_id = session['customer']
            db.commit()
            logger.info(f"Updated user {user_id} to {tier} tier")
            return True
        except Exception as e:
            logger.error(f"Failed to update user subscription: {str(e)}")
            db.rollback()
            return False
    
    def _handle_subscription_updated(self, event: Dict[str, Any], db: Session) -> bool:
        """Handle subscription updates."""
        subscription = event['data']['object']
        customer_id = subscription['customer']
        
        user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
        if not user:
            logger.error(f"User with customer ID {customer_id} not found")
            return False
        
        # Determine tier based on subscription status and price
        if subscription['status'] != 'active':
            user.subscription_tier = SubscriptionTier.FREE
        else:
            # You would determine tier based on the price ID in the subscription
            # For now, we'll keep the current tier if active
            pass
        
        db.commit()
        return True
    
    def _handle_subscription_cancelled(self, event: Dict[str, Any], db: Session) -> bool:
        """Handle subscription cancellation."""
        subscription = event['data']['object']
        customer_id = subscription['customer']
        
        user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
        if not user:
            logger.error(f"User with customer ID {customer_id} not found")
            return False
        
        user.subscription_tier = SubscriptionTier.FREE
        db.commit()
        logger.info(f"Downgraded user {user.id} to FREE tier due to cancellation")
        return True
    
    def _handle_payment_failed(self, event: Dict[str, Any], db: Session) -> bool:
        """Handle failed payments."""
        # For now, just log the event
        # In production, you might want to send notifications or take other actions
        logger.warning(f"Payment failed for invoice: {event['data']['object']['id']}")
        return True

# Create singleton instance
stripe_service = StripeService()