import re
from typing import Optional
from pydantic import validator
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    """Utility class for input validation and sanitization."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        if not email or len(email) > 254:
            return False
        
        # Basic email regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
    
    @staticmethod
    def validate_password(password: str) -> tuple[bool, str]:
        """Validate password strength."""
        if not password:
            return False, "Password is required"
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if len(password) > 128:
            return False, "Password must be less than 128 characters"
        
        # Check for at least one letter and one number
        has_letter = bool(re.search(r'[a-zA-Z]', password))
        has_number = bool(re.search(r'\d', password))
        
        if not (has_letter and has_number):
            return False, "Password must contain at least one letter and one number"
        
        return True, "Password is valid"
    
    @staticmethod
    def sanitize_string(text: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not text:
            return ""
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\'\\\x00-\x1f\x7f-\x9f]', '', text)
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def validate_asset_symbol(symbol: str) -> bool:
        """Validate trading asset symbol."""
        if not symbol:
            return False
        
        # Allow alphanumeric characters, dots, and dashes
        # Length between 1-10 characters
        pattern = r'^[A-Za-z0-9.-]{1,10}$'
        return bool(re.match(pattern, symbol))
    
    @staticmethod
    def validate_trading_action(action: str) -> bool:
        """Validate trading action."""
        if not action:
            return False
        
        valid_actions = ['buy', 'sell']
        return action.lower() in valid_actions
    
    @staticmethod
    def validate_time_horizon(horizon: str) -> bool:
        """Validate time horizon."""
        if not horizon:
            return True  # Optional field
        
        valid_horizons = ['short-term', 'medium-term', 'long-term']
        return horizon.lower() in valid_horizons
    
    @staticmethod
    def validate_quantity(quantity: str) -> bool:
        """Validate quantity/amount string."""
        if not quantity:
            return True  # Optional field
        
        # Allow reasonable formats like "100", "100 shares", "$1000", "1000 USD"
        if len(quantity) > 50:
            return False
        
        # Check for suspicious content
        suspicious_patterns = [
            r'<script', r'javascript:', r'on\w+\s*=', 
            r'union\s+select', r'drop\s+table'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, quantity, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def validate_context(context: str) -> bool:
        """Validate context/description text."""
        if not context:
            return True  # Optional field
        
        # Limit length
        if len(context) > 2000:
            return False
        
        # Check for suspicious content
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'data:text/html',
            r'vbscript:',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected in context: {pattern}")
                return False
        
        return True

# Pydantic validators for models
def email_validator(v):
    """Pydantic validator for email."""
    if not InputValidator.validate_email(v):
        raise ValueError('Invalid email format')
    return v

def password_validator(v):
    """Pydantic validator for password."""
    is_valid, message = InputValidator.validate_password(v)
    if not is_valid:
        raise ValueError(message)
    return v

def asset_validator(v):
    """Pydantic validator for asset symbol."""
    if not InputValidator.validate_asset_symbol(v):
        raise ValueError('Invalid asset symbol format')
    return v

def action_validator(v):
    """Pydantic validator for trading action."""
    if not InputValidator.validate_trading_action(v):
        raise ValueError('Invalid trading action')
    return v.lower()

def time_horizon_validator(v):
    """Pydantic validator for time horizon."""
    if v and not InputValidator.validate_time_horizon(v):
        raise ValueError('Invalid time horizon')
    return v

def quantity_validator(v):
    """Pydantic validator for quantity."""
    if v and not InputValidator.validate_quantity(v):
        raise ValueError('Invalid quantity format')
    return InputValidator.sanitize_string(v, 50)

def context_validator(v):
    """Pydantic validator for context."""
    if v and not InputValidator.validate_context(v):
        raise ValueError('Invalid context content')
    return InputValidator.sanitize_string(v, 2000)