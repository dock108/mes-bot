from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import time
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any
import re

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    def __init__(self):
        # Rate limiting storage (in production, use Redis)
        self.rate_limit_storage: Dict[str, list] = defaultdict(list)
        self.failed_attempts: Dict[str, list] = defaultdict(list)
        
        # Security patterns
        self.suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',  # JavaScript injection
            r'on\w+\s*=',  # Event handlers
            r'union\s+select',  # SQL injection
            r'drop\s+table',  # SQL injection
            r'insert\s+into',  # SQL injection
            r'delete\s+from',  # SQL injection
        ]
        
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            # Apply security checks
            await self._check_rate_limits(request)
            await self._check_suspicious_content(request)
            
            # Process the request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            # Log request
            process_time = time.time() - start_time
            await self._log_request(request, response, process_time)
            
            return response
            
        except HTTPException as e:
            # Log security violations
            await self._log_security_violation(request, str(e.detail))
            raise e
        except Exception as e:
            logger.error(f"Security middleware error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
    
    async def _check_rate_limits(self, request: Request):
        """Check rate limits per IP address."""
        client_ip = self._get_client_ip(request)
        now = datetime.utcnow()
        
        # Clean old entries
        self.rate_limit_storage[client_ip] = [
            timestamp for timestamp in self.rate_limit_storage[client_ip]
            if now - timestamp < timedelta(minutes=1)
        ]
        
        # Check limits
        request_count = len(self.rate_limit_storage[client_ip])
        
        # Basic rate limiting: 60 requests per minute per IP
        if request_count >= 60:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Add current request
        self.rate_limit_storage[client_ip].append(now)
        
        # Check for failed login attempts
        if request.url.path == "/api/auth/login":
            await self._check_failed_login_attempts(request, client_ip)
    
    async def _check_failed_login_attempts(self, request: Request, client_ip: str):
        """Check for brute force login attempts."""
        now = datetime.utcnow()
        
        # Clean old failed attempts (last 15 minutes)
        self.failed_attempts[client_ip] = [
            timestamp for timestamp in self.failed_attempts[client_ip]
            if now - timestamp < timedelta(minutes=15)
        ]
        
        # Check if too many failed attempts
        if len(self.failed_attempts[client_ip]) >= 5:
            logger.warning(f"Too many failed login attempts from IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Please try again in 15 minutes."
            )
    
    async def _check_suspicious_content(self, request: Request):
        """Check for suspicious content in request."""
        try:
            # Check URL path
            path = str(request.url.path)
            query = str(request.url.query) if request.url.query else ""
            
            for pattern in self.suspicious_patterns:
                if re.search(pattern, path + query, re.IGNORECASE):
                    logger.warning(f"Suspicious pattern detected in URL: {pattern}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid request format"
                    )
            
            # Check request body if present
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get("content-type", "")
                if "application/json" in content_type:
                    body = await request.body()
                    if body:
                        body_str = body.decode('utf-8')
                        for pattern in self.suspicious_patterns:
                            if re.search(pattern, body_str, re.IGNORECASE):
                                logger.warning(f"Suspicious pattern detected in body: {pattern}")
                                raise HTTPException(
                                    status_code=status.HTTP_400_BAD_REQUEST,
                                    detail="Invalid request content"
                                )
                        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking suspicious content: {str(e)}")
    
    def _add_security_headers(self, response):
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address, considering proxies."""
        # Check for forwarded headers (when behind a proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    async def _log_request(self, request: Request, response, process_time: float):
        """Log request details."""
        client_ip = self._get_client_ip(request)
        
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "client_ip": client_ip,
            "status_code": response.status_code,
            "process_time": round(process_time, 4),
            "user_agent": request.headers.get("User-Agent", ""),
        }
        
        # Log at appropriate level
        if response.status_code >= 500:
            logger.error(f"Request failed: {log_data}")
        elif response.status_code >= 400:
            logger.warning(f"Client error: {log_data}")
        else:
            logger.info(f"Request completed: {log_data}")
    
    async def _log_security_violation(self, request: Request, detail: str):
        """Log security violations."""
        client_ip = self._get_client_ip(request)
        
        violation_data = {
            "violation_type": "security_check_failed",
            "client_ip": client_ip,
            "method": request.method,
            "url": str(request.url),
            "detail": detail,
            "user_agent": request.headers.get("User-Agent", ""),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.warning(f"Security violation: {violation_data}")
    
    def record_failed_login(self, client_ip: str):
        """Record a failed login attempt."""
        now = datetime.utcnow()
        self.failed_attempts[client_ip].append(now)
        logger.warning(f"Failed login attempt from IP: {client_ip}")

# Create singleton instance
security_middleware = SecurityMiddleware()