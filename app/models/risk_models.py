"""
Database models for Risk Analytics system
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.models import Base


class RiskMetric(Base):
    """Store calculated risk metrics over time"""
    __tablename__ = 'risk_metrics'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)

    # Value at Risk metrics
    var_95 = Column(Float, nullable=False)
    cvar_95 = Column(Float, nullable=False)

    # Greeks aggregation
    portfolio_delta = Column(Float, default=0.0)
    portfolio_gamma = Column(Float, default=0.0)
    portfolio_vega = Column(Float, default=0.0)
    portfolio_theta = Column(Float, default=0.0)
    portfolio_rho = Column(Float, default=0.0)

    # Risk scores and ratios
    correlation_score = Column(Float, default=0.0)
    risk_score = Column(Integer, default=0)  # 0-100 scale

    # Performance metrics
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float, default=0.0)
    calmar_ratio = Column(Float, default=0.0)
    kelly_fraction = Column(Float, default=0.0)

    # Market regime
    regime_state = Column(String(20), default='normal')  # normal, trending, volatile, crisis

    # Position context
    num_positions = Column(Integer, default=0)
    total_exposure = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'portfolio_greeks': {
                'delta': self.portfolio_delta,
                'gamma': self.portfolio_gamma,
                'vega': self.portfolio_vega,
                'theta': self.portfolio_theta,
                'rho': self.portfolio_rho
            },
            'risk_score': self.risk_score,
            'correlation_score': self.correlation_score,
            'performance': {
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio
            },
            'kelly_fraction': self.kelly_fraction,
            'regime_state': self.regime_state,
            'positions': self.num_positions,
            'exposure': self.total_exposure,
            'unrealized_pnl': self.unrealized_pnl
        }


class RiskAlert(Base):
    """Store risk alerts and notifications"""
    __tablename__ = 'risk_alerts'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)

    # Alert details
    alert_type = Column(String(50), nullable=False)  # var_breach, drawdown, correlation, etc.
    severity = Column(String(20), nullable=False)  # info, warning, critical
    message = Column(Text, nullable=False)

    # Metrics that triggered alert
    metric_name = Column(String(50))
    metric_value = Column(Float)
    threshold_value = Column(Float)

    # Action and acknowledgment
    action_suggested = Column(Text)
    action_taken = Column(String(100))
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)

    # Notification status
    discord_sent = Column(Boolean, default=False)
    telegram_sent = Column(Boolean, default=False)
    email_sent = Column(Boolean, default=False)

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'metric': {
                'name': self.metric_name,
                'value': self.metric_value,
                'threshold': self.threshold_value
            },
            'action_suggested': self.action_suggested,
            'action_taken': self.action_taken,
            'acknowledged': self.acknowledged,
            'notifications': {
                'discord': self.discord_sent,
                'telegram': self.telegram_sent,
                'email': self.email_sent
            }
        }


class StressTestResult(Base):
    """Store stress test scenario results"""
    __tablename__ = 'stress_test_results'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)

    # Scenario details
    scenario_name = Column(String(100), nullable=False)
    scenario_type = Column(String(50))  # flash_crash, volatility_spike, etc.

    # Risk metrics
    probability = Column(Float, nullable=False)
    expected_loss = Column(Float, nullable=False)
    max_loss = Column(Float, nullable=False)
    var_impact = Column(Float)  # Change in VaR

    # Recovery estimates
    recovery_hours = Column(Float)
    confidence_level = Column(Float, default=0.95)

    # Affected positions (stored as JSON string)
    affected_positions = Column(Text)
    num_affected = Column(Integer, default=0)

    # Scenario parameters (stored as JSON)
    scenario_params = Column(Text)

    def to_dict(self):
        """Convert to dictionary for API responses"""
        import json
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'scenario': {
                'name': self.scenario_name,
                'type': self.scenario_type,
                'probability': self.probability
            },
            'losses': {
                'expected': self.expected_loss,
                'maximum': self.max_loss,
                'var_impact': self.var_impact
            },
            'recovery_hours': self.recovery_hours,
            'affected_positions': json.loads(self.affected_positions) if self.affected_positions else [],
            'parameters': json.loads(self.scenario_params) if self.scenario_params else {}
        }


class RiskLimit(Base):
    """Define and track risk limits"""
    __tablename__ = 'risk_limits'

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Limit definition
    limit_name = Column(String(100), nullable=False, unique=True)
    limit_type = Column(String(50), nullable=False)  # var, drawdown, exposure, etc.
    limit_value = Column(Float, nullable=False)

    # Current status
    current_value = Column(Float, default=0.0)
    utilization_pct = Column(Float, default=0.0)

    # Control settings
    is_active = Column(Boolean, default=True)
    is_hard_limit = Column(Boolean, default=False)  # If true, auto-stop trading
    warning_threshold = Column(Float, default=0.8)  # Warn at 80% utilization

    # Breach tracking
    last_breach = Column(DateTime)
    breach_count = Column(Integer, default=0)

    def check_limit(self, value: float) -> tuple[bool, float]:
        """
        Check if value breaches limit

        Returns:
            (is_breached, utilization_percentage)
        """
        if not self.is_active:
            return False, 0.0

        utilization = abs(value / self.limit_value) if self.limit_value != 0 else 0
        self.current_value = value
        self.utilization_pct = utilization * 100

        is_breached = utilization >= 1.0

        if is_breached:
            self.last_breach = datetime.now()
            self.breach_count += 1

        return is_breached, self.utilization_pct

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'name': self.limit_name,
            'type': self.limit_type,
            'limit': self.limit_value,
            'current': self.current_value,
            'utilization': self.utilization_pct,
            'is_active': self.is_active,
            'is_hard_limit': self.is_hard_limit,
            'warning_at': self.warning_threshold * 100,
            'last_breach': self.last_breach.isoformat() if self.last_breach else None,
            'breach_count': self.breach_count
        }


class PositionRisk(Base):
    """Track risk metrics for individual positions"""
    __tablename__ = 'position_risks'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)

    # Position identification
    position_id = Column(String(100), nullable=False)
    symbol = Column(String(50), nullable=False)

    # Position details
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)

    # Greeks
    delta = Column(Float, default=0.0)
    gamma = Column(Float, default=0.0)
    vega = Column(Float, default=0.0)
    theta = Column(Float, default=0.0)
    rho = Column(Float, default=0.0)

    # Risk metrics
    position_var = Column(Float, default=0.0)
    position_risk_score = Column(Integer, default=0)
    max_loss = Column(Float)
    probability_profit = Column(Float)

    # P&L
    unrealized_pnl = Column(Float, default=0.0)
    unrealized_pnl_pct = Column(Float, default=0.0)

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'position_id': self.position_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'prices': {
                'entry': self.entry_price,
                'current': self.current_price
            },
            'greeks': {
                'delta': self.delta,
                'gamma': self.gamma,
                'vega': self.vega,
                'theta': self.theta,
                'rho': self.rho
            },
            'risk': {
                'var': self.position_var,
                'score': self.position_risk_score,
                'max_loss': self.max_loss,
                'prob_profit': self.probability_profit
            },
            'pnl': {
                'unrealized': self.unrealized_pnl,
                'unrealized_pct': self.unrealized_pnl_pct
            }
        }
