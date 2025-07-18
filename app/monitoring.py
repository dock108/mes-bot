"""
Enhanced Monitoring System with Notification Integration
"""

import asyncio
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.config import config
from app.models import DecisionHistory, MarketData, Trade, get_session_maker
from app.notification_service import (
    NotificationLevel,
    send_error_alert,
    send_performance_alert,
    send_system_alert,
    send_trade_alert,
)

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of system alerts"""

    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    HIGH_DRAWDOWN = "high_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    SYSTEM_ERROR = "system_error"
    CONNECTION_ERROR = "connection_error"
    MODEL_PERFORMANCE = "model_performance"
    DAILY_SUMMARY = "daily_summary"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    UNUSUAL_MARKET_ACTIVITY = "unusual_market_activity"


@dataclass
class AlertThresholds:
    """Configuration for alert thresholds"""

    # Performance thresholds
    max_daily_loss_pct: float = 0.15  # 15% daily loss
    max_drawdown_pct: float = 0.25  # 25% drawdown
    consecutive_loss_limit: int = 5
    low_win_rate_threshold: float = 0.2  # 20% win rate

    # System thresholds
    max_errors_per_hour: int = 10
    connection_timeout_minutes: int = 5
    model_accuracy_threshold: float = 0.4  # 40% accuracy

    # Market thresholds
    high_volatility_threshold: float = 0.5  # 50% volatility
    unusual_volume_multiplier: float = 3.0  # 3x normal volume


class TradingMonitor:
    """Enhanced trading monitor with notification integration"""

    def __init__(self, database_url: str):
        self.session_maker = get_session_maker(database_url)
        self.thresholds = AlertThresholds()
        self.error_count = 0
        self.last_error_reset = datetime.utcnow()
        self.last_connection_check = datetime.utcnow()
        self.monitoring_active = True

        # Performance tracking
        self.daily_stats = {}
        self.performance_history = []

        logger.info("Trading monitor initialized with notification integration")

    async def start_monitoring(self):
        """Start continuous monitoring"""
        logger.info("Starting enhanced trading monitoring...")

        # Send startup notification
        await send_system_alert(
            title="Trading System Started",
            message="Enhanced trading system with ML capabilities has started successfully",
            level=NotificationLevel.INFO,
            context={"timestamp": datetime.utcnow().isoformat()},
        )

        # Start monitoring tasks
        await asyncio.gather(
            self._monitor_performance(),
            self._monitor_system_health(),
            self._monitor_market_conditions(),
            self._daily_summary_task(),
        )

    async def _monitor_performance(self):
        """Monitor trading performance and send alerts"""
        while self.monitoring_active:
            try:
                await self._check_performance_alerts()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await send_error_alert(
                    title="Performance Monitor Error",
                    message=f"Error in performance monitoring: {str(e)}",
                    error=e,
                )
                await asyncio.sleep(60)  # Wait before retrying

    async def _monitor_system_health(self):
        """Monitor system health and send alerts"""
        while self.monitoring_active:
            try:
                await self._check_system_health()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in system health monitoring: {e}")
                await asyncio.sleep(60)

    async def _monitor_market_conditions(self):
        """Monitor market conditions and send alerts"""
        while self.monitoring_active:
            try:
                await self._check_market_conditions()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in market monitoring: {e}")
                await asyncio.sleep(60)

    async def _daily_summary_task(self):
        """Send daily summary at market close"""
        while self.monitoring_active:
            try:
                now = datetime.utcnow()
                # Wait until market close (around 4 PM ET = 21:00 UTC)
                if now.hour == 21 and now.minute == 0:
                    await self._send_daily_summary()
                    await asyncio.sleep(3600)  # Wait an hour to avoid duplicate
                else:
                    await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in daily summary task: {e}")
                await asyncio.sleep(300)

    async def _check_performance_alerts(self):
        """Check performance metrics and send alerts if needed"""
        session = self.session_maker()
        try:
            # Get today's performance
            today = datetime.utcnow().date()
            today_trades = session.query(Trade).filter(Trade.date == today).all()

            if not today_trades:
                return

            # Calculate metrics
            total_pnl = sum(trade.realized_pnl or 0 for trade in today_trades)
            closed_trades = [t for t in today_trades if t.status == "CLOSED"]

            if closed_trades:
                win_rate = len([t for t in closed_trades if (t.realized_pnl or 0) > 0]) / len(
                    closed_trades
                )
                avg_pnl = sum(t.realized_pnl or 0 for t in closed_trades) / len(closed_trades)
            else:
                win_rate = 0
                avg_pnl = 0

            # Check thresholds
            daily_loss_pct = abs(total_pnl) / config.trading.start_cash if total_pnl < 0 else 0

            # High drawdown alert
            if daily_loss_pct > self.thresholds.max_daily_loss_pct:
                await send_performance_alert(
                    title="High Daily Loss Alert",
                    message=f"Daily loss of ${total_pnl:.2f} ({daily_loss_pct:.1%}) exceeds threshold",
                    level=NotificationLevel.CRITICAL,
                    metrics={
                        "daily_pnl": total_pnl,
                        "daily_loss_pct": daily_loss_pct,
                        "threshold": self.thresholds.max_daily_loss_pct,
                        "trades_today": len(today_trades),
                    },
                )

            # Low win rate alert
            if len(closed_trades) >= 5 and win_rate < self.thresholds.low_win_rate_threshold:
                await send_performance_alert(
                    title="Low Win Rate Alert",
                    message=f"Win rate of {win_rate:.1%} is below threshold with {len(closed_trades)} trades",
                    level=NotificationLevel.WARNING,
                    metrics={
                        "win_rate": win_rate,
                        "threshold": self.thresholds.low_win_rate_threshold,
                        "trades_completed": len(closed_trades),
                        "avg_pnl": avg_pnl,
                    },
                )

            # Consecutive losses alert
            consecutive_losses = self._count_consecutive_losses(closed_trades)
            if consecutive_losses >= self.thresholds.consecutive_loss_limit:
                await send_performance_alert(
                    title="Consecutive Losses Alert",
                    message=f"{consecutive_losses} consecutive losses detected",
                    level=NotificationLevel.ERROR,
                    metrics={
                        "consecutive_losses": consecutive_losses,
                        "threshold": self.thresholds.consecutive_loss_limit,
                        "recent_trades": len(closed_trades),
                    },
                )

        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
        finally:
            session.close()

    async def _check_system_health(self):
        """Check system health metrics"""
        try:
            # Check error rate
            current_time = datetime.utcnow()
            if (current_time - self.last_error_reset).total_seconds() > 3600:  # Reset hourly
                self.error_count = 0
                self.last_error_reset = current_time

            if self.error_count > self.thresholds.max_errors_per_hour:
                await send_system_alert(
                    title="High Error Rate Alert",
                    message=f"System has encountered {self.error_count} errors in the last hour",
                    level=NotificationLevel.ERROR,
                    context={
                        "error_count": self.error_count,
                        "threshold": self.thresholds.max_errors_per_hour,
                        "time_window": "1 hour",
                    },
                )

            # Check ML model performance
            await self._check_ml_model_performance()

        except Exception as e:
            logger.error(f"Error checking system health: {e}")

    async def _check_ml_model_performance(self):
        """Check ML model performance"""
        try:
            session = self.session_maker()

            # Get recent ML decisions
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            recent_decisions = (
                session.query(DecisionHistory)
                .filter(
                    DecisionHistory.timestamp >= cutoff_time,
                    DecisionHistory.actual_outcome.isnot(None),
                )
                .all()
            )

            if len(recent_decisions) < 10:  # Need enough data
                return

            # Calculate accuracy
            correct_predictions = 0
            for decision in recent_decisions:
                # Simple accuracy: positive outcome predicted correctly
                predicted_positive = decision.confidence > 0.5
                actual_positive = decision.actual_outcome > 0
                if predicted_positive == actual_positive:
                    correct_predictions += 1

            accuracy = correct_predictions / len(recent_decisions)

            if accuracy < self.thresholds.model_accuracy_threshold:
                await send_system_alert(
                    title="Low ML Model Accuracy",
                    message=f"ML model accuracy of {accuracy:.1%} is below threshold",
                    level=NotificationLevel.WARNING,
                    context={
                        "accuracy": accuracy,
                        "threshold": self.thresholds.model_accuracy_threshold,
                        "sample_size": len(recent_decisions),
                        "time_window": "24 hours",
                    },
                )

        except Exception as e:
            logger.error(f"Error checking ML model performance: {e}")
        finally:
            session.close()

    async def _check_market_conditions(self):
        """Check for unusual market conditions"""
        try:
            session = self.session_maker()

            # Get recent market data
            recent_data = (
                session.query(MarketData)
                .filter(MarketData.timestamp >= datetime.utcnow() - timedelta(hours=1))
                .order_by(desc(MarketData.timestamp))
                .limit(60)
                .all()
            )

            if len(recent_data) < 10:
                return

            # Check for high volatility
            if (
                recent_data[0].atm_iv
                and recent_data[0].atm_iv > self.thresholds.high_volatility_threshold
            ):
                await send_system_alert(
                    title="High Market Volatility",
                    message=f"ATM IV of {recent_data[0].atm_iv:.1%} indicates high volatility",
                    level=NotificationLevel.WARNING,
                    context={
                        "current_iv": recent_data[0].atm_iv,
                        "threshold": self.thresholds.high_volatility_threshold,
                        "vix_level": recent_data[0].vix_level,
                    },
                )

            # Check for unusual volume
            avg_volume = sum(d.volume for d in recent_data) / len(recent_data)
            current_volume = recent_data[0].volume

            if current_volume > avg_volume * self.thresholds.unusual_volume_multiplier:
                await send_system_alert(
                    title="Unusual Market Volume",
                    message=f"Current volume {current_volume:.0f} is {current_volume/avg_volume:.1f}x normal",
                    level=NotificationLevel.INFO,
                    context={
                        "current_volume": current_volume,
                        "avg_volume": avg_volume,
                        "multiplier": current_volume / avg_volume,
                        "threshold": self.thresholds.unusual_volume_multiplier,
                    },
                )

        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
        finally:
            session.close()

    async def _send_daily_summary(self):
        """Send daily performance summary"""
        try:
            session = self.session_maker()

            # Get today's data
            today = datetime.utcnow().date()
            today_trades = session.query(Trade).filter(Trade.date == today).all()

            # Calculate metrics
            total_pnl = sum(trade.realized_pnl or 0 for trade in today_trades)
            closed_trades = [t for t in today_trades if t.status == "CLOSED"]
            open_trades = [t for t in today_trades if t.status == "OPEN"]

            if closed_trades:
                win_rate = len([t for t in closed_trades if (t.realized_pnl or 0) > 0]) / len(
                    closed_trades
                )
                avg_pnl = sum(t.realized_pnl or 0 for t in closed_trades) / len(closed_trades)
                best_trade = max(closed_trades, key=lambda t: t.realized_pnl or 0)
                worst_trade = min(closed_trades, key=lambda t: t.realized_pnl or 0)
            else:
                win_rate = 0
                avg_pnl = 0
                best_trade = None
                worst_trade = None

            # Get ML decisions
            ml_decisions = (
                session.query(DecisionHistory)
                .filter(
                    func.date(DecisionHistory.timestamp) == today, DecisionHistory.action == "ENTER"
                )
                .count()
            )

            # Create summary
            summary_metrics = {
                "date": today.isoformat(),
                "total_trades": len(today_trades),
                "closed_trades": len(closed_trades),
                "open_trades": len(open_trades),
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "best_trade_pnl": best_trade.realized_pnl if best_trade else 0,
                "worst_trade_pnl": worst_trade.realized_pnl if worst_trade else 0,
                "ml_decisions": ml_decisions,
            }

            # Determine alert level
            if total_pnl < -config.trading.daily_loss_limit * 0.5:
                level = NotificationLevel.ERROR
            elif total_pnl < 0:
                level = NotificationLevel.WARNING
            else:
                level = NotificationLevel.INFO

            await send_performance_alert(
                title="Daily Trading Summary",
                message=f"Today's P&L: ${total_pnl:.2f} | Win Rate: {win_rate:.1%} | Trades: {len(closed_trades)}",
                level=level,
                metrics=summary_metrics,
            )

        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
        finally:
            session.close()

    def _count_consecutive_losses(self, trades: List[Trade]) -> int:
        """Count consecutive losses from most recent trades"""
        if not trades:
            return 0

        # Sort by entry time (most recent first)
        sorted_trades = sorted(trades, key=lambda t: t.entry_time, reverse=True)

        consecutive = 0
        for trade in sorted_trades:
            if (trade.realized_pnl or 0) < 0:
                consecutive += 1
            else:
                break

        return consecutive

    async def record_error(self, error: Exception, context: Dict = None):
        """Record error and potentially send alert"""
        self.error_count += 1

        # Send critical error alert
        await send_error_alert(
            title="System Error Detected",
            message=f"Error in trading system: {str(error)}",
            error=error,
            context=context or {},
        )

    async def record_trade_opened(self, trade: Trade):
        """Record trade opening and send notification"""
        await send_trade_alert(
            title="New Trade Opened",
            message=f"Opened strangle: {trade.call_strike}C/{trade.put_strike}P for ${trade.total_premium:.2f}",
            level=NotificationLevel.INFO,
            context={
                "trade_id": trade.id,
                "underlying_price": trade.underlying_price_at_entry,
                "call_strike": trade.call_strike,
                "put_strike": trade.put_strike,
                "total_premium": trade.total_premium,
                "implied_move": trade.implied_move,
            },
        )

    async def record_trade_closed(self, trade: Trade):
        """Record trade closing and send notification"""
        pnl = trade.realized_pnl or 0
        level = NotificationLevel.INFO if pnl > 0 else NotificationLevel.WARNING

        await send_trade_alert(
            title="Trade Closed",
            message=f"Closed trade #{trade.id}: P&L ${pnl:.2f} ({pnl/trade.total_premium:.1%})",
            level=level,
            context={
                "trade_id": trade.id,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                "realized_pnl": pnl,
                "return_pct": pnl / trade.total_premium if trade.total_premium > 0 else 0,
                "hold_time_hours": (
                    (trade.exit_time - trade.entry_time).total_seconds() / 3600
                    if trade.exit_time
                    else 0
                ),
            },
        )

    async def stop_monitoring(self):
        """Stop monitoring and send shutdown notification"""
        self.monitoring_active = False

        await send_system_alert(
            title="Trading System Stopped",
            message="Trading system has been stopped gracefully",
            level=NotificationLevel.INFO,
            context={"timestamp": datetime.utcnow().isoformat()},
        )

    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status"""
        return {
            "monitoring_active": self.monitoring_active,
            "error_count_last_hour": self.error_count,
            "last_error_reset": self.last_error_reset.isoformat(),
            "thresholds": {
                "max_daily_loss_pct": self.thresholds.max_daily_loss_pct,
                "max_drawdown_pct": self.thresholds.max_drawdown_pct,
                "consecutive_loss_limit": self.thresholds.consecutive_loss_limit,
                "low_win_rate_threshold": self.thresholds.low_win_rate_threshold,
                "max_errors_per_hour": self.thresholds.max_errors_per_hour,
                "model_accuracy_threshold": self.thresholds.model_accuracy_threshold,
            },
        }


# Global monitor instance
monitor = None


def initialize_monitor(database_url: str):
    """Initialize global monitor"""
    global monitor
    monitor = TradingMonitor(database_url)
    return monitor


def get_monitor() -> Optional[TradingMonitor]:
    """Get global monitor instance"""
    return monitor
