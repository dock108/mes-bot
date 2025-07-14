"""
Risk management system for the MES 0DTE Lotto-Grid Options Bot
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session

from app.config import config
from app.models import Trade, DailySummary, get_session_maker

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk controls and position limits"""
    
    def __init__(self, database_url: str):
        self.session_maker = get_session_maker(database_url)
        self.daily_start_equity = None
        self.session_start_time = datetime.utcnow()
        
    def get_current_session(self) -> Session:
        """Get database session"""
        return self.session_maker()
    
    def set_daily_start_equity(self, equity: float):
        """Set the starting equity for the day"""
        self.daily_start_equity = equity
        logger.info(f"Daily start equity set to ${equity:,.2f}")
    
    def can_open_new_trade(self, premium_cost: float, current_equity: float) -> Tuple[bool, str]:
        """Check if a new trade can be opened based on risk rules"""
        
        # Check if we're in trading hours
        # This would be checked by the main bot, but adding here for completeness
        
        # Rule 1: Maximum open trades limit
        open_trades_count = self._get_open_trades_count()
        if open_trades_count >= config.trading.max_open_trades:
            return False, f"Maximum open trades limit reached ({open_trades_count}/{config.trading.max_open_trades})"
        
        # Rule 2: Premium limit per strangle
        if premium_cost > config.trading.max_premium_per_strangle:
            return False, f"Premium cost ${premium_cost:.2f} exceeds limit ${config.trading.max_premium_per_strangle}"
        
        # Rule 3: Available funds check
        required_cash = premium_cost * 1.1  # 10% buffer
        if current_equity < required_cash:
            return False, f"Insufficient funds: need ${required_cash:.2f}, have ${current_equity:.2f}"
        
        # Rule 4: Maximum drawdown check
        if self.daily_start_equity:
            current_drawdown = self.daily_start_equity - current_equity
            if current_drawdown > config.trading.max_drawdown:
                return False, f"Maximum drawdown exceeded: ${current_drawdown:.2f} > ${config.trading.max_drawdown}"
        
        # Rule 5: Minimum time between trades
        if not self._check_minimum_time_between_trades():
            return False, f"Minimum time between trades not met ({config.trading.min_time_between_trades} minutes)"
        
        return True, "Trade approved"
    
    def _get_open_trades_count(self) -> int:
        """Get count of currently open trades"""
        session = self.get_current_session()
        try:
            count = session.query(Trade).filter(Trade.status == 'OPEN').count()
            return count
        finally:
            session.close()
    
    def _check_minimum_time_between_trades(self) -> bool:
        """Check if minimum time has passed since last trade"""
        session = self.get_current_session()
        try:
            # Get the most recent trade
            last_trade = session.query(Trade).order_by(Trade.entry_time.desc()).first()
            
            if not last_trade:
                return True  # No previous trades
            
            time_since_last = datetime.utcnow() - last_trade.entry_time
            min_time_delta = timedelta(minutes=config.trading.min_time_between_trades)
            
            return time_since_last >= min_time_delta
            
        finally:
            session.close()
    
    def get_current_exposure(self) -> Dict[str, float]:
        """Get current risk exposure metrics"""
        session = self.get_current_session()
        try:
            # Get all open trades
            open_trades = session.query(Trade).filter(Trade.status == 'OPEN').all()
            
            total_premium_at_risk = sum(trade.total_premium for trade in open_trades)
            unrealized_pnl = sum(trade.unrealized_pnl or 0 for trade in open_trades)
            
            # Calculate current drawdown
            current_drawdown = 0
            if self.daily_start_equity:
                daily_pnl = self._get_daily_pnl()
                current_equity = self.daily_start_equity + daily_pnl
                current_drawdown = max(0, self.daily_start_equity - current_equity)
            
            return {
                'open_trades_count': len(open_trades),
                'total_premium_at_risk': total_premium_at_risk,
                'unrealized_pnl': unrealized_pnl,
                'current_drawdown': current_drawdown,
                'daily_pnl': self._get_daily_pnl()
            }
            
        finally:
            session.close()
    
    def _get_daily_pnl(self) -> float:
        """Get today's realized + unrealized P&L"""
        session = self.get_current_session()
        try:
            today = datetime.utcnow().date()
            
            # Get today's trades
            today_trades = session.query(Trade).filter(Trade.date == today).all()
            
            total_pnl = 0
            for trade in today_trades:
                if trade.realized_pnl:
                    total_pnl += trade.realized_pnl
                elif trade.unrealized_pnl:
                    total_pnl += trade.unrealized_pnl
                else:
                    # For open trades without unrealized P&L, assume worst case (full premium loss)
                    if trade.status == 'OPEN':
                        total_pnl -= trade.total_premium
            
            return total_pnl
            
        finally:
            session.close()
    
    def should_halt_trading(self, current_equity: float) -> Tuple[bool, str]:
        """Check if trading should be halted due to risk limits"""
        
        # Check maximum drawdown
        if self.daily_start_equity:
            current_drawdown = self.daily_start_equity - current_equity
            if current_drawdown > config.trading.max_drawdown:
                return True, f"Maximum daily drawdown exceeded: ${current_drawdown:.2f}"
        
        # Check if account equity is critically low
        critical_equity_threshold = config.trading.start_cash * 0.5  # 50% of starting equity
        if current_equity < critical_equity_threshold:
            return True, f"Account equity critically low: ${current_equity:.2f}"
        
        # Check for too many consecutive losses (optional additional safety)
        consecutive_losses = self._get_consecutive_losses()
        if consecutive_losses >= 5:  # Configurable threshold
            return True, f"Too many consecutive losses: {consecutive_losses}"
        
        return False, "Trading can continue"
    
    def _get_consecutive_losses(self) -> int:
        """Get count of consecutive losing trades"""
        session = self.get_current_session()
        try:
            # Get recent closed trades in reverse chronological order
            recent_trades = session.query(Trade).filter(
                Trade.status.in_(['CLOSED_LOSS', 'EXPIRED', 'CLOSED_WIN'])
            ).order_by(Trade.exit_time.desc()).limit(10).all()
            
            consecutive_losses = 0
            for trade in recent_trades:
                if trade.status in ['CLOSED_LOSS', 'EXPIRED']:
                    consecutive_losses += 1
                else:
                    break  # Hit a winner, stop counting
            
            return consecutive_losses
            
        finally:
            session.close()
    
    def get_position_sizing_recommendation(self, available_equity: float, 
                                         option_premium: float) -> int:
        """Recommend position size based on available equity and risk"""
        
        # Basic Kelly criterion approach (simplified)
        # For 0DTE options with high win rate but small wins vs large losses
        
        # Conservative approach: never risk more than 2% of equity per trade
        max_risk_per_trade = available_equity * 0.02
        
        # But also respect the configured maximum premium per strangle
        max_premium = min(max_risk_per_trade, config.trading.max_premium_per_strangle)
        
        # Calculate suggested position size
        if option_premium > 0:
            suggested_size = int(max_premium / option_premium)
            return max(1, suggested_size)  # At least 1 contract
        
        return 1
    
    def should_close_all_positions(self, current_equity: float, current_time: datetime) -> Tuple[bool, str]:
        """Check if all positions should be closed (e.g., at end of day)"""
        
        # Check if it's close to market close (15:58 ET for 0DTE options)
        # This would typically be handled by the main bot's scheduler
        # but adding here for risk management perspective
        
        # Emergency close conditions
        if self.daily_start_equity:
            # Close if losses exceed 50% of daily equity (extreme risk)
            extreme_loss_threshold = self.daily_start_equity * 0.5
            if current_equity < extreme_loss_threshold:
                return True, "Emergency close: extreme losses"
        
        # Check if we're in a scenario where we should close all positions
        # (This would be called by the bot's end-of-day routine)
        
        return False, "No emergency close needed"
    
    def update_daily_summary(self):
        """Update or create today's daily summary"""
        session = self.get_current_session()
        try:
            today = datetime.utcnow().date()
            
            # Get or create daily summary
            daily_summary = session.query(DailySummary).filter(
                DailySummary.date == today
            ).first()
            
            if not daily_summary:
                daily_summary = DailySummary(date=today)
                session.add(daily_summary)
                session.flush()  # Ensure the record is in database before updating
            
            # Get today's trades
            today_trades = session.query(Trade).filter(Trade.date == today).all()
            
            # Calculate metrics
            total_trades = len(today_trades)
            winning_trades = len([t for t in today_trades if t.status == 'CLOSED_WIN'])
            losing_trades = len([t for t in today_trades if t.status in ['CLOSED_LOSS', 'EXPIRED']])
            
            gross_profit = sum(t.realized_pnl for t in today_trades if t.realized_pnl and t.realized_pnl > 0) or 0
            gross_loss = sum(t.realized_pnl for t in today_trades if t.realized_pnl and t.realized_pnl < 0) or 0
            net_pnl = gross_profit + gross_loss
            
            # Update summary
            daily_summary.total_trades = total_trades
            daily_summary.winning_trades = winning_trades
            daily_summary.losing_trades = losing_trades
            daily_summary.gross_profit = gross_profit
            daily_summary.gross_loss = abs(gross_loss)
            daily_summary.net_pnl = net_pnl
            daily_summary.max_concurrent_trades = max(
                self._get_max_concurrent_trades_today(), 
                daily_summary.max_concurrent_trades or 0
            )
            
            session.commit()
            logger.info(f"Updated daily summary: {total_trades} trades, ${net_pnl:.2f} P&L")
            
        except Exception as e:
            logger.error(f"Error updating daily summary: {e}")
            session.rollback()
        finally:
            session.close()
    
    def _get_max_concurrent_trades_today(self) -> int:
        """Calculate maximum concurrent trades for today"""
        # This is a simplified version - would need more complex logic
        # to track exact concurrent positions throughout the day
        return self._get_open_trades_count()
    
    def log_risk_event(self, event_type: str, message: str, details: Optional[str] = None):
        """Log a risk management event"""
        logger.warning(f"RISK EVENT [{event_type}]: {message}")
        if details:
            logger.warning(f"RISK DETAILS: {details}")
    
    def get_risk_metrics_summary(self) -> Dict:
        """Get a summary of current risk metrics for monitoring"""
        exposure = self.get_current_exposure()
        
        return {
            'timestamp': datetime.utcnow(),
            'open_trades': exposure['open_trades_count'],
            'max_trades_limit': config.trading.max_open_trades,
            'total_premium_at_risk': exposure['total_premium_at_risk'],
            'current_drawdown': exposure['current_drawdown'],
            'max_drawdown_limit': config.trading.max_drawdown,
            'daily_pnl': exposure['daily_pnl'],
            'unrealized_pnl': exposure['unrealized_pnl'],
            'trading_halted': False  # Would be set by main bot
        }