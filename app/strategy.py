"""
Core trading strategy logic for MES 0DTE Lotto-Grid Options Bot
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session

from app.config import config
from app.models import Trade, get_session_maker
from app.ib_client import IBClient
from app.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class LottoGridStrategy:
    """
    Implements the 0DTE MES options strangle strategy:
    - Calculate implied move from ATM straddle
    - Place OTM strangles when realized volatility < implied
    - Target 4x profit, accept 100% loss
    - Manage risk with position limits
    """
    
    def __init__(self, ib_client: IBClient, risk_manager: RiskManager, database_url: str):
        self.ib_client = ib_client
        self.risk_manager = risk_manager
        self.session_maker = get_session_maker(database_url)
        
        # Strategy state
        self.implied_move = None
        self.underlying_price = None
        self.last_trade_time = None
        self.daily_high = None
        self.daily_low = None
        self.session_start_time = datetime.utcnow()
        
        # Price history for volatility calculation
        self.price_history = []
        self.max_history_length = 100  # Keep last 100 price points
    
    async def initialize_daily_session(self) -> bool:
        """Initialize strategy for the trading day"""
        try:
            logger.info("Initializing daily trading session...")
            
            # Get current MES price
            mes_contract = await self.ib_client.get_mes_contract()
            current_price = await self.ib_client.get_current_price(mes_contract)
            
            if not current_price:
                logger.error("Could not get current MES price")
                return False
            
            self.underlying_price = current_price
            self.daily_high = current_price
            self.daily_low = current_price
            
            # Calculate implied move from ATM straddle
            expiry = self.ib_client.get_today_expiry_string()
            call_price, put_price, implied_move = await self.ib_client.get_atm_straddle_price(
                current_price, expiry
            )
            
            self.implied_move = implied_move
            
            logger.info(f"Daily session initialized:")
            logger.info(f"  MES Price: ${current_price:.2f}")
            logger.info(f"  ATM Call: ${call_price:.2f}")
            logger.info(f"  ATM Put: ${put_price:.2f}")
            logger.info(f"  Implied Move: ${implied_move:.2f}")
            
            # Set starting equity for risk management
            account_values = await self.ib_client.get_account_values()
            if 'NetLiquidation' in account_values:
                self.risk_manager.set_daily_start_equity(account_values['NetLiquidation'])
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize daily session: {e}")
            return False
    
    def update_price_history(self, price: float):
        """Update price history for volatility calculations"""
        timestamp = datetime.utcnow()
        self.price_history.append((timestamp, price))
        
        # Keep only recent history
        if len(self.price_history) > self.max_history_length:
            self.price_history = self.price_history[-self.max_history_length:]
        
        # Update daily high/low
        if self.daily_high is None or price > self.daily_high:
            self.daily_high = price
        if self.daily_low is None or price < self.daily_low:
            self.daily_low = price
    
    def calculate_realized_range(self, lookback_minutes: int = 60) -> float:
        """Calculate realized price range over lookback period"""
        if len(self.price_history) < 2:
            return 0.0
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=lookback_minutes)
        recent_prices = [
            price for timestamp, price in self.price_history 
            if timestamp >= cutoff_time
        ]
        
        if len(recent_prices) < 2:
            return 0.0
        
        return max(recent_prices) - min(recent_prices)
    
    def should_place_trade(self) -> Tuple[bool, str]:
        """Determine if conditions are met to place a new strangle"""
        
        if not self.implied_move:
            return False, "Implied move not calculated"
        
        # Check minimum time between trades
        if self.last_trade_time:
            time_since_last = datetime.utcnow() - self.last_trade_time
            min_gap = timedelta(minutes=config.trading.min_time_between_trades)
            if time_since_last < min_gap:
                remaining = min_gap - time_since_last
                return False, f"Too soon since last trade (wait {remaining.seconds // 60} more minutes)"
        
        # Check if realized volatility is low enough
        realized_range = self.calculate_realized_range(60)  # Last 60 minutes
        volatility_threshold = self.implied_move * config.trading.volatility_threshold
        
        if realized_range >= volatility_threshold:
            return False, f"Realized range {realized_range:.2f} >= threshold {volatility_threshold:.2f}"
        
        # Check market hours
        if not self.ib_client.is_market_hours():
            return False, "Outside market hours"
        
        logger.info(f"Trade conditions met:")
        logger.info(f"  Realized range (60m): {realized_range:.2f}")
        logger.info(f"  Volatility threshold: {volatility_threshold:.2f}")
        logger.info(f"  Implied move: {self.implied_move:.2f}")
        
        return True, "Conditions met for new strangle"
    
    def calculate_strike_levels(self, current_price: float) -> List[Tuple[float, float]]:
        """Calculate strike levels for strangles based on implied move"""
        if not self.implied_move:
            return []
        
        strike_pairs = []
        
        # Level 1: 1.25x implied move
        offset_1 = self.implied_move * config.trading.implied_move_multiplier_1
        call_strike_1 = self._round_to_strike(current_price + offset_1)
        put_strike_1 = self._round_to_strike(current_price - offset_1)
        strike_pairs.append((call_strike_1, put_strike_1))
        
        # Level 2: 1.5x implied move  
        offset_2 = self.implied_move * config.trading.implied_move_multiplier_2
        call_strike_2 = self._round_to_strike(current_price + offset_2)
        put_strike_2 = self._round_to_strike(current_price - offset_2)
        strike_pairs.append((call_strike_2, put_strike_2))
        
        return strike_pairs
    
    def _round_to_strike(self, price: float) -> float:
        """Round price to nearest valid option strike (25-point increments for MES)"""
        return round(price / 25) * 25
    
    async def place_strangle_trade(self, call_strike: float, put_strike: float) -> Optional[Dict]:
        """Place a single strangle trade"""
        try:
            logger.info(f"Attempting to place strangle: {call_strike}C / {put_strike}P")
            
            # Get current account equity
            account_values = await self.ib_client.get_account_values()
            current_equity = account_values.get('NetLiquidation', 0)
            
            # Get estimated premium cost
            expiry = self.ib_client.get_today_expiry_string()
            call_contract = await self.ib_client.get_mes_option_contract(expiry, call_strike, 'C')
            put_contract = await self.ib_client.get_mes_option_contract(expiry, put_strike, 'P')
            
            call_price = await self.ib_client.get_current_price(call_contract)
            put_price = await self.ib_client.get_current_price(put_contract)
            
            if not call_price or not put_price:
                logger.warning("Could not get option prices for strangle")
                return None
            
            estimated_premium = (call_price + put_price) * 5  # MES multiplier
            
            # Risk check
            can_trade, reason = self.risk_manager.can_open_new_trade(estimated_premium, current_equity)
            if not can_trade:
                logger.warning(f"Risk check failed: {reason}")
                return None
            
            # Place the strangle
            strangle_result = await self.ib_client.place_strangle(
                self.underlying_price,
                call_strike,
                put_strike,
                expiry,
                config.trading.max_premium_per_strangle
            )
            
            # Record trade in database
            trade_record = await self._record_trade(strangle_result)
            
            self.last_trade_time = datetime.utcnow()
            
            logger.info(f"Strangle placed successfully:")
            logger.info(f"  Call: {call_strike} @ ${call_price:.2f}")
            logger.info(f"  Put: {put_strike} @ ${put_price:.2f}")
            logger.info(f"  Total Premium: ${estimated_premium:.2f}")
            logger.info(f"  Trade ID: {trade_record.id}")
            
            return {
                'trade_record': trade_record,
                'strangle_result': strangle_result
            }
            
        except Exception as e:
            logger.error(f"Error placing strangle trade: {e}")
            return None
    
    async def _record_trade(self, strangle_result: Dict) -> Trade:
        """Record trade in database"""
        session = self.session_maker()
        try:
            trade = Trade(
                date=datetime.utcnow().date(),
                entry_time=datetime.utcnow(),
                underlying_symbol='MES',
                underlying_price_at_entry=self.underlying_price,
                implied_move=self.implied_move,
                call_strike=strangle_result['call_strike'],
                put_strike=strangle_result['put_strike'],
                call_premium=strangle_result['call_price'],
                put_premium=strangle_result['put_price'],
                total_premium=strangle_result['total_premium'],
                status='OPEN',
                call_status='OPEN',
                put_status='OPEN'
            )
            
            # Store IB order IDs if available
            if strangle_result.get('call_trades'):
                call_entry_trade = strangle_result['call_trades'][0]  # Entry order
                trade.call_order_id = call_entry_trade.order.orderId
                if len(strangle_result['call_trades']) > 1:
                    trade.call_tp_order_id = strangle_result['call_trades'][1].order.orderId
            
            if strangle_result.get('put_trades'):
                put_entry_trade = strangle_result['put_trades'][0]  # Entry order
                trade.put_order_id = put_entry_trade.order.orderId
                if len(strangle_result['put_trades']) > 1:
                    trade.put_tp_order_id = strangle_result['put_trades'][1].order.orderId
            
            session.add(trade)
            session.commit()
            
            logger.info(f"Trade recorded with ID: {trade.id}")
            return trade
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    async def execute_trading_cycle(self) -> bool:
        """Execute one trading cycle - check conditions and place trades if appropriate"""
        try:
            # Update current price
            mes_contract = await self.ib_client.get_mes_contract()
            current_price = await self.ib_client.get_current_price(mes_contract)
            
            if not current_price:
                logger.warning("Could not get current price")
                return False
            
            self.underlying_price = current_price
            self.update_price_history(current_price)
            
            # Check if we should place a trade
            should_trade, reason = self.should_place_trade()
            
            if not should_trade:
                logger.debug(f"Not placing trade: {reason}")
                return False
            
            # Calculate strike levels
            strike_pairs = self.calculate_strike_levels(current_price)
            if not strike_pairs:
                logger.warning("No strike levels calculated")
                return False
            
            # For now, use the first strike level (1.25x implied move)
            # Could be enhanced to rotate between levels or use multiple
            call_strike, put_strike = strike_pairs[0]
            
            # Place the strangle
            result = await self.place_strangle_trade(call_strike, put_strike)
            
            return result is not None
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return False
    
    async def update_open_positions(self):
        """Update P&L and status of open positions"""
        session = self.session_maker()
        try:
            # Get open trades from database
            open_trades = session.query(Trade).filter(Trade.status == 'OPEN').all()
            
            for trade in open_trades:
                await self._update_trade_pnl(trade, session)
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            session.rollback()
        finally:
            session.close()
    
    async def _update_trade_pnl(self, trade: Trade, session: Session):
        """Update P&L for a single trade"""
        try:
            expiry = self.ib_client.get_today_expiry_string()
            
            # Get current option prices
            call_contract = await self.ib_client.get_mes_option_contract(
                expiry, trade.call_strike, 'C'
            )
            put_contract = await self.ib_client.get_mes_option_contract(
                expiry, trade.put_strike, 'P'
            )
            
            current_call_price = await self.ib_client.get_current_price(call_contract)
            current_put_price = await self.ib_client.get_current_price(put_contract)
            
            if current_call_price is not None and current_put_price is not None:
                # Calculate unrealized P&L
                call_pnl = (current_call_price - trade.call_premium) * 5  # MES multiplier
                put_pnl = (current_put_price - trade.put_premium) * 5
                total_unrealized_pnl = call_pnl + put_pnl
                
                trade.unrealized_pnl = total_unrealized_pnl
                
                logger.debug(f"Trade {trade.id} P&L: Call ${call_pnl:.2f}, Put ${put_pnl:.2f}, Total ${total_unrealized_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating trade {trade.id} P&L: {e}")
    
    async def flatten_all_positions(self) -> bool:
        """Close all open positions at market (end-of-day routine)"""
        logger.info("Flattening all open positions...")
        
        session = self.session_maker()
        try:
            open_trades = session.query(Trade).filter(Trade.status == 'OPEN').all()
            
            if not open_trades:
                logger.info("No open positions to flatten")
                return True
            
            for trade in open_trades:
                await self._close_trade_at_market(trade, session)
            
            session.commit()
            logger.info(f"Flattened {len(open_trades)} positions")
            return True
            
        except Exception as e:
            logger.error(f"Error flattening positions: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    async def _close_trade_at_market(self, trade: Trade, session: Session):
        """Close a single trade at market prices"""
        try:
            expiry = self.ib_client.get_today_expiry_string()
            
            # Cancel any pending profit-taking orders
            if trade.call_tp_order_id:
                await self.ib_client.cancel_order(trade.call_tp_order_id)
            if trade.put_tp_order_id:
                await self.ib_client.cancel_order(trade.put_tp_order_id)
            
            # Close positions at market if they have value
            total_pnl = 0
            
            if trade.call_status == 'OPEN':
                call_contract = await self.ib_client.get_mes_option_contract(
                    expiry, trade.call_strike, 'C'
                )
                call_close_trade = await self.ib_client.close_position_at_market(call_contract, 1)
                if call_close_trade:
                    # Would need to wait for fill and record exit price
                    trade.call_status = 'EXPIRED'
            
            if trade.put_status == 'OPEN':
                put_contract = await self.ib_client.get_mes_option_contract(
                    expiry, trade.put_strike, 'P'
                )
                put_close_trade = await self.ib_client.close_position_at_market(put_contract, 1)
                if put_close_trade:
                    trade.put_status = 'EXPIRED'
            
            # Mark trade as closed
            trade.status = 'EXPIRED'
            trade.exit_time = datetime.utcnow()
            trade.realized_pnl = trade.unrealized_pnl or -trade.total_premium
            
            logger.info(f"Closed trade {trade.id} with P&L: ${trade.realized_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing trade {trade.id}: {e}")
    
    def get_strategy_status(self) -> Dict:
        """Get current strategy status for monitoring"""
        return {
            'timestamp': datetime.utcnow(),
            'underlying_price': self.underlying_price,
            'implied_move': self.implied_move,
            'daily_high': self.daily_high,
            'daily_low': self.daily_low,
            'daily_range': (self.daily_high - self.daily_low) if (self.daily_high and self.daily_low) else 0,
            'realized_range_60m': self.calculate_realized_range(60),
            'last_trade_time': self.last_trade_time,
            'session_start': self.session_start_time,
            'price_history_length': len(self.price_history)
        }