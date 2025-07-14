"""
Test data generator for UAT testing
"""
import random
from datetime import datetime, date, timedelta
from typing import List
from faker import Faker

from app.models import Trade, DailySummary, BacktestResult, get_session_maker


class TestDataGenerator:
    """Generate realistic test data for UAT testing"""
    
    def __init__(self):
        self.fake = Faker()
        Faker.seed(42)  # Consistent test data
        random.seed(42)
    
    def populate_database(self, database_url: str):
        """Populate database with comprehensive test data"""
        session_maker = get_session_maker(database_url)
        session = session_maker()
        
        try:
            # Generate trades for the last 30 days
            trades = self.generate_trades(30)
            session.add_all(trades)
            
            # Generate daily summaries
            summaries = self.generate_daily_summaries(30)
            session.add_all(summaries)
            
            # Generate backtest results
            backtests = self.generate_backtest_results(5)
            session.add_all(backtests)
            
            session.commit()
            
        finally:
            session.close()
    
    def generate_trades(self, days: int) -> List[Trade]:
        """Generate realistic trade data"""
        trades = []
        current_date = date.today()
        
        for day_offset in range(days):
            trade_date = current_date - timedelta(days=day_offset)
            
            # Skip weekends
            if trade_date.weekday() >= 5:
                continue
            
            # Generate 2-8 trades per day
            num_trades = random.randint(2, 8)
            
            for trade_num in range(num_trades):
                trade = self._create_single_trade(trade_date, trade_num)
                trades.append(trade)
        
        return trades
    
    def _create_single_trade(self, trade_date: date, trade_num: int) -> Trade:
        """Create a single realistic trade"""
        # Base MES price around 4200
        base_price = 4200 + random.uniform(-100, 100)
        
        # Entry time during trading hours
        entry_hour = random.randint(9, 15)
        entry_minute = random.randint(0, 59)
        entry_time = datetime.combine(
            trade_date, 
            datetime.min.time().replace(hour=entry_hour, minute=entry_minute)
        )
        
        # Strike prices (25-point increments)
        implied_move = random.uniform(15, 30)
        multiplier = random.choice([1.25, 1.5])
        offset = implied_move * multiplier
        
        call_strike = round((base_price + offset) / 25) * 25
        put_strike = round((base_price - offset) / 25) * 25
        
        # Option premiums
        call_premium = random.uniform(1.5, 4.0)
        put_premium = random.uniform(1.5, 4.0)
        total_premium = (call_premium + put_premium) * 5  # MES multiplier
        
        # Determine trade outcome (realistic win rate ~25%)
        is_winner = random.random() < 0.25
        
        if is_winner:
            # Winning trade - one leg hits 4x target
            if random.random() < 0.5:
                # Call winner
                exit_time = entry_time + timedelta(hours=random.randint(1, 4))
                call_exit_price = call_premium * 4.0
                put_exit_price = 0.0
                realized_pnl = (call_exit_price - call_premium + put_exit_price - put_premium) * 5
                status = 'CLOSED_WIN'
                call_status = 'CLOSED_PROFIT'
                put_status = 'EXPIRED'
            else:
                # Put winner
                exit_time = entry_time + timedelta(hours=random.randint(1, 4))
                call_exit_price = 0.0
                put_exit_price = put_premium * 4.0
                realized_pnl = (call_exit_price - call_premium + put_exit_price - put_premium) * 5
                status = 'CLOSED_WIN'
                call_status = 'EXPIRED'
                put_status = 'CLOSED_PROFIT'
        else:
            # Losing trade - expires worthless
            exit_time = entry_time.replace(hour=16, minute=0)  # Market close
            call_exit_price = 0.0
            put_exit_price = 0.0
            realized_pnl = -total_premium
            status = 'EXPIRED'
            call_status = 'EXPIRED'
            put_status = 'EXPIRED'
        
        # Some trades still open (today's trades)
        if trade_date == date.today() and random.random() < 0.3:
            exit_time = None
            call_exit_price = None
            put_exit_price = None
            realized_pnl = None
            status = 'OPEN'
            call_status = 'OPEN'
            put_status = 'OPEN'
            
            # Add unrealized P&L for open trades
            unrealized_pnl = random.uniform(-total_premium * 0.8, total_premium * 0.5)
        else:
            unrealized_pnl = None
        
        return Trade(
            date=trade_date,
            entry_time=entry_time,
            exit_time=exit_time,
            underlying_symbol='MES',
            underlying_price_at_entry=base_price,
            implied_move=implied_move,
            call_strike=call_strike,
            put_strike=put_strike,
            call_premium=call_premium,
            put_premium=put_premium,
            total_premium=total_premium,
            call_exit_price=call_exit_price,
            put_exit_price=put_exit_price,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            status=status,
            call_status=call_status,
            put_status=put_status,
            call_order_id=random.randint(10000, 99999),
            put_order_id=random.randint(10000, 99999),
        )
    
    def generate_daily_summaries(self, days: int) -> List[DailySummary]:
        """Generate daily summary data"""
        summaries = []
        current_date = date.today()
        
        cumulative_pnl = 0
        
        for day_offset in range(days):
            summary_date = current_date - timedelta(days=day_offset)
            
            # Skip weekends
            if summary_date.weekday() >= 5:
                continue
            
            # Realistic daily trading stats
            total_trades = random.randint(2, 8)
            winning_trades = random.randint(0, min(3, total_trades))
            losing_trades = total_trades - winning_trades
            
            # P&L calculation
            gross_profit = winning_trades * random.uniform(50, 200)  # Winners average ~$125
            gross_loss = losing_trades * random.uniform(15, 30)      # Losers average ~$22.50
            net_pnl = gross_profit - gross_loss
            cumulative_pnl += net_pnl
            
            # Market data
            opening_price = 4200 + random.uniform(-50, 50)
            daily_range = random.uniform(10, 60)
            closing_price = opening_price + random.uniform(-daily_range/2, daily_range/2)
            high_price = max(opening_price, closing_price) + random.uniform(0, daily_range/3)
            low_price = min(opening_price, closing_price) - random.uniform(0, daily_range/3)
            
            summary = DailySummary(
                date=summary_date,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                gross_profit=gross_profit,
                gross_loss=gross_loss,
                net_pnl=net_pnl,
                max_drawdown=max(0, random.uniform(0, 100)),
                max_concurrent_trades=random.randint(1, 6),
                opening_price=opening_price,
                closing_price=closing_price,
                high_price=high_price,
                low_price=low_price,
                implied_move=random.uniform(15, 30)
            )
            
            summaries.append(summary)
        
        return summaries
    
    def generate_backtest_results(self, count: int) -> List[BacktestResult]:
        """Generate backtest result data"""
        results = []
        
        for i in range(count):
            start_date = date.today() - timedelta(days=random.randint(30, 180))
            end_date = start_date + timedelta(days=random.randint(30, 90))
            
            initial_capital = 5000.0
            total_trades = random.randint(50, 200)
            winning_trades = int(total_trades * random.uniform(0.15, 0.35))  # 15-35% win rate
            losing_trades = total_trades - winning_trades
            
            # Calculate return
            total_return = random.uniform(-0.3, 0.8)  # -30% to +80%
            final_capital = initial_capital * (1 + total_return)
            
            result = BacktestResult(
                name=f"Backtest_{i+1}_{start_date.strftime('%Y%m%d')}",
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                max_trades=15,
                profit_target=4.0,
                implied_move_mult_1=1.25,
                implied_move_mult_2=1.5,
                volatility_threshold=0.67,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=winning_trades / total_trades,
                final_capital=final_capital,
                total_return=total_return,
                max_drawdown=random.uniform(50, 500),
                sharpe_ratio=random.uniform(-0.5, 2.0),
                execution_time=random.uniform(10, 120)
            )
            
            results.append(result)
        
        return results