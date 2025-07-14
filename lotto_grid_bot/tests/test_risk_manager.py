"""
Tests for the risk management system
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.risk_manager import RiskManager
from app.models import Base, Trade, DailySummary
from app.config import config


class TestRiskManager:
    """Test cases for RiskManager"""
    
    @pytest.fixture
    def in_memory_db(self):
        """Create in-memory SQLite database for testing"""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine
    
    @pytest.fixture
    def risk_manager(self, tmp_path):
        """Create risk manager with temporary file database"""
        # Use a temporary file database that persists for the test
        db_file = tmp_path / "test_risk.db"
        db_url = f"sqlite:///{db_file}"
        return RiskManager(db_url)
    
    @pytest.fixture
    def sample_trades(self, risk_manager):
        """Create sample trades for testing"""
        session = risk_manager.get_current_session()
        
        # Create some open trades
        trade1 = Trade(
            date=datetime.utcnow().date(),
            entry_time=datetime.utcnow() - timedelta(hours=1),
            underlying_price_at_entry=4200.0,
            call_strike=4225.0,
            put_strike=4175.0,
            call_premium=2.5,
            put_premium=3.0,
            total_premium=27.5,
            implied_move=20.0,
            status='OPEN'
        )
        
        trade2 = Trade(
            date=datetime.utcnow().date(),
            entry_time=datetime.utcnow() - timedelta(hours=2),
            underlying_price_at_entry=4190.0,
            call_strike=4215.0,
            put_strike=4165.0,
            call_premium=3.0,
            put_premium=2.5,
            total_premium=27.5,
            implied_move=18.0,
            status='OPEN'
        )
        
        # Create a closed winning trade
        trade3 = Trade(
            date=datetime.utcnow().date(),
            entry_time=datetime.utcnow() - timedelta(hours=3),
            exit_time=datetime.utcnow() - timedelta(hours=1),
            underlying_price_at_entry=4180.0,
            call_strike=4205.0,
            put_strike=4155.0,
            call_premium=2.0,
            put_premium=2.5,
            total_premium=22.5,
            realized_pnl=67.5,  # 3x profit
            implied_move=15.0,
            status='CLOSED_WIN'
        )
        
        session.add_all([trade1, trade2, trade3])
        session.commit()
        session.close()
        
        return [trade1, trade2, trade3]
    
    def test_set_daily_start_equity(self, risk_manager):
        """Test setting daily start equity"""
        risk_manager.set_daily_start_equity(5000.0)
        assert risk_manager.daily_start_equity == 5000.0
    
    def test_can_open_new_trade_success(self, risk_manager, sample_trades):
        """Test successful trade approval"""
        risk_manager.set_daily_start_equity(5000.0)
        
        can_trade, reason = risk_manager.can_open_new_trade(25.0, 4800.0)
        
        assert can_trade is True
        assert "Trade approved" in reason
    
    def test_can_open_new_trade_max_trades_limit(self, risk_manager, sample_trades):
        """Test trade rejection due to max trades limit"""
        # Mock config to have very low max trades
        original_max_trades = config.trading.max_open_trades
        config.trading.max_open_trades = 1
        
        try:
            can_trade, reason = risk_manager.can_open_new_trade(25.0, 5000.0)
            
            assert can_trade is False
            assert "Maximum open trades limit" in reason
        finally:
            config.trading.max_open_trades = original_max_trades
    
    def test_can_open_new_trade_premium_limit(self, risk_manager, sample_trades):
        """Test trade rejection due to premium limit"""
        can_trade, reason = risk_manager.can_open_new_trade(50.0, 5000.0)  # Exceeds $25 limit
        
        assert can_trade is False
        assert "Premium cost" in reason and "exceeds limit" in reason
    
    def test_can_open_new_trade_insufficient_funds(self, risk_manager, sample_trades):
        """Test trade rejection due to insufficient funds"""
        can_trade, reason = risk_manager.can_open_new_trade(25.0, 20.0)  # Very low equity
        
        assert can_trade is False
        assert "Insufficient funds" in reason
    
    def test_can_open_new_trade_max_drawdown(self, risk_manager, sample_trades):
        """Test trade rejection due to max drawdown"""
        risk_manager.set_daily_start_equity(5000.0)
        
        # Simulate large drawdown
        can_trade, reason = risk_manager.can_open_new_trade(25.0, 4000.0)  # $1000 drawdown > $750 limit
        
        assert can_trade is False
        assert "Maximum drawdown exceeded" in reason
    
    def test_get_current_exposure(self, risk_manager, sample_trades):
        """Test current exposure calculation"""
        exposure = risk_manager.get_current_exposure()
        
        assert exposure['open_trades_count'] == 2  # Two open trades
        assert exposure['total_premium_at_risk'] == 55.0  # 27.5 + 27.5
        assert isinstance(exposure['daily_pnl'], float)
    
    def test_should_halt_trading_max_drawdown(self, risk_manager, sample_trades):
        """Test trading halt due to max drawdown"""
        risk_manager.set_daily_start_equity(5000.0)
        
        should_halt, reason = risk_manager.should_halt_trading(4000.0)  # $1000 drawdown
        
        assert should_halt is True
        assert "Maximum daily drawdown exceeded" in reason
    
    def test_should_halt_trading_critical_equity(self, risk_manager, sample_trades):
        """Test trading halt due to critically low equity"""
        should_halt, reason = risk_manager.should_halt_trading(1000.0)  # Very low equity
        
        assert should_halt is True
        assert "Account equity critically low" in reason
    
    def test_should_halt_trading_consecutive_losses(self, risk_manager):
        """Test trading halt due to consecutive losses"""
        session = risk_manager.get_current_session()
        
        # Create multiple losing trades
        for i in range(6):
            trade = Trade(
                date=datetime.utcnow().date(),
                entry_time=datetime.utcnow() - timedelta(hours=i+1),
                exit_time=datetime.utcnow() - timedelta(hours=i),
                underlying_price_at_entry=4200.0,
                call_strike=4225.0,
                put_strike=4175.0,
                call_premium=2.5,
                put_premium=3.0,
                total_premium=27.5,
                realized_pnl=-27.5,  # Full loss
                implied_move=20.0,
                status='EXPIRED'
            )
            session.add(trade)
        
        session.commit()
        session.close()
        
        should_halt, reason = risk_manager.should_halt_trading(4500.0)
        
        assert should_halt is True
        assert "Too many consecutive losses" in reason
    
    def test_get_position_sizing_recommendation(self, risk_manager):
        """Test position sizing recommendations"""
        # Test normal case
        size = risk_manager.get_position_sizing_recommendation(5000.0, 25.0)
        assert size == 1  # Should recommend 1 contract
        
        # Test with very expensive option
        size = risk_manager.get_position_sizing_recommendation(5000.0, 150.0)
        assert size == 1  # Should still be at least 1
        
        # Test with cheap option and high equity
        size = risk_manager.get_position_sizing_recommendation(10000.0, 10.0)
        assert size >= 1
    
    def test_update_daily_summary(self, risk_manager):
        """Test daily summary update"""
        # Create sample trades directly in this test to avoid session issues
        session = risk_manager.get_current_session()
        today = datetime.utcnow().date()
        
        # Create some test trades
        trade1 = Trade(
            date=today,
            entry_time=datetime.utcnow() - timedelta(hours=1),
            underlying_price_at_entry=4200.0,
            call_strike=4225.0,
            put_strike=4175.0,
            call_premium=2.5,
            put_premium=3.0,
            total_premium=27.5,
            implied_move=20.0,
            status='OPEN'
        )
        
        trade2 = Trade(
            date=today,
            entry_time=datetime.utcnow() - timedelta(hours=3),
            exit_time=datetime.utcnow() - timedelta(hours=1),
            underlying_price_at_entry=4180.0,
            call_strike=4205.0,
            put_strike=4155.0,
            call_premium=2.0,
            put_premium=2.5,
            total_premium=22.5,
            realized_pnl=67.5,  # 3x profit
            implied_move=15.0,
            status='CLOSED_WIN'
        )
        
        session.add_all([trade1, trade2])
        session.commit()
        session.close()
        
        # Now test the daily summary update
        risk_manager.update_daily_summary()
        
        # Check the results
        session = risk_manager.get_current_session()
        summary = session.query(DailySummary).filter(DailySummary.date == today).first()
        
        assert summary is not None
        assert summary.total_trades == 2  # Two trades created in this test
        assert summary.winning_trades == 1  # One winning trade
        assert summary.losing_trades == 0  # No losing trades (1 is still open)
        assert summary.gross_profit == 67.5  # Profit from winning trade
        
        session.close()
    
    def test_log_risk_event(self, risk_manager, caplog):
        """Test risk event logging"""
        with caplog.at_level('WARNING'):
            risk_manager.log_risk_event("TEST_EVENT", "Test message", "Test details")
        
        assert "RISK EVENT [TEST_EVENT]: Test message" in caplog.text
        assert "RISK DETAILS: Test details" in caplog.text
    
    def test_get_risk_metrics_summary(self, risk_manager, sample_trades):
        """Test risk metrics summary"""
        risk_manager.set_daily_start_equity(5000.0)
        
        metrics = risk_manager.get_risk_metrics_summary()
        
        assert 'timestamp' in metrics
        assert metrics['open_trades'] == 2
        assert metrics['max_trades_limit'] == config.trading.max_open_trades
        assert metrics['total_premium_at_risk'] == 55.0
        assert 'current_drawdown' in metrics
        assert 'daily_pnl' in metrics
        assert metrics['trading_halted'] is False
    
    @patch('app.risk_manager.config.trading.min_time_between_trades', 60)
    def test_minimum_time_between_trades(self, risk_manager):
        """Test minimum time between trades enforcement"""
        session = risk_manager.get_current_session()
        
        # Create a recent trade
        recent_trade = Trade(
            date=datetime.utcnow().date(),
            entry_time=datetime.utcnow() - timedelta(minutes=30),  # 30 minutes ago
            underlying_price_at_entry=4200.0,
            call_strike=4225.0,
            put_strike=4175.0,
            call_premium=2.5,
            put_premium=3.0,
            total_premium=27.5,
            implied_move=20.0,
            status='OPEN'
        )
        session.add(recent_trade)
        session.commit()
        session.close()
        
        # Should not allow new trade (need 60 minutes gap)
        can_trade, reason = risk_manager.can_open_new_trade(25.0, 5000.0)
        
        assert can_trade is False
        assert "Minimum time between trades not met" in reason