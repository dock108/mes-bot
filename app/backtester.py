"""
Backtesting engine for MES 0DTE Lotto-Grid Options Bot
Uses synthetic option pricing with Black-Scholes model
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from sqlalchemy.orm import Session

from app.config import config
from app.data_providers.vix_provider import VIXProvider
from app.models import BacktestResult, get_session_maker

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Individual backtest trade record"""

    entry_time: datetime
    underlying_price: float
    call_strike: float
    put_strike: float
    call_premium: float
    put_premium: float
    total_premium: float
    implied_move: float

    # Exit info
    exit_time: Optional[datetime] = None
    call_exit_price: Optional[float] = None
    put_exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED_WIN, CLOSED_LOSS, EXPIRED

    def __post_init__(self):
        self.id = id(self)  # Simple ID for tracking


@dataclass
class DecisionTrace:
    """Record of each trading decision point"""

    timestamp: datetime
    price: float
    decision: str  # "TRADED", "NO_TRADE", "NEAR_MISS"
    reasons: List[str]
    market_conditions: Dict[str, float]
    near_miss_score: float = 0.0  # 0-1, how close to trading
    potential_trade: Optional[Dict] = None  # What trade would have looked like


class BlackScholesCalculator:
    """Black-Scholes option pricing calculator"""

    @staticmethod
    def option_price(
        S: float, K: float, T: float, r: float, sigma: float, option_type: str
    ) -> float:
        """
        Calculate Black-Scholes option price
        S: Stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        """
        if T <= 0:
            # At expiration, return intrinsic value
            if option_type.lower() == "call":
                return max(0, S - K)
            else:
                return max(0, K - S)

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(0, price)  # Ensure non-negative price


class LottoGridBacktester:
    """Backtesting engine for the Lotto Grid strategy"""

    def __init__(self, database_url: str):
        self.session_maker = get_session_maker(database_url)
        self.bs_calculator = BlackScholesCalculator()
        self.vix_provider = None  # Initialize when needed

    def fetch_historical_data(
        self, symbol: str, start_date: date, end_date: date, interval: str = "5m"
    ) -> pd.DataFrame:
        """Fetch historical price data from Yahoo Finance"""
        # Try MES=F first, then fall back to ES=F, then SPY if not available
        for ticker_symbol in ["MES=F", "ES=F", "SPY"]:
            try:
                logger.info(f"Fetching futures data ({ticker_symbol}) from Yahoo Finance")
                ticker = yf.Ticker(ticker_symbol)

                # For intraday data, limit to last 60 days
                if interval in ["1m", "5m", "15m", "30m", "60m"]:
                    max_start = date.today() - timedelta(days=59)
                    if start_date < max_start:
                        logger.warning(
                            f"Adjusting start date from {start_date} to {max_start} for intraday data"
                        )
                        start_date = max_start

                # Download data
                data = ticker.history(
                    start=start_date,
                    end=end_date + timedelta(days=1),
                    interval=interval,
                    prepost=False,  # Regular trading hours only
                )

                if not data.empty:
                    # Clean and prepare data
                    data = data.dropna()
                    data.index = pd.to_datetime(data.index)
                    logger.info(f"Successfully fetched {len(data)} records using {ticker_symbol}")
                    return data
                else:
                    logger.warning(f"No data returned for {ticker_symbol}")

            except Exception as e:
                logger.warning(f"Failed to fetch data with {ticker_symbol}: {str(e)}")
                continue

        # If we get here, all tickers failed
        raise ValueError(
            f"Unable to fetch data from MES=F, ES=F, or SPY. "
            f"Note: Yahoo Finance only provides intraday data for the last 60 days."
        )

    def calculate_implied_volatility(
        self,
        price_data: pd.DataFrame,
        window: int = 20,
        trading_date: Optional[date] = None,
    ) -> float:
        """Calculate implied volatility from historical price movements, with VIX adjustment"""
        try:
            # Use close prices
            closes = price_data["Close"]

            # Calculate returns
            returns = np.log(closes / closes.shift(1)).dropna()

            # Calculate rolling volatility (annualized)
            # For intraday data, need to scale appropriately
            if len(returns) < window:
                window = len(returns)

            volatility = returns.rolling(window=window).std().iloc[-1]

            # Annualize based on interval
            # For 5-minute data: 252 trading days * 78 intervals per day (6.5 hours * 12)
            trading_periods_per_year = 252 * 78  # Approximate for 5m data
            realized_vol = volatility * np.sqrt(trading_periods_per_year)

            # Try to get VIX data for adjustment
            if trading_date and self.vix_provider:
                try:
                    vix_value = self.vix_provider.get_vix_value(trading_date)
                    vix_vol = vix_value / 100.0  # VIX is in percentage points

                    # Blend realized vol with VIX-implied vol (30% VIX, 70% realized)
                    # This helps account for forward-looking volatility expectations
                    blended_vol = 0.7 * realized_vol + 0.3 * vix_vol

                    logger.debug(
                        f"Volatility blend: Realized={realized_vol:.2%}, "
                        f"VIX={vix_vol:.2%}, Blended={blended_vol:.2%}"
                    )
                    return max(0.05, min(2.0, blended_vol))  # 5% to 200%

                except Exception as e:
                    logger.debug(f"Could not get VIX data for {trading_date}: {e}")
                    # Fall back to realized vol only

            # Ensure reasonable bounds
            return max(0.05, min(2.0, realized_vol))  # 5% to 200%

        except Exception as e:
            logger.warning(f"Error calculating implied volatility: {e}")
            return 0.15  # Default to 15%

    def calculate_atm_straddle_price(
        self, underlying_price: float, time_to_expiry: float, volatility: float
    ) -> Tuple[float, float, float]:
        """Calculate ATM straddle price and implied move"""
        # Round to nearest strike (25-point increments for MES)
        atm_strike = round(underlying_price / 25) * 25

        risk_free_rate = 0.01  # Assume 1% risk-free rate

        call_price = self.bs_calculator.option_price(
            underlying_price, atm_strike, time_to_expiry, risk_free_rate, volatility, "call"
        )

        put_price = self.bs_calculator.option_price(
            underlying_price, atm_strike, time_to_expiry, risk_free_rate, volatility, "put"
        )

        # Implied move ≈ straddle price
        implied_move = call_price + put_price

        return call_price, put_price, implied_move

    def calculate_strike_levels(
        self, underlying_price: float, implied_move: float
    ) -> List[Tuple[float, float]]:
        """Calculate strike levels based on implied move"""
        strike_pairs = []

        # Level 1: 1.25x implied move
        offset_1 = implied_move * config.trading.implied_move_multiplier_1
        call_strike_1 = self._round_to_strike(underlying_price + offset_1)
        put_strike_1 = self._round_to_strike(underlying_price - offset_1)
        strike_pairs.append((call_strike_1, put_strike_1))

        # Level 2: 1.5x implied move
        offset_2 = implied_move * config.trading.implied_move_multiplier_2
        call_strike_2 = self._round_to_strike(underlying_price + offset_2)
        put_strike_2 = self._round_to_strike(underlying_price - offset_2)
        strike_pairs.append((call_strike_2, put_strike_2))

        return strike_pairs

    def _round_to_strike(self, price: float) -> float:
        """Round price to nearest valid strike (25-point increments)"""
        return round(price / 25) * 25

    def should_place_trade(
        self,
        price_data: pd.DataFrame,
        current_idx: int,
        implied_move: float,
        last_trade_time: Optional[datetime] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Determine if conditions are met to place a trade, return decision details"""
        current_time = price_data.index[current_idx]
        current_price = price_data.iloc[current_idx]["Close"]
        reasons = []
        passed_checks = 0
        total_checks = 3

        # Initialize market conditions
        market_conditions = {
            "current_price": current_price,
            "implied_move": implied_move,
            "volatility_threshold": config.trading.volatility_threshold,
        }

        # Check 1: Minimum time between trades
        can_trade_time = True
        if last_trade_time:
            time_diff = current_time - last_trade_time
            minutes_since_last = time_diff.total_seconds() / 60
            market_conditions["minutes_since_last_trade"] = minutes_since_last

            if time_diff < timedelta(minutes=config.trading.min_time_between_trades):
                can_trade_time = False
                reasons.append(
                    f"Only {minutes_since_last:.1f} minutes since last trade "
                    f"(need {config.trading.min_time_between_trades})"
                )
            else:
                passed_checks += 1
        else:
            passed_checks += 1
            market_conditions["minutes_since_last_trade"] = float("inf")

        # Check 2: Sufficient data
        lookback_periods = 12  # 60 minutes / 5 minutes
        start_idx = max(0, current_idx - lookback_periods)
        recent_data = price_data.iloc[start_idx : current_idx + 1]

        has_sufficient_data = len(recent_data) >= 2
        if not has_sufficient_data:
            reasons.append(f"Insufficient data: only {len(recent_data)} periods available")
        else:
            passed_checks += 1

        # Check 3: Volatility condition
        volatility_check_passed = False
        if has_sufficient_data:
            realized_range = recent_data["High"].max() - recent_data["Low"].min()
            volatility_threshold = implied_move * config.trading.volatility_threshold
            volatility_ratio = realized_range / implied_move if implied_move > 0 else 0

            market_conditions.update(
                {
                    "realized_range": realized_range,
                    "volatility_threshold_value": volatility_threshold,
                    "volatility_ratio": volatility_ratio,
                    "realized_high": recent_data["High"].max(),
                    "realized_low": recent_data["Low"].min(),
                }
            )

            volatility_check_passed = bool(realized_range < volatility_threshold)
            if not volatility_check_passed:
                reasons.append(
                    f"Realized volatility ({volatility_ratio:.1%}) > "
                    f"threshold ({config.trading.volatility_threshold:.1%})"
                )
            else:
                passed_checks += 1
                reasons.append(
                    f"Volatility condition met: {volatility_ratio:.1%} < "
                    f"{config.trading.volatility_threshold:.1%}"
                )

        # Calculate near-miss score
        near_miss_score = passed_checks / total_checks

        # Determine if we should trade
        should_trade = bool(can_trade_time and has_sufficient_data and volatility_check_passed)

        decision_info = {
            "can_trade": should_trade,
            "reasons": reasons,
            "market_conditions": market_conditions,
            "near_miss_score": near_miss_score,
            "checks_passed": f"{passed_checks}/{total_checks}",
        }

        return should_trade, decision_info

    def place_strangle(
        self,
        underlying_price: float,
        call_strike: float,
        put_strike: float,
        time_to_expiry: float,
        volatility: float,
        entry_time: datetime,
        implied_move: float,
    ) -> Optional[BacktestTrade]:
        """Simulate placing a strangle trade"""
        try:
            risk_free_rate = 0.01

            # Calculate option premiums
            call_premium = self.bs_calculator.option_price(
                underlying_price, call_strike, time_to_expiry, risk_free_rate, volatility, "call"
            )

            put_premium = self.bs_calculator.option_price(
                underlying_price, put_strike, time_to_expiry, risk_free_rate, volatility, "put"
            )

            total_premium = (call_premium + put_premium) * 5  # MES multiplier

            # Check premium limit
            if total_premium > config.trading.max_premium_per_strangle:
                logger.debug(
                    f"Premium ${total_premium:.2f} exceeds limit ${config.trading.max_premium_per_strangle}"
                )
                return None

            trade = BacktestTrade(
                entry_time=entry_time,
                underlying_price=underlying_price,
                call_strike=call_strike,
                put_strike=put_strike,
                call_premium=call_premium,
                put_premium=put_premium,
                total_premium=total_premium,
                implied_move=implied_move,
            )

            return trade

        except Exception as e:
            logger.error(f"Error placing strangle: {e}")
            return None

    def update_trade_pnl(
        self, trade: BacktestTrade, current_price: float, time_to_expiry: float, volatility: float
    ) -> bool:
        """Update trade P&L and check for exit conditions"""
        try:
            risk_free_rate = 0.01

            # Calculate current option values
            current_call_price = self.bs_calculator.option_price(
                current_price, trade.call_strike, time_to_expiry, risk_free_rate, volatility, "call"
            )

            current_put_price = self.bs_calculator.option_price(
                current_price, trade.put_strike, time_to_expiry, risk_free_rate, volatility, "put"
            )

            # Check profit targets (4x premium)
            call_target = trade.call_premium * config.trading.profit_target_multiplier
            put_target = trade.put_premium * config.trading.profit_target_multiplier

            profit_hit = False

            if current_call_price >= call_target:
                # Call hit profit target
                call_pnl = (call_target - trade.call_premium) * 5
                put_pnl = (0 - trade.put_premium) * 5  # Assume put expires worthless
                total_pnl = call_pnl + put_pnl

                trade.call_exit_price = call_target
                trade.put_exit_price = 0
                trade.realized_pnl = total_pnl
                trade.status = "CLOSED_WIN"
                profit_hit = True

            elif current_put_price >= put_target:
                # Put hit profit target
                put_pnl = (put_target - trade.put_premium) * 5
                call_pnl = (0 - trade.call_premium) * 5  # Assume call expires worthless
                total_pnl = call_pnl + put_pnl

                trade.call_exit_price = 0
                trade.put_exit_price = put_target
                trade.realized_pnl = total_pnl
                trade.status = "CLOSED_WIN"
                profit_hit = True

            return profit_hit

        except Exception as e:
            logger.error(f"Error updating trade P&L: {e}")
            return False

    def close_trade_at_expiry(self, trade: BacktestTrade, final_price: float):
        """Close trade at expiration"""
        # Calculate intrinsic values
        call_intrinsic = max(0, final_price - trade.call_strike)
        put_intrinsic = max(0, trade.put_strike - final_price)

        # If already closed with profit, don't override
        if trade.status == "CLOSED_WIN":
            return

        # Calculate final P&L
        call_pnl = (call_intrinsic - trade.call_premium) * 5
        put_pnl = (put_intrinsic - trade.put_premium) * 5
        total_pnl = call_pnl + put_pnl

        trade.call_exit_price = call_intrinsic
        trade.put_exit_price = put_intrinsic
        trade.realized_pnl = total_pnl

        if total_pnl > 0:
            trade.status = "CLOSED_WIN"
        else:
            trade.status = "EXPIRED"

    async def run_backtest(
        self, start_date: date, end_date: date, initial_capital: float = 5000, symbol: str = "MES"
    ) -> Dict:
        """Run complete backtest"""
        logger.info(f"Starting backtest from {start_date} to {end_date}")

        # Validate dates
        if end_date > date.today():
            raise ValueError(f"End date {end_date} cannot be in the future")

        if start_date >= end_date:
            raise ValueError(f"Start date {start_date} must be before end date {end_date}")

        # Initialize VIX provider if available
        try:
            self.vix_provider = VIXProvider()
            logger.info("VIX data provider initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize VIX provider: {e}")
            logger.warning("Backtesting will continue without VIX data")
            self.vix_provider = None

        try:
            # Fetch historical data
            price_data = self.fetch_historical_data(symbol, start_date, end_date, "5m")

            # Initialize backtest state
            capital = initial_capital
            trades = []
            daily_pnl = []
            open_trades = []
            max_drawdown = 0
            peak_capital = initial_capital
            decision_traces = []  # Collect all trading decisions

            # Group by trading days
            daily_groups = price_data.groupby(price_data.index.date)

            for trading_date, day_data in daily_groups:
                logger.info(f"Processing {trading_date} with {len(day_data)} data points")

                # Skip weekends
                if trading_date.weekday() >= 5:
                    logger.info(f"  Skipping weekend: {trading_date}")
                    continue

                # Filter to market hours (9:30 AM - 4:00 PM ET)
                day_data = day_data.between_time("09:30", "16:00")
                if day_data.empty:
                    logger.info(f"  No data during market hours for {trading_date}")
                    continue

                logger.info(f"  Market hours data points: {len(day_data)}")

                # Calculate daily implied volatility (with VIX adjustment if available)
                volatility = self.calculate_implied_volatility(day_data, trading_date=trading_date)

                # Initialize daily session (at 9:35)
                session_start_idx = None
                for idx, (timestamp, row) in enumerate(day_data.iterrows()):
                    if timestamp.time() >= pd.Timestamp("09:35").time():
                        session_start_idx = idx
                        break

                if session_start_idx is None:
                    logger.warning(f"  No data after 9:35 AM for {trading_date}")
                    continue

                logger.info(
                    f"  Session start index: {session_start_idx}, time: {day_data.index[session_start_idx]}"
                )

                # Calculate implied move from ATM straddle
                opening_price = day_data.iloc[session_start_idx]["Close"]
                time_to_expiry = 6.5 / 24  # Hours until 4 PM in fraction of day

                call_price, put_price, implied_move = self.calculate_atm_straddle_price(
                    opening_price, time_to_expiry, volatility
                )

                last_trade_time = None
                day_start_capital = capital

                # Process each time period
                for idx, (timestamp, row) in enumerate(day_data.iterrows()):
                    current_price = row["Close"]
                    current_time = timestamp

                    # Calculate time to expiry (in years)
                    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
                    time_to_expiry = max(
                        0, (market_close - current_time).total_seconds() / (365.25 * 24 * 3600)
                    )

                    # Update open trades
                    trades_to_remove = []
                    for trade in open_trades:
                        if time_to_expiry <= 0:
                            # Expire trade
                            self.close_trade_at_expiry(trade, current_price)
                            trade.exit_time = current_time
                            trades.append(trade)
                            capital += trade.realized_pnl
                            trades_to_remove.append(trade)
                        else:
                            # Check for profit exit
                            profit_hit = self.update_trade_pnl(
                                trade, current_price, time_to_expiry, volatility
                            )
                            if profit_hit:
                                trade.exit_time = current_time
                                trades.append(trade)
                                capital += trade.realized_pnl
                                trades_to_remove.append(trade)

                    # Remove closed trades
                    for trade in trades_to_remove:
                        open_trades.remove(trade)

                    # Check for new trade entry (after 9:35 and before 3:30)
                    # Always check decisions during trading hours for trace
                    if (
                        idx >= session_start_idx
                        and current_time.time() < pd.Timestamp("15:30").time()
                        and time_to_expiry > 0.5 / 24
                    ):
                        should_trade, decision_info = self.should_place_trade(
                            day_data, idx, implied_move, last_trade_time
                        )

                        if idx % 12 == 0:  # Log every hour
                            logger.debug(
                                f"  Decision at {current_time}: should_trade={should_trade}, "
                                f"near_miss={decision_info['near_miss_score']:.2f}"
                            )

                        # Add position limit check to decision info
                        if len(open_trades) >= config.trading.max_open_trades:
                            decision_info["reasons"].append(
                                f"Max positions reached ({config.trading.max_open_trades})"
                            )
                            decision_info["near_miss_score"] *= 0.9

                        # Calculate potential trade details
                        strike_pairs = self.calculate_strike_levels(current_price, implied_move)
                        if strike_pairs:
                            call_strike, put_strike = strike_pairs[0]
                            decision_info["potential_trade"] = {
                                "call_strike": call_strike,
                                "put_strike": put_strike,
                                "estimated_premium": implied_move * 0.1,  # Rough estimate
                            }

                        # Create decision trace
                        decision = (
                            "TRADED"
                            if should_trade
                            else (
                                "NEAR_MISS"
                                if decision_info["near_miss_score"] > 0.5
                                else "NO_TRADE"
                            )
                        )
                        trace = DecisionTrace(
                            timestamp=current_time,
                            price=current_price,
                            decision=decision,
                            reasons=decision_info["reasons"],
                            market_conditions=decision_info["market_conditions"],
                            near_miss_score=decision_info["near_miss_score"],
                            potential_trade=decision_info.get("potential_trade"),
                        )
                        decision_traces.append(trace)

                        if should_trade:
                            # Calculate strikes
                            strike_pairs = self.calculate_strike_levels(current_price, implied_move)

                            if strike_pairs:
                                call_strike, put_strike = strike_pairs[0]  # Use first level

                                trade = self.place_strangle(
                                    current_price,
                                    call_strike,
                                    put_strike,
                                    time_to_expiry,
                                    volatility,
                                    current_time,
                                    implied_move,
                                )

                                if trade and capital >= trade.total_premium:
                                    open_trades.append(trade)
                                    capital -= trade.total_premium
                                    last_trade_time = current_time

                                    logger.debug(
                                        f"Placed strangle at {current_time}: "
                                        f"{call_strike}C/{put_strike}P for ${trade.total_premium:.2f}"
                                    )
                                else:
                                    # Update trace if trade failed
                                    trace.decision = "NO_TRADE"
                                    if not trade:
                                        trace.reasons.append("Failed to calculate option premiums")
                                    elif capital < trade.total_premium:
                                        trace.reasons.append(
                                            f"Insufficient capital: ${capital:.2f} < ${trade.total_premium:.2f}"
                                        )

                # End of day - close any remaining trades
                for trade in open_trades:
                    self.close_trade_at_expiry(trade, day_data.iloc[-1]["Close"])
                    trade.exit_time = day_data.index[-1]
                    trades.append(trade)
                    capital += trade.realized_pnl

                open_trades.clear()

                # Calculate daily metrics
                day_pnl = capital - day_start_capital
                daily_pnl.append({"date": trading_date, "pnl": day_pnl, "capital": capital})

                # Update drawdown
                if capital > peak_capital:
                    peak_capital = capital
                else:
                    drawdown = peak_capital - capital
                    max_drawdown = max(max_drawdown, drawdown)

            # Calculate final metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.realized_pnl > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            total_return = (capital - initial_capital) / initial_capital

            # Calculate Sharpe ratio
            if daily_pnl:
                daily_returns = [d["pnl"] / initial_capital for d in daily_pnl]
                sharpe_ratio = (
                    np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                    if np.std(daily_returns) > 0
                    else 0
                )
            else:
                sharpe_ratio = 0

            results = {
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital,
                "final_capital": capital,
                "total_return": total_return,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "trades": trades,
                "daily_pnl": daily_pnl,
                "decision_traces": decision_traces,  # Include decision trace data
            }

            logger.info(f"Backtest completed:")
            logger.info(f"  Total return: {total_return:.2%}")
            logger.info(f"  Win rate: {win_rate:.2%}")
            logger.info(f"  Max drawdown: ${max_drawdown:.2f}")
            logger.info(f"  Total trades: {total_trades}")
            logger.info(f"  Decision traces collected: {len(decision_traces)}")

            return results

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise

    async def save_backtest_result(self, results: Dict, name: str) -> int:
        """Save backtest results to database"""
        session = self.session_maker()
        try:
            backtest_result = BacktestResult(
                name=name,
                start_date=results["start_date"],
                end_date=results["end_date"],
                initial_capital=results["initial_capital"],
                max_trades=config.trading.max_open_trades,
                profit_target=config.trading.profit_target_multiplier,
                implied_move_mult_1=config.trading.implied_move_multiplier_1,
                implied_move_mult_2=config.trading.implied_move_multiplier_2,
                volatility_threshold=config.trading.volatility_threshold,
                total_trades=results["total_trades"],
                winning_trades=results["winning_trades"],
                losing_trades=results["losing_trades"],
                win_rate=results["win_rate"],
                final_capital=results["final_capital"],
                total_return=results["total_return"],
                max_drawdown=results["max_drawdown"],
                sharpe_ratio=results["sharpe_ratio"],
                execution_time=0,  # Would be calculated in practice
            )

            session.add(backtest_result)
            session.commit()

            logger.info(f"Saved backtest result '{name}' with ID: {backtest_result.id}")
            return backtest_result.id

        except Exception as e:
            logger.error(f"Error saving backtest result: {e}")
            session.rollback()
            raise
        finally:
            session.close()
