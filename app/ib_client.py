"""
Interactive Brokers API client using ib_insync
"""

import asyncio
import logging
import math
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import pytz
from ib_insync import IB, Contract, Future, Option, Order, Trade
from ib_insync.objects import AccountValue, PortfolioItem

from app.circuit_breaker import CircuitOpenError, circuit_breaker
from app.config import config
from app.ib_connection_manager import ConnectionState, IBConnectionManager, with_connection_check
from app.instruments import instrument_manager
from app.models import Trade as TradeModel

logger = logging.getLogger(__name__)


class IBClient:
    """Interactive Brokers API client wrapper"""

    def __init__(self):
        self.ib = IB()
        self.connected = False
        self.contracts_cache = {}
        self.market_data_subscriptions = set()

        # Set up event handlers
        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error
        self.ib.orderStatusEvent += self._on_order_status

        # Initialize connection manager
        self.connection_manager = IBConnectionManager(self)

        # Set up connection callbacks
        self.connection_manager.on_connected_callback = self._on_connection_restored
        self.connection_manager.on_disconnected_callback = self._on_connection_lost
        self.connection_manager.on_error_callback = self._on_connection_error

    async def connect(self) -> bool:
        """Connect to IB Gateway/TWS with resilience"""
        return await self.connection_manager.connect()

    async def connect_direct(self) -> bool:
        """Direct connection method for connection manager to use"""
        try:
            await self.ib.connectAsync(
                host=config.ib.host, port=config.ib.port, clientId=config.ib.client_id, timeout=20
            )
            logger.info(f"Connected to IB Gateway at {config.ib.host}:{config.ib.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False

    async def disconnect(self):
        """Disconnect from IB"""
        if self.connected:
            self.ib.disconnect()
            logger.info("Disconnected from IB Gateway")

    def _on_connected(self):
        """Handle connection event"""
        self.connected = True
        logger.info("IB connection established")

    def _on_disconnected(self):
        """Handle disconnection event"""
        self.connected = False
        logger.warning("IB connection lost")

    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle IB errors"""
        if errorCode in [2104, 2106, 2158]:  # Normal status messages
            logger.debug(f"IB Status: {errorString}")
        else:
            logger.error(f"IB Error {errorCode}: {errorString} (Contract: {contract})")

    def _on_order_status(self, trade):
        """Handle order status updates"""
        logger.info(f"Order status update: {trade.order.orderId} - {trade.orderStatus.status}")

    async def get_contract_for_symbol(self, symbol: str) -> Contract:
        """Get contract for any supported instrument symbol"""
        spec = instrument_manager.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")

        if spec.instrument_type.value == "futures":
            return await self.get_futures_contract(symbol)
        else:
            raise ValueError(f"Unsupported instrument type: {spec.instrument_type}")

    async def get_futures_contract(self, symbol: str, expiry: Optional[str] = None) -> Contract:
        """Get futures contract for symbol"""
        if symbol == "MES":
            return await self.get_mes_contract(expiry)
        else:
            return await self.get_front_month_contract_for_symbol(symbol, expiry)

    async def get_front_month_contract_for_symbol(
        self, symbol: str, expiry: Optional[str] = None
    ) -> Contract:
        """Get front month contract for any futures symbol"""
        spec = instrument_manager.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")

        if expiry:
            # Use specific expiry
            contract = Future(symbol, expiry, spec.exchange)
            qualified = await self.ib.qualifyContractsAsync(contract)
            if qualified:
                return qualified[0]
            else:
                raise ValueError(f"Could not qualify {symbol} contract for expiry {expiry}")
        else:
            # Use auto-detection (similar to MES logic)
            return await self._get_front_month_contract_generic(symbol)

    async def _get_front_month_contract_generic(self, symbol: str) -> Contract:
        """Generic front month contract detection for any symbol"""
        spec = instrument_manager.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")

        try:
            # Request available contracts from IB
            base_contract = Future(symbol, "", spec.exchange)
            contracts = await self.ib.reqContractDetailsAsync(base_contract)

            if not contracts:
                logger.warning(f"No {symbol} contracts found, falling back to estimation")
                return await self._get_estimated_front_month_contract_generic(symbol)

            # Filter and sort by expiry date
            current_date = datetime.now().date()
            rollover_threshold = current_date + timedelta(days=config.ib.contract_rollover_days)

            valid_contracts = []
            for contract_details in contracts:
                expiry_str = contract_details.contract.lastTradeDateOrContractMonth
                if expiry_str:
                    try:
                        expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()
                        if expiry_date > rollover_threshold:
                            valid_contracts.append((expiry_date, contract_details.contract))
                    except ValueError:
                        continue

            if not valid_contracts:
                logger.warning(f"No valid {symbol} contracts found, falling back to estimation")
                return await self._get_estimated_front_month_contract_generic(symbol)

            # Sort by expiry date and pick the nearest one
            valid_contracts.sort(key=lambda x: x[0])
            selected_expiry, selected_contract = valid_contracts[0]

            logger.info(
                f"Auto-selected {symbol} contract: "
                f"{selected_contract.lastTradeDateOrContractMonth} "
                f"(expires {selected_expiry})"
            )
            return selected_contract

        except Exception as e:
            logger.error(f"Error auto-detecting {symbol} contract: {e}")
            logger.info(f"Falling back to estimated front month contract for {symbol}")
            return await self._get_estimated_front_month_contract_generic(symbol)

    async def _get_estimated_front_month_contract_generic(self, symbol: str) -> Contract:
        """Fallback estimation for any symbol"""
        spec = instrument_manager.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")

        current_date = datetime.now()
        rollover_threshold = current_date + timedelta(days=config.ib.contract_rollover_days)

        # Simple estimation logic
        if rollover_threshold.month != current_date.month:
            target_month = rollover_threshold
        else:
            target_month = current_date

        contract_month = target_month.strftime("%Y%m")

        logger.warning(f"Estimating {symbol} contract month as {contract_month}")

        contract = Future(symbol, contract_month, spec.exchange)
        qualified = await self.ib.qualifyContractsAsync(contract)

        if qualified:
            return qualified[0]
        else:
            raise ValueError(
                f"Could not qualify estimated {symbol} contract for month {contract_month}"
            )

    async def get_front_month_contract(self) -> Contract:
        """Get the front month MES contract automatically"""
        # Check if a specific contract month is configured
        if config.ib.mes_contract_month:
            logger.info(f"Using configured MES contract month: {config.ib.mes_contract_month}")
            return await self.get_mes_contract(config.ib.mes_contract_month)

        try:
            # Request available MES contracts from IB
            base_contract = Future("MES", "", "GLOBEX")
            contracts = await self.ib.reqContractDetailsAsync(base_contract)

            if not contracts:
                logger.warning("No MES contracts found, falling back to current month estimation")
                return await self._get_estimated_front_month_contract()

            # Filter to get valid contracts and sort by expiry date
            current_date = datetime.now().date()
            rollover_threshold = current_date + timedelta(days=config.ib.contract_rollover_days)

            valid_contracts = []
            for contract_details in contracts:
                expiry_str = contract_details.contract.lastTradeDateOrContractMonth
                if expiry_str:
                    # Parse expiry date (format: YYYYMMDD)
                    try:
                        expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()
                        if expiry_date > rollover_threshold:
                            valid_contracts.append((expiry_date, contract_details.contract))
                    except ValueError:
                        continue

            if not valid_contracts:
                logger.warning("No valid MES contracts found, falling back to estimation")
                return await self._get_estimated_front_month_contract()

            # Sort by expiry date and pick the nearest one
            valid_contracts.sort(key=lambda x: x[0])
            selected_expiry, selected_contract = valid_contracts[0]

            logger.info(
                f"Auto-selected MES contract: {selected_contract.lastTradeDateOrContractMonth} "
                f"(expires {selected_expiry})"
            )
            return selected_contract

        except Exception as e:
            logger.error(f"Error auto-detecting MES contract: {e}")
            logger.info("Falling back to estimated front month contract")
            return await self._get_estimated_front_month_contract()

    async def _get_estimated_front_month_contract(self) -> Contract:
        """Fallback method to estimate front month contract when IB query fails"""
        current_date = datetime.now()

        # MES contracts typically expire on the 3rd Friday of the month
        # If we're past the rollover threshold, use next month
        rollover_threshold = current_date + timedelta(days=config.ib.contract_rollover_days)

        # Simple estimation: if we're in the last few days of the month, use next month
        if rollover_threshold.month != current_date.month:
            target_month = rollover_threshold
        else:
            target_month = current_date

        # Format as YYYYMM for the contract month
        contract_month = target_month.strftime("%Y%m")

        logger.warning(
            f"Estimating MES contract month as {contract_month} "
            f"(current date: {current_date.date()}, "
            f"rollover threshold: {rollover_threshold.date()})"
        )

        contract = Future("MES", contract_month, "GLOBEX")
        qualified = await self.ib.qualifyContractsAsync(contract)

        if qualified:
            return qualified[0]
        else:
            raise ValueError(f"Could not qualify estimated MES contract for month {contract_month}")

    async def get_mes_contract(self, expiry: Optional[str] = None) -> Contract:
        """Get MES futures contract"""
        if expiry is None:
            # Use automatic front month detection
            return await self.get_front_month_contract()
        else:
            contract = Future("MES", expiry, "GLOBEX")

        # Qualify the contract
        qualified = await self.ib.qualifyContractsAsync(contract)
        if qualified:
            return qualified[0]
        else:
            raise ValueError(f"Could not qualify MES contract for expiry {expiry}")

    async def get_option_contract(
        self, symbol: str, expiry: str, strike: float, right: str
    ) -> Contract:
        """Get option contract for any instrument"""
        spec = instrument_manager.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")

        cache_key = f"{symbol}_{expiry}_{strike}_{right}"

        if cache_key in self.contracts_cache:
            return self.contracts_cache[cache_key]

        # Use option_symbol if different from underlying symbol
        option_symbol = spec.option_symbol or symbol
        contract = Option(option_symbol, expiry, strike, right, spec.exchange)
        qualified = await self.ib.qualifyContractsAsync(contract)

        if qualified:
            self.contracts_cache[cache_key] = qualified[0]
            return qualified[0]
        else:
            raise ValueError(f"Could not qualify {symbol} option: {expiry} {strike} {right}")

    async def get_mes_option_contract(self, expiry: str, strike: float, right: str) -> Contract:
        """Get MES option contract (backwards compatibility)"""
        return await self.get_option_contract("MES", expiry, strike, right)

    async def _on_connection_restored(self):
        """Handle connection restoration"""
        logger.info("Connection restored - resuming operations")
        # Re-subscribe to market data if needed
        # Could add logic to refresh positions/orders here

    async def _on_connection_lost(self):
        """Handle connection loss"""
        logger.warning("Connection lost - attempting recovery")
        self.connected = False

    async def _on_connection_error(self, error_msg: str):
        """Handle connection errors"""
        logger.error(f"Connection error: {error_msg}")
        # Could add alerting logic here

    @with_connection_check
    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    async def get_current_price(self, contract: Contract) -> Optional[float]:
        """Get current market price for a contract"""
        try:
            ticker = self.ib.reqMktData(contract, "", False, False)
            await asyncio.sleep(2)  # Wait for data

            if ticker.marketPrice() and ticker.marketPrice() > 0:
                return float(ticker.marketPrice())
            elif ticker.close and ticker.close > 0:
                return float(ticker.close)
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting price for {contract}: {e}")
            return None
        finally:
            self.ib.cancelMktData(contract)

    @with_connection_check
    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    async def get_market_data(self, contract: Contract) -> Optional[Dict[str, float]]:
        """Get comprehensive market data for a contract"""
        try:
            ticker = self.ib.reqMktData(contract, "", False, False)
            await asyncio.sleep(2)  # Wait for data to populate

            # Get various price points
            bid = ticker.bid if ticker.bid > 0 else None
            ask = ticker.ask if ticker.ask > 0 else None
            last = ticker.last if ticker.last > 0 else None
            close = ticker.close if ticker.close > 0 else None

            # Calculate mid price
            if bid and ask:
                mid = (bid + ask) / 2
            elif last:
                mid = last
            elif close:
                mid = close
            else:
                return None

            # Get volume data
            volume = ticker.volume if hasattr(ticker, "volume") and ticker.volume >= 0 else 0

            # For options, get implied volatility
            iv = None
            if contract.secType == "OPT" and hasattr(ticker, "modelGreeks"):
                if ticker.modelGreeks and hasattr(ticker.modelGreeks, "impliedVol"):
                    iv = ticker.modelGreeks.impliedVol

            return {
                "bid": bid or mid - 0.25,  # Approximate if not available
                "ask": ask or mid + 0.25,  # Approximate if not available
                "mid": mid,
                "last": last,
                "close": close,
                "volume": volume,
                "implied_volatility": iv,
                "bid_size": ticker.bidSize if hasattr(ticker, "bidSize") else 0,
                "ask_size": ticker.askSize if hasattr(ticker, "askSize") else 0,
            }

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
        finally:
            self.ib.cancelMktData(contract)

    @with_connection_check
    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    async def get_option_implied_volatility(self, contract: Contract) -> Optional[float]:
        """Get implied volatility for an option contract"""
        try:
            if contract.secType != "OPT":
                logger.warning("Contract is not an option")
                return None

            ticker = self.ib.reqMktData(contract, "", False, False)
            await asyncio.sleep(3)  # Wait longer for Greeks calculation

            if hasattr(ticker, "modelGreeks") and ticker.modelGreeks:
                if hasattr(ticker.modelGreeks, "impliedVol"):
                    iv = ticker.modelGreeks.impliedVol
                    if iv and iv > 0:
                        return float(iv)

            # Fallback: estimate from option price
            if ticker.marketPrice() and ticker.marketPrice() > 0:
                # Very rough approximation for ATM options
                option_price = float(ticker.marketPrice())
                underlying_price = await self.get_current_price(await self.get_mes_contract())
                if underlying_price:
                    time_to_expiry = 0.02  # Rough estimate for 0DTE in years
                    # Simplified IV approximation for ATM options
                    iv_estimate = (option_price / underlying_price) / (
                        0.4 * math.sqrt(time_to_expiry)
                    )
                    return min(2.0, max(0.05, iv_estimate))  # Cap between 5% and 200%

            return None

        except Exception as e:
            logger.error(f"Error getting option implied volatility: {e}")
            return None
        finally:
            self.ib.cancelMktData(contract)

    @with_connection_check
    async def get_atm_straddle_price(
        self, symbol: str, underlying_price: float, expiry: str
    ) -> Tuple[float, float, float]:
        """Get ATM straddle price for any instrument"""
        spec = instrument_manager.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")

        try:
            # Round to nearest strike using instrument-specific increment
            atm_strike = instrument_manager.round_to_strike(symbol, underlying_price)

            call_contract = await self.get_option_contract(symbol, expiry, atm_strike, "C")
            put_contract = await self.get_option_contract(symbol, expiry, atm_strike, "P")

            call_price = await self.get_current_price(call_contract)
            put_price = await self.get_current_price(put_contract)

            if call_price and put_price:
                straddle_price = call_price + put_price
                implied_move = straddle_price  # Simplified: straddle price ≈ implied daily move
                return call_price, put_price, implied_move
            else:
                raise ValueError(f"Could not get ATM option prices for {symbol}")

        except Exception as e:
            logger.error(f"Error getting ATM straddle price for {symbol}: {e}")
            raise

    async def get_atm_straddle_price_legacy(
        self, underlying_price: float, expiry: str
    ) -> Tuple[float, float, float]:
        """Get ATM straddle price and calculate implied move (legacy MES method)"""
        return await self.get_atm_straddle_price("MES", underlying_price, expiry)

    @with_connection_check
    @circuit_breaker(failure_threshold=3, recovery_timeout=30)
    async def place_bracket_order(
        self,
        contract: Contract,
        quantity: int,
        limit_price: float,
        take_profit_price: float,
        stop_loss_price: Optional[float] = None,
    ) -> List[Trade]:
        """Place a bracket order (entry + take profit + optional stop loss)"""
        try:
            # Create bracket orders
            bracket_orders = self.ib.bracketOrder(
                action="BUY",
                quantity=quantity,
                limitPrice=limit_price,
                takeProfitPrice=take_profit_price,
                stopLossPrice=stop_loss_price or 0.01,  # Minimal stop loss
            )

            # Place all orders
            trades = []
            for order in bracket_orders:
                trade = self.ib.placeOrder(contract, order)
                trades.append(trade)
                logger.info(f"Placed {order.orderType} order: {trade.order.orderId}")

            return trades

        except Exception as e:
            logger.error(f"Error placing bracket order: {e}")
            raise

    @with_connection_check
    @circuit_breaker(failure_threshold=3, recovery_timeout=30)
    async def place_strangle(
        self,
        symbol: str,
        underlying_price: float,
        call_strike: float,
        put_strike: float,
        expiry: str,
        max_premium: float,
    ) -> Dict:
        """Place a complete strangle trade for any instrument"""
        spec = instrument_manager.get_instrument(symbol)
        if not spec:
            raise ValueError(f"Unknown instrument: {symbol}")

        try:
            # Get option contracts
            call_contract = await self.get_option_contract(symbol, expiry, call_strike, "C")
            put_contract = await self.get_option_contract(symbol, expiry, put_strike, "P")

            # Get current option prices
            call_price = await self.get_current_price(call_contract)
            put_price = await self.get_current_price(put_contract)

            if not call_price or not put_price:
                raise ValueError(f"Could not get option prices for {symbol}")

            # Calculate total premium using instrument-specific multiplier
            total_premium = (call_price + put_price) * spec.option_multiplier

            if total_premium > max_premium:
                raise ValueError(f"Total premium ${total_premium:.2f} exceeds max ${max_premium}")

            # Calculate take profit prices
            call_tp_price = call_price * config.trading.profit_target_multiplier
            put_tp_price = put_price * config.trading.profit_target_multiplier

            # Place bracket orders for both legs
            call_trades = await self.place_bracket_order(
                call_contract, 1, call_price, call_tp_price
            )
            put_trades = await self.place_bracket_order(put_contract, 1, put_price, put_tp_price)

            return {
                "call_trades": call_trades,
                "put_trades": put_trades,
                "call_price": call_price,
                "put_price": put_price,
                "total_premium": total_premium,
                "call_strike": call_strike,
                "put_strike": put_strike,
                "underlying_price": underlying_price,
                "symbol": symbol,
            }

        except Exception as e:
            logger.error(f"Error placing strangle for {symbol}: {e}")
            raise

    async def place_strangle_legacy(
        self,
        underlying_price: float,
        call_strike: float,
        put_strike: float,
        expiry: str,
        max_premium: float,
    ) -> Dict:
        """Place a complete strangle trade (legacy MES method)"""
        return await self.place_strangle(
            "MES", underlying_price, call_strike, put_strike, expiry, max_premium
        )

    @with_connection_check
    @circuit_breaker(failure_threshold=3, recovery_timeout=30)
    async def cancel_order(self, order_id: int) -> bool:
        """Cancel an order by ID"""
        try:
            order = None
            for trade in self.ib.openTrades():
                if trade.order.orderId == order_id:
                    order = trade.order
                    break

            if order:
                self.ib.cancelOrder(order)
                logger.info(f"Cancelled order {order_id}")
                return True
            else:
                logger.warning(f"Order {order_id} not found for cancellation")
                return False

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    @with_connection_check
    @circuit_breaker(failure_threshold=3, recovery_timeout=30)
    async def close_position_at_market(self, contract: Contract, quantity: int) -> Optional[Trade]:
        """Close a position with market order"""
        try:
            market_order = Order()
            market_order.action = "SELL"
            market_order.orderType = "MKT"
            market_order.totalQuantity = quantity

            trade = self.ib.placeOrder(contract, market_order)
            logger.info(f"Placed market order to close position: {trade.order.orderId}")
            return trade

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None

    @with_connection_check
    async def get_account_values(self) -> Dict[str, float]:
        """Get account values"""
        try:
            account_values = {}
            for value in self.ib.accountSummary():
                if value.tag in ["NetLiquidation", "AvailableFunds", "ExcessLiquidity"]:
                    account_values[value.tag] = float(value.value)

            return account_values

        except Exception as e:
            logger.error(f"Error getting account values: {e}")
            return {}

    @with_connection_check
    async def get_open_positions(self) -> List[PortfolioItem]:
        """Get open positions"""
        try:
            return self.ib.portfolio()
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    @with_connection_check
    async def get_open_orders(self) -> List[Trade]:
        """Get open orders"""
        try:
            return self.ib.openTrades()
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    def is_market_hours(self) -> bool:
        """Check if currently in market hours"""
        et_tz = pytz.timezone("US/Eastern")
        now_et = datetime.now(et_tz)

        # Skip weekends
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check time range (9:30 AM to 4:00 PM ET)
        market_open = time(
            config.market_hours.market_open_hour, config.market_hours.market_open_minute
        )
        market_close = time(
            config.market_hours.market_close_hour, config.market_hours.market_close_minute
        )

        current_time = now_et.time()
        return market_open <= current_time <= market_close

    def get_today_expiry_string(self) -> str:
        """Get today's date in IB expiry format (YYYYMMDD)"""
        return date.today().strftime("%Y%m%d")

    def is_connected(self) -> bool:
        """Check if connected to IB"""
        if hasattr(self, "connection_manager") and self.connection_manager:
            return self.connection_manager.state == ConnectionState.CONNECTED
        return self.connected

    def get_connection_status(self) -> dict:
        """Get detailed connection status"""
        if hasattr(self, "connection_manager") and self.connection_manager:
            return self.connection_manager.get_connection_info()
        return {
            "state": "CONNECTED" if self.connected else "DISCONNECTED",
            "is_healthy": self.connected,
        }

    @with_connection_check
    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    async def get_option_chain_data(
        self, underlying_symbol: str, expiry: str, num_strikes: int = 10
    ) -> Optional[Dict[str, Dict]]:
        """Get option chain data for put/call ratio and gamma calculations"""
        try:
            # Get current underlying price
            underlying_contract = await self.get_mes_contract()
            underlying_price = await self.get_current_price(underlying_contract)
            if not underlying_price:
                return None

            # Round to nearest strike
            center_strike = self._round_to_strike(underlying_price)

            # Get strikes around ATM
            strikes = []
            strike_increment = 5.0  # MES strike increment
            for i in range(-num_strikes // 2, num_strikes // 2 + 1):
                strike = center_strike + (i * strike_increment)
                strikes.append(strike)

            chain_data = {
                "underlying_price": underlying_price,
                "strikes": {},
                "total_call_volume": 0,
                "total_put_volume": 0,
                "total_call_oi": 0,
                "total_put_oi": 0,
                "net_gamma": 0,
            }

            # Collect data for each strike
            for strike in strikes:
                strike_data = {"strike": strike}

                # Get call data
                try:
                    call_contract = await self.get_mes_option_contract(expiry, strike, "C")
                    call_data = await self.get_market_data(call_contract)
                    if call_data:
                        strike_data["call_bid"] = call_data["bid"]
                        strike_data["call_ask"] = call_data["ask"]
                        strike_data["call_volume"] = call_data["volume"]
                        strike_data["call_iv"] = call_data["implied_volatility"] or 0.2
                        chain_data["total_call_volume"] += call_data["volume"]
                except Exception as e:
                    logger.debug(f"Error getting call data for strike {strike}: {e}")

                # Get put data
                try:
                    put_contract = await self.get_mes_option_contract(expiry, strike, "P")
                    put_data = await self.get_market_data(put_contract)
                    if put_data:
                        strike_data["put_bid"] = put_data["bid"]
                        strike_data["put_ask"] = put_data["ask"]
                        strike_data["put_volume"] = put_data["volume"]
                        strike_data["put_iv"] = put_data["implied_volatility"] or 0.2
                        chain_data["total_put_volume"] += put_data["volume"]
                except Exception as e:
                    logger.debug(f"Error getting put data for strike {strike}: {e}")

                # Estimate gamma contribution (simplified)
                if "call_iv" in strike_data and "put_iv" in strike_data:
                    moneyness = (underlying_price - strike) / underlying_price
                    # Simple gamma approximation for ATM options
                    if abs(moneyness) < 0.02:  # Near ATM
                        gamma_weight = 1.0
                    else:
                        gamma_weight = max(0, 1 - abs(moneyness) * 20)

                    # Net gamma (calls positive, puts negative)
                    call_gamma = gamma_weight * strike_data.get("call_volume", 0)
                    put_gamma = -gamma_weight * strike_data.get("put_volume", 0)
                    chain_data["net_gamma"] += call_gamma + put_gamma

                chain_data["strikes"][strike] = strike_data

            # Calculate put/call ratio
            if chain_data["total_call_volume"] > 0:
                chain_data["put_call_ratio"] = (
                    chain_data["total_put_volume"] / chain_data["total_call_volume"]
                )
            else:
                chain_data["put_call_ratio"] = 1.0

            logger.info(
                f"Collected option chain data: {len(strikes)} strikes, "
                f"P/C ratio: {chain_data['put_call_ratio']:.2f}"
            )
            return chain_data

        except Exception as e:
            logger.error(f"Error getting option chain data: {e}")
            return None
