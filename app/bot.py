"""
Main trading bot engine for MES 0DTE Lotto-Grid Options Bot
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime, time, timedelta
from typing import Optional, Dict
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import config
from app.models import create_database, SystemLog, get_session_maker
from app.ib_client import IBClient
from app.risk_manager import RiskManager
from app.strategy import LottoGridStrategy

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{config.logging.log_dir}/{config.logging.bot_log_file}"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self):
        return (datetime.utcnow() - self.last_failure_time).total_seconds() > self.recovery_timeout

    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class LottoGridBot:
    """Main trading bot orchestrator"""

    def __init__(self):
        self.ib_client = IBClient()
        self.risk_manager = RiskManager(config.database.url)
        self.strategy = LottoGridStrategy(self.ib_client, self.risk_manager, config.database.url)
        self.scheduler = AsyncIOScheduler()
        self.session_maker = get_session_maker(config.database.url)

        self.running = False
        self.trading_halted = False
        self.daily_initialized = False

        # Error handling and recovery
        self.circuit_breaker = CircuitBreaker()
        self.error_count = 0
        self.last_health_check = datetime.utcnow()
        self.health_check_interval = 60  # seconds
        self.max_errors_per_hour = 10
        self.error_timestamps = []

        # Performance monitoring
        self.performance_metrics = {
            "start_time": datetime.utcnow(),
            "trades_count": 0,
            "errors_count": 0,
            "uptime_seconds": 0,
            "memory_usage_mb": 0,
        }

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop())

    async def start(self):
        """Start the trading bot"""
        try:
            logger.info("Starting MES 0DTE Lotto-Grid Options Bot...")

            # Validate configuration
            config.validate()
            logger.info("Configuration validated")

            # Create database
            create_database(config.database.url)
            logger.info("Database initialized")

            # Connect to IB
            if not await self.ib_client.connect():
                raise RuntimeError("Failed to connect to Interactive Brokers")

            # Verify market data permissions
            await self._verify_market_data()

            # Set up scheduled tasks
            self._setup_scheduler()

            # Start scheduler
            self.scheduler.start()
            logger.info("Scheduler started")

            self.running = True
            self._log_system_event("BOT_START", "Trading bot started successfully")

            # Main event loop
            await self._main_loop()

        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            self._log_system_event("BOT_ERROR", f"Bot startup failed: {e}")
            await self.stop()

    async def stop(self):
        """Stop the trading bot gracefully"""
        if not self.running:
            return

        logger.info("Stopping trading bot...")
        self.running = False

        try:
            # Flatten all positions if in trading hours
            if self.ib_client.is_market_hours():
                await self.strategy.flatten_all_positions()

            # Stop scheduler
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)

            # Disconnect from IB
            await self.ib_client.disconnect()

            # Final risk metrics update
            self.risk_manager.update_daily_summary()

            self._log_system_event("BOT_STOP", "Trading bot stopped gracefully")
            logger.info("Trading bot stopped")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def _setup_scheduler(self):
        """Set up scheduled tasks"""
        # Market hours in ET
        et_tz = pytz.timezone("US/Eastern")

        # Daily initialization at 9:35 AM ET
        self.scheduler.add_job(
            self._daily_initialization,
            CronTrigger(hour=9, minute=35, timezone=et_tz),
            id="daily_init",
            replace_existing=True,
        )

        # Trading cycle every 5 minutes during market hours (9:30 AM - 4:00 PM ET)
        self.scheduler.add_job(
            self._trading_cycle,
            CronTrigger(
                hour="9-15", minute="*/5", day_of_week="0-4", timezone=et_tz  # Monday-Friday
            ),
            id="trading_cycle",
            replace_existing=True,
        )

        # Position updates every minute during market hours
        self.scheduler.add_job(
            self._update_positions,
            CronTrigger(hour="9-15", minute="*", day_of_week="0-4", timezone=et_tz),
            id="position_updates",
            replace_existing=True,
        )

        # Flatten all positions at 15:58 ET
        self.scheduler.add_job(
            self._end_of_day_flatten,
            CronTrigger(hour=15, minute=58, timezone=et_tz),
            id="eod_flatten",
            replace_existing=True,
        )

        # Daily summary at market close
        self.scheduler.add_job(
            self._daily_summary,
            CronTrigger(hour=16, minute=5, timezone=et_tz),
            id="daily_summary",
            replace_existing=True,
        )

        # Risk monitoring every 30 seconds during trading hours
        self.scheduler.add_job(
            self._risk_monitoring,
            "interval",
            seconds=30,
            id="risk_monitoring",
            replace_existing=True,
        )

    async def _main_loop(self):
        """Main event loop"""
        while self.running:
            try:
                # Check if we need to initialize for the day
                if self.ib_client.is_market_hours() and not self.daily_initialized:
                    await self._daily_initialization()

                # Sleep for a bit to avoid busy waiting
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    async def _daily_initialization(self):
        """Initialize strategy for the trading day"""
        try:
            if self.daily_initialized:
                logger.debug("Daily initialization already completed")
                return

            logger.info("Starting daily initialization...")

            if not self.ib_client.is_market_hours():
                logger.info("Outside market hours, skipping daily initialization")
                return

            # Initialize strategy
            success = await self.strategy.initialize_daily_session()

            if success:
                self.daily_initialized = True
                self.trading_halted = False
                self._log_system_event("DAILY_INIT", "Daily initialization completed")
                logger.info("Daily initialization completed successfully")
            else:
                logger.error("Daily initialization failed")
                self._log_system_event("DAILY_INIT_FAILED", "Daily initialization failed")

        except Exception as e:
            logger.error(f"Error in daily initialization: {e}")
            self._log_system_event("DAILY_INIT_ERROR", f"Daily initialization error: {e}")

    async def _trading_cycle(self):
        """Execute trading cycle"""
        try:
            if not self.daily_initialized or self.trading_halted:
                return

            if not self.ib_client.is_market_hours():
                return

            logger.debug("Executing trading cycle...")

            # Execute strategy
            trade_placed = await self.strategy.execute_trading_cycle()

            if trade_placed:
                logger.info("New strangle trade placed")
                self._log_system_event("TRADE_PLACED", "New strangle trade executed")

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self._log_system_event("TRADING_CYCLE_ERROR", f"Trading cycle error: {e}")

    async def _update_positions(self):
        """Update open positions P&L"""
        try:
            if not self.daily_initialized:
                return

            await self.strategy.update_open_positions()

        except Exception as e:
            logger.error(f"Error updating positions: {e}")

    async def _risk_monitoring(self):
        """Monitor risk metrics and halt trading if necessary"""
        try:
            if not self.daily_initialized or not self.ib_client.is_market_hours():
                return

            # Get current account equity
            account_values = await self.ib_client.get_account_values()
            current_equity = account_values.get("NetLiquidation", 0)

            # Check if trading should be halted
            should_halt, reason = self.risk_manager.should_halt_trading(current_equity)

            if should_halt and not self.trading_halted:
                self.trading_halted = True
                self._log_system_event("TRADING_HALTED", f"Trading halted: {reason}")
                logger.warning(f"TRADING HALTED: {reason}")

                # Optionally flatten all positions immediately
                if "extreme" in reason.lower():
                    await self.strategy.flatten_all_positions()

            # Log risk metrics periodically
            risk_metrics = self.risk_manager.get_risk_metrics_summary()
            logger.debug(f"Risk metrics: {risk_metrics}")

        except Exception as e:
            logger.error(f"Error in risk monitoring: {e}")

    async def _end_of_day_flatten(self):
        """Flatten all positions at end of day"""
        try:
            logger.info("Executing end-of-day flatten...")

            success = await self.strategy.flatten_all_positions()

            if success:
                self._log_system_event("EOD_FLATTEN", "End-of-day flatten completed")
                logger.info("End-of-day flatten completed")
            else:
                self._log_system_event("EOD_FLATTEN_FAILED", "End-of-day flatten failed")
                logger.error("End-of-day flatten failed")

        except Exception as e:
            logger.error(f"Error in end-of-day flatten: {e}")
            self._log_system_event("EOD_FLATTEN_ERROR", f"End-of-day flatten error: {e}")

    async def _daily_summary(self):
        """Generate daily summary and reset for next day"""
        try:
            logger.info("Generating daily summary...")

            # Update daily summary in database
            self.risk_manager.update_daily_summary()

            # Reset daily flags
            self.daily_initialized = False
            self.trading_halted = False

            # Log strategy status
            status = self.strategy.get_strategy_status()
            logger.info(f"Daily summary completed. Strategy status: {status}")

            self._log_system_event(
                "DAILY_SUMMARY", "Daily summary generated and reset for next day"
            )

        except Exception as e:
            logger.error(f"Error in daily summary: {e}")
            self._log_system_event("DAILY_SUMMARY_ERROR", f"Daily summary error: {e}")

    async def _verify_market_data(self):
        """Verify market data permissions"""
        try:
            logger.info("Verifying market data permissions...")

            # Try to get MES contract and price
            mes_contract = await self.ib_client.get_mes_contract()
            price = await self.ib_client.get_current_price(mes_contract)

            if price:
                logger.info(f"Market data verified. Current MES price: ${price:.2f}")
            else:
                logger.warning("Could not get MES price - check market data permissions")

        except Exception as e:
            logger.error(f"Market data verification failed: {e}")
            raise

    def _log_system_event(
        self, event_type: str, message: str, details: Optional[str] = None, level: str = "INFO"
    ):
        """Log system event to database with error tracking"""
        try:
            session = self.session_maker()
            log_entry = SystemLog(
                level=level,
                module="bot_engine",
                message=f"[{event_type}] {message}",
                details=details,
            )
            session.add(log_entry)
            session.commit()
            session.close()

            # Track error rates
            if level in ["ERROR", "CRITICAL"]:
                self._track_error()

        except Exception as e:
            logger.error(f"Failed to log system event: {e}")

    def _track_error(self):
        """Track error occurrence for rate limiting"""
        now = datetime.utcnow()
        self.error_timestamps.append(now)
        self.error_count += 1
        self.performance_metrics["errors_count"] += 1

        # Clean old error timestamps (older than 1 hour)
        hour_ago = now - timedelta(hours=1)
        self.error_timestamps = [ts for ts in self.error_timestamps if ts > hour_ago]

        # Check if error rate is too high
        if len(self.error_timestamps) > self.max_errors_per_hour:
            self._handle_high_error_rate()

    def _handle_high_error_rate(self):
        """Handle high error rate situation"""
        logger.critical(
            f"High error rate detected: {len(self.error_timestamps)} errors in last hour"
        )
        self.trading_halted = True

    def get_health_status(self) -> Dict:
        """Get comprehensive health status"""
        now = datetime.utcnow()
        uptime = (now - self.performance_metrics["start_time"]).total_seconds()

        # Get memory usage (with fallback if psutil not available)
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            memory_mb = 0  # Fallback if psutil not available

        # Update performance metrics
        self.performance_metrics["uptime_seconds"] = uptime
        self.performance_metrics["memory_usage_mb"] = memory_mb

        health_status = {
            "status": "healthy" if self.running and not self.trading_halted else "unhealthy",
            "uptime_hours": uptime / 3600,
            "memory_usage_mb": memory_mb,
            "error_rate_per_hour": len(self.error_timestamps),
            "circuit_breaker_state": self.circuit_breaker.state,
            "trading_halted": self.trading_halted,
            "last_health_check": self.last_health_check.isoformat(),
            "performance_metrics": self.performance_metrics.copy(),
        }

        # Add component health checks
        try:
            # Check database connectivity
            session = self.session_maker()
            session.execute("SELECT 1")
            session.close()
            health_status["database_status"] = "healthy"
        except Exception as e:
            health_status["database_status"] = f"unhealthy: {str(e)}"

        # Check IB connection
        health_status["ib_connection_status"] = (
            "healthy"
            if hasattr(self.ib_client, "is_connected") and self.ib_client.is_connected()
            else "unknown"
        )

        return health_status

    def get_status(self) -> dict:
        """Get current bot status for monitoring"""
        return {
            "running": self.running,
            "connected_to_ib": self.ib_client.connected,
            "daily_initialized": self.daily_initialized,
            "trading_halted": self.trading_halted,
            "in_market_hours": self.ib_client.is_market_hours(),
            "strategy_status": self.strategy.get_strategy_status(),
            "risk_metrics": self.risk_manager.get_risk_metrics_summary(),
        }


async def main():
    """Main entry point"""
    bot = LottoGridBot()

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
