#!/usr/bin/env python3
"""
Live test script for IB connection resilience
This script simulates connection issues and verifies recovery
"""

import asyncio
import logging
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append("..")

from app.circuit_breaker import CircuitOpenError  # noqa: E402
from app.ib_client import IBClient  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_connection_resilience():
    """Test connection resilience features"""

    ib_client = IBClient()

    logger.info("=" * 80)
    logger.info("Starting IB Connection Resilience Test")
    logger.info("=" * 80)

    # Test 1: Initial connection
    logger.info("\nTest 1: Initial Connection")
    logger.info("-" * 40)

    try:
        connected = await ib_client.connect()
        if connected:
            logger.info("✅ Successfully connected to IB")
            status = ib_client.get_connection_status()
            logger.info(f"Connection status: {status}")
        else:
            logger.error("❌ Failed to connect to IB")
            return
    except Exception as e:
        logger.error(f"❌ Connection error: {e}")
        return

    # Test 2: Basic operations with connection check
    logger.info("\nTest 2: Basic Operations with Connection Check")
    logger.info("-" * 40)

    try:
        # Get MES contract
        contract = await ib_client.get_mes_contract()
        logger.info(f"✅ Retrieved MES contract: {contract.localSymbol}")

        # Get current price
        price = await ib_client.get_current_price(contract)
        if price:
            logger.info(f"✅ Current MES price: ${price:.2f}")
        else:
            logger.warning("⚠️  Could not get current price")

        # Get account values
        account_values = await ib_client.get_account_values()
        if account_values:
            logger.info(f"✅ Account values retrieved: {list(account_values.keys())}")
        else:
            logger.warning("⚠️  Could not get account values")

    except ConnectionError as e:
        logger.error(f"❌ Connection error during operations: {e}")
    except CircuitOpenError as e:
        logger.error(f"❌ Circuit breaker open: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")

    # Test 3: Connection status monitoring
    logger.info("\nTest 3: Connection Status Monitoring")
    logger.info("-" * 40)

    # Monitor connection for 30 seconds
    start_time = datetime.now()
    monitor_duration = 30  # seconds

    logger.info(f"Monitoring connection for {monitor_duration} seconds...")
    logger.info("(You can test disconnection by stopping IB Gateway/TWS)")

    while (datetime.now() - start_time).total_seconds() < monitor_duration:
        try:
            status = ib_client.get_connection_status()
            logger.info(
                f"Connection state: {status.get('state', 'UNKNOWN')} | "
                f"Healthy: {status.get('is_healthy', False)} | "
                f"Quality: {status.get('connection_quality', 'N/A')} | "
                f"Latency: {status.get('average_latency_ms', 0):.1f}ms"
            )

            # Try a simple operation to test circuit breaker
            if ib_client.is_connected():
                await ib_client.get_account_values()

        except ConnectionError:
            logger.warning("⚠️  Connection lost - waiting for recovery...")
        except CircuitOpenError as e:
            logger.warning(f"⚠️  Circuit breaker open: {e}")
        except Exception as e:
            logger.error(f"❌ Error during monitoring: {e}")

        await asyncio.sleep(5)

    # Test 4: Cleanup
    logger.info("\nTest 4: Cleanup")
    logger.info("-" * 40)

    try:
        await ib_client.disconnect()
        logger.info("✅ Disconnected successfully")
    except Exception as e:
        logger.error(f"❌ Error during disconnect: {e}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)

    if hasattr(ib_client, "connection_manager") and ib_client.connection_manager:
        final_status = ib_client.get_connection_status()
        logger.info(f"Total connections: {final_status.get('total_connections', 0)}")
        logger.info(f"Total disconnections: {final_status.get('total_disconnections', 0)}")
        logger.info(f"Uptime percentage: {final_status.get('uptime_percentage', 0):.1f}%")

    logger.info("\nTest completed!")


async def test_circuit_breaker_behavior():
    """Test circuit breaker behavior"""

    logger.info("\n" + "=" * 80)
    logger.info("Testing Circuit Breaker Behavior")
    logger.info("=" * 80)

    ib_client = IBClient()

    # Connect first
    await ib_client.connect()

    # Simulate failures by requesting invalid contract
    logger.info("\nSimulating failures to trigger circuit breaker...")

    for i in range(10):
        try:
            # Try to get price for invalid contract
            invalid_contract = await ib_client.get_mes_contract("999999")  # Invalid expiry
            await ib_client.get_current_price(invalid_contract)
        except Exception as e:
            logger.info(f"Attempt {i+1}: {type(e).__name__}: {e}")

        await asyncio.sleep(1)

    await ib_client.disconnect()


async def main():
    """Main test runner"""

    # Run connection resilience test
    await test_connection_resilience()

    # Optional: Run circuit breaker test
    # await test_circuit_breaker_behavior()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"\nUnhandled exception: {e}")
        raise
