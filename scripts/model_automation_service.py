#!/usr/bin/env python
"""
Model Automation Service

This script runs the automated model training and management system as a service.
It handles scheduled retraining, performance monitoring, and model deployment.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import config  # noqa: E402
from app.model_automation import ModelAutomationEngine  # noqa: E402
from app.model_versioning import ModelVersionManager  # noqa: E402
from app.notification_service import (  # noqa: E402
    NotificationLevel,
    send_system_alert,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/model_automation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class ModelAutomationService:
    """
    Service wrapper for the model automation engine
    """

    def __init__(self):
        self.automation_engine = ModelAutomationEngine(config.database.url)
        self.version_manager = ModelVersionManager(config.database.url)
        self.running = False
        self.shutdown_event = asyncio.Event()

    async def start(self):
        """Start the automation service"""
        try:
            logger.info("Starting Model Automation Service...")

            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Start the automation engine
            await self.automation_engine.start_automation()

            self.running = True

            await send_system_alert(
                "Model Automation Service Started",
                "The automated model training and management service is now running",
                NotificationLevel.INFO,
                context={
                    "service_version": "1.0.0",
                    "database_url": config.database.url,
                    "automation_enabled": True,
                },
            )

            logger.info("Model Automation Service started successfully")

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except Exception as e:
            logger.error(f"Error starting automation service: {e}")
            await send_system_alert(
                "Model Automation Service Error",
                f"Failed to start automation service: {str(e)}",
                NotificationLevel.ERROR,
                context={"error": str(e)},
            )
            raise

    async def stop(self):
        """Stop the automation service"""
        try:
            logger.info("Stopping Model Automation Service...")
            self.running = False

            # Stop the automation engine
            await self.automation_engine.stop_automation()

            await send_system_alert(
                "Model Automation Service Stopped",
                "The automated model training and management service has been stopped",
                NotificationLevel.INFO,
            )

            logger.info("Model Automation Service stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping automation service: {e}")

        finally:
            self.shutdown_event.set()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop())

    async def get_status(self) -> dict:
        """Get current status of the automation service"""
        try:
            automation_status = self.automation_engine.get_automation_status()

            # Get model version information
            model_versions = {}
            for model_type in ["entry", "exit", "strike"]:
                versions = self.version_manager.list_model_versions(model_type, limit=5)
                model_versions[model_type] = {
                    "total_versions": len(versions),
                    "production_version": next(
                        (v["version"] for v in versions if v["is_production"]), None
                    ),
                    "latest_version": versions[0]["version"] if versions else None,
                }

            return {
                "service_running": self.running,
                "automation_engine": automation_status,
                "model_versions": model_versions,
                "timestamp": asyncio.get_event_loop().time(),
            }

        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {"error": str(e)}

    async def manual_operations(self):
        """Provide manual operation interface"""
        while self.running:
            try:
                print("\n" + "=" * 50)
                print("Model Automation Service - Manual Operations")
                print("=" * 50)
                print("1. Trigger manual retraining of all models")
                print("2. Show model version status")
                print("3. Deploy specific model version")
                print("4. Rollback model to previous version")
                print("5. Show automation status")
                print("6. Exit manual operations")
                print("=" * 50)

                choice = input("Enter your choice (1-6): ").strip()

                if choice == "1":
                    await self._manual_retrain_all()
                elif choice == "2":
                    await self._show_model_versions()
                elif choice == "3":
                    await self._deploy_model_version()
                elif choice == "4":
                    await self._rollback_model()
                elif choice == "5":
                    await self._show_automation_status()
                elif choice == "6":
                    break
                else:
                    print("Invalid choice. Please try again.")

                input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in manual operations: {e}")
                print(f"Error: {e}")

    async def _manual_retrain_all(self):
        """Manually trigger retraining of all models"""
        print("\nTriggering manual retraining of all models...")
        results = await self.automation_engine.manual_retrain_all_models()

        print("Retraining Results:")
        for model_type, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {model_type}: {status}")

    async def _show_model_versions(self):
        """Show model version information"""
        print("\nModel Version Status:")
        print("-" * 60)

        for model_type in ["entry", "exit", "strike"]:
            print(f"\n{model_type.upper()} MODELS:")
            versions = self.version_manager.list_model_versions(model_type, limit=10)

            if not versions:
                print("  No versions found")
                continue

            for v in versions:
                status_icon = "🚀" if v["is_production"] else "📋"
                print(
                    f"  {status_icon} {v['version']} ({v['algorithm']}) - "
                    f"Score: {v['performance_score']:.3f} - "
                    f"Created: {v['created_at'][:19]}"
                )

    async def _deploy_model_version(self):
        """Deploy a specific model version to production"""
        model_type = input("Enter model type (entry/exit/strike): ").strip().lower()
        if model_type not in ["entry", "exit", "strike"]:
            print("Invalid model type")
            return

        version = input("Enter version to deploy: ").strip()
        if not version:
            print("Version cannot be empty")
            return

        print(f"Deploying {model_type} model version {version} to production...")
        success = self.version_manager.deploy_model_to_production(model_type, version)

        if success:
            print("✅ Deployment successful")
        else:
            print("❌ Deployment failed")

    async def _rollback_model(self):
        """Rollback model to previous version"""
        model_type = input("Enter model type to rollback (entry/exit/strike): ").strip().lower()
        if model_type not in ["entry", "exit", "strike"]:
            print("Invalid model type")
            return

        print(f"Rolling back {model_type} model to previous version...")
        success = self.version_manager.rollback_model(model_type)

        if success:
            print("✅ Rollback successful")
        else:
            print("❌ Rollback failed")

    async def _show_automation_status(self):
        """Show current automation status"""
        status = await self.get_status()

        print("\nAutomation Service Status:")
        print("-" * 40)
        print(f"Service Running: {'✅' if status['service_running'] else '❌'}")

        if "automation_engine" in status:
            engine_status = status["automation_engine"]
            print(f"Automation Engine: {'✅' if engine_status['running'] else '❌'}")
            print(f"Active Tasks: {engine_status['active_tasks']}")

            print("\nLast Retraining:")
            for model_type, timestamp in engine_status["last_retraining"].items():
                if timestamp:
                    print(f"  {model_type}: {timestamp[:19]}")
                else:
                    print(f"  {model_type}: Never")


async def main():
    """Main service entry point"""
    try:
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)

        service = ModelAutomationService()

        # Check if running in interactive mode
        if len(sys.argv) > 1 and sys.argv[1] == "--manual":
            # Manual operations mode
            await service.start()
            await service.manual_operations()
            await service.stop()
        else:
            # Service mode - run continuously
            await service.start()

    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
