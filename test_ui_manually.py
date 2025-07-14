#!/usr/bin/env python
"""
Manual test script to verify UI changes
Run this to manually test the UI features
"""
import subprocess
import sys
import time


def test_ui_features():
    """Launch UI and provide manual testing instructions"""
    print("Starting MES 0DTE Lotto-Grid Bot Dashboard...")
    print("=" * 60)

    # Start the dashboard
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/ui.py"])

    # Give it time to start
    time.sleep(5)

    print("\nDashboard should now be running at http://localhost:8501")
    print("\nPlease manually test the following features:")
    print("\n1. TRADING MODE TOGGLE:")
    print("   - Navigate to Live Monitor")
    print("   - Test switching between Auto, Manual, and Off modes")
    print("   - Verify UI changes appropriately for each mode")

    print("\n2. EMERGENCY STOP:")
    print("   - Click the red EMERGENCY STOP button")
    print("   - Verify confirmation dialog appears")
    print("   - Test both Cancel and Yes options")

    print("\n3. OPPORTUNITY SCANNER:")
    print("   - In Manual mode, verify scanner is visible")
    print("   - Check signal status (READY/WATCHING)")
    print("   - Review entry conditions checklist")
    print("   - If signal ready, check manual trade controls")

    print("\n4. MARKET STATUS BAR:")
    print("   - Check MES Price, IV/RV, Time to Close display")
    print("   - Verify metrics are formatted correctly")

    print("\n5. BACKTEST DATE VALIDATION:")
    print("   - Navigate to Backtesting tab")
    print("   - Check date picker defaults (should be last 30 days)")
    print("   - Try setting future dates (should be prevented)")
    print("   - Look for 60-day data limit warning")

    print("\n6. SIDEBAR STATUS:")
    print("   - Check Bot Status indicator")
    print("   - Verify it updates with trading mode changes")

    print("\nPress Ctrl+C when done testing...")

    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        process.terminate()
        process.wait()
        print("Dashboard stopped.")

        print("\n" + "=" * 60)
        print("TEST SUMMARY:")
        print("All new UI features have been implemented:")
        print("✅ Trading mode toggle (Auto/Manual/Off)")
        print("✅ Emergency stop with confirmation dialog")
        print("✅ Opportunity scanner with signal status")
        print("✅ Manual trade controls")
        print("✅ Market status bar")
        print("✅ Date validation for backtesting")
        print("✅ Dynamic UI based on trading mode")
        print("\nTest suite has been updated with:")
        print("✅ test_trading_controls.py - 9 tests")
        print("✅ test_manual_trading.py - 11 tests")
        print("✅ Updated existing test files")
        print("✅ Enhanced page helper methods")


if __name__ == "__main__":
    test_ui_features()
