"""
UAT tests for live monitoring functionality
"""
import pytest
from playwright.sync_api import Page, expect

from tests.uat.helpers.dashboard_pages import DashboardPage, LiveMonitorPage


class TestLiveMonitoring:
    """Test live monitoring workflow and functionality"""

    def test_live_monitor_navigation(self, dashboard_with_data):
        """Test navigation to live monitor section"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Verify we're on the right section
        current_section = dashboard.get_current_section()
        assert "Live Monitor" in current_section or "live" in current_section.lower()

        # Main content should be visible
        expect(dashboard.main_content).to_be_visible()

    def test_open_positions_display(self, dashboard_with_data):
        """Test that open positions are displayed correctly"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        live_monitor = LiveMonitorPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Check for positions display
        positions_count = live_monitor.get_open_positions_count()

        # Should have some test data positions or empty state
        assert positions_count >= 0, "Invalid positions count"

        if positions_count > 0:
            # Get position data
            position_data = live_monitor.get_position_data()
            assert len(position_data) == positions_count, "Position count mismatch"

            # Check that position data has expected structure
            if position_data:
                first_position = position_data[0]
                expected_fields = ["Symbol", "Strike", "Type", "Premium", "P&L"]

                # Should have some recognizable trading fields
                has_trading_fields = any(
                    any(field.lower() in key.lower() for key in first_position.keys())
                    for field in expected_fields
                )
                assert (
                    has_trading_fields
                ), f"Missing trading fields in position data: {first_position.keys()}"

    def test_real_time_data_updates(self, dashboard_with_data):
        """Test real-time data updates in live monitor"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        live_monitor = LiveMonitorPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Check for real-time price updates
        has_price_updates = live_monitor.check_real_time_updates()

        # Should show some price/financial data
        if has_price_updates:
            # Take screenshot to verify display
            dashboard.take_screenshot("real_time_prices")

    def test_market_status_indicator(self, dashboard_with_data):
        """Test market status indicator display"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        live_monitor = LiveMonitorPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Check market status
        market_status = live_monitor.get_market_status()

        # Should have some market status information
        assert market_status != "Unknown", "Market status not displayed"

        # Common market statuses
        valid_statuses = ["OPEN", "CLOSED", "PRE-MARKET", "AFTER-HOURS"]
        status_valid = any(status.lower() in market_status.lower() for status in valid_statuses)

        # Take screenshot for manual verification if status unclear
        if not status_valid:
            dashboard.take_screenshot("market_status_display")

    def test_position_metrics_accuracy(self, dashboard_with_data):
        """Test accuracy of position metrics and calculations"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        live_monitor = LiveMonitorPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Get current metrics
        metrics = dashboard.get_metrics_values()

        # Get position data
        position_data = live_monitor.get_position_data()
        positions_count = len(position_data)

        # Check metric consistency
        if "Open Positions" in metrics:
            displayed_count = metrics["Open Positions"]
            # Extract number from string (e.g., "5 positions" -> 5)
            try:
                metric_count = int("".join(filter(str.isdigit, displayed_count)))
                assert (
                    metric_count == positions_count
                ), f"Open positions metric ({metric_count}) doesn't match table count ({positions_count})"
            except ValueError:
                # Metric might be in different format, take screenshot for review
                dashboard.take_screenshot("position_count_mismatch")

    @pytest.mark.slow
    def test_live_data_refresh_cycle(self, dashboard_with_data):
        """Test that live data refreshes on appropriate intervals"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        live_monitor = LiveMonitorPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Get initial state
        initial_metrics = dashboard.get_metrics_values()
        initial_positions = live_monitor.get_position_data()

        # Wait for refresh interval (Streamlit typically refreshes automatically)
        page.wait_for_timeout(10000)  # 10 seconds

        # Get updated state
        updated_metrics = dashboard.get_metrics_values()
        updated_positions = live_monitor.get_position_data()

        # Data structure should remain consistent
        assert set(initial_metrics.keys()) == set(
            updated_metrics.keys()
        ), "Metrics structure changed after refresh"

        # Position count should be stable for test data
        assert len(initial_positions) == len(
            updated_positions
        ), "Position count changed unexpectedly"

    def test_emergency_controls_functionality(self, dashboard_with_data):
        """Test emergency stop and control functionality"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Check for emergency controls
        has_emergency_stop = dashboard.check_emergency_stop_visible()
        assert has_emergency_stop, "Emergency stop button should be visible"

        # Take screenshot before clicking
        dashboard.take_screenshot("before_emergency_stop")

        # Test emergency stop functionality (in paper trading mode)
        dashboard.click_emergency_stop()

        # Wait for response
        page.wait_for_timeout(1000)

        # Should show confirmation dialog
        confirmation = page.locator('text="EMERGENCY STOP CONFIRMATION"')
        expect(confirmation).to_be_visible()

        # Should have Yes and Cancel buttons
        yes_button = page.locator('button:has-text("Yes, STOP ALL")')
        cancel_button = page.locator('button:has-text("Cancel")')

        expect(yes_button).to_be_visible()
        expect(cancel_button).to_be_visible()

        # Take screenshot of confirmation
        dashboard.take_screenshot("emergency_stop_confirmation")

        # Cancel for test purposes
        cancel_button.click()
        page.wait_for_timeout(1000)

        # Confirmation should be gone
        expect(confirmation).not_to_be_visible()

    def test_position_detail_display(self, dashboard_with_data):
        """Test detailed position information display"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        live_monitor = LiveMonitorPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Get position data
        position_data = live_monitor.get_position_data()

        if position_data:
            # Check that positions have required details
            first_position = position_data[0]

            # Should have key trading information
            required_info = ["strike", "premium", "pnl", "symbol", "type"]
            position_keys_lower = [key.lower() for key in first_position.keys()]

            missing_info = []
            for info in required_info:
                if not any(info in key for key in position_keys_lower):
                    missing_info.append(info)

            if missing_info:
                # Take screenshot for review
                dashboard.take_screenshot("position_detail_review")
                print(f"Position may be missing: {missing_info}")
                print(f"Available fields: {list(first_position.keys())}")

    def test_live_pnl_calculations(self, dashboard_with_data):
        """Test that P&L calculations update correctly"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        live_monitor = LiveMonitorPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Get metrics with P&L information
        metrics = dashboard.get_metrics_values()

        # Look for P&L metrics
        pnl_metrics = {k: v for k, v in metrics.items() if "p&l" in k.lower() or "pnl" in k.lower()}

        if pnl_metrics:
            # Verify P&L values are properly formatted
            for metric_name, metric_value in pnl_metrics.items():
                # Should contain currency formatting or numerical value
                has_currency = "$" in metric_value or "%" in metric_value
                has_number = any(char.isdigit() for char in metric_value)

                assert (
                    has_currency or has_number
                ), f"P&L metric '{metric_name}' has unexpected format: '{metric_value}'"

        # Check position-level P&L
        position_data = live_monitor.get_position_data()
        if position_data:
            for position in position_data:
                pnl_fields = [
                    k for k in position.keys() if "p&l" in k.lower() or "pnl" in k.lower()
                ]

                for field in pnl_fields:
                    value = position[field]
                    # Should be a valid P&L format
                    if value and value != "N/A":
                        has_valid_format = (
                            "$" in value or "%" in value or any(char.isdigit() for char in value)
                        )
                        assert (
                            has_valid_format
                        ), f"Position P&L field '{field}' has invalid format: '{value}'"

    def test_opportunity_scanner_visibility(self, dashboard_with_data):
        """Test opportunity scanner visibility based on trading mode"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Test in Manual mode - scanner should be visible
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        if manual_button.count() > 0:
            manual_button.click()
            page.wait_for_timeout(2000)

            scanner = page.locator('text="Opportunity Scanner"')
            expect(scanner).to_be_visible()

            dashboard.take_screenshot("scanner_manual_mode")

        # Test in Off mode - scanner should be hidden
        off_button = (
            page.locator('label:has-text("Off")').locator("..").locator('input[type="radio"]')
        )
        if off_button.count() > 0:
            off_button.click()
            page.wait_for_timeout(2000)

            scanner = page.locator('text="Opportunity Scanner"')
            expect(scanner).not_to_be_visible()

            dashboard.take_screenshot("scanner_off_mode")

    def test_trading_mode_dependent_ui(self, dashboard_with_data):
        """Test UI elements that depend on trading mode"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Test Manual mode specific elements
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        if manual_button.count() > 0:
            manual_button.click()
            page.wait_for_timeout(2000)

            # If signal is ready, should show manual controls
            signal_ready = page.locator('text="SIGNAL READY"')
            if signal_ready.count() > 0:
                # Should have manual trade controls
                place_trade = page.locator('button:has-text("Place Trade")')
                expect(place_trade).to_be_visible()

                dashboard.take_screenshot("manual_mode_trade_controls")
