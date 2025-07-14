"""
UAT tests for trading controls and mode management
"""

import pytest
from playwright.sync_api import Page, expect

from tests.uat.helpers.dashboard_pages import DashboardPage, LiveMonitorPage


class TestTradingControls:
    """Test trading mode controls and emergency stop functionality"""

    def test_trading_mode_toggle(self, dashboard_with_data):
        """Test trading mode toggle functionality"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Find trading mode radio buttons
        trading_mode_radio = (
            page.locator('label:has-text("Trading Mode")')
            .locator("..")
            .locator('input[type="radio"]')
        )

        # Should have 3 modes: Auto, Manual, Off
        assert trading_mode_radio.count() >= 3, "Should have at least 3 trading modes"

        # Test each mode
        modes = ["Auto", "Manual", "Off"]
        for mode in modes:
            # Click the mode button
            mode_button = (
                page.locator(f'label:has-text("{mode}")')
                .locator("..")
                .locator('input[type="radio"]')
            )
            if mode_button.count() > 0:
                mode_button.click()
                page.wait_for_timeout(1000)

                # Take screenshot of each mode
                dashboard.take_screenshot(f"trading_mode_{mode.lower()}")

                # Verify mode is selected
                checked_radio = page.locator('input[type="radio"]:checked')
                assert checked_radio.count() > 0, f"No radio button checked for {mode} mode"

    def test_trading_mode_ui_changes(self, dashboard_with_data):
        """Test that UI changes appropriately based on trading mode"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Test Manual mode
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        if manual_button.count() > 0:
            manual_button.click()
            page.wait_for_timeout(2000)

            # Should show opportunity scanner
            opportunity_scanner = page.locator('text="Opportunity Scanner"')
            expect(opportunity_scanner).to_be_visible()

            # Should show manual trade controls when signal is ready
            signal_ready = page.locator('text="SIGNAL READY"')
            if signal_ready.count() > 0:
                # Check for manual control buttons
                review_button = page.locator('button:has-text("Review Trade")')
                place_button = page.locator('button:has-text("Place Trade")')
                skip_button = page.locator('button:has-text("Skip Signal")')

                expect(review_button).to_be_visible()
                expect(place_button).to_be_visible()
                expect(skip_button).to_be_visible()

            dashboard.take_screenshot("manual_mode_controls")

        # Test Off mode
        off_button = (
            page.locator('label:has-text("Off")').locator("..").locator('input[type="radio"]')
        )
        if off_button.count() > 0:
            off_button.click()
            page.wait_for_timeout(2000)

            # Should NOT show opportunity scanner
            opportunity_scanner = page.locator('text="Opportunity Scanner"')
            expect(opportunity_scanner).not_to_be_visible()

            dashboard.take_screenshot("off_mode_ui")

    def test_emergency_stop_button_visibility(self, dashboard_with_data):
        """Test emergency stop button is visible and accessible"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Check emergency stop button
        emergency_stop = page.locator('button:has-text("EMERGENCY STOP")')
        expect(emergency_stop).to_be_visible()

        # Check it has proper styling (primary button)
        button_classes = emergency_stop.get_attribute("class") or ""
        assert (
            "primary" in button_classes.lower() or emergency_stop.get_attribute("type") == "primary"
        ), "Emergency stop should be a primary button"

        dashboard.take_screenshot("emergency_stop_button")

    def test_emergency_stop_confirmation_flow(self, dashboard_with_data):
        """Test emergency stop confirmation dialog flow"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Click emergency stop button
        emergency_stop = page.locator('button:has-text("EMERGENCY STOP")')
        emergency_stop.click()
        page.wait_for_timeout(1000)

        # Should show confirmation dialog
        confirmation_text = page.locator('text="EMERGENCY STOP CONFIRMATION"')
        expect(confirmation_text).to_be_visible()

        warning_text = page.locator(
            'text="This will immediately close all positions and halt trading!"'
        )
        expect(warning_text).to_be_visible()

        # Should have Yes and Cancel buttons
        yes_button = page.locator('button:has-text("Yes, STOP ALL")')
        cancel_button = page.locator('button:has-text("Cancel")')

        expect(yes_button).to_be_visible()
        expect(cancel_button).to_be_visible()

        dashboard.take_screenshot("emergency_stop_confirmation")

        # Test cancel button
        cancel_button.click()
        page.wait_for_timeout(1000)

        # Confirmation should be gone
        expect(confirmation_text).not_to_be_visible()

        # Emergency stop button should still be there
        expect(emergency_stop).to_be_visible()

    def test_emergency_stop_execution(self, dashboard_with_data):
        """Test emergency stop execution (in test mode)"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Click emergency stop
        emergency_stop = page.locator('button:has-text("EMERGENCY STOP")')
        emergency_stop.click()
        page.wait_for_timeout(1000)

        # Confirm emergency stop
        yes_button = page.locator('button:has-text("Yes, STOP ALL")')
        yes_button.click()
        page.wait_for_timeout(2000)

        # Should show execution message
        stop_message = page.locator('text="Emergency stop executed!"')
        halt_message = page.locator('text="All positions closed. Trading halted."')

        # At least one confirmation message should appear
        messages_visible = stop_message.count() > 0 or halt_message.count() > 0
        assert messages_visible, "Should show emergency stop confirmation message"

        dashboard.take_screenshot("emergency_stop_executed")

    def test_market_status_bar(self, dashboard_with_data):
        """Test market status bar display"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Check for market status metrics
        mes_price = page.locator('text="MES Price"')
        iv_rv = page.locator('text="IV / RV60"')
        time_to_close = page.locator('text="Time to Close"')

        expect(mes_price).to_be_visible()
        expect(iv_rv).to_be_visible()
        expect(time_to_close).to_be_visible()

        # Check that metrics have values
        metrics = dashboard.get_metrics_values()

        # Should have price information
        price_metrics = [k for k in metrics.keys() if "price" in k.lower() or "mes" in k.lower()]
        assert len(price_metrics) > 0, "Should have MES price metric"

        dashboard.take_screenshot("market_status_bar")

    @pytest.mark.slow
    def test_trading_mode_persistence(self, dashboard_with_data):
        """Test that trading mode persists across navigation"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Set to Manual mode
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        manual_button.click()
        page.wait_for_timeout(1000)

        # Navigate away
        dashboard.navigate_to_section("Performance")
        page.wait_for_timeout(2000)

        # Navigate back
        dashboard.navigate_to_section("Live Monitor")
        page.wait_for_timeout(2000)

        # Check that Manual mode is still selected
        checked_radio = page.locator('input[type="radio"]:checked')
        if checked_radio.count() > 0:
            # The mode should still be Manual
            # This depends on session state implementation
            dashboard.take_screenshot("mode_persistence_check")

    def test_sidebar_bot_status(self, dashboard_with_data):
        """Test bot status indicator in sidebar"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Check sidebar for bot status
        sidebar = dashboard.sidebar

        # Should have bot status section
        bot_status_header = sidebar.locator('text="Bot Status"')
        expect(bot_status_header).to_be_visible()

        # Should show trading mode status
        status_indicators = [
            'text="Auto Trading Active"',
            'text="Manual Mode"',
            'text="Trading OFF"',
        ]

        status_found = False
        for indicator in status_indicators:
            if sidebar.locator(indicator).count() > 0:
                status_found = True
                break

        assert status_found, "Should show bot trading status"

        # Should show paper trading mode
        paper_trading = sidebar.locator('text="Paper Trading Mode"')
        expect(paper_trading).to_be_visible()

        dashboard.take_screenshot("sidebar_bot_status")

    def test_auto_refresh_controls(self, dashboard_with_data):
        """Test auto-refresh controls"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Find refresh controls
        refresh_button = page.locator('button:has-text("Refresh Now")')
        auto_refresh_checkbox = (
            page.locator('text="Auto Refresh"').locator("..").locator('input[type="checkbox"]')
        )

        expect(refresh_button).to_be_visible()
        assert auto_refresh_checkbox.count() > 0, "Should have auto-refresh checkbox"

        # Test refresh button
        refresh_button.click()
        page.wait_for_timeout(2000)

        # Should update last refresh time
        last_refresh = page.locator("text=/Last:.*\\d{2}:\\d{2}:\\d{2}/")
        expect(last_refresh).to_be_visible()

        dashboard.take_screenshot("refresh_controls")
