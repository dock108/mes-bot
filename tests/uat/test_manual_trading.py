"""
UAT tests for manual trading mode and opportunity scanner
"""

import pytest
from playwright.sync_api import Page, expect

from tests.uat.helpers.dashboard_pages import DashboardPage, LiveMonitorPage


class TestManualTradingMode:
    """Test manual trading mode functionality and opportunity scanner"""

    def test_opportunity_scanner_display(self, dashboard_with_data):
        """Test opportunity scanner display in manual/auto modes"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Set to Manual mode
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        manual_button.click()
        page.wait_for_timeout(2000)

        # Check opportunity scanner is visible
        scanner_header = page.locator('h3:has-text("Opportunity Scanner")')
        expect(scanner_header).to_be_visible()

        # Check for signal status
        signal_status = page.locator("text=/SIGNAL READY|WATCHING.../")
        expect(signal_status).to_be_visible()

        dashboard.take_screenshot("opportunity_scanner_manual_mode")

        # Switch to Auto mode - scanner should still be visible
        auto_button = (
            page.locator('label:has-text("Auto")').locator("..").locator('input[type="radio"]')
        )
        auto_button.click()
        page.wait_for_timeout(2000)

        expect(scanner_header).to_be_visible()
        dashboard.take_screenshot("opportunity_scanner_auto_mode")

        # Switch to Off mode - scanner should be hidden
        off_button = (
            page.locator('label:has-text("Off")').locator("..").locator('input[type="radio"]')
        )
        off_button.click()
        page.wait_for_timeout(2000)

        expect(scanner_header).not_to_be_visible()
        dashboard.take_screenshot("opportunity_scanner_off_mode")

    def test_signal_status_indicators(self, dashboard_with_data):
        """Test signal status display"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor in Manual mode
        dashboard.navigate_to_section("Live Monitor")
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        manual_button.click()
        page.wait_for_timeout(2000)

        # Check for signal status indicators
        signal_ready = page.locator('text="SIGNAL READY"')
        signal_watching = page.locator('text="WATCHING..."')

        # Should have one of the status indicators
        has_status = signal_ready.count() > 0 or signal_watching.count() > 0
        assert has_status, "Should display signal status"

        # Check for visual indicators (emojis)
        green_dot = page.locator("text=/🟢/")
        yellow_dot = page.locator("text=/🟡/")

        has_indicator = green_dot.count() > 0 or yellow_dot.count() > 0
        assert has_indicator, "Should have visual status indicator"

        dashboard.take_screenshot("signal_status_indicators")

    def test_entry_conditions_checklist(self, dashboard_with_data):
        """Test entry conditions display"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor in Manual mode
        dashboard.navigate_to_section("Live Monitor")
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        manual_button.click()
        page.wait_for_timeout(2000)

        # Check for entry conditions
        conditions_header = page.locator('text="Entry Conditions:"')
        expect(conditions_header).to_be_visible()

        # Check for specific conditions
        expected_conditions = [
            "RV < 67% of IV",
            "Time between trades OK",
            "Risk limits OK",
            "Premium in range",
        ]

        for condition in expected_conditions:
            condition_elem = page.locator(f'text="{condition}"')
            expect(condition_elem).to_be_visible()

            # Check for checkmark or X
            parent = condition_elem.locator("..")
            checkmark = parent.locator("text=/✅/")
            x_mark = parent.locator("text=/❌/")

            has_indicator = checkmark.count() > 0 or x_mark.count() > 0
            assert has_indicator, f"Condition '{condition}' should have status indicator"

        dashboard.take_screenshot("entry_conditions_checklist")

    def test_suggested_trade_display(self, dashboard_with_data):
        """Test suggested trade information display"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor in Manual mode
        dashboard.navigate_to_section("Live Monitor")
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        manual_button.click()
        page.wait_for_timeout(2000)

        # If signal is ready, check suggested trade
        signal_ready = page.locator('text="SIGNAL READY"')
        if signal_ready.count() > 0:
            # Check for suggested trade header
            suggested_trade = page.locator('text="Suggested Trade:"')
            expect(suggested_trade).to_be_visible()

            # Check for strike information
            call_strike = page.locator("text=/Call Strike/")
            put_strike = page.locator("text=/Put Strike/")

            expect(call_strike).to_be_visible()
            expect(put_strike).to_be_visible()

            # Check for premium information
            total_premium = page.locator("text=/Total Premium/")
            target = page.locator("text=/Target/")

            expect(total_premium).to_be_visible()
            expect(target).to_be_visible()

            # Check formatting includes prices
            price_pattern = page.locator("text=/\\$\\d+/")
            assert price_pattern.count() > 0, "Should display prices in suggested trade"

            dashboard.take_screenshot("suggested_trade_display")

    def test_manual_trade_controls(self, dashboard_with_data):
        """Test manual trade control buttons"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor in Manual mode
        dashboard.navigate_to_section("Live Monitor")
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        manual_button.click()
        page.wait_for_timeout(2000)

        # Check if signal is ready
        signal_ready = page.locator('text="SIGNAL READY"')
        if signal_ready.count() > 0:
            # Should show manual controls
            controls_header = page.locator('text="Manual Trade Controls:"')
            expect(controls_header).to_be_visible()

            # Check for control buttons
            review_button = page.locator('button:has-text("Review Trade")')
            place_button = page.locator('button:has-text("Place Trade")')
            skip_button = page.locator('button:has-text("Skip Signal")')

            expect(review_button).to_be_visible()
            expect(place_button).to_be_visible()
            expect(skip_button).to_be_visible()

            # Place Trade should be primary button
            place_button_type = place_button.get_attribute("type")
            assert place_button_type == "primary", "Place Trade should be primary button"

            dashboard.take_screenshot("manual_trade_controls")

    def test_manual_controls_not_in_auto_mode(self, dashboard_with_data):
        """Test that manual controls are hidden in auto mode"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor in Auto mode
        dashboard.navigate_to_section("Live Monitor")
        auto_button = (
            page.locator('label:has-text("Auto")').locator("..").locator('input[type="radio"]')
        )
        auto_button.click()
        page.wait_for_timeout(2000)

        # Manual controls should not be visible
        controls_header = page.locator('text="Manual Trade Controls:"')
        expect(controls_header).not_to_be_visible()

        # Control buttons should not be visible
        review_button = page.locator('button:has-text("Review Trade")')
        place_button = page.locator('button:has-text("Place Trade")')
        skip_button = page.locator('button:has-text("Skip Signal")')

        expect(review_button).not_to_be_visible()
        expect(place_button).not_to_be_visible()
        expect(skip_button).not_to_be_visible()

        dashboard.take_screenshot("auto_mode_no_manual_controls")

    def test_review_trade_functionality(self, dashboard_with_data):
        """Test review trade button functionality"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor in Manual mode
        dashboard.navigate_to_section("Live Monitor")
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        manual_button.click()
        page.wait_for_timeout(2000)

        # If signal is ready, test review button
        signal_ready = page.locator('text="SIGNAL READY"')
        if signal_ready.count() > 0:
            review_button = page.locator('button:has-text("Review Trade")')
            review_button.click()
            page.wait_for_timeout(1000)

            # Should show some review information
            # This might be an info box or expanded details
            info_box = page.locator('[role="alert"]').filter(has_text="review")
            if info_box.count() > 0:
                dashboard.take_screenshot("trade_review_info")

    def test_place_trade_functionality(self, dashboard_with_data):
        """Test place trade button functionality"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor in Manual mode
        dashboard.navigate_to_section("Live Monitor")
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        manual_button.click()
        page.wait_for_timeout(2000)

        # If signal is ready, test place button
        signal_ready = page.locator('text="SIGNAL READY"')
        if signal_ready.count() > 0:
            place_button = page.locator('button:has-text("Place Trade")')
            place_button.click()
            page.wait_for_timeout(1000)

            # Should show success message
            success_box = page.locator('[role="alert"]').filter(has_text="Trade")
            if success_box.count() > 0:
                dashboard.take_screenshot("trade_placed_confirmation")

    def test_skip_signal_functionality(self, dashboard_with_data):
        """Test skip signal button functionality"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor in Manual mode
        dashboard.navigate_to_section("Live Monitor")
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        manual_button.click()
        page.wait_for_timeout(2000)

        # If signal is ready, test skip button
        signal_ready = page.locator('text="SIGNAL READY"')
        if signal_ready.count() > 0:
            skip_button = page.locator('button:has-text("Skip Signal")')
            skip_button.click()
            page.wait_for_timeout(1000)

            # Should show skip confirmation
            skip_info = page.locator('[role="alert"]').filter(has_text="skipped")
            if skip_info.count() > 0:
                dashboard.take_screenshot("signal_skipped_confirmation")

    def test_strike_information_format(self, dashboard_with_data):
        """Test strike price information formatting"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor in Manual mode
        dashboard.navigate_to_section("Live Monitor")
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        manual_button.click()
        page.wait_for_timeout(2000)

        # Check if signal is ready
        signal_ready = page.locator('text="SIGNAL READY"')
        if signal_ready.count() > 0:
            # Look for strike price patterns
            call_pattern = page.locator("text=/\\$\\d+[,.]?\\d*\\s*@\\s*\\$\\d+[,.]?\\d*/")
            put_pattern = page.locator("text=/\\$\\d+[,.]?\\d*\\s*@\\s*\\$\\d+[,.]?\\d*/")

            has_strike_info = call_pattern.count() > 0 or put_pattern.count() > 0
            assert has_strike_info, "Should display strike prices with premium"

            # Check for multiplier information (4x)
            multiplier = page.locator("text=/\\(4x\\)|4x/")
            if multiplier.count() > 0:
                dashboard.take_screenshot("strike_multiplier_display")

    @pytest.mark.slow
    def test_opportunity_scanner_updates(self, dashboard_with_data):
        """Test that opportunity scanner updates appropriately"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor in Manual mode
        dashboard.navigate_to_section("Live Monitor")
        manual_button = (
            page.locator('label:has-text("Manual")').locator("..").locator('input[type="radio"]')
        )
        manual_button.click()
        page.wait_for_timeout(2000)

        # Get initial state
        initial_status = page.locator("text=/SIGNAL READY|WATCHING.../").text_content()

        # Get initial entry conditions
        initial_conditions = []
        for elem in page.locator("text=/✅|❌/").all():
            initial_conditions.append(elem.text_content())

        # Wait for potential update
        page.wait_for_timeout(5000)

        # Scanner should still be functional
        scanner_header = page.locator('h3:has-text("Opportunity Scanner")')
        expect(scanner_header).to_be_visible()

        # Status should still be displayed
        current_status = page.locator("text=/SIGNAL READY|WATCHING.../")
        expect(current_status).to_be_visible()

        dashboard.take_screenshot("opportunity_scanner_after_wait")
