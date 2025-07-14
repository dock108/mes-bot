"""
UAT tests for dashboard UI components and navigation
"""

import pytest
from playwright.sync_api import Page, expect

from tests.uat.helpers.dashboard_pages import DashboardPage


class TestDashboardUI:
    """Test dashboard UI components and basic functionality"""

    def test_dashboard_loads_successfully(self, dashboard_with_data):
        """Test that dashboard loads and displays main components"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Check that sidebar is visible
        expect(dashboard.sidebar).to_be_visible()

        # Check that main content area is visible
        expect(dashboard.main_content).to_be_visible()

        # Check that we have navigation options
        nav_options = dashboard.get_sidebar_options()
        assert len(nav_options) > 0, "No navigation options found"

        # Verify essential sections are present
        expected_sections = ["Live Monitor", "Performance", "Backtest", "Configuration"]
        for section in expected_sections:
            assert any(section in option for option in nav_options), f"Missing section: {section}"

    def test_navigation_between_sections(self, dashboard_with_data):
        """Test navigation between different dashboard sections"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        sections_to_test = ["Live Monitor", "Performance", "Backtest", "Configuration"]

        for section in sections_to_test:
            # Navigate to section
            dashboard.navigate_to_section(section)

            # Verify section is selected
            current_section = dashboard.get_current_section()
            assert (
                section in current_section or current_section in section
            ), f"Failed to navigate to {section}. Current: {current_section}"

            # Wait a moment for content to load
            page.wait_for_timeout(1000)

            # Verify main content area has updated
            expect(dashboard.main_content).to_be_visible()

    def test_dashboard_metrics_display(self, dashboard_with_data):
        """Test that dashboard displays key metrics"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Should start on Live Monitor by default or navigate there
        dashboard.navigate_to_section("Live Monitor")

        # Get displayed metrics
        metrics = dashboard.get_metrics_values()

        # Should have some metrics displayed
        assert len(metrics) > 0, "No metrics found on dashboard"

        # Check for expected metric types (values may vary)
        expected_metrics = ["Total P&L", "Today's P&L", "Open Positions", "Win Rate"]
        found_metrics = list(metrics.keys())

        # At least some expected metrics should be present
        metrics_found = any(
            any(expected in found for found in found_metrics) for expected in expected_metrics
        )
        assert metrics_found, f"No expected metrics found. Found: {found_metrics}"

    def test_emergency_stop_button(self, dashboard_with_data):
        """Test emergency stop button visibility and functionality"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor where emergency stop should be
        dashboard.navigate_to_section("Live Monitor")

        # Check if emergency stop button is visible
        has_emergency_stop = dashboard.check_emergency_stop_visible()
        assert has_emergency_stop, "Emergency stop button should be visible"

        # Test clicking emergency stop
        dashboard.click_emergency_stop()
        page.wait_for_timeout(1000)

        # Should show confirmation dialog
        confirmation = page.locator('text="EMERGENCY STOP CONFIRMATION"')
        expect(confirmation).to_be_visible()

        # Should have cancel option
        cancel_button = page.locator('button:has-text("Cancel")')
        expect(cancel_button).to_be_visible()

        # Take screenshot of confirmation dialog
        dashboard.take_screenshot("emergency_stop_confirmation_dialog")

        # Cancel the emergency stop
        cancel_button.click()
        page.wait_for_timeout(1000)

        # Confirmation should be gone
        expect(confirmation).not_to_be_visible()

    def test_responsive_layout(self, dashboard_with_data):
        """Test that dashboard layout is responsive"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Test different viewport sizes
        viewports = [
            {"width": 1920, "height": 1080},  # Desktop
            {"width": 1366, "height": 768},  # Laptop
            {"width": 768, "height": 1024},  # Tablet
        ]

        for viewport in viewports:
            page.set_viewport_size(viewport)
            page.wait_for_timeout(1000)  # Wait for layout adjustment

            # Check that key elements are still visible
            expect(dashboard.sidebar).to_be_visible()
            expect(dashboard.main_content).to_be_visible()

            # Navigation should still work
            dashboard.navigate_to_section("Performance")
            expect(dashboard.main_content).to_be_visible()

    @pytest.mark.slow
    def test_data_refresh_functionality(self, dashboard_with_data):
        """Test that dashboard data refreshes appropriately"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Get initial metrics
        initial_metrics = dashboard.get_metrics_values()

        # Wait for potential refresh (Streamlit auto-refresh)
        page.wait_for_timeout(5000)

        # Get metrics again
        updated_metrics = dashboard.get_metrics_values()

        # Metrics should still be displayed (values may or may not change)
        assert len(updated_metrics) > 0, "Metrics disappeared after refresh"

        # Keys should remain consistent
        assert set(initial_metrics.keys()) == set(
            updated_metrics.keys()
        ), "Metric structure changed unexpectedly"

    def test_error_handling_display(self, dashboard_with_data):
        """Test that dashboard handles and displays errors gracefully"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate through all sections to check for error messages
        sections = ["Live Monitor", "Performance", "Backtest", "Configuration"]

        for section in sections:
            dashboard.navigate_to_section(section)
            page.wait_for_timeout(2000)  # Wait for content to load

            # Check for error indicators
            error_elements = page.locator("text=/Error|Exception|Failed/")
            if error_elements.count() > 0:
                # Take screenshot for debugging
                dashboard.take_screenshot(f"error_in_{section.lower().replace(' ', '_')}")

                # Get error text for reporting
                error_text = error_elements.first.text_content()
                print(f"Error found in {section}: {error_text}")

    def test_loading_states(self, page_with_server):
        """Test that loading states are handled properly"""
        page = page_with_server
        dashboard = DashboardPage(page)

        # Check initial loading
        dashboard.wait_for_load()

        # Navigate to different sections and check loading
        sections = ["Performance", "Backtest"]

        for section in sections:
            dashboard.navigate_to_section(section)

            # Give time for any loading indicators
            page.wait_for_timeout(1000)

            # Main content should be visible after loading
            expect(dashboard.main_content).to_be_visible()

            # Should not have infinite loading spinners
            loading_spinners = page.locator('[data-testid="stSpinner"]')
            # Allow some loading time but shouldn't be stuck
            page.wait_for_timeout(5000)

            # Content should load eventually
            expect(dashboard.main_content).to_be_visible()

    def test_market_status_display(self, dashboard_with_data):
        """Test market status bar and metrics display"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Check for market status metrics
        mes_price = page.locator('text="MES Price"')
        iv_rv = page.locator('text="IV / RV60"')
        time_to_close = page.locator('text="Time to Close"')

        # All market status items should be visible
        expect(mes_price).to_be_visible()
        expect(iv_rv).to_be_visible()
        expect(time_to_close).to_be_visible()

        # Take screenshot of market status
        dashboard.take_screenshot("market_status_display")

    def test_trading_mode_selector(self, dashboard_with_data):
        """Test trading mode radio button selector"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Live Monitor
        dashboard.navigate_to_section("Live Monitor")

        # Check for trading mode selector
        trading_mode_label = page.locator('text="Trading Mode"')
        expect(trading_mode_label).to_be_visible()

        # Should have radio buttons for modes
        auto_radio = page.locator('label:has-text("Auto")')
        manual_radio = page.locator('label:has-text("Manual")')
        off_radio = page.locator('label:has-text("Off")')

        expect(auto_radio).to_be_visible()
        expect(manual_radio).to_be_visible()
        expect(off_radio).to_be_visible()

        # One should be selected
        checked_radio = page.locator('input[type="radio"]:checked')
        assert checked_radio.count() > 0, "One trading mode should be selected"

        dashboard.take_screenshot("trading_mode_selector")
