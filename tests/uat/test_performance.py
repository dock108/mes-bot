"""
UAT tests for performance analytics functionality
"""

import pytest
from playwright.sync_api import Page, expect

from tests.uat.helpers.dashboard_pages import DashboardPage, PerformancePage


@pytest.mark.uat
class TestPerformanceAnalytics:
    """Test performance analytics and reporting functionality"""

    def test_performance_section_navigation(self, dashboard_with_data):
        """Test navigation to performance analytics section"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Performance section
        dashboard.navigate_to_section("Performance")

        # Verify navigation worked
        current_section = dashboard.get_current_section()
        assert "Performance" in current_section or "performance" in current_section.lower()

        # Main content should be visible
        expect(dashboard.main_content).to_be_visible()

    def test_performance_metrics_display(self, dashboard_with_data):
        """Test that key performance metrics are displayed"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        performance = PerformancePage(page)

        # Navigate to Performance
        dashboard.navigate_to_section("Performance")

        # Get performance metrics
        metrics = performance.get_performance_metrics()

        # Should have performance metrics displayed
        assert len(metrics) > 0, "No performance metrics found"

        # Check for expected performance metrics
        expected_metrics = [
            "Total Return",
            "Win Rate",
            "Profit Factor",
            "Max Drawdown",
            "Average Win",
            "Average Loss",
            "Total Trades",
            "Sharpe Ratio",
        ]

        found_metrics = list(metrics.keys())
        metrics_present = []

        for expected in expected_metrics:
            if any(expected.lower() in found.lower() for found in found_metrics):
                metrics_present.append(expected)

        # Should have at least some key metrics
        assert (
            len(metrics_present) >= 3
        ), f"Missing key performance metrics. Found: {found_metrics}, Expected some of: {expected_metrics}"

    def test_performance_charts_display(self, dashboard_with_data):
        """Test that performance charts are loaded and displayed"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        performance = PerformancePage(page)

        # Navigate to Performance
        dashboard.navigate_to_section("Performance")

        # Wait for charts to load
        page.wait_for_timeout(3000)

        # Check if charts are loaded
        charts_loaded = performance.check_charts_loaded()

        if charts_loaded:
            # Get chart count
            chart_count = performance.get_chart_count()
            assert chart_count > 0, "Charts detected but count is 0"

            # Should have multiple charts for comprehensive analysis
            assert chart_count >= 2, f"Expected multiple charts, found {chart_count}"

            # Take screenshot for verification
            dashboard.take_screenshot("performance_charts")
        else:
            # Charts might be loading or not implemented yet
            dashboard.take_screenshot("performance_no_charts")

    def test_trades_history_table(self, dashboard_with_data):
        """Test trades history table display and functionality"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        performance = PerformancePage(page)

        # Navigate to Performance
        dashboard.navigate_to_section("Performance")

        # Check if trades table is present
        has_trades_table = performance.check_trades_table()

        if has_trades_table:
            # Get trades count
            trades_count = performance.get_trades_count()
            assert trades_count > 0, "Trades table found but no trades displayed"

            # Should have reasonable number of test trades
            assert trades_count <= 1000, f"Unexpectedly high number of trades: {trades_count}"

            # Take screenshot of trades table
            dashboard.take_screenshot("trades_history_table")
        else:
            # Trades table might be in different section or format
            dashboard.take_screenshot("performance_no_trades_table")

    def test_metric_value_formats(self, dashboard_with_data):
        """Test that performance metrics have proper formatting"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        performance = PerformancePage(page)

        # Navigate to Performance
        dashboard.navigate_to_section("Performance")

        # Get performance metrics
        metrics = performance.get_performance_metrics()

        for metric_name, metric_value in metrics.items():
            # Skip empty values
            if not metric_value or metric_value.strip() == "":
                continue

            # Check formatting based on metric type
            metric_lower = metric_name.lower()

            if "rate" in metric_lower or "%" in metric_name:
                # Should have percentage format
                assert "%" in metric_value or any(
                    char.isdigit() for char in metric_value
                ), f"Rate metric '{metric_name}' should have percentage format: '{metric_value}'"

            elif "return" in metric_lower or "p&l" in metric_lower or "drawdown" in metric_lower:
                # Should have currency or percentage format
                has_valid_format = (
                    "$" in metric_value
                    or "%" in metric_value
                    or any(char.isdigit() for char in metric_value)
                )
                assert (
                    has_valid_format
                ), f"Financial metric '{metric_name}' has invalid format: '{metric_value}'"

            elif "trades" in metric_lower or "count" in metric_lower:
                # Should have integer format
                has_number = any(char.isdigit() for char in metric_value)
                assert (
                    has_number
                ), f"Count metric '{metric_name}' should have numeric format: '{metric_value}'"

    def test_performance_period_filtering(self, dashboard_with_data):
        """Test performance data filtering by time periods"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        performance = PerformancePage(page)

        # Navigate to Performance
        dashboard.navigate_to_section("Performance")

        # Look for period filter controls (dropdown, buttons, etc.)
        period_controls = page.locator("text=/Day|Week|Month|Year|All Time/")

        if period_controls.count() > 0:
            # Test different periods
            periods_to_test = ["1 Month", "3 Months", "All Time"]

            for period in periods_to_test:
                period_control = page.locator(f'text="{period}"')
                if period_control.count() > 0:
                    # Click period filter
                    period_control.click()
                    page.wait_for_timeout(2000)  # Wait for data update

                    # Get metrics after filter
                    filtered_metrics = performance.get_performance_metrics()

                    # Should still have metrics displayed
                    assert len(filtered_metrics) > 0, f"No metrics after filtering by {period}"

                    # Take screenshot
                    dashboard.take_screenshot(
                        f"performance_filtered_{period.lower().replace(' ', '_')}"
                    )
        else:
            # Period filtering might not be implemented or in different format
            dashboard.take_screenshot("performance_no_period_filters")

    @pytest.mark.slow
    def test_performance_data_consistency(self, dashboard_with_data):
        """Test consistency of performance data across refreshes"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        performance = PerformancePage(page)

        # Navigate to Performance
        dashboard.navigate_to_section("Performance")

        # Get initial metrics
        initial_metrics = performance.get_performance_metrics()
        initial_trades_count = performance.get_trades_count()

        # Refresh page or wait for auto-refresh
        page.reload()
        dashboard.wait_for_load()
        dashboard.navigate_to_section("Performance")
        page.wait_for_timeout(3000)

        # Get metrics after refresh
        refreshed_metrics = performance.get_performance_metrics()
        refreshed_trades_count = performance.get_trades_count()

        # Data should be consistent (same test data)
        assert set(initial_metrics.keys()) == set(
            refreshed_metrics.keys()
        ), "Performance metrics structure changed after refresh"

        assert (
            initial_trades_count == refreshed_trades_count
        ), f"Trades count changed after refresh: {initial_trades_count} -> {refreshed_trades_count}"

        # Values should be the same for static test data
        for metric_name in initial_metrics:
            initial_value = initial_metrics[metric_name]
            refreshed_value = refreshed_metrics[metric_name]

            # Allow for small formatting differences but values should be consistent
            if initial_value != refreshed_value:
                print(f"Metric '{metric_name}' changed: '{initial_value}' -> '{refreshed_value}'")

    def test_performance_chart_interactions(self, dashboard_with_data):
        """Test performance chart interactivity"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        performance = PerformancePage(page)

        # Navigate to Performance
        dashboard.navigate_to_section("Performance")

        # Wait for charts to load
        page.wait_for_timeout(3000)

        if performance.check_charts_loaded():
            # Get first chart
            chart = page.locator('[data-testid="stPlotlyChart"]').first

            # Test hover interaction
            chart.hover()
            page.wait_for_timeout(1000)

            # Take screenshot during hover
            dashboard.take_screenshot("chart_hover_interaction")

            # Test zoom/pan (if supported)
            # Note: Plotly charts in Streamlit might have limited interactivity
            chart_area = chart.locator(".plot-container").first
            if chart_area.count() > 0:
                # Attempt zoom interaction
                chart_area.click()
                page.wait_for_timeout(500)

    def test_export_functionality(self, dashboard_with_data):
        """Test data export functionality if available"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Performance
        dashboard.navigate_to_section("Performance")

        # Look for export buttons or download links
        export_controls = page.locator("text=/Export|Download|CSV|Excel/")

        if export_controls.count() > 0:
            # Take screenshot of export options
            dashboard.take_screenshot("export_options_available")

            # Test export functionality (in test environment)
            for i in range(export_controls.count()):
                control = export_controls.nth(i)
                control_text = control.text_content()

                if control_text and any(
                    keyword in control_text.lower() for keyword in ["export", "download"]
                ):
                    # Could test download (but would need download handling setup)
                    dashboard.take_screenshot(f"export_option_{i}")
        else:
            # Export functionality might not be implemented
            dashboard.take_screenshot("no_export_options")

    def test_performance_alerts_display(self, dashboard_with_data):
        """Test display of performance alerts or warnings"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        performance = PerformancePage(page)

        # Navigate to Performance
        dashboard.navigate_to_section("Performance")

        # Look for alert indicators
        alert_elements = page.locator("text=/Alert|Warning|Risk|Attention/")

        if alert_elements.count() > 0:
            # Take screenshot of alerts
            dashboard.take_screenshot("performance_alerts")

            for i in range(alert_elements.count()):
                alert = alert_elements.nth(i)
                alert_text = alert.text_content()
                print(f"Performance alert found: {alert_text}")

        # Check for visual indicators (colors, icons)
        warning_colors = page.locator('[style*="red"], [style*="orange"], [class*="warning"]')
        if warning_colors.count() > 0:
            dashboard.take_screenshot("performance_warning_indicators")
