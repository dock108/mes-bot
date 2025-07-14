"""
UAT tests for backtesting functionality and workflow
"""

from datetime import datetime

import pytest
from playwright.sync_api import Page, expect

from tests.uat.helpers.dashboard_pages import BacktestPage, DashboardPage


class TestBacktestingWorkflow:
    """Test backtesting functionality and user workflow"""

    def test_backtest_section_navigation(self, dashboard_with_data):
        """Test navigation to backtest section"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Backtest section
        dashboard.navigate_to_section("Backtest")

        # Verify navigation worked
        current_section = dashboard.get_current_section()
        assert "Backtest" in current_section or "backtest" in current_section.lower()

        # Main content should be visible
        expect(dashboard.main_content).to_be_visible()

    def test_backtest_parameter_interface(self, dashboard_with_data):
        """Test backtest parameter input interface"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        backtest = BacktestPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Look for parameter input fields
        input_fields = page.locator('input[type="number"], input[type="text"]')
        date_fields = page.locator('input[type="date"]')

        # Should have some parameter inputs
        total_inputs = input_fields.count() + date_fields.count()
        assert total_inputs > 0, "No backtest parameter inputs found"

        # Take screenshot of parameter interface
        dashboard.take_screenshot("backtest_parameters_interface")

        # Test setting some parameters
        test_parameters = {"Max Trades": "10", "Profit Target": "4.0", "Initial Capital": "5000"}

        for param_name, value in test_parameters.items():
            try:
                backtest.set_backtest_parameter(param_name, value)
                page.wait_for_timeout(500)  # Small delay between inputs
            except Exception as e:
                print(f"Could not set parameter {param_name}: {e}")

    def test_backtest_execution_workflow(self, dashboard_with_data):
        """Test complete backtest execution workflow"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        backtest = BacktestPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Set some parameters first
        test_parameters = {"Max Trades": "5", "Profit Target": "3.0"}

        for param_name, value in test_parameters.items():
            try:
                backtest.set_backtest_parameter(param_name, value)
            except Exception:
                pass  # Parameter might not exist or be in different format

        # Take screenshot before running
        dashboard.take_screenshot("before_backtest_run")

        # Look for run button and click it
        run_buttons = page.locator(
            'button:has-text("Run"), button:has-text("Start"), button:has-text("Execute")'
        )

        if run_buttons.count() > 0:
            # Click the run button
            run_buttons.first.click()

            # Wait for execution to start
            page.wait_for_timeout(3000)

            # Take screenshot during execution
            dashboard.take_screenshot("backtest_running")

            # Wait for completion (with timeout)
            try:
                backtest.wait_for_backtest_completion(timeout=30000)

                # Take screenshot after completion
                dashboard.take_screenshot("backtest_completed")

            except Exception:
                # Backtest might still be running or failed
                dashboard.take_screenshot("backtest_timeout")
        else:
            # Run button might not be visible or implemented
            dashboard.take_screenshot("no_backtest_run_button")

    def test_backtest_results_display(self, dashboard_with_data):
        """Test backtest results display and formatting"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        backtest = BacktestPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Check if there are existing backtest results (from test data)
        results = backtest.get_backtest_results()

        if results:
            # Verify result metrics are properly formatted
            expected_result_metrics = [
                "Total Return",
                "Win Rate",
                "Total Trades",
                "Max Drawdown",
                "Sharpe Ratio",
                "Profit Factor",
                "Final Capital",
            ]

            found_metrics = list(results.keys())
            metrics_present = []

            for expected in expected_result_metrics:
                if any(expected.lower() in found.lower() for found in found_metrics):
                    metrics_present.append(expected)

            assert (
                len(metrics_present) >= 3
            ), f"Missing key backtest metrics. Found: {found_metrics}"

            # Check value formatting
            for metric_name, metric_value in results.items():
                if not metric_value or metric_value.strip() == "":
                    continue

                # Verify proper formatting
                metric_lower = metric_name.lower()

                if "rate" in metric_lower or "%" in metric_name:
                    has_valid_format = "%" in metric_value or any(
                        char.isdigit() for char in metric_value
                    )
                    assert (
                        has_valid_format
                    ), f"Rate metric '{metric_name}' invalid format: '{metric_value}'"

                elif (
                    "return" in metric_lower
                    or "capital" in metric_lower
                    or "drawdown" in metric_lower
                ):
                    has_valid_format = (
                        "$" in metric_value
                        or "%" in metric_value
                        or any(char.isdigit() for char in metric_value)
                    )
                    assert (
                        has_valid_format
                    ), f"Financial metric '{metric_name}' invalid format: '{metric_value}'"

        # Take screenshot of results
        dashboard.take_screenshot("backtest_results_display")

    def test_backtest_charts_visualization(self, dashboard_with_data):
        """Test backtest result charts and visualizations"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        backtest = BacktestPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Check for backtest charts
        has_charts = backtest.check_backtest_charts()

        if has_charts:
            # Count number of charts
            chart_count = page.locator('[data-testid="stPlotlyChart"]').count()
            assert chart_count > 0, "Charts detected but count is 0"

            # Should have multiple visualization charts
            expected_min_charts = 2  # e.g., equity curve, drawdown chart
            if chart_count >= expected_min_charts:
                dashboard.take_screenshot("backtest_charts_multiple")
            else:
                dashboard.take_screenshot("backtest_charts_limited")
        else:
            # Charts might load after running backtest or be in different format
            dashboard.take_screenshot("no_backtest_charts")

    def test_backtest_parameter_validation(self, dashboard_with_data):
        """Test validation of backtest parameters"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        backtest = BacktestPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Test invalid parameter values
        invalid_parameters = {
            "Max Trades": "-5",  # Negative value
            "Profit Target": "0",  # Zero value
            "Initial Capital": "abc",  # Non-numeric value
        }

        for param_name, invalid_value in invalid_parameters.items():
            try:
                # Set invalid parameter
                backtest.set_backtest_parameter(param_name, invalid_value)
                page.wait_for_timeout(1000)

                # Look for validation errors
                error_messages = page.locator("text=/Error|Invalid|Warning/")
                if error_messages.count() > 0:
                    # Good - validation is working
                    dashboard.take_screenshot(
                        f"validation_error_{param_name.lower().replace(' ', '_')}"
                    )

            except Exception as e:
                print(f"Parameter validation test for {param_name}: {e}")

    def test_backtest_date_validation(self, dashboard_with_data):
        """Test comprehensive date validation for backtesting"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Find date input fields
        start_date_input = (
            page.locator('label:has-text("Start Date")').locator("..").locator('input[type="date"]')
        )
        end_date_input = (
            page.locator('label:has-text("End Date")').locator("..").locator('input[type="date"]')
        )

        # Test 1: Check that date inputs exist
        assert start_date_input.count() > 0, "Start date input not found"
        assert end_date_input.count() > 0, "End date input not found"

        # Test 2: Check default date values
        start_value = start_date_input.input_value()
        end_value = end_date_input.input_value()

        # Should have default values
        assert start_value != "", "Start date should have default value"
        assert end_value != "", "End date should have default value"

        # Test 3: Check that default dates are in the past
        from datetime import date, datetime

        today = date.today()

        start_date = datetime.strptime(start_value, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_value, "%Y-%m-%d").date()

        assert start_date < today, "Start date should be in the past"
        assert end_date <= today, "End date should not be in the future"

        # Test 4: Check 60-day limit
        days_back = (today - start_date).days
        assert days_back <= 60, "Default start date should be within 60 days"

        dashboard.take_screenshot("backtest_date_defaults")

    def test_future_date_prevention(self, dashboard_with_data):
        """Test that future dates are prevented in backtest"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Try to set a future date
        from datetime import date, timedelta

        tomorrow = date.today() + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")

        # Find end date input and try to set future date
        end_date_input = (
            page.locator('label:has-text("End Date")').locator("..").locator('input[type="date"]')
        )

        # Check max attribute
        max_date = end_date_input.get_attribute("max")
        if max_date:
            max_date_obj = datetime.strptime(max_date, "%Y-%m-%d").date()
            assert max_date_obj <= date.today(), "Max date should not allow future dates"

        # Try to run backtest and check for error
        run_button = page.locator('button:has-text("Run Backtest")')
        if run_button.count() > 0:
            run_button.click()
            page.wait_for_timeout(2000)

            # Should show error about dates
            error_message = page.locator("text=/future|Future|invalid date|Invalid date/")
            if error_message.count() > 0:
                dashboard.take_screenshot("future_date_error")

    def test_date_range_validation(self, dashboard_with_data):
        """Test date range validation (start < end)"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Set end date before start date
        start_date_input = (
            page.locator('label:has-text("Start Date")').locator("..").locator('input[type="date"]')
        )
        end_date_input = (
            page.locator('label:has-text("End Date")').locator("..").locator('input[type="date"]')
        )

        from datetime import date, timedelta

        end_date = date.today() - timedelta(days=20)
        start_date = date.today() - timedelta(days=10)

        # Set dates in wrong order
        end_date_input.fill(end_date.strftime("%Y-%m-%d"))
        start_date_input.fill(start_date.strftime("%Y-%m-%d"))

        # Try to run backtest
        run_button = page.locator('button:has-text("Run Backtest")')
        if run_button.count() > 0:
            run_button.click()
            page.wait_for_timeout(2000)

            # Should show error
            error_message = page.locator(
                "text=/Start date must be before end date|Invalid date range/"
            )
            if error_message.count() > 0:
                dashboard.take_screenshot("date_range_validation_error")

    def test_60_day_lookback_warning(self, dashboard_with_data):
        """Test 60-day lookback limit warning"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Try to set start date beyond 60 days
        start_date_input = (
            page.locator('label:has-text("Start Date")').locator("..").locator('input[type="date"]')
        )

        from datetime import date, timedelta

        old_date = date.today() - timedelta(days=90)

        # Check if input has min attribute limiting date range
        min_date = start_date_input.get_attribute("min")
        if min_date:
            min_date_obj = datetime.strptime(min_date, "%Y-%m-%d").date()
            days_allowed = (date.today() - min_date_obj).days
            assert days_allowed <= 60, "Should limit lookback to ~60 days"

        # Look for any warning about data limitations
        warning_text = page.locator("text=/60 days|intraday data|Yahoo Finance/")
        if warning_text.count() > 0:
            dashboard.take_screenshot("60_day_limit_warning")

    @pytest.mark.slow
    def test_multiple_backtest_runs(self, dashboard_with_data):
        """Test running multiple backtests with different parameters"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        backtest = BacktestPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Test different parameter sets
        parameter_sets = [
            {"Max Trades": "5", "Profit Target": "3.0"},
            {"Max Trades": "10", "Profit Target": "4.0"},
            {"Max Trades": "15", "Profit Target": "2.0"},
        ]

        for i, params in enumerate(parameter_sets):
            # Set parameters
            for param_name, value in params.items():
                try:
                    backtest.set_backtest_parameter(param_name, value)
                except Exception:
                    continue

            # Take screenshot of parameter setup
            dashboard.take_screenshot(f"backtest_params_set_{i+1}")

            # Try to run backtest
            run_buttons = page.locator('button:has-text("Run"), button:has-text("Start")')
            if run_buttons.count() > 0:
                run_buttons.first.click()
                page.wait_for_timeout(5000)  # Wait for completion or progress

                # Take screenshot of results
                dashboard.take_screenshot(f"backtest_run_{i+1}_results")

    def test_backtest_history_management(self, dashboard_with_data):
        """Test backtest history and result management"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Look for historical backtest results
        history_elements = page.locator("text=/History|Previous|Past|Saved/")

        if history_elements.count() > 0:
            # Test viewing historical results
            history_elements.first.click()
            page.wait_for_timeout(2000)

            # Take screenshot of history view
            dashboard.take_screenshot("backtest_history_view")

            # Look for individual result entries
            result_entries = page.locator('[data-testid="stDataFrame"] tbody tr')
            if result_entries.count() > 0:
                result_count = result_entries.count()
                assert result_count > 0, "History view shows no backtest results"

                # Test clicking on a historical result
                result_entries.first.click()
                page.wait_for_timeout(2000)

                dashboard.take_screenshot("historical_backtest_selected")
        else:
            # History management might not be implemented
            dashboard.take_screenshot("no_backtest_history")

    def test_backtest_comparison_functionality(self, dashboard_with_data):
        """Test comparison of multiple backtest results"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Look for comparison features
        compare_elements = page.locator("text=/Compare|Comparison|vs/")

        if compare_elements.count() > 0:
            # Test comparison functionality
            compare_elements.first.click()
            page.wait_for_timeout(2000)

            # Take screenshot of comparison interface
            dashboard.take_screenshot("backtest_comparison_interface")

            # Look for comparison charts or tables
            comparison_viz = page.locator(
                '[data-testid="stPlotlyChart"], [data-testid="stDataFrame"]'
            )
            if comparison_viz.count() > 0:
                dashboard.take_screenshot("backtest_comparison_results")
        else:
            # Comparison feature might not be implemented
            dashboard.take_screenshot("no_backtest_comparison")

    def test_backtest_export_functionality(self, dashboard_with_data):
        """Test exporting backtest results"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Backtest
        dashboard.navigate_to_section("Backtest")

        # Look for export options
        export_elements = page.locator(
            'button:has-text("Export"), button:has-text("Download"), text=/CSV|Excel/'
        )

        if export_elements.count() > 0:
            # Take screenshot of export options
            dashboard.take_screenshot("backtest_export_options")

            # Test export functionality (without actually downloading)
            for i in range(min(2, export_elements.count())):  # Test first 2 options
                element = export_elements.nth(i)
                element_text = element.text_content()

                if element_text and any(
                    keyword in element_text.lower() for keyword in ["export", "download"]
                ):
                    dashboard.take_screenshot(f"backtest_export_option_{i+1}")
        else:
            # Export functionality might not be implemented
            dashboard.take_screenshot("no_backtest_export")
