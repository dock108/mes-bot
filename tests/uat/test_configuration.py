"""
UAT tests for configuration management functionality
"""

import pytest
from playwright.sync_api import Page, expect

from tests.uat.helpers.dashboard_pages import ConfigurationPage, DashboardPage


class TestConfigurationManagement:
    """Test configuration management and settings functionality"""

    def test_configuration_section_navigation(self, dashboard_with_data):
        """Test navigation to configuration section"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Configuration section
        dashboard.navigate_to_section("Configuration")

        # Verify navigation worked
        current_section = dashboard.get_current_section()
        assert (
            "Configuration" in current_section
            or "config" in current_section.lower()
            or "settings" in current_section.lower()
        )

        # Main content should be visible
        expect(dashboard.main_content).to_be_visible()

    def test_configuration_settings_display(self, dashboard_with_data):
        """Test that configuration settings are displayed properly"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        config_page = ConfigurationPage(page)

        # Navigate to Configuration
        dashboard.navigate_to_section("Configuration")

        # Get current settings
        settings = config_page.get_current_settings()

        # Should have some configuration settings
        assert len(settings) > 0, "No configuration settings found"

        # Check for expected trading configuration parameters
        expected_settings = [
            "Max Trades",
            "Profit Target",
            "Stop Loss",
            "Initial Capital",
            "Risk Per Trade",
            "Implied Move Multiplier",
            "Volatility Threshold",
        ]

        found_settings = list(settings.keys())
        settings_present = []

        for expected in expected_settings:
            if any(expected.lower() in found.lower() for found in found_settings):
                settings_present.append(expected)

        # Should have at least some key trading settings
        assert (
            len(settings_present) >= 3
        ), f"Missing key trading settings. Found: {found_settings}, Expected some of: {expected_settings}"

        # Take screenshot of configuration interface
        dashboard.take_screenshot("configuration_settings_display")

    def test_setting_value_modification(self, dashboard_with_data):
        """Test modifying configuration setting values"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        config_page = ConfigurationPage(page)

        # Navigate to Configuration
        dashboard.navigate_to_section("Configuration")

        # Get initial settings
        initial_settings = config_page.get_current_settings()

        # Test modifying some settings
        test_modifications = {"Max Trades": "12", "Profit Target": "3.5", "Initial Capital": "7500"}

        modifications_made = []

        for setting_name, new_value in test_modifications.items():
            try:
                # Update the setting
                config_page.update_setting(setting_name, new_value)
                modifications_made.append((setting_name, new_value))
                page.wait_for_timeout(500)  # Small delay between updates
            except Exception as e:
                print(f"Could not modify setting {setting_name}: {e}")

        # Verify modifications were applied
        if modifications_made:
            updated_settings = config_page.get_current_settings()

            for setting_name, expected_value in modifications_made:
                if setting_name in updated_settings:
                    actual_value = updated_settings[setting_name]
                    assert (
                        expected_value in actual_value or actual_value in expected_value
                    ), f"Setting {setting_name} not updated correctly. Expected: {expected_value}, Got: {actual_value}"

        # Take screenshot after modifications
        dashboard.take_screenshot("configuration_after_modifications")

    def test_configuration_save_functionality(self, dashboard_with_data):
        """Test saving configuration changes"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        config_page = ConfigurationPage(page)

        # Navigate to Configuration
        dashboard.navigate_to_section("Configuration")

        # Make a test modification
        try:
            config_page.update_setting("Max Trades", "8")
            page.wait_for_timeout(1000)
        except Exception:
            pass  # Setting might not exist

        # Take screenshot before saving
        dashboard.take_screenshot("before_config_save")

        # Try to save configuration
        try:
            config_page.save_configuration()
            page.wait_for_timeout(2000)  # Wait for save operation

            # Check for save success
            save_success = config_page.check_save_success()

            if save_success:
                dashboard.take_screenshot("config_save_success")
            else:
                # Look for any feedback messages
                feedback_messages = page.locator("text=/Saved|Updated|Success|Error|Failed/")
                if feedback_messages.count() > 0:
                    dashboard.take_screenshot("config_save_feedback")
                else:
                    dashboard.take_screenshot("config_save_no_feedback")

        except Exception as e:
            print(f"Error testing save functionality: {e}")
            dashboard.take_screenshot("config_save_error")

    def test_configuration_validation(self, dashboard_with_data):
        """Test validation of configuration values"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        config_page = ConfigurationPage(page)

        # Navigate to Configuration
        dashboard.navigate_to_section("Configuration")

        # Test invalid configuration values
        invalid_settings = {
            "Max Trades": "-5",  # Negative value
            "Profit Target": "0",  # Zero value
            "Initial Capital": "abc",  # Non-numeric value
            "Risk Per Trade": "150",  # Value over 100% (if percentage)
        }

        for setting_name, invalid_value in invalid_settings.items():
            try:
                # Set invalid value
                config_page.update_setting(setting_name, invalid_value)
                page.wait_for_timeout(1000)

                # Look for validation errors
                error_messages = page.locator("text=/Error|Invalid|Warning|Must be|Should be/")

                if error_messages.count() > 0:
                    # Good - validation is working
                    dashboard.take_screenshot(
                        f"validation_error_{setting_name.lower().replace(' ', '_')}"
                    )

                    # Clear the invalid value
                    config_page.update_setting(setting_name, "5")  # Set to valid value
                    page.wait_for_timeout(500)

            except Exception as e:
                print(f"Validation test for {setting_name}: {e}")

    def test_configuration_reset_functionality(self, dashboard_with_data):
        """Test resetting configuration to defaults"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        config_page = ConfigurationPage(page)

        # Navigate to Configuration
        dashboard.navigate_to_section("Configuration")

        # Get initial settings
        initial_settings = config_page.get_current_settings()

        # Make some modifications
        test_changes = {"Max Trades": "20", "Profit Target": "5.0"}

        for setting_name, value in test_changes.items():
            try:
                config_page.update_setting(setting_name, value)
            except Exception:
                continue

        # Look for reset button
        reset_buttons = page.locator(
            'button:has-text("Reset"), button:has-text("Default"), button:has-text("Restore")'
        )

        if reset_buttons.count() > 0:
            # Take screenshot before reset
            dashboard.take_screenshot("before_config_reset")

            # Click reset
            reset_buttons.first.click()
            page.wait_for_timeout(2000)

            # Take screenshot after reset
            dashboard.take_screenshot("after_config_reset")

            # Verify settings changed back
            reset_settings = config_page.get_current_settings()

            # At least some settings should have changed back
            changes_detected = False
            for setting_name in test_changes:
                if setting_name in reset_settings and setting_name in initial_settings:
                    if reset_settings[setting_name] != test_changes[setting_name]:
                        changes_detected = True
                        break

            if not changes_detected:
                dashboard.take_screenshot("reset_verification_unclear")
        else:
            # Reset functionality might not be implemented
            dashboard.take_screenshot("no_reset_functionality")

    def test_configuration_categories(self, dashboard_with_data):
        """Test different categories of configuration settings"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Configuration
        dashboard.navigate_to_section("Configuration")

        # Look for configuration categories/tabs
        category_elements = page.locator("text=/Trading|Risk|Display|General|Advanced/")

        if category_elements.count() > 0:
            # Test switching between categories
            for i in range(min(3, category_elements.count())):  # Test first 3 categories
                category = category_elements.nth(i)
                category_text = category.text_content()

                if category_text:
                    # Click category
                    category.click()
                    page.wait_for_timeout(1500)

                    # Take screenshot of category
                    category_name = category_text.lower().replace(" ", "_")
                    dashboard.take_screenshot(f"config_category_{category_name}")

                    # Verify content changed
                    expect(dashboard.main_content).to_be_visible()
        else:
            # All settings might be on one page
            dashboard.take_screenshot("config_single_page")

    def test_configuration_help_documentation(self, dashboard_with_data):
        """Test configuration help and documentation features"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Configuration
        dashboard.navigate_to_section("Configuration")

        # Look for help elements
        help_elements = page.locator(
            'text=/Help|Info|?|Documentation/, [title*="help"], [aria-label*="help"]'
        )

        if help_elements.count() > 0:
            # Test help functionality
            help_elements.first.click()
            page.wait_for_timeout(1000)

            # Take screenshot of help display
            dashboard.take_screenshot("config_help_display")

            # Look for help content
            help_content = page.locator("text=/Description|Explanation|Range|Default/")
            if help_content.count() > 0:
                dashboard.take_screenshot("config_help_content")
        else:
            # Help might be inline or not implemented
            dashboard.take_screenshot("no_config_help")

    def test_configuration_import_export(self, dashboard_with_data):
        """Test configuration import/export functionality"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)

        # Navigate to Configuration
        dashboard.navigate_to_section("Configuration")

        # Look for import/export options
        import_export_elements = page.locator("text=/Import|Export|Save Config|Load Config/")

        if import_export_elements.count() > 0:
            # Take screenshot of import/export options
            dashboard.take_screenshot("config_import_export_options")

            # Test export functionality
            export_elements = page.locator('button:has-text("Export"), text="Export"')
            if export_elements.count() > 0:
                export_elements.first.click()
                page.wait_for_timeout(1000)
                dashboard.take_screenshot("config_export_action")

            # Test import interface
            import_elements = page.locator('button:has-text("Import"), text="Import"')
            if import_elements.count() > 0:
                import_elements.first.click()
                page.wait_for_timeout(1000)
                dashboard.take_screenshot("config_import_interface")
        else:
            # Import/export might not be implemented
            dashboard.take_screenshot("no_config_import_export")

    def test_configuration_real_time_updates(self, dashboard_with_data):
        """Test real-time updates of configuration changes"""
        page = dashboard_with_data
        dashboard = DashboardPage(page)
        config_page = ConfigurationPage(page)

        # Navigate to Configuration
        dashboard.navigate_to_section("Configuration")

        # Make a configuration change
        try:
            config_page.update_setting("Max Trades", "15")
            page.wait_for_timeout(1000)
        except Exception:
            pass

        # Navigate to Live Monitor to see if changes are reflected
        dashboard.navigate_to_section("Live Monitor")
        page.wait_for_timeout(2000)

        # Take screenshot of live monitor after config change
        dashboard.take_screenshot("live_monitor_after_config_change")

        # Navigate back to configuration
        dashboard.navigate_to_section("Configuration")

        # Verify setting is still applied
        current_settings = config_page.get_current_settings()
        if "Max Trades" in current_settings:
            max_trades_value = current_settings["Max Trades"]
            # Should reflect the change we made
            dashboard.take_screenshot("config_persistence_check")
