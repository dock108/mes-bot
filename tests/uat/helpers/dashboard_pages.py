"""
Page Object Model for Streamlit Dashboard UAT Testing
"""
from playwright.sync_api import Page, Locator
from typing import Dict, List, Optional
import time


class BasePage:
    """Base page with common functionality"""
    
    def __init__(self, page: Page):
        self.page = page
    
    def wait_for_load(self, timeout: int = 30000):
        """Wait for page to fully load"""
        self.page.wait_for_selector('[data-testid="stSidebar"]', timeout=timeout)
        time.sleep(1)  # Additional wait for dynamic content
    
    def take_screenshot(self, name: str):
        """Take screenshot for debugging"""
        self.page.screenshot(path=f"test-results/screenshots/{name}.png")
    
    def get_sidebar_options(self) -> List[str]:
        """Get all sidebar navigation options"""
        sidebar = self.page.locator('[data-testid="stSidebar"]')
        options = sidebar.locator('label').all_text_contents()
        return [opt.strip() for opt in options if opt.strip()]


class DashboardPage(BasePage):
    """Main dashboard page interactions"""
    
    def __init__(self, page: Page):
        super().__init__(page)
        self.sidebar = page.locator('[data-testid="stSidebar"]')
        # Try multiple selectors for main content area
        main_selectors = [
            '[data-testid="stAppViewContainer"]',  # Streamlit main app container
            '[data-testid="main"]', 
            '.main', 
            '.stApp'
        ]
        self.main_content = None
        for selector in main_selectors:
            locator = page.locator(selector).first  # Use .first to avoid strict mode violations
            if locator.count() > 0:
                self.main_content = locator
                break
        
        # Fallback to page if no main content found
        if self.main_content is None:
            self.main_content = page
    
    def navigate_to_section(self, section_name: str):
        """Navigate to a specific dashboard section"""
        # Create mapping between test names and UI names (with emojis)
        section_mapping = {
            "Live Monitor": "📊 Live Monitor",
            "Performance": "📈 Performance", 
            "Backtest": "🔄 Backtesting",
            "Backtesting": "🔄 Backtesting",
            "Configuration": "⚙️ Configuration"
        }
        
        # Try both the original name and the mapped UI name
        names_to_try = [section_name]
        if section_name in section_mapping:
            names_to_try.append(section_mapping[section_name])
        
        for name in names_to_try:
            # Try multiple ways to find and click the navigation element
            navigation_attempts = [
                f'input[value="{name}"]',  # Radio button by exact value
                f'label:has-text("{name}")',  # Label containing text
                f'text="{name}"',  # Direct text match
                f'[data-testid="stRadio"] label:has-text("{name}")',  # Streamlit radio label
                f'div:has-text("{name}") input[type="radio"]',  # Radio input near text
                f'label:text-is("{name}")',  # Exact text match for label
            ]
            
            for selector in navigation_attempts:
                element = self.sidebar.locator(selector)
                if element.count() > 0:
                    try:
                        element.click()
                        time.sleep(3)  # Wait for content to load
                        return
                    except Exception:
                        continue
        
        # Fallback: try partial text matching for any section name variations
        for name in names_to_try:
            # Look for any element containing the section name (without emoji)
            clean_name = section_name.replace("📊 ", "").replace("📈 ", "").replace("🔄 ", "").replace("⚙️ ", "")
            sidebar_text = self.sidebar.locator(f':has-text("{clean_name}")')
            if sidebar_text.count() > 0:
                try:
                    sidebar_text.first.click()
                    time.sleep(3)
                    return
                except Exception:
                    continue
    
    def get_current_section(self) -> str:
        """Get currently selected section"""
        # Streamlit radio buttons use numeric indices, so we need to map them to section names
        index_to_section = {
            "0": "Live Monitor",
            "1": "Performance", 
            "2": "Backtest",
            "3": "Configuration"
        }
        
        # Approach 1: Look for checked radio button and get its index
        checked_radio = self.sidebar.locator('input[type="radio"]:checked')
        if checked_radio.count() > 0:
            raw_value = checked_radio.get_attribute('value') or ""
            
            # If it's a numeric index, map it to section name
            if raw_value in index_to_section:
                return index_to_section[raw_value]
            
            # If it's the full text name, try to map it
            ui_to_test_mapping = {
                "📊 Live Monitor": "Live Monitor",
                "📈 Performance": "Performance", 
                "🔄 Backtesting": "Backtest",
                "⚙️ Configuration": "Configuration"
            }
            
            if raw_value in ui_to_test_mapping:
                return ui_to_test_mapping[raw_value]
            
            # Try to clean emoji prefixes
            if raw_value:
                clean_value = raw_value
                emoji_prefixes = ["📊 ", "📈 ", "🔄 ", "⚙️ "]
                for prefix in emoji_prefixes:
                    if clean_value.startswith(prefix):
                        clean_value = clean_value[len(prefix):]
                        break
                
                if clean_value == "Backtesting":
                    clean_value = "Backtest"
                    
                return clean_value
        
        # Approach 2: Look for selected label in Streamlit radio group
        radio_labels = self.sidebar.locator('[data-testid="stRadio"] label')
        for i in range(radio_labels.count()):
            label = radio_labels.nth(i)
            # Check if this label appears to be selected (common Streamlit patterns)
            if label.get_attribute('data-checked') == 'true' or 'selected' in (label.get_attribute('class') or ''):
                if str(i) in index_to_section:
                    return index_to_section[str(i)]
        
        # Approach 3: Fallback to looking at visible content
        all_sidebar_text = self.sidebar.text_content()
        if "Live Monitor" in all_sidebar_text:
            return "Live Monitor"
        elif "Performance" in all_sidebar_text:
            return "Performance"  
        elif "Backtesting" in all_sidebar_text or "Backtest" in all_sidebar_text:
            return "Backtest"
        elif "Configuration" in all_sidebar_text:
            return "Configuration"
            
        return ""
    
    def get_metrics_values(self) -> Dict[str, str]:
        """Extract metric values from dashboard"""
        metrics = {}
        
        # Try multiple selectors for Streamlit metrics
        possible_selectors = [
            '[data-testid="stMetric"]',  # Modern Streamlit metric selector
            '[data-testid="metric"]',    # Alternative metric selector
            '.metric-container',         # CSS class selector
            '[data-testid="stMetricValue"]'  # Direct metric value selector
        ]
        
        for selector in possible_selectors:
            metric_containers = self.main_content.locator(selector)
            if metric_containers.count() > 0:
                for i in range(metric_containers.count()):
                    container = metric_containers.nth(i)
                    # Try to extract label and value from metric container
                    label_text = container.locator('label, .metric-label, [data-testid="stMetricLabel"]').text_content()
                    value_text = container.locator('.metric-value, [data-testid="stMetricValue"]').text_content()
                    
                    if label_text and value_text:
                        metrics[label_text.strip()] = value_text.strip()
                
                if metrics:  # If we found metrics, stop trying other selectors
                    break
        
        # Fallback: try to find any text that looks like metrics
        if not metrics:
            all_text = self.main_content.text_content()
            # Look for common metric patterns like "Total P&L: $123.45"
            import re
            metric_patterns = re.findall(r'([A-Z][^:]+):\s*\$?([0-9,.-]+%?)', all_text)
            for label, value in metric_patterns:
                metrics[label.strip()] = value.strip()
        
        return metrics
    
    def check_emergency_stop_visible(self) -> bool:
        """Check if emergency stop button is visible"""
        return self.main_content.locator('text="EMERGENCY STOP"').count() > 0
    
    def click_emergency_stop(self):
        """Click the emergency stop button"""
        stop_button = self.main_content.locator('button:has-text("EMERGENCY STOP")')
        if stop_button.count() > 0:
            stop_button.click()
            time.sleep(1)
    
    def get_trading_mode(self) -> str:
        """Get current trading mode (Auto/Manual/Off)"""
        checked_radio = self.main_content.locator('input[type="radio"]:checked')
        if checked_radio.count() > 0:
            value = checked_radio.get_attribute('value')
            # Map numeric values to mode names if needed
            mode_map = {"0": "Auto", "1": "Manual", "2": "Off"}
            if value in mode_map:
                return mode_map[value]
            return value
        return ""
    
    def set_trading_mode(self, mode: str):
        """Set trading mode to Auto/Manual/Off"""
        mode_button = self.main_content.locator(f'label:has-text("{mode}")').locator('..').locator('input[type="radio"]')
        if mode_button.count() > 0:
            mode_button.click()
            time.sleep(1)
    
    def check_opportunity_scanner_visible(self) -> bool:
        """Check if opportunity scanner is visible"""
        return self.main_content.locator('text="Opportunity Scanner"').count() > 0
    
    def get_signal_status(self) -> str:
        """Get current signal status (SIGNAL READY/WATCHING...)"""
        ready = self.main_content.locator('text="SIGNAL READY"')
        watching = self.main_content.locator('text="WATCHING..."')
        
        if ready.count() > 0:
            return "SIGNAL READY"
        elif watching.count() > 0:
            return "WATCHING..."
        return ""
    
    def get_entry_conditions(self) -> Dict[str, bool]:
        """Get entry conditions and their status"""
        conditions = {}
        
        # Look for condition text with checkmarks/X marks
        condition_elements = self.main_content.locator('text=/RV < 67% of IV|Time between trades OK|Risk limits OK|Premium in range/')
        
        for i in range(condition_elements.count()):
            condition = condition_elements.nth(i)
            text = condition.text_content().strip()
            
            # Check if it has a checkmark or X
            parent = condition.locator('..')
            has_checkmark = parent.locator('text="✅"').count() > 0
            
            conditions[text] = has_checkmark
        
        return conditions
    
    def click_manual_trade_button(self, button_text: str):
        """Click a manual trade control button (Review Trade/Place Trade/Skip Signal)"""
        button = self.main_content.locator(f'button:has-text("{button_text}")')
        if button.count() > 0:
            button.click()
            time.sleep(1)


class LiveMonitorPage(BasePage):
    """Live monitoring section interactions"""
    
    def __init__(self, page: Page):
        super().__init__(page)
        self.main_content = page.locator('[data-testid="main"]')
    
    def get_open_positions_count(self) -> int:
        """Get number of open positions displayed"""
        positions_table = self.main_content.locator('[data-testid="stDataFrame"]')
        if positions_table.count() > 0:
            rows = positions_table.locator('tbody tr')
            return rows.count()
        return 0
    
    def get_position_data(self) -> List[Dict[str, str]]:
        """Extract open positions data"""
        positions = []
        table = self.main_content.locator('[data-testid="stDataFrame"]')
        
        if table.count() > 0:
            headers = table.locator('thead th').all_text_contents()
            rows = table.locator('tbody tr')
            
            for i in range(rows.count()):
                row = rows.nth(i)
                cells = row.locator('td').all_text_contents()
                
                if len(cells) == len(headers):
                    position_data = dict(zip(headers, cells))
                    positions.append(position_data)
        
        return positions
    
    def check_real_time_updates(self) -> bool:
        """Check if real-time price updates are working"""
        # Look for price elements and check if they update
        price_elements = self.main_content.locator('text=/\\$[0-9,]+\\.?[0-9]*/')
        return price_elements.count() > 0
    
    def get_market_status(self) -> str:
        """Get current market status indicator"""
        status_elem = self.main_content.locator('text=/Market Status|OPEN|CLOSED/')
        if status_elem.count() > 0:
            return status_elem.text_content().strip()
        return "Unknown"


class PerformancePage(BasePage):
    """Performance analytics section interactions"""
    
    def __init__(self, page: Page):
        super().__init__(page)
        self.main_content = page.locator('[data-testid="main"]')
    
    def get_performance_metrics(self) -> Dict[str, str]:
        """Extract performance metrics"""
        metrics = {}
        
        # Get metric containers
        metric_containers = self.main_content.locator('[data-testid="metric-container"]')
        for i in range(metric_containers.count()):
            container = metric_containers.nth(i)
            label = container.locator('[data-testid="metric-label"]').text_content()
            value = container.locator('[data-testid="metric-value"]').text_content()
            
            if label and value:
                metrics[label.strip()] = value.strip()
        
        return metrics
    
    def check_charts_loaded(self) -> bool:
        """Check if performance charts are loaded"""
        # Look for Plotly charts
        plotly_charts = self.main_content.locator('[data-testid="stPlotlyChart"]')
        return plotly_charts.count() > 0
    
    def get_chart_count(self) -> int:
        """Get number of charts displayed"""
        charts = self.main_content.locator('[data-testid="stPlotlyChart"]')
        return charts.count()
    
    def check_trades_table(self) -> bool:
        """Check if trades table is visible"""
        table = self.main_content.locator('[data-testid="stDataFrame"]')
        return table.count() > 0
    
    def get_trades_count(self) -> int:
        """Get number of trades in the table"""
        table = self.main_content.locator('[data-testid="stDataFrame"]')
        if table.count() > 0:
            rows = table.locator('tbody tr')
            return rows.count()
        return 0


class BacktestPage(BasePage):
    """Backtesting section interactions"""
    
    def __init__(self, page: Page):
        super().__init__(page)
        self.main_content = page.locator('[data-testid="main"]')
    
    def set_backtest_parameter(self, param_name: str, value: str):
        """Set a backtest parameter value"""
        # Find input field by label
        label = self.main_content.locator(f'text="{param_name}"')
        if label.count() > 0:
            # Find associated input
            input_field = label.locator('..').locator('input').first
            if input_field.count() > 0:
                input_field.fill(value)
    
    def click_run_backtest(self):
        """Click the run backtest button"""
        run_button = self.main_content.locator('button:has-text("Run Backtest")')
        if run_button.count() > 0:
            run_button.click()
            time.sleep(2)  # Wait for execution to start
    
    def wait_for_backtest_completion(self, timeout: int = 60000):
        """Wait for backtest to complete"""
        # Wait for results to appear or progress to complete
        self.page.wait_for_selector(
            'text="Backtest Complete" | [data-testid="stPlotlyChart"]',
            timeout=timeout
        )
    
    def get_backtest_results(self) -> Dict[str, str]:
        """Extract backtest results"""
        results = {}
        
        # Look for result metrics
        metric_containers = self.main_content.locator('[data-testid="metric-container"]')
        for i in range(metric_containers.count()):
            container = metric_containers.nth(i)
            label = container.locator('[data-testid="metric-label"]').text_content()
            value = container.locator('[data-testid="metric-value"]').text_content()
            
            if label and value:
                results[label.strip()] = value.strip()
        
        return results
    
    def check_backtest_charts(self) -> bool:
        """Check if backtest result charts are displayed"""
        charts = self.main_content.locator('[data-testid="stPlotlyChart"]')
        return charts.count() > 0
    
    def set_date_input(self, label: str, date_value: str):
        """Set a date input field value"""
        date_input = self.main_content.locator(f'label:has-text("{label}")').locator('..').locator('input[type="date"]')
        if date_input.count() > 0:
            date_input.fill(date_value)
    
    def get_date_input_value(self, label: str) -> str:
        """Get a date input field value"""
        date_input = self.main_content.locator(f'label:has-text("{label}")').locator('..').locator('input[type="date"]')
        if date_input.count() > 0:
            return date_input.input_value()
        return ""
    
    def get_date_input_constraints(self, label: str) -> Dict[str, str]:
        """Get date input min/max constraints"""
        date_input = self.main_content.locator(f'label:has-text("{label}")').locator('..').locator('input[type="date"]')
        if date_input.count() > 0:
            return {
                "min": date_input.get_attribute("min") or "",
                "max": date_input.get_attribute("max") or ""
            }
        return {"min": "", "max": ""}


class ConfigurationPage(BasePage):
    """Configuration section interactions"""
    
    def __init__(self, page: Page):
        super().__init__(page)
        self.main_content = page.locator('[data-testid="main"]')
    
    def get_current_settings(self) -> Dict[str, str]:
        """Get current configuration settings"""
        settings = {}
        
        # Find all input fields and their labels
        inputs = self.main_content.locator('input')
        for i in range(inputs.count()):
            input_field = inputs.nth(i)
            input_type = input_field.get_attribute('type')
            
            if input_type in ['number', 'text']:
                # Find associated label
                label_elem = input_field.locator('..').locator('label').first
                if label_elem.count() > 0:
                    label = label_elem.text_content().strip()
                    value = input_field.input_value()
                    settings[label] = value
        
        return settings
    
    def update_setting(self, setting_name: str, value: str):
        """Update a configuration setting"""
        # Find input by associated label
        label = self.main_content.locator(f'text="{setting_name}"')
        if label.count() > 0:
            input_field = label.locator('..').locator('input').first
            if input_field.count() > 0:
                input_field.fill(value)
    
    def save_configuration(self):
        """Save configuration changes"""
        save_button = self.main_content.locator('button:has-text("Save")')
        if save_button.count() > 0:
            save_button.click()
            time.sleep(1)
    
    def check_save_success(self) -> bool:
        """Check if configuration was saved successfully"""
        success_message = self.main_content.locator('text="Configuration saved"')
        return success_message.count() > 0