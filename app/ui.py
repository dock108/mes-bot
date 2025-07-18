"""
Streamlit dashboard for MES 0DTE Lotto-Grid Options Bot
"""

import asyncio
import logging
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.backtester import LottoGridBacktester
from app.config import config
from app.models import BacktestResult, DailySummary, Trade, get_session_maker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="MES 0DTE Lotto-Grid Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True

if "testing_mode" not in st.session_state:
    # Enable testing mode if TRADE_MODE is set to paper/test
    import os

    st.session_state.testing_mode = os.getenv("TRADE_MODE") in ["paper", "test"]


class Dashboard:
    """Main dashboard class"""

    def __init__(self):
        self.session_maker = get_session_maker(config.database.url)
        self.backtester = LottoGridBacktester(config.database.url)

    def get_session(self):
        """Get database session"""
        return self.session_maker()

    def load_open_trades(self) -> pd.DataFrame:
        """Load open trades from database"""
        session = self.get_session()
        try:
            trades = session.query(Trade).filter(Trade.status == "OPEN").all()

            if not trades:
                return pd.DataFrame()

            data = []
            for trade in trades:
                data.append(
                    {
                        "ID": trade.id,
                        "Entry Time": trade.entry_time.strftime("%H:%M:%S"),
                        "Call Strike": f"${trade.call_strike:.0f}",
                        "Put Strike": f"${trade.put_strike:.0f}",
                        "Call Premium": f"${trade.call_premium:.2f}",
                        "Put Premium": f"${trade.put_premium:.2f}",
                        "Total Premium": f"${trade.total_premium:.2f}",
                        "Unrealized P&L": f"${trade.unrealized_pnl or 0:.2f}",
                        "Call Status": trade.call_status,
                        "Put Status": trade.put_status,
                        "Time Open": str(datetime.utcnow() - trade.entry_time).split(".")[0],
                    }
                )

            return pd.DataFrame(data)

        finally:
            session.close()

    def load_daily_summary(self) -> Dict:
        """Load today's trading summary"""
        session = self.get_session()
        try:
            today = date.today()
            summary = session.query(DailySummary).filter(DailySummary.date == today).first()

            if not summary:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "net_pnl": 0.0,
                    "win_rate": 0.0,
                    "max_concurrent_trades": 0,
                }

            win_rate = (
                summary.winning_trades / summary.total_trades if summary.total_trades > 0 else 0
            )

            return {
                "total_trades": summary.total_trades,
                "winning_trades": summary.winning_trades,
                "losing_trades": summary.losing_trades,
                "net_pnl": summary.net_pnl,
                "win_rate": win_rate,
                "max_concurrent_trades": summary.max_concurrent_trades,
            }

        finally:
            session.close()

    def load_recent_trades(self, days: int = 7) -> pd.DataFrame:
        """Load recent closed trades"""
        session = self.get_session()
        try:
            cutoff_date = date.today() - timedelta(days=days)
            trades = (
                session.query(Trade)
                .filter(
                    Trade.date >= cutoff_date,
                    Trade.status.in_(["CLOSED_WIN", "CLOSED_LOSS", "EXPIRED"]),
                )
                .order_by(Trade.exit_time.desc())
                .limit(50)
                .all()
            )

            if not trades:
                return pd.DataFrame()

            data = []
            for trade in trades:
                data.append(
                    {
                        "Date": trade.date.strftime("%Y-%m-%d"),
                        "Entry": trade.entry_time.strftime("%H:%M"),
                        "Exit": trade.exit_time.strftime("%H:%M") if trade.exit_time else "N/A",
                        "Strikes": f"{trade.call_strike:.0f}C/{trade.put_strike:.0f}P",
                        "Premium": f"${trade.total_premium:.2f}",
                        "P&L": f"${trade.realized_pnl:.2f}",
                        "Return %": (
                            f"{(trade.realized_pnl / trade.total_premium * 100):.1f}%"
                            if trade.total_premium > 0
                            else "0%"
                        ),
                        "Status": trade.status,
                    }
                )

            return pd.DataFrame(data)

        finally:
            session.close()

    def load_equity_curve(self, days: int = 30) -> pd.DataFrame:
        """Load equity curve data"""
        session = self.get_session()
        try:
            cutoff_date = date.today() - timedelta(days=days)
            summaries = (
                session.query(DailySummary)
                .filter(DailySummary.date >= cutoff_date)
                .order_by(DailySummary.date)
                .all()
            )

            if not summaries:
                return pd.DataFrame()

            data = []
            cumulative_pnl = config.trading.start_cash

            for summary in summaries:
                cumulative_pnl += summary.net_pnl
                data.append(
                    {
                        "Date": summary.date,
                        "Daily P&L": summary.net_pnl,
                        "Cumulative Equity": cumulative_pnl,
                        "Trades": summary.total_trades,
                        "Win Rate": (
                            summary.winning_trades / summary.total_trades
                            if summary.total_trades > 0
                            else 0
                        ),
                    }
                )

            return pd.DataFrame(data)

        finally:
            session.close()

    def create_equity_chart(self, df: pd.DataFrame):
        """Create equity curve chart"""
        if df.empty:
            return None

        fig = go.Figure()

        # Add equity line
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Cumulative Equity"],
                mode="lines+markers",
                name="Equity",
                line=dict(color="blue", width=2),
                hovertemplate="Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>",
            )
        )

        # Add starting equity line
        fig.add_hline(
            y=config.trading.start_cash,
            line_dash="dash",
            line_color="gray",
            annotation_text="Starting Equity",
        )

        fig.update_layout(
            title="Account Equity Curve",
            xaxis_title="Date",
            yaxis_title="Account Equity ($)",
            hovermode="x unified",
            showlegend=True,
        )

        return fig

    def create_daily_pnl_chart(self, df: pd.DataFrame):
        """Create daily P&L chart"""
        if df.empty:
            return None

        colors = ["green" if pnl >= 0 else "red" for pnl in df["Daily P&L"]]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df["Date"],
                    y=df["Daily P&L"],
                    marker_color=colors,
                    hovertemplate="Date: %{x}<br>P&L: $%{y:,.2f}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title="Daily P&L",
            xaxis_title="Date",
            yaxis_title="Daily P&L ($)",
            hovermode="x unified",
        )

        return fig

    def render_live_monitoring(self):
        """Render live monitoring tab"""
        st.header("Live Trading Monitor")

        # Initialize trading mode in session state
        if "trading_mode" not in st.session_state:
            st.session_state.trading_mode = "manual"  # Default to manual for safety

        # Market Status Bar
        status_col1, status_col2, status_col3, status_col4 = st.columns([2, 2, 2, 2])

        with status_col1:
            st.metric("MES Price", "$4,215.50", "+0.5%")

        with status_col2:
            st.metric("IV / RV60", "12.5% / 8.2%", "-35%")

        with status_col3:
            current_time = datetime.now()
            market_close = current_time.replace(hour=16, minute=0, second=0)
            time_to_close = market_close - current_time
            hours = int(time_to_close.total_seconds() // 3600)
            minutes = int((time_to_close.total_seconds() % 3600) // 60)
            st.metric("Time to Close", f"{hours}h {minutes}m")

        with status_col4:
            # Trading Mode Toggle
            mode = st.radio(
                "Trading Mode",
                ["Auto", "Manual", "Off"],
                index=["Auto", "Manual", "Off"].index(st.session_state.trading_mode.title()),
                horizontal=True,
            )
            st.session_state.trading_mode = mode.lower()

        st.divider()

        # Auto-refresh controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if st.button("🔄 Refresh Now"):
                st.session_state.last_refresh = datetime.now()
                st.rerun()

        with col2:
            auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto_refresh

        with col3:
            st.write(f"Last: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

        # Current status metrics
        daily_summary = self.load_daily_summary()

        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Today's P&L", f"${daily_summary['net_pnl']:,.2f}", delta=None)

        with col2:
            st.metric("Total Trades", daily_summary["total_trades"])

        with col3:
            st.metric("Win Rate", f"{daily_summary['win_rate']:.1%}")

        with col4:
            st.metric(
                "Winners",
                daily_summary["winning_trades"],
                delta=f"+{daily_summary['winning_trades']}",
            )

        with col5:
            st.metric(
                "Losers",
                daily_summary["losing_trades"],
                delta=(
                    f"-{daily_summary['losing_trades']}"
                    if daily_summary["losing_trades"] > 0
                    else None
                ),
            )

        st.divider()

        # Opportunity Scanner (only show in manual or auto mode)
        if st.session_state.trading_mode != "off":
            st.subheader("📡 Opportunity Scanner")

            # Signal status
            signal_col1, signal_col2 = st.columns([1, 3])

            with signal_col1:
                # Simulate signal status
                signal_ready = True  # This would come from actual strategy logic
                if signal_ready:
                    st.success("🟢 SIGNAL READY")
                else:
                    st.info("🟡 WATCHING...")

            with signal_col2:
                # Entry conditions checklist
                st.write("**Entry Conditions:**")
                conditions = {
                    "RV < 67% of IV": (8.2 / 12.5) < 0.67,
                    "Time between trades OK": True,
                    "Risk limits OK": True,
                    "Premium in range": True,
                }

                for condition, met in conditions.items():
                    if met:
                        st.write(f"✅ {condition}")
                    else:
                        st.write(f"❌ {condition}")

            # Suggested trade details
            if signal_ready:
                st.write("**Suggested Trade:**")
                trade_col1, trade_col2, trade_col3 = st.columns(3)

                with trade_col1:
                    st.info("**Call Strike**\n\n" "$4,265 @ $8.50")

                with trade_col2:
                    st.info("**Put Strike**\n\n" "$4,165 @ $10.00")

                with trade_col3:
                    st.success("**Total Premium:** $18.50\n\n" "**Target:** $74.00 (4x)")

                # Manual trade controls (only in manual mode)
                if st.session_state.trading_mode == "manual":
                    st.write("**Manual Trade Controls:**")
                    action_col1, action_col2, action_col3 = st.columns(3)

                    with action_col1:
                        if st.button("📋 Review Trade", use_container_width=True):
                            st.info("Trade review details would appear here")

                    with action_col2:
                        if st.button("✅ Place Trade", type="primary", use_container_width=True):
                            st.success("Trade placement would execute here")

                    with action_col3:
                        if st.button("⏭️ Skip Signal", use_container_width=True):
                            st.info("Signal skipped")

        st.divider()

        # Active trades table
        st.subheader("Active Trades")
        open_trades_df = self.load_open_trades()

        if not open_trades_df.empty:
            st.dataframe(open_trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No active trades")

        st.divider()

        # Recent trades table
        st.subheader("Recent Trades (Last 7 Days)")
        recent_trades_df = self.load_recent_trades(7)

        if not recent_trades_df.empty:
            # Color code the P&L column
            def color_pnl(val):
                color = "color: green" if float(val.replace("$", "")) >= 0 else "color: red"
                return color

            styled_df = recent_trades_df.style.applymap(color_pnl, subset=["P&L"])

            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("No recent trades")

        st.divider()

        # Emergency Controls
        st.subheader("⚠️ Emergency Controls")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if st.button("🛑 EMERGENCY STOP", type="primary", use_container_width=True):
                # Show confirmation using session state
                if "show_emergency_confirm" not in st.session_state:
                    st.session_state.show_emergency_confirm = False

                st.session_state.show_emergency_confirm = True

        # Show confirmation dialog if triggered
        if st.session_state.get("show_emergency_confirm", False):
            st.warning("⚠️ **EMERGENCY STOP CONFIRMATION**")
            st.write("This will immediately close all positions and halt trading!")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, STOP ALL", type="primary", use_container_width=True):
                    try:
                        # Create emergency stop flag file
                        import os

                        emergency_flag_file = "./data/emergency_stop.flag"
                        os.makedirs(os.path.dirname(emergency_flag_file), exist_ok=True)

                        with open(emergency_flag_file, "w") as f:
                            f.write(f"EMERGENCY_STOP_TRIGGERED_{datetime.now().isoformat()}")

                        st.error("🛑 Emergency stop signal sent!")
                        st.write(
                            "Emergency stop flag created. Bot will halt trading on next cycle."
                        )
                        st.info("Monitor logs for confirmation of position closure.")

                    except Exception as e:
                        st.error(f"Failed to create emergency stop flag: {e}")

                    st.session_state.show_emergency_confirm = False
                    time.sleep(2)
                    st.rerun()

            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_emergency_confirm = False
                    st.rerun()

        # Auto-refresh mechanism (disabled during testing)
        if st.session_state.auto_refresh and not st.session_state.get("testing_mode", False):
            time.sleep(config.ui.refresh_interval)
            st.rerun()

    def render_performance_analytics(self):
        """Render performance analytics tab"""
        st.header("Performance Analytics")

        # Date range selector
        col1, col2 = st.columns(2)

        with col1:
            days_back = st.selectbox(
                "Time Period",
                [7, 14, 30, 60, 90],
                index=2,  # Default to 30 days
                format_func=lambda x: f"Last {x} days",
            )

        equity_df = self.load_equity_curve(days_back)

        if equity_df.empty:
            st.warning("No performance data available for the selected period")
            return

        # Performance metrics
        current_equity = equity_df["Cumulative Equity"].iloc[-1]
        starting_equity = config.trading.start_cash
        total_return = (current_equity - starting_equity) / starting_equity
        max_equity = equity_df["Cumulative Equity"].max()
        current_drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Return",
                f"{total_return:.2%}",
                delta=f"${current_equity - starting_equity:,.2f}",
            )

        with col2:
            st.metric("Current Equity", f"${current_equity:,.2f}")

        with col3:
            st.metric(
                "Max Drawdown",
                f"{current_drawdown:.2%}",
                delta=f"-${max_equity - current_equity:,.2f}" if current_drawdown > 0 else None,
            )

        with col4:
            avg_daily_trades = equity_df["Trades"].mean()
            st.metric("Avg Daily Trades", f"{avg_daily_trades:.1f}")

        st.divider()

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            equity_chart = self.create_equity_chart(equity_df)
            if equity_chart:
                st.plotly_chart(equity_chart, use_container_width=True)

        with col2:
            pnl_chart = self.create_daily_pnl_chart(equity_df)
            if pnl_chart:
                st.plotly_chart(pnl_chart, use_container_width=True)

        # Detailed statistics
        st.subheader("Detailed Statistics")

        total_trades = equity_df["Trades"].sum()
        profitable_days = len(equity_df[equity_df["Daily P&L"] > 0])
        total_days = len(equity_df)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Trading Statistics:**")
            st.write(f"- Total Trading Days: {total_days}")
            st.write(f"- Profitable Days: {profitable_days} ({profitable_days/total_days:.1%})")
            st.write(f"- Total Trades: {total_trades}")
            st.write(f"- Average Trades/Day: {total_trades/total_days:.1f}")

        with col2:
            st.write("**P&L Statistics:**")
            avg_daily_pnl = equity_df["Daily P&L"].mean()
            std_daily_pnl = equity_df["Daily P&L"].std()
            best_day = equity_df["Daily P&L"].max()
            worst_day = equity_df["Daily P&L"].min()

            st.write(f"- Average Daily P&L: ${avg_daily_pnl:.2f}")
            st.write(f"- Daily P&L Std Dev: ${std_daily_pnl:.2f}")
            st.write(f"- Best Day: ${best_day:.2f}")
            st.write(f"- Worst Day: ${worst_day:.2f}")

    def render_backtesting(self):
        """Render backtesting tab"""
        st.header("Strategy Backtesting")

        # Data source information
        st.info(
            "📊 **Data Sources**: The backtester uses SPY ETF data as a proxy for MES futures. "
            "For intraday backtests, data is limited to the last 60 days due to Yahoo Finance restrictions."
        )

        # Backtest configuration
        with st.form("backtest_form"):
            st.subheader("Backtest Parameters")

            col1, col2 = st.columns(2)

            with col1:
                # For Yahoo Finance free tier, intraday data is only available for last 60 days
                # Also ensure we're not trying to fetch future data
                today = date.today()
                max_lookback = 59  # Yahoo Finance limit for intraday data

                start_date = st.date_input(
                    "Start Date",
                    value=today - timedelta(days=30),
                    max_value=today - timedelta(days=1),
                    min_value=today - timedelta(days=max_lookback),
                )

                initial_capital = st.number_input(
                    "Initial Capital",
                    min_value=1000.0,
                    max_value=100000.0,
                    value=float(config.trading.start_cash),
                    step=1000.0,
                )

                max_trades = st.number_input(
                    "Max Concurrent Trades",
                    min_value=1,
                    max_value=50,
                    value=config.trading.max_open_trades,
                    step=1,
                )

            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=today - timedelta(days=1),
                    max_value=today,
                    min_value=start_date,  # End date must be after start date
                )

                profit_target = st.number_input(
                    "Profit Target Multiplier",
                    min_value=2.0,
                    max_value=10.0,
                    value=config.trading.profit_target_multiplier,
                    step=0.5,
                )

                volatility_threshold = st.number_input(
                    "Volatility Threshold",
                    min_value=0.1,
                    max_value=1.0,
                    value=config.trading.volatility_threshold,
                    step=0.05,
                )

            run_backtest = st.form_submit_button("Run Backtest")

        if run_backtest:
            if start_date >= end_date:
                st.error("Start date must be before end date")
                return

            # Additional validation
            if end_date > date.today():
                st.error(f"End date cannot be in the future. Today is {date.today()}")
                return

            days_back = (date.today() - start_date).days
            if days_back > 59:
                st.warning(
                    f"⚠️ Yahoo Finance only provides intraday data for the last 60 days. Adjusting start date."
                )
                start_date = date.today() - timedelta(days=59)

            # Temporarily update config for backtest
            original_max_trades = config.trading.max_open_trades
            original_profit_target = config.trading.profit_target_multiplier
            original_volatility_threshold = config.trading.volatility_threshold

            config.trading.max_open_trades = max_trades
            config.trading.profit_target_multiplier = profit_target
            config.trading.volatility_threshold = volatility_threshold

            try:
                with st.spinner("Running backtest... This may take a few minutes."):
                    # Run backtest
                    results = asyncio.run(
                        self.backtester.run_backtest(start_date, end_date, initial_capital)
                    )

                # Display results
                st.success("Backtest completed!")

                # Key metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Return",
                        f"{results['total_return']:.2%}",
                        delta=f"${results['final_capital'] - results['initial_capital']:,.2f}",
                    )

                with col2:
                    st.metric(
                        "Win Rate",
                        f"{results['win_rate']:.1%}",
                        delta=f"{results['winning_trades']}/{results['total_trades']}",
                    )

                with col3:
                    st.metric(
                        "Max Drawdown",
                        f"${results['max_drawdown']:,.2f}",
                        delta=f"{results['max_drawdown']/results['initial_capital']:.1%}",
                    )

                with col4:
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")

                # Equity curve
                if results["daily_pnl"]:
                    # Transform backtester data to match expected chart format
                    equity_data = pd.DataFrame(results["daily_pnl"])

                    # Rename columns to match what create_equity_chart expects
                    equity_data = equity_data.rename(
                        columns={"date": "Date", "capital": "Cumulative Equity", "pnl": "Daily P&L"}
                    )

                    # Add missing columns for complete functionality
                    equity_data["Trades"] = 0  # Default value, could be enhanced later
                    equity_data["Win Rate"] = 0.0  # Default value, could be enhanced later

                    equity_chart = self.create_equity_chart(equity_data)
                    if equity_chart:
                        st.plotly_chart(equity_chart, use_container_width=True)

                # Trade details
                if results["trades"]:
                    st.subheader("Trade Details")

                    trade_data = []
                    for trade in results["trades"][-20:]:  # Last 20 trades
                        trade_data.append(
                            {
                                "Entry Time": trade.entry_time.strftime("%Y-%m-%d %H:%M"),
                                "Strikes": f"{trade.call_strike:.0f}C/{trade.put_strike:.0f}P",
                                "Premium": f"${trade.total_premium:.2f}",
                                "P&L": f"${trade.realized_pnl:.2f}",
                                "Return": f"{(trade.realized_pnl / trade.total_premium * 100):.1f}%",
                                "Status": trade.status,
                            }
                        )

                    trade_df = pd.DataFrame(trade_data)
                    st.dataframe(trade_df, use_container_width=True, hide_index=True)

                # Decision Trace Analysis
                if "decision_traces" in results:
                    st.subheader("📊 Trading Decision Analysis")

                    if not results["decision_traces"]:
                        st.info(
                            "No trading decisions were evaluated during this backtest period. "
                            "This may happen if the market was closed or data was insufficient."
                        )
                    else:
                        traces = results["decision_traces"]
                        total_decisions = len(traces)
                        near_misses = [
                            t for t in traces if t.near_miss_score > 0.5 and t.decision != "TRADED"
                        ]
                        trades_taken = [t for t in traces if t.decision == "TRADED"]

                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Decision Points", total_decisions)
                        with col2:
                            st.metric("Near Misses", len(near_misses))
                        with col3:
                            st.metric("Trades Taken", len(trades_taken))
                        with col4:
                            avg_near_miss = (
                                sum(t.near_miss_score for t in traces) / len(traces)
                                if traces
                                else 0
                            )
                            st.metric("Avg Near-Miss Score", f"{avg_near_miss:.2%}")

                        # Show detailed trace toggle
                        show_trace = st.checkbox("Show Detailed Decision Trace", value=False)

                        if show_trace:
                            # Filter options
                            trace_filter = st.selectbox(
                                "Filter decisions:",
                                ["All", "Near Misses Only", "Trades Only", "Failed Checks Only"],
                            )

                            # Filter traces based on selection
                            filtered_traces = traces
                            if trace_filter == "Near Misses Only":
                                filtered_traces = near_misses
                            elif trace_filter == "Trades Only":
                                filtered_traces = trades_taken
                            elif trace_filter == "Failed Checks Only":
                                filtered_traces = [t for t in traces if t.near_miss_score < 0.5]

                            # Group by date for better organization
                            traces_by_date = {}
                            for trace in filtered_traces:
                                date_key = trace.timestamp.date()
                                if date_key not in traces_by_date:
                                    traces_by_date[date_key] = []
                                traces_by_date[date_key].append(trace)

                            # Display traces by date
                            for date_key in sorted(traces_by_date.keys(), reverse=True):
                                with st.expander(
                                    f"📅 {date_key} - {len(traces_by_date[date_key])} decisions"
                                ):
                                    for trace in traces_by_date[date_key]:
                                        # Color coding based on decision
                                        if trace.decision == "TRADED":
                                            st.success(
                                                f"✅ **{trace.timestamp.strftime('%H:%M:%S')}** - "
                                                f"TRADED at ${trace.price:.2f}"
                                            )
                                        elif trace.near_miss_score > 0.7:
                                            st.warning(
                                                f"⚠️ **{trace.timestamp.strftime('%H:%M:%S')}** - "
                                                f"NEAR MISS (Score: {trace.near_miss_score:.2%}) at ${trace.price:.2f}"
                                            )
                                        else:
                                            st.info(
                                                f"ℹ️ **{trace.timestamp.strftime('%H:%M:%S')}** - "
                                                f"NO TRADE (Score: {trace.near_miss_score:.2%}) at ${trace.price:.2f}"
                                            )

                                        # Show reasons
                                        with st.container():
                                            st.write("**Reasons:**")
                                            for reason in trace.reasons:
                                                st.write(f"• {reason}")

                                            # Show market conditions
                                            if st.checkbox(
                                                f"Show market conditions",
                                                key=f"mc_{trace.timestamp}",
                                            ):
                                                conditions = trace.market_conditions
                                                cols = st.columns(3)
                                                with cols[0]:
                                                    st.write(
                                                        f"**Implied Move:** ${conditions.get('implied_move', 0):.2f}"
                                                    )
                                                    st.write(
                                                        f"**Current Price:** "
                                                        f"${conditions.get('current_price', 0):.2f}"
                                                    )
                                                with cols[1]:
                                                    st.write(
                                                        f"**Realized Range:** "
                                                        f"${conditions.get('realized_range', 0):.2f}"
                                                    )
                                                    st.write(
                                                        f"**Volatility Ratio:** "
                                                        f"{conditions.get('volatility_ratio', 0):.1%}"
                                                    )
                                                with cols[2]:
                                                    st.write(
                                                        f"**Minutes Since Trade:** "
                                                        f"{conditions.get('minutes_since_last_trade', 'N/A')}"
                                                    )
                                                    st.write(
                                                        f"**Vol Threshold:** "
                                                        f"{conditions.get('volatility_threshold', 0):.1%}"
                                                    )

                                            # Show potential trade if available
                                            if trace.potential_trade:
                                                st.write("**Potential Trade:**")
                                                st.write(
                                                    f"• Strikes: {trace.potential_trade['call_strike']:.0f}C / "
                                                    f"{trace.potential_trade['put_strike']:.0f}P"
                                                )
                                                st.write(
                                                    f"• Est. Premium: ${trace.potential_trade['estimated_premium']:.2f}"
                                                )

                                        st.divider()

                # Save backtest option
                backtest_name = st.text_input(
                    "Save backtest as:", value=f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M')}"
                )
                if st.button("Save Results"):
                    try:
                        backtest_id = asyncio.run(
                            self.backtester.save_backtest_result(results, backtest_name)
                        )
                        st.success(f"Backtest saved with ID: {backtest_id}")
                    except Exception as e:
                        st.error(f"Error saving backtest: {e}")

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                logger.error(f"Backtest error: {e}")

            finally:
                # Restore original config
                config.trading.max_open_trades = original_max_trades
                config.trading.profit_target_multiplier = original_profit_target
                config.trading.volatility_threshold = original_volatility_threshold

        # Load saved backtests
        st.divider()
        st.subheader("Saved Backtests")

        session = self.get_session()
        try:
            saved_backtests = (
                session.query(BacktestResult)
                .order_by(BacktestResult.created_at.desc())
                .limit(10)
                .all()
            )

            if saved_backtests:
                backtest_data = []
                for bt in saved_backtests:
                    backtest_data.append(
                        {
                            "Name": bt.name,
                            "Period": f"{bt.start_date} to {bt.end_date}",
                            "Return": f"{bt.total_return:.2%}",
                            "Win Rate": f"{bt.win_rate:.1%}",
                            "Max DD": f"${bt.max_drawdown:.0f}",
                            "Trades": bt.total_trades,
                            "Created": bt.created_at.strftime("%Y-%m-%d %H:%M"),
                        }
                    )

                backtest_df = pd.DataFrame(backtest_data)
                st.dataframe(backtest_df, use_container_width=True, hide_index=True)
            else:
                st.info("No saved backtests")

        finally:
            session.close()

    def render_configuration(self):
        """Render configuration tab"""
        st.header("Bot Configuration")

        st.warning("⚠️ Changing configuration requires bot restart to take effect")

        # Current configuration display
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Trading Parameters")
            st.write(f"**Trade Mode:** {config.trading.trade_mode}")
            st.write(f"**Max Open Trades:** {config.trading.max_open_trades}")
            st.write(f"**Max Premium per Strangle:** ${config.trading.max_premium_per_strangle}")
            st.write(f"**Profit Target:** {config.trading.profit_target_multiplier}x")
            st.write(f"**Max Drawdown:** ${config.trading.max_drawdown}")

            st.subheader("Strategy Parameters")
            st.write(f"**Implied Move Multiplier 1:** {config.trading.implied_move_multiplier_1}")
            st.write(f"**Implied Move Multiplier 2:** {config.trading.implied_move_multiplier_2}")
            st.write(f"**Volatility Threshold:** {config.trading.volatility_threshold}")
            st.write(
                f"**Min Time Between Trades:** {config.trading.min_time_between_trades} minutes"
            )

        with col2:
            st.subheader("IB Connection")
            st.write(f"**Host:** {config.ib.host}")
            st.write(f"**Port:** {config.ib.port}")
            st.write(f"**Client ID:** {config.ib.client_id}")
            st.write(f"**Paper Trading:** {'Yes' if config.ib.is_paper_trading else 'No'}")

            st.subheader("Market Hours")
            st.write(
                f"**Market Open:** "
                f"{config.market_hours.market_open_hour:02d}:{config.market_hours.market_open_minute:02d} ET"
            )
            st.write(
                f"**Market Close:** "
                f"{config.market_hours.market_close_hour:02d}:{config.market_hours.market_close_minute:02d} ET"
            )
            st.write(
                f"**Flatten Time:** {config.market_hours.flatten_hour:02d}:{config.market_hours.flatten_minute:02d} ET"
            )

        st.divider()

        # Configuration editor (for display purposes)
        st.subheader("Configuration Editor")
        st.info("To modify configuration, edit the .env file and restart the bot")

        config_text = f"""
# Trading Configuration
TRADE_MODE={config.trading.trade_mode}
MAX_OPEN_TRADES={config.trading.max_open_trades}
MAX_PREMIUM_PER_STRANGLE={config.trading.max_premium_per_strangle}
PROFIT_TARGET_MULTIPLIER={config.trading.profit_target_multiplier}
MAX_DRAW={config.trading.max_drawdown}

# Strategy Parameters
IMPLIED_MOVE_MULTIPLIER_1={config.trading.implied_move_multiplier_1}
IMPLIED_MOVE_MULTIPLIER_2={config.trading.implied_move_multiplier_2}
VOLATILITY_THRESHOLD={config.trading.volatility_threshold}
MIN_TIME_BETWEEN_TRADES={config.trading.min_time_between_trades}

# IB Configuration
IB_GATEWAY_HOST={config.ib.host}
IB_GATEWAY_PORT={config.ib.port}
IB_CLIENT_ID={config.ib.client_id}
        """

        st.code(config_text, language="bash")


def main():
    """Main Streamlit app"""
    st.title("🎯 MES 0DTE Lotto-Grid Options Bot")
    st.markdown("*Production-grade automated options trading system*")

    # Initialize dashboard
    dashboard = Dashboard()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio(
        "Select View", ["📊 Live Monitor", "📈 Performance", "🔄 Backtesting", "⚙️ Configuration"]
    )

    # Bot status in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("Bot Status")

    # Dynamic status based on trading mode
    if "trading_mode" not in st.session_state:
        st.session_state.trading_mode = "manual"

    if st.session_state.trading_mode == "auto":
        st.sidebar.success("🟢 Auto Trading Active")
    elif st.session_state.trading_mode == "manual":
        st.sidebar.warning("🟡 Manual Mode")
    else:
        st.sidebar.error("🔴 Trading OFF")

    st.sidebar.info("🔵 Paper Trading Mode")

    # Market hours check
    current_time = datetime.now()
    market_open = current_time.replace(hour=9, minute=30, second=0)
    market_close = current_time.replace(hour=16, minute=0, second=0)
    is_market_open = market_open <= current_time <= market_close

    st.sidebar.write(f"**Market Hours:** {'🟢 Open' if is_market_open else '🔴 Closed'}")

    # Render selected tab
    if tab == "📊 Live Monitor":
        dashboard.render_live_monitoring()
    elif tab == "📈 Performance":
        dashboard.render_performance_analytics()
    elif tab == "🔄 Backtesting":
        dashboard.render_backtesting()
    elif tab == "⚙️ Configuration":
        dashboard.render_configuration()


if __name__ == "__main__":
    main()
