"""
Intelligent Alert System for Risk Management

This module provides multi-channel alerting with smart prioritization,
dynamic thresholds, and automated mitigation suggestions.
"""

import os
import json
import asyncio
import smtplib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of risk alerts"""
    VAR_BREACH = "var_breach"
    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_BREACH = "drawdown_breach"
    CORRELATION_HIGH = "correlation_high"
    VOLATILITY_SPIKE = "volatility_spike"
    REGIME_CHANGE = "regime_change"
    POSITION_LIMIT = "position_limit"
    MARGIN_CALL = "margin_call"
    PATTERN_DETECTED = "pattern_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    SYSTEM_ERROR = "system_error"


@dataclass
class Alert:
    """Container for alert information"""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    suggested_actions: List[str] = None
    metadata: Dict[str, Any] = None
    channels: List[str] = None


class AlertChannel:
    """Base class for alert channels"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)

    async def send(self, alert: Alert) -> bool:
        """Send alert through this channel"""
        raise NotImplementedError


class DiscordChannel(AlertChannel):
    """Discord webhook alert channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get('webhook_url')

    async def send(self, alert: Alert) -> bool:
        """Send alert to Discord"""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            import aiohttp

            # Format Discord embed
            embed = {
                "title": f"{self._get_emoji(alert.severity)} {alert.title}",
                "description": alert.message,
                "color": self._get_color(alert.severity),
                "fields": [],
                "timestamp": alert.timestamp.isoformat(),
                "footer": {"text": "MES Risk Analytics"}
            }

            # Add metric fields
            if alert.metric_value is not None:
                embed["fields"].append({
                    "name": "Current Value",
                    "value": f"{alert.metric_value:.2f}",
                    "inline": True
                })

            if alert.threshold_value is not None:
                embed["fields"].append({
                    "name": "Threshold",
                    "value": f"{alert.threshold_value:.2f}",
                    "inline": True
                })

            # Add suggested actions
            if alert.suggested_actions:
                actions_text = "\n".join(f"• {action}" for action in alert.suggested_actions)
                embed["fields"].append({
                    "name": "Recommended Actions",
                    "value": actions_text,
                    "inline": False
                })

            payload = {
                "embeds": [embed],
                "username": "Risk Alert Bot"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    return response.status == 204

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    def _get_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for severity level"""
        return {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🆘"
        }.get(severity, "📢")

    def _get_color(self, severity: AlertSeverity) -> int:
        """Get color code for severity"""
        return {
            AlertSeverity.INFO: 0x3498db,  # Blue
            AlertSeverity.WARNING: 0xf39c12,  # Orange
            AlertSeverity.CRITICAL: 0xe74c3c,  # Red
            AlertSeverity.EMERGENCY: 0x9b59b6  # Purple
        }.get(severity, 0x95a5a6)  # Gray default


class TelegramChannel(AlertChannel):
    """Telegram bot alert channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bot_token = config.get('bot_token')
        self.chat_id = config.get('chat_id')

    async def send(self, alert: Alert) -> bool:
        """Send alert to Telegram"""
        if not self.enabled or not self.bot_token or not self.chat_id:
            return False

        try:
            import aiohttp

            # Format message with Markdown
            message = f"*{self._get_emoji(alert.severity)} {alert.title}*\n\n"
            message += f"{alert.message}\n\n"

            if alert.metric_value is not None:
                message += f"📊 *Current:* {alert.metric_value:.2f}\n"
            if alert.threshold_value is not None:
                message += f"🎯 *Threshold:* {alert.threshold_value:.2f}\n"

            if alert.suggested_actions:
                message += "\n*Recommended Actions:*\n"
                for action in alert.suggested_actions:
                    message += f"• {action}\n"

            message += f"\n_Alert ID: {alert.alert_id}_"

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    def _get_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for severity level"""
        return {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🆘"
        }.get(severity, "📢")


class EmailChannel(AlertChannel):
    """Email alert channel"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.sender_email = config.get('sender_email')
        self.sender_password = config.get('sender_password')
        self.recipient_emails = config.get('recipient_emails', [])

    async def send(self, alert: Alert) -> bool:
        """Send alert via email"""
        if not self.enabled or not self.sender_email or not self.recipient_emails:
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)

            # Create HTML content
            html = f"""
            <html>
                <body style="font-family: Arial, sans-serif;">
                    <h2 style="color: {self._get_color(alert.severity)};">
                        {alert.title}
                    </h2>
                    <p>{alert.message}</p>

                    <table style="border-collapse: collapse; margin: 20px 0;">
                        <tr>
                            <td style="padding: 5px;"><strong>Alert Type:</strong></td>
                            <td style="padding: 5px;">{alert.alert_type.value}</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px;"><strong>Severity:</strong></td>
                            <td style="padding: 5px;">{alert.severity.value}</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px;"><strong>Timestamp:</strong></td>
                            <td style="padding: 5px;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                        </tr>
            """

            if alert.metric_value is not None:
                html += f"""
                        <tr>
                            <td style="padding: 5px;"><strong>Current Value:</strong></td>
                            <td style="padding: 5px;">{alert.metric_value:.2f}</td>
                        </tr>
                """

            if alert.threshold_value is not None:
                html += f"""
                        <tr>
                            <td style="padding: 5px;"><strong>Threshold:</strong></td>
                            <td style="padding: 5px;">{alert.threshold_value:.2f}</td>
                        </tr>
                """

            html += "</table>"

            if alert.suggested_actions:
                html += "<h3>Recommended Actions:</h3><ul>"
                for action in alert.suggested_actions:
                    html += f"<li>{action}</li>"
                html += "</ul>"

            html += f"""
                    <hr>
                    <p style="color: #666; font-size: 12px;">
                        Alert ID: {alert.alert_id}<br>
                        MES Risk Analytics System
                    </p>
                </body>
            </html>
            """

            msg.attach(MIMEText(html, 'html'))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _get_color(self, severity: AlertSeverity) -> str:
        """Get color for severity"""
        return {
            AlertSeverity.INFO: "#3498db",
            AlertSeverity.WARNING: "#f39c12",
            AlertSeverity.CRITICAL: "#e74c3c",
            AlertSeverity.EMERGENCY: "#9b59b6"
        }.get(severity, "#95a5a6")


class RiskAlertSystem:
    """
    Multi-channel alert system with smart prioritization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.channels = self._initialize_channels()
        self.alert_history = []
        self.alert_cooldowns = {}  # Prevent alert spam
        self.dynamic_thresholds = {}

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'channels': {
                'discord': {
                    'enabled': os.getenv('DISCORD_ALERTS_ENABLED', 'false').lower() == 'true',
                    'webhook_url': os.getenv('DISCORD_WEBHOOK_URL')
                },
                'telegram': {
                    'enabled': os.getenv('TELEGRAM_ALERTS_ENABLED', 'false').lower() == 'true',
                    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                    'chat_id': os.getenv('TELEGRAM_CHAT_ID')
                },
                'email': {
                    'enabled': os.getenv('EMAIL_ALERTS_ENABLED', 'false').lower() == 'true',
                    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                    'sender_email': os.getenv('ALERT_EMAIL_FROM'),
                    'sender_password': os.getenv('ALERT_EMAIL_PASSWORD'),
                    'recipient_emails': os.getenv('ALERT_EMAIL_TO', '').split(',')
                }
            },
            'thresholds': {
                'var_breach': 0.95,
                'drawdown_warning': 0.7,
                'drawdown_breach': 0.9,
                'correlation_high': 0.8,
                'volatility_spike': 2.0,
                'risk_score_warning': 60,
                'risk_score_critical': 80
            },
            'cooldown_minutes': {
                AlertSeverity.INFO: 30,
                AlertSeverity.WARNING: 15,
                AlertSeverity.CRITICAL: 5,
                AlertSeverity.EMERGENCY: 0
            }
        }

    def _initialize_channels(self) -> Dict[str, AlertChannel]:
        """Initialize alert channels"""
        channels = {}

        if self.config['channels']['discord']['enabled']:
            channels['discord'] = DiscordChannel(self.config['channels']['discord'])

        if self.config['channels']['telegram']['enabled']:
            channels['telegram'] = TelegramChannel(self.config['channels']['telegram'])

        if self.config['channels']['email']['enabled']:
            channels['email'] = EmailChannel(self.config['channels']['email'])

        return channels

    def configure_thresholds(self, thresholds: Dict[str, float]):
        """
        Dynamic threshold adjustment based on market conditions

        Args:
            thresholds: New threshold values
        """
        self.config['thresholds'].update(thresholds)
        logger.info(f"Updated alert thresholds: {thresholds}")

    def adjust_thresholds_for_regime(self, regime: str):
        """
        Adjust thresholds based on market regime

        Args:
            regime: Current market regime
        """
        adjustments = {
            'crisis': {
                'var_breach': 0.90,
                'drawdown_warning': 0.5,
                'volatility_spike': 1.5,
                'risk_score_warning': 40,
                'risk_score_critical': 60
            },
            'volatile': {
                'var_breach': 0.92,
                'drawdown_warning': 0.6,
                'volatility_spike': 1.75,
                'risk_score_warning': 50,
                'risk_score_critical': 70
            },
            'normal': {
                'var_breach': 0.95,
                'drawdown_warning': 0.7,
                'volatility_spike': 2.0,
                'risk_score_warning': 60,
                'risk_score_critical': 80
            },
            'quiet': {
                'var_breach': 0.97,
                'drawdown_warning': 0.8,
                'volatility_spike': 2.5,
                'risk_score_warning': 70,
                'risk_score_critical': 90
            }
        }

        if regime in adjustments:
            self.configure_thresholds(adjustments[regime])

    async def send_alert(self, alert: Alert) -> bool:
        """
        Send alert through configured channels

        Args:
            alert: Alert to send

        Returns:
            Success status
        """
        # Check cooldown
        if not self._check_cooldown(alert):
            logger.debug(f"Alert {alert.alert_type} still in cooldown period")
            return False

        # Prioritize alert
        priority_channels = self.prioritize_alert(alert)

        # Send through channels
        success = False
        for channel_name in priority_channels:
            if channel_name in self.channels:
                channel_success = await self.channels[channel_name].send(alert)
                success = success or channel_success

        # Update history and cooldown
        if success:
            self.alert_history.append(alert)
            self._update_cooldown(alert)

        return success

    def _check_cooldown(self, alert: Alert) -> bool:
        """Check if alert is in cooldown period"""
        key = f"{alert.alert_type.value}_{alert.severity.value}"

        if key in self.alert_cooldowns:
            last_sent = self.alert_cooldowns[key]
            cooldown_minutes = self.config['cooldown_minutes'].get(alert.severity, 15)

            if datetime.now() - last_sent < timedelta(minutes=cooldown_minutes):
                return False

        return True

    def _update_cooldown(self, alert: Alert):
        """Update cooldown timestamp"""
        key = f"{alert.alert_type.value}_{alert.severity.value}"
        self.alert_cooldowns[key] = datetime.now()

    def prioritize_alert(self, alert: Alert) -> List[str]:
        """
        ML-based alert prioritization

        Args:
            alert: Alert to prioritize

        Returns:
            Ordered list of channels to use
        """
        # Default channel priority based on severity
        if alert.severity == AlertSeverity.EMERGENCY:
            return ['discord', 'telegram', 'email']  # All channels
        elif alert.severity == AlertSeverity.CRITICAL:
            return ['discord', 'telegram']  # Fast channels
        elif alert.severity == AlertSeverity.WARNING:
            return ['discord']  # Primary channel
        else:
            return ['email']  # Low priority channel

    def suggest_mitigation(self, alert_type: AlertType,
                          metrics: Dict[str, float]) -> List[str]:
        """
        Automated mitigation strategy suggestions

        Args:
            alert_type: Type of alert
            metrics: Current risk metrics

        Returns:
            List of suggested actions
        """
        suggestions = []

        if alert_type == AlertType.VAR_BREACH:
            suggestions.append("Close highest risk positions immediately")
            suggestions.append("Reduce overall position size by 50%")
            suggestions.append("Implement tighter stop losses")

        elif alert_type == AlertType.DRAWDOWN_WARNING:
            pct_to_limit = metrics.get('drawdown_pct_to_limit', 0)
            if pct_to_limit > 0.9:
                suggestions.append("URGENT: Close all positions to prevent breach")
            elif pct_to_limit > 0.8:
                suggestions.append("Close losing positions immediately")
            else:
                suggestions.append("Avoid new positions until recovery")

        elif alert_type == AlertType.VOLATILITY_SPIKE:
            suggestions.append("Widen stop losses to avoid premature exits")
            suggestions.append("Reduce position sizes to account for increased volatility")
            suggestions.append("Consider hedging with opposite positions")

        elif alert_type == AlertType.CORRELATION_HIGH:
            suggestions.append("Diversify into uncorrelated assets")
            suggestions.append("Reduce correlated position sizes")
            suggestions.append("Consider market-neutral strategies")

        elif alert_type == AlertType.REGIME_CHANGE:
            new_regime = metrics.get('new_regime', 'unknown')
            if new_regime == 'crisis':
                suggestions.append("Switch to defensive mode immediately")
                suggestions.append("Close all speculative positions")
            elif new_regime == 'volatile':
                suggestions.append("Reduce leverage and position sizes")
                suggestions.append("Increase cash reserves")

        elif alert_type == AlertType.MARGIN_CALL:
            suggestions.append("CRITICAL: Add funds or close positions immediately")
            suggestions.append("Review margin requirements for all positions")

        return suggestions

    def create_alert(self, alert_type: AlertType, severity: AlertSeverity,
                    title: str, message: str, **kwargs) -> Alert:
        """
        Factory method to create alerts

        Args:
            alert_type: Type of alert
            severity: Severity level
            title: Alert title
            message: Alert message
            **kwargs: Additional alert parameters

        Returns:
            Alert object
        """
        import uuid

        alert = Alert(
            alert_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            metric_value=kwargs.get('metric_value'),
            threshold_value=kwargs.get('threshold_value'),
            suggested_actions=kwargs.get('suggested_actions', []),
            metadata=kwargs.get('metadata', {}),
            channels=kwargs.get('channels')
        )

        # Add mitigation suggestions if not provided
        if not alert.suggested_actions:
            metrics = kwargs.get('metrics', {})
            alert.suggested_actions = self.suggest_mitigation(alert_type, metrics)

        return alert
