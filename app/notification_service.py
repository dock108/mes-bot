"""
Notification Service for Critical Trading Events
"""

import asyncio
import json
import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Dict, List, Optional, Set

import aiohttp
import requests
from jinja2 import Template

from app.config import config

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification priority levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Available notification channels"""

    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class NotificationConfig:
    """Configuration for notification channels"""

    # Email settings
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    from_email: str = ""
    to_emails: List[str] = None

    # SMS settings (Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_to_numbers: List[str] = None

    # Slack settings
    slack_webhook_url: str = ""
    slack_channel: str = "#trading-alerts"

    # Webhook settings
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = None

    # Rate limiting
    rate_limit_minutes: int = 5
    max_notifications_per_period: int = 10

    # Channel configuration by level
    channel_config: Dict[NotificationLevel, List[NotificationChannel]] = None

    def __post_init__(self):
        if self.to_emails is None:
            self.to_emails = []
        if self.twilio_to_numbers is None:
            self.twilio_to_numbers = []
        if self.webhook_headers is None:
            self.webhook_headers = {}
        if self.channel_config is None:
            # Default channel configuration
            self.channel_config = {
                NotificationLevel.INFO: [NotificationChannel.CONSOLE],
                NotificationLevel.WARNING: [NotificationChannel.CONSOLE, NotificationChannel.EMAIL],
                NotificationLevel.ERROR: [
                    NotificationChannel.CONSOLE,
                    NotificationChannel.EMAIL,
                    NotificationChannel.SLACK,
                ],
                NotificationLevel.CRITICAL: [
                    NotificationChannel.CONSOLE,
                    NotificationChannel.EMAIL,
                    NotificationChannel.SMS,
                    NotificationChannel.SLACK,
                ],
            }


@dataclass
class NotificationMessage:
    """Container for notification message"""

    level: NotificationLevel
    title: str
    message: str
    timestamp: datetime
    context: Dict = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


class NotificationService:
    """Service for sending notifications through multiple channels"""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.rate_limiter = {}  # Track notifications per channel
        self.notification_history = []
        self.enabled_channels = self._determine_enabled_channels()

        # Email templates
        self.email_templates = {
            "trade_alert": self._get_trade_alert_template(),
            "error_alert": self._get_error_alert_template(),
            "performance_alert": self._get_performance_alert_template(),
            "system_alert": self._get_system_alert_template(),
        }

        logger.info(f"Notification service initialized with channels: {self.enabled_channels}")

    def _determine_enabled_channels(self) -> Set[NotificationChannel]:
        """Determine which notification channels are properly configured"""
        enabled = set()

        # Check email configuration
        if self.config.smtp_user and self.config.smtp_password and self.config.to_emails:
            enabled.add(NotificationChannel.EMAIL)

        # Check SMS configuration
        if (
            self.config.twilio_account_sid
            and self.config.twilio_auth_token
            and self.config.twilio_from_number
            and self.config.twilio_to_numbers
        ):
            enabled.add(NotificationChannel.SMS)

        # Check Slack configuration
        if self.config.slack_webhook_url:
            enabled.add(NotificationChannel.SLACK)

        # Check webhook configuration
        if self.config.webhook_url:
            enabled.add(NotificationChannel.WEBHOOK)

        # Console is always available
        enabled.add(NotificationChannel.CONSOLE)

        return enabled

    def _mask_phone_number(self, phone_number: str) -> str:
        """Return a masked version of the phone number, e.g., ********1234"""
        # Only show last 4 digits (or fewer if number is short)
        num_digits = 4
        if not phone_number or len(phone_number) <= num_digits:
            return "*" * (len(phone_number) if phone_number else 0)
        return "*" * (len(phone_number) - num_digits) + phone_number[-num_digits:]

    async def send_notification(self, notification: NotificationMessage) -> bool:
        """Send notification through appropriate channels"""
        try:
            # Check rate limits
            if not self._check_rate_limit(notification.level):
                logger.warning(f"Rate limit exceeded for {notification.level.value} notifications")
                return False

            # Get channels for this notification level
            channels = self.config.channel_config.get(notification.level, [])
            channels = [ch for ch in channels if ch in self.enabled_channels]

            if not channels:
                logger.warning(f"No enabled channels for {notification.level.value} notifications")
                return False

            # Send to all configured channels
            results = []
            for channel in channels:
                try:
                    if channel == NotificationChannel.CONSOLE:
                        result = await self._send_console_notification(notification)
                    elif channel == NotificationChannel.EMAIL:
                        result = await self._send_email_notification(notification)
                    elif channel == NotificationChannel.SMS:
                        result = await self._send_sms_notification(notification)
                    elif channel == NotificationChannel.SLACK:
                        result = await self._send_slack_notification(notification)
                    elif channel == NotificationChannel.WEBHOOK:
                        result = await self._send_webhook_notification(notification)
                    else:
                        result = False

                    results.append(result)

                except Exception as e:
                    logger.error(f"Error sending {channel.value} notification: {e}")
                    results.append(False)

            # Record notification
            self.notification_history.append(
                {
                    "timestamp": notification.timestamp,
                    "level": notification.level.value,
                    "title": notification.title,
                    "channels": [ch.value for ch in channels],
                    "success": any(results),
                }
            )

            # Update rate limiter
            self._update_rate_limit(notification.level)

            return any(results)

        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False

    async def _send_console_notification(self, notification: NotificationMessage) -> bool:
        """Send console notification"""
        try:
            level_colors = {
                NotificationLevel.INFO: "\033[36m",  # Cyan
                NotificationLevel.WARNING: "\033[33m",  # Yellow
                NotificationLevel.ERROR: "\033[31m",  # Red
                NotificationLevel.CRITICAL: "\033[91m",  # Bright Red
            }

            reset_color = "\033[0m"
            color = level_colors.get(notification.level, "")

            message = (
                f"{color}[{notification.level.value.upper()}] {notification.title}{reset_color}"
            )
            print(f"🚨 {message}")
            print(f"   {notification.message}")
            if notification.context:
                print(f"   Context: {json.dumps(notification.context, indent=2)}")

            return True

        except Exception as e:
            logger.error(f"Error sending console notification: {e}")
            return False

    async def _send_email_notification(self, notification: NotificationMessage) -> bool:
        """Send email notification"""
        try:
            if not self.config.to_emails:
                return False

            # Determine email template
            template_name = self._determine_email_template(notification)
            template = self.email_templates.get(template_name, self.email_templates["system_alert"])

            # Render email content
            html_content = template.render(
                title=notification.title,
                message=notification.message,
                level=notification.level.value,
                timestamp=notification.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                context=notification.context or {},
            )

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{notification.level.value.upper()}] {notification.title}"
            msg["From"] = self.config.from_email or self.config.smtp_user
            msg["To"] = ", ".join(self.config.to_emails)

            # Add HTML part
            html_part = MIMEText(html_content, "html")
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)

            logger.debug(f"Email notification sent to {len(self.config.to_emails)} recipients")
            return True

        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False

    async def _send_sms_notification(self, notification: NotificationMessage) -> bool:
        """Send SMS notification using Twilio"""
        try:
            if not self.config.twilio_to_numbers:
                return False

            # Prepare SMS message (keep it short)
            sms_message = f"[{notification.level.value.upper()}] {notification.title}\n{notification.message[:100]}..."

            # Send to each number
            success_count = 0
            for phone_number in self.config.twilio_to_numbers:
                try:
                    # Use Twilio API
                    url = f"https://api.twilio.com/2010-04-01/Accounts/{self.config.twilio_account_sid}/Messages.json"

                    data = {
                        "From": self.config.twilio_from_number,
                        "To": phone_number,
                        "Body": sms_message,
                    }

                    response = requests.post(
                        url,
                        data=data,
                        auth=(self.config.twilio_account_sid, self.config.twilio_auth_token),
                    )

                    if response.status_code == 201:
                        success_count += 1
                    else:
                        masked_phone = self._mask_phone_number(phone_number)
                        logger.error(
                            f"Failed to send SMS to {masked_phone}: Status code {response.status_code}"
                        )

                except Exception as e:
                    masked_phone = self._mask_phone_number(phone_number)
                    logger.error(f"Error sending SMS to {masked_phone}: {e}")

            return success_count > 0

        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
            return False

    async def _send_slack_notification(self, notification: NotificationMessage) -> bool:
        """Send Slack notification"""
        try:
            if not self.config.slack_webhook_url:
                return False

            # Prepare Slack message
            color_map = {
                NotificationLevel.INFO: "good",
                NotificationLevel.WARNING: "warning",
                NotificationLevel.ERROR: "danger",
                NotificationLevel.CRITICAL: "danger",
            }

            slack_payload = {
                "channel": self.config.slack_channel,
                "username": "Trading Bot",
                "icon_emoji": ":robot_face:",
                "attachments": [
                    {
                        "color": color_map.get(notification.level, "warning"),
                        "title": f"[{notification.level.value.upper()}] {notification.title}",
                        "text": notification.message,
                        "timestamp": int(notification.timestamp.timestamp()),
                        "fields": (
                            [
                                {
                                    "title": "Context",
                                    "value": json.dumps(notification.context or {}, indent=2),
                                    "short": False,
                                }
                            ]
                            if notification.context
                            else []
                        ),
                    }
                ],
            }

            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.slack_webhook_url,
                    json=slack_payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        logger.debug("Slack notification sent successfully")
                        return True
                    else:
                        logger.error(f"Failed to send Slack notification: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False

    async def _send_webhook_notification(self, notification: NotificationMessage) -> bool:
        """Send webhook notification"""
        try:
            if not self.config.webhook_url:
                return False

            # Prepare webhook payload
            webhook_payload = {
                "level": notification.level.value,
                "title": notification.title,
                "message": notification.message,
                "timestamp": notification.timestamp.isoformat(),
                "context": notification.context or {},
            }

            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=webhook_payload,
                    headers=self.config.webhook_headers,
                ) as response:
                    if response.status in [200, 201, 202]:
                        logger.debug("Webhook notification sent successfully")
                        return True
                    else:
                        logger.error(f"Failed to send webhook notification: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False

    def _check_rate_limit(self, level: NotificationLevel) -> bool:
        """Check if notification level is within rate limits"""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(minutes=self.config.rate_limit_minutes)

        # Count recent notifications for this level
        recent_count = sum(
            1
            for notif in self.notification_history
            if (
                notif["level"] == level.value
                and datetime.fromisoformat(notif["timestamp"]) > cutoff_time
            )
        )

        return recent_count < self.config.max_notifications_per_period

    def _update_rate_limit(self, level: NotificationLevel):
        """Update rate limiter after sending notification"""
        # Clean up old entries
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(minutes=self.config.rate_limit_minutes * 2)

        self.notification_history = [
            notif
            for notif in self.notification_history
            if datetime.fromisoformat(notif["timestamp"]) > cutoff_time
        ]

    def _determine_email_template(self, notification: NotificationMessage) -> str:
        """Determine which email template to use"""
        if "trade" in notification.title.lower():
            return "trade_alert"
        elif "error" in notification.title.lower() or notification.level == NotificationLevel.ERROR:
            return "error_alert"
        elif "performance" in notification.title.lower():
            return "performance_alert"
        else:
            return "system_alert"

    def _get_trade_alert_template(self) -> Template:
        """Get trade alert email template"""
        template_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f8ff; padding: 20px; border-radius: 5px; }
                .content { padding: 20px; }
                .context { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-top: 20px; }
                .level-{{ level }} { border-left: 5px solid
                    {% if level == 'critical' %}#ff4444
                    {% elif level == 'error' %}#ff6666
                    {% elif level == 'warning' %}#ffaa44
                    {% else %}#44aaff{% endif %}; }
            </style>
        </head>
        <body>
            <div class="header level-{{ level }}">
                <h2>🚨 Trading Alert: {{ title }}</h2>
                <p><strong>Level:</strong> {{ level|upper }}</p>
                <p><strong>Time:</strong> {{ timestamp }}</p>
            </div>

            <div class="content">
                <p>{{ message }}</p>

                {% if context %}
                <div class="context">
                    <h3>Additional Context:</h3>
                    <pre>{{ context | tojson(indent=2) }}</pre>
                </div>
                {% endif %}
            </div>

            <div style="margin-top: 30px; font-size: 12px; color: #666;">
                <p>This alert was generated by the MES Trading Bot notification system.</p>
            </div>
        </body>
        </html>
        """
        return Template(template_html)

    def _get_error_alert_template(self) -> Template:
        """Get error alert email template"""
        template_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header {
                    background-color: #ffe6e6;
                    padding: 20px;
                    border-radius: 5px;
                    border-left: 5px solid #ff4444;
                }
                .content { padding: 20px; }
                .context { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-top: 20px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h2>❌ System Error: {{ title }}</h2>
                <p><strong>Level:</strong> {{ level|upper }}</p>
                <p><strong>Time:</strong> {{ timestamp }}</p>
            </div>

            <div class="content">
                <p>{{ message }}</p>

                {% if context %}
                <div class="context">
                    <h3>Error Details:</h3>
                    <pre>{{ context | tojson(indent=2) }}</pre>
                </div>
                {% endif %}
            </div>

            <div style="margin-top: 30px; font-size: 12px; color: #666;">
                <p>This error alert was generated by the MES Trading Bot monitoring system.</p>
            </div>
        </body>
        </html>
        """
        return Template(template_html)

    def _get_performance_alert_template(self) -> Template:
        """Get performance alert email template"""
        template_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header {
                    background-color: #e6f3ff;
                    padding: 20px;
                    border-radius: 5px;
                    border-left: 5px solid #0066cc;
                }
                .content { padding: 20px; }
                .context { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-top: 20px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h2>📊 Performance Alert: {{ title }}</h2>
                <p><strong>Level:</strong> {{ level|upper }}</p>
                <p><strong>Time:</strong> {{ timestamp }}</p>
            </div>

            <div class="content">
                <p>{{ message }}</p>

                {% if context %}
                <div class="context">
                    <h3>Performance Metrics:</h3>
                    <pre>{{ context | tojson(indent=2) }}</pre>
                </div>
                {% endif %}
            </div>

            <div style="margin-top: 30px; font-size: 12px; color: #666;">
                <p>This performance alert was generated by the MES Trading Bot monitoring system.</p>
            </div>
        </body>
        </html>
        """
        return Template(template_html)

    def _get_system_alert_template(self) -> Template:
        """Get system alert email template"""
        template_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f8ff; padding: 20px; border-radius: 5px; }
                .content { padding: 20px; }
                .context { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-top: 20px; }
                .level-{{ level }} { border-left: 5px solid
                    {% if level == 'critical' %}#ff4444
                    {% elif level == 'error' %}#ff6666
                    {% elif level == 'warning' %}#ffaa44
                    {% else %}#44aaff{% endif %}; }
            </style>
        </head>
        <body>
            <div class="header level-{{ level }}">
                <h2>🔔 System Alert: {{ title }}</h2>
                <p><strong>Level:</strong> {{ level|upper }}</p>
                <p><strong>Time:</strong> {{ timestamp }}</p>
            </div>

            <div class="content">
                <p>{{ message }}</p>

                {% if context %}
                <div class="context">
                    <h3>System Information:</h3>
                    <pre>{{ context | tojson(indent=2) }}</pre>
                </div>
                {% endif %}
            </div>

            <div style="margin-top: 30px; font-size: 12px; color: #666;">
                <p>This system alert was generated by the MES Trading Bot monitoring system.</p>
            </div>
        </body>
        </html>
        """
        return Template(template_html)

    def get_notification_stats(self) -> Dict:
        """Get notification statistics"""
        total_notifications = len(self.notification_history)

        # Count by level
        level_counts = {}
        for level in NotificationLevel:
            level_counts[level.value] = sum(
                1 for notif in self.notification_history if notif["level"] == level.value
            )

        # Recent notifications (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_count = sum(
            1
            for notif in self.notification_history
            if datetime.fromisoformat(notif["timestamp"]) > recent_cutoff
        )

        return {
            "total_notifications": total_notifications,
            "level_counts": level_counts,
            "recent_24h": recent_count,
            "enabled_channels": [ch.value for ch in self.enabled_channels],
            "rate_limit_config": {
                "minutes": self.config.rate_limit_minutes,
                "max_per_period": self.config.max_notifications_per_period,
            },
        }


# Notification helper functions
async def send_trade_alert(
    title: str,
    message: str,
    level: NotificationLevel = NotificationLevel.WARNING,
    context: Dict = None,
):
    """Send a trade-related alert"""
    if hasattr(config, "notification_service") and config.notification_service:
        notification = NotificationMessage(
            level=level,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            context=context or {},
        )
        await config.notification_service.send_notification(notification)
    else:
        logger.warning(f"Notification service not configured: {title} - {message}")


async def send_error_alert(title: str, message: str, error: Exception = None, context: Dict = None):
    """Send an error alert"""
    error_context = context or {}
    if error:
        error_context["error_type"] = type(error).__name__
        error_context["error_message"] = str(error)

    if hasattr(config, "notification_service") and config.notification_service:
        notification = NotificationMessage(
            level=NotificationLevel.ERROR,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            context=error_context,
        )
        await config.notification_service.send_notification(notification)
    else:
        logger.error(f"Notification service not configured: {title} - {message}")


async def send_performance_alert(
    title: str, message: str, metrics: Dict, level: NotificationLevel = NotificationLevel.INFO
):
    """Send a performance alert with metrics"""
    if hasattr(config, "notification_service") and config.notification_service:
        notification = NotificationMessage(
            level=level,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            context={"metrics": metrics},
        )
        await config.notification_service.send_notification(notification)
    else:
        logger.info(f"Notification service not configured: {title} - {message}")


async def send_system_alert(
    title: str,
    message: str,
    level: NotificationLevel = NotificationLevel.INFO,
    context: Dict = None,
):
    """Send a system alert"""
    if hasattr(config, "notification_service") and config.notification_service:
        notification = NotificationMessage(
            level=level,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            context=context or {},
        )
        await config.notification_service.send_notification(notification)
    else:
        logger.info(f"Notification service not configured: {title} - {message}")
