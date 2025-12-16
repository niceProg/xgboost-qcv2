#!/usr/bin/env python3
"""
Signal notification system untuk XGBoost trading alerts.
Mengirim notifikasi ke berbagai channels (Telegram, Discord, Slack, Email).
"""

import os
import sys
import json
import logging
import asyncio
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/signal_notifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SignalNotifier:
    """Handle trading signal notifications."""

    def __init__(self, config_file: str = 'notification_config.json'):
        self.config = self.load_config(config_file)
        self.notification_history = []

    def load_config(self, config_file: str) -> Dict:
        """Load notification configuration."""
        default_config = {
            "enabled": True,
            "rate_limit": {
                "max_signals_per_hour": 10,
                "cooldown_minutes": 5
            },
            "channels": {
                "telegram": {
                    "enabled": False,
                    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
                    "chat_id": os.getenv("TELEGRAM_CHAT_ID")
                },
                "discord": {
                    "enabled": False,
                    "webhook_url": os.getenv("DISCORD_WEBHOOK_URL")
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": os.getenv("SLACK_WEBHOOK_URL")
                },
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": os.getenv("EMAIL_USERNAME"),
                    "password": os.getenv("EMAIL_PASSWORD"),
                    "recipients": []
                }
            },
            "filters": {
                "min_confidence": 0.7,
                "signals_only": True,  # Only send BUY/SELL, not HOLD
                "business_hours_only": False,
                "max_price_change": 0.1  # Max 10% price change for alerts
            }
        }

        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in user_config:
                        user_config[key] = default_config[key]
                    elif isinstance(default_config[key], dict):
                        for subkey in default_config[key]:
                            if subkey not in user_config[key]:
                                user_config[key][subkey] = default_config[key][subkey]
                return user_config
            except Exception as e:
                logger.error(f"Error loading config: {e}. Using defaults.")
                return default_config
        else:
            # Create default config
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config

    def should_send_notification(self, signal_data: Dict) -> bool:
        """Check if notification should be sent based on filters."""
        if not self.config['enabled']:
            return False

        # Check confidence threshold
        if signal_data.get('confidence', 0) < self.config['filters']['min_confidence']:
            logger.info(f"Signal confidence too low: {signal_data.get('confidence', 0):.2f}")
            return False

        # Check if signal only
        if self.config['filters']['signals_only']:
            signal = signal_data.get('signal', 'HOLD')
            if signal == 'HOLD':
                logger.info("HOLD signal - not sending notification")
                return False

        # Check business hours
        if self.config['filters']['business_hours_only']:
            now = datetime.now()
            if now.hour < 9 or now.hour > 17:
                logger.info("Outside business hours")
                return False

        # Check rate limiting
        if self.is_rate_limited():
            logger.info("Rate limited - not sending notification")
            return False

        return True

    def is_rate_limited(self) -> bool:
        """Check if we're rate limited."""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        cooldown_minutes = self.config['rate_limit']['cooldown_minutes']
        cooldown_time = now - timedelta(minutes=cooldown_minutes)

        # Clean old notifications
        self.notification_history = [
            n for n in self.notification_history
            if n['timestamp'] > one_hour_ago
        ]

        # Check hourly limit
        if len(self.notification_history) >= self.config['rate_limit']['max_signals_per_hour']:
            return True

        # Check cooldown
        recent_notifications = [
            n for n in self.notification_history
            if n['timestamp'] > cooldown_time
        ]
        if len(recent_notifications) > 0:
            return True

        return False

    def send_signal_notification(self, signal_data: Dict):
        """Send signal notification to all enabled channels."""
        if not self.should_send_notification(signal_data):
            return False

        logger.info(f"🚀 Sending signal notification: {signal_data.get('signal')}")

        # Create formatted message
        message = self.format_signal_message(signal_data)

        # Send to all enabled channels
        success_count = 0
        channels = self.config['channels']

        if channels['telegram']['enabled']:
            if self.send_telegram_notification(message):
                success_count += 1

        if channels['discord']['enabled']:
            if self.send_discord_notification(message):
                success_count += 1

        if channels['slack']['enabled']:
            if self.send_slack_notification(message):
                success_count += 1

        if channels['email']['enabled']:
            if self.send_email_notification(message, signal_data):
                success_count += 1

        if success_count > 0:
            # Record notification
            self.notification_history.append({
                'timestamp': datetime.now(),
                'signal': signal_data.get('signal'),
                'symbol': signal_data.get('symbol'),
                'confidence': signal_data.get('confidence'),
                'channels_sent': success_count
            })
            return True
        else:
            logger.error("Failed to send notification to any channel")
            return False

    def format_signal_message(self, signal_data: Dict) -> str:
        """Format trading signal message."""
        signal = signal_data.get('signal', 'UNKNOWN')
        symbol = signal_data.get('symbol', 'UNKNOWN')
        confidence = signal_data.get('confidence', 0)
        price = signal_data.get('price', 0)
        prediction_prob = signal_data.get('prediction_probability', 0)
        recommendation = signal_data.get('recommendation', {})

        # Create emoji based on signal
        signal_emoji = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '🟡'
        }.get(signal, '⚪')

        # Format confidence
        confidence_pct = confidence * 100
        confidence_emoji = '🔥' if confidence > 0.8 else '⚡' if confidence > 0.6 else '💡'

        message = f"""
{signal_emoji} **TRADING SIGNAL ALERT** {signal_emoji}

📊 **Asset**: {symbol}
💰 **Price**: ${price:,.2f}
🎯 **Signal**: {signal}
{confidence_emoji} **Confidence**: {confidence_pct:.1f}% ({prediction_prob:.3f})
📈 **Prediction Score**: {prediction_prob:.3f}
⏰ **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} WIB

📋 **Recommendation**:
• Action: {recommendation.get('action', 'HOLD')}
• Position Size: {recommendation.get('suggested_position_size', 0):.1%}
• Stop Loss: ${recommendation.get('stop_loss', 0):,.2f}
• Take Profit: ${recommendation.get('take_profit', 0):,.2f}
• Risk Level: {recommendation.get('risk_level', 'Unknown')}
• Holding Period: {recommendation.get('holding_period', 'Unknown')}

💡 **Reasoning**: {recommendation.get('reasoning', 'No reasoning provided')}

---
🤖 *XGBoost Real-time Trading System*
⚡ *Powered by Machine Learning*
"""

        return message

    def send_telegram_notification(self, message: str) -> bool:
        """Send notification via Telegram."""
        try:
            bot_token = self.config['channels']['telegram']['bot_token']
            chat_id = self.config['channels']['telegram']['chat_id']

            if not bot_token or not chat_id:
                logger.error("Telegram bot token or chat ID not configured")
                return False

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("✅ Telegram notification sent")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False

    def send_discord_notification(self, message: str) -> bool:
        """Send notification via Discord webhook."""
        try:
            webhook_url = self.config['channels']['discord']['webhook_url']
            if not webhook_url:
                logger.error("Discord webhook URL not configured")
                return False

            # Discord uses different markdown format
            discord_message = self.convert_to_discord_format(message)

            payload = {
                'content': discord_message,
                'username': 'XGBoost Trading Bot',
                'avatar_url': 'https://i.imgur.com/4M34hi2.png'  # Optional avatar
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info("✅ Discord notification sent")
            return True

        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def send_slack_notification(self, message: str) -> bool:
        """Send notification via Slack webhook."""
        try:
            webhook_url = self.config['channels']['slack']['webhook_url']
            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False

            # Slack uses different formatting
            slack_message = {
                'text': '🚀 Trading Signal Alert',
                'blocks': [
                    {
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': message
                        }
                    }
                ]
            }

            response = requests.post(webhook_url, json=slack_message, timeout=10)
            response.raise_for_status()

            logger.info("✅ Slack notification sent")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    def send_email_notification(self, message: str, signal_data: Dict) -> bool:
        """Send notification via email."""
        try:
            email_config = self.config['channels']['email']
            smtp_server = email_config['smtp_server']
            smtp_port = email_config['smtp_port']
            username = email_config['username']
            password = email_config['password']
            recipients = email_config['recipients']

            if not all([smtp_server, username, password, recipients]):
                logger.error("Email configuration incomplete")
                return False

            # Create email
            msg = MimeMultipart()
            signal = signal_data.get('signal', 'UNKNOWN')
            symbol = signal_data.get('symbol', 'UNKNOWN')

            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"🚀 Trading Signal: {signal} {symbol} - {datetime.now().strftime('%H:%M')}"

            # Convert markdown to plain text for email
            email_body = self.convert_to_plain_text(message)
            msg.attach(MimeText(email_body, 'plain'))

            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()

            logger.info("✅ Email notification sent")
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    def convert_to_discord_format(self, message: str) -> str:
        """Convert markdown message to Discord format."""
        # Discord uses different markdown syntax
        discord_msg = message.replace('**', '**').replace('•', '•')  # Keep bold, bullets work
        return discord_msg

    def convert_to_plain_text(self, message: str) -> str:
        """Convert markdown message to plain text."""
        import re

        # Remove markdown formatting
        plain_text = re.sub(r'\*\*(.*?)\*\*', r'\1', message)  # Remove bold
        plain_text = re.sub(r'\*(.*?)\*', r'\1', message)  # Remove italic
        plain_text = re.sub(r'`(.*?)`', r'\1', message)  # Remove code
        plain_text = re.sub(r'### (.*)', r'\1:', plain_text)  # Headers
        plain_text = re.sub(r'## (.*)', r'\1:', plain_text)
        plain_text = re.sub(r'# (.*)', r'\1:', plain_text)
        plain_text = plain_text.replace('•', '- ')  # Convert bullets

        return plain_text

    def send_system_notification(self, message: str, level: str = 'info'):
        """Send system notification (not trading signals)."""
        if not self.config['enabled']:
            return

        level_emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅'
        }.get(level, 'ℹ️')

        system_message = f"{level_emoji} **System Notification**\n\n{message}\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Send to Telegram (system notifications bypass rate limiting)
        if self.config['channels']['telegram']['enabled']:
            self.send_telegram_notification(system_message)

    def get_notification_stats(self) -> Dict:
        """Get notification statistics."""
        now = datetime.now()
        one_day_ago = now - timedelta(days=1)
        one_hour_ago = now - timedelta(hours=1)

        notifications_last_hour = [
            n for n in self.notification_history
            if n['timestamp'] > one_hour_ago
        ]

        notifications_last_day = [
            n for n in self.notification_history
            if n['timestamp'] > one_day_ago
        ]

        return {
            'total_notifications': len(self.notification_history),
            'last_hour': len(notifications_last_hour),
            'last_24h': len(notifications_last_day),
            'rate_limit_reached': len(notifications_last_hour) >= self.config['rate_limit']['max_signals_per_hour'],
            'last_notification': self.notification_history[-1]['timestamp'].isoformat() if self.notification_history else None
        }


# CLI interface
def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Signal notification system')
    parser.add_argument('--test', action='store_true', help='Send test notification')
    parser.add_argument('--stats', action='store_true', help='Show notification statistics')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--signal', type=str, help='Send signal notification (JSON format)')

    args = parser.parse_args()

    notifier = SignalNotifier(config_file=args.config or 'notification_config.json')

    if args.test:
        # Send test notification
        test_signal = {
            'signal': 'BUY',
            'symbol': 'BTCUSDT',
            'price': 42500.0,
            'confidence': 0.85,
            'prediction_probability': 0.85,
            'recommendation': {
                'action': 'BUY',
                'suggested_position_size': 1.0,
                'stop_loss': 41800.0,
                'take_profit': 44200.0,
                'risk_level': 'Low',
                'reasoning': 'Strong buy signal with high confidence'
            }
        }
        success = notifier.send_signal_notification(test_signal)
        print(f"Test notification {'sent' if success else 'failed'}")

    elif args.stats:
        stats = notifier.get_notification_stats()
        print("Notification Statistics:")
        print(json.dumps(stats, indent=2))

    elif args.signal:
        try:
            signal_data = json.loads(args.signal)
            success = notifier.send_signal_notification(signal_data)
            print(f"Signal notification {'sent' if success else 'failed'}")
        except json.JSONDecodeError:
            print("Invalid JSON format for signal data")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()