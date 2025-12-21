#!/usr/bin/env python3
"""
Real-time Database Monitor for XGBoost Trading System.
Monitors new 2025 data in newera database and triggers training.
"""

import os
import json
import logging
import time
import pymysql
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import schedule

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Create logs directory if it doesn't exist
os.makedirs('./logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/realtime_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealtimeDatabaseMonitor:
    """Monitor new data arrival in newera database using smart event-driven approach."""

    def __init__(self, config_file: str = None):
        self.config_file = config_file or './config/realtime_config.json'
        self.state_file = './state/last_processed.json'

        # Tables to monitor - Define BEFORE loading state
        self.tables = [
            'cg_spot_price_history',
            'cg_funding_rate_history',
            'cg_futures_basis_history',
            'cg_spot_aggregated_taker_volume_history',
            'cg_long_short_global_account_ratio_history',
            'cg_long_short_top_account_ratio_history'
        ]

        # Database configuration
        self.db_config = {
            'host': os.getenv('TRADING_DB_HOST', 'localhost'),
            'port': int(os.getenv('TRADING_DB_PORT', 3306)),
            'user': os.getenv('TRADING_DB_USER'),
            'password': os.getenv('TRADING_DB_PASSWORD'),
            'database': os.getenv('TRADING_DB_NAME', 'newera'),
            'charset': 'utf8mb4'
        }

        # Monitoring configuration
        self.config = self.load_config()
        self.state = self.load_state()

        # Smart table configurations - based on created_at for efficient monitoring
        self.table_configs = {
            'cg_spot_price_history': {
                'time_col': 'time',
                'key_cols': ['time', 'exchange', 'symbol', 'interval'],
                'min_new_records': 10,    # MINIMUM 10 records untuk trigger
                'priority': 'HIGH',
                'max_check_interval': 60,   # Check every 1 minute
                'urgent_threshold': 20,   # Trigger urgent if >20 records
                'business_hours_only': False,
                'check_window': 300       # Check last 5 minutes for new records
            },
            'cg_funding_rate_history': {
                'time_col': 'time',
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'min_new_records': 10,    # MINIMUM 10 records
                'priority': 'HIGH',
                'max_check_interval': 60,   # Check every 1 minute
                'urgent_threshold': 20,
                'business_hours_only': False,
                'check_window': 300
            },
            'cg_futures_basis_history': {
                'time_col': 'time',
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'min_new_records': 10,    # MINIMUM 10 records
                'priority': 'HIGH',
                'max_check_interval': 300,
                'urgent_threshold': 20,
                'business_hours_only': False,
                'check_window': 300
            },
            'cg_spot_aggregated_taker_volume_history': {
                'time_col': 'time',
                'key_cols': ['time', 'exchange_name', 'symbol', 'interval'],
                'exchange_col': 'exchange_name',
                'min_new_records': 10,    # MINIMUM 10 records
                'priority': 'HIGH',
                'max_check_interval': 60,   # Check every 1 minute
                'urgent_threshold': 20,
                'business_hours_only': False,
                'check_window': 300
            },
            'cg_long_short_global_account_ratio_history': {
                'time_col': 'time',
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'min_new_records': 10,    # MINIMUM 10 records
                'priority': 'MEDIUM',
                'max_check_interval': 60,   # Check every 1 minute
                'urgent_threshold': 20,
                'business_hours_only': False,
                'check_window': 300
            },
            'cg_long_short_top_account_ratio_history': {
                'time_col': 'time',
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'min_new_records': 10,    # MINIMUM 10 records
                'priority': 'MEDIUM',
                'max_check_interval': 60,   # Check every 1 minute
                'urgent_threshold': 20,
                'business_hours_only': False,
                'check_window': 300
            }
        }

        # Smart monitoring state
        self.table_activity = {}  # Track activity patterns
        self.adaptive_intervals = {}  # Dynamic check intervals

        # Notification tracking (no artificial limits)
        self.last_notification_time = None
        self.notification_count = 0
        self.rate_limit_hits = 0  # Track when we hit Telegram rate limits

        # Time-based notification settings
        self.min_notification_interval = 180  # 3 minutes between guaranteed notifications
        self.last_data_detection_time = None  # Track when we last detected data

        # Status
        self.running = False
        self.connection = None

        # Initialize smart monitoring
        self.initialize_smart_monitoring()

    def initialize_smart_monitoring(self):
        """Initialize smart monitoring with adaptive intervals."""
        for table in self.tables:
            self.table_activity[table] = {
                'last_check': datetime.min,
                'last_data_found': None,
                'data_frequency': 0,  # Records per hour
                'active_periods': [],  # When this table is most active
                'adaptive_interval': self.table_configs[table]['max_check_interval']
            }
            self.adaptive_intervals[table] = self.table_configs[table]['max_check_interval']

    def is_business_hours(self) -> bool:
        """Check if current time is during active trading hours."""
        import pytz
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        now = datetime.now(jakarta_tz)
        # Convert to UTC for business hours calculation
        utc_now = now.astimezone(pytz.UTC)
        # Major crypto market active hours (UTC)
        # 9:00 AM - 11:59 PM UTC covers most global markets
        return 9 <= utc_now.hour <= 23

    def get_adaptive_check_interval(self, table: str) -> int:
        """Calculate optimal check interval based on activity patterns."""
        config = self.table_configs[table]
        activity = self.table_activity[table]

        # Skip if business hours only and not in business hours
        if config['business_hours_only'] and not self.is_business_hours():
            return config['max_check_interval'] * 2  # Double interval outside business hours

        # Real-time adaptive logic - always check frequently for active tables
        if config['priority'] == 'HIGH':
            return min(config['max_check_interval'], 30)  # High priority: max 30 seconds
        elif config['priority'] == 'MEDIUM':
            return min(config['max_check_interval'], 60)  # Medium priority: max 1 minute
        else:
            return config['max_check_interval']  # Low priority: as configured

    def should_check_table_now(self, table: str) -> bool:
        """Periodic checking logic - check based on configured interval."""
        # Get last check time dari state
        last_check_time = self.state.get('last_check_time', {}).get(table)

        if not last_check_time:
            # Belum pernah di-check, check sekarang
            return True

        # Convert last_check_time ke datetime
        if isinstance(last_check_time, str):
            try:
                last_check_dt = datetime.fromisoformat(last_check_time)
            except:
                return True  # Error, check sekarang
        else:
            return True

        # Calculate time since last check (make timezone-aware)
        import pytz
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        current_time = datetime.now(jakarta_tz)

        # Make sure last_check_dt is timezone-aware
        if last_check_dt.tzinfo is None:
            last_check_dt = jakarta_tz.localize(last_check_dt)

        time_since_check = (current_time - last_check_dt).total_seconds()

        # Check if enough time has passed based on table config
        check_interval = self.table_configs[table]['max_check_interval']

        return time_since_check >= check_interval

    def check_urgent_conditions(self, table: str) -> bool:
        """Check for urgent conditions that require immediate attention."""
        try:
            if not self.connection:
                if not self.connect_to_database():
                    return False

            cursor = self.connection.cursor()
            config = self.table_configs[table]

            # Quick check for recent high-volume data using time column
            exchange_col = config.get('exchange_col', 'exchange')  # Default to 'exchange'
            urgent_query = f"""
            SELECT COUNT(*) as recent_count
            FROM {table}
            WHERE {config['time_col']} > UNIX_TIMESTAMP(DATE_SUB(NOW(), INTERVAL 2 MINUTE)) * 1000
            AND `{exchange_col}` IN %s
            """

            cursor.execute(urgent_query, (self.config['exchanges'],))
            result = cursor.fetchone()

            if result and result[0] >= config['urgent_threshold']:
                logger.warning(f"🚨 URGENT: {result[0]} new records in {table} within 2 minutes!")
                cursor.close()
                return True

            cursor.close()
            return False

        except Exception as e:
            logger.error(f"Error checking urgent conditions for {table}: {e}")
            return False

    def load_config(self) -> Dict:
        """Load monitoring configuration."""
        default_config = {
            "monitor_pairs": ["BTCUSDT", "ETHUSDT"],
            "monitor_intervals": ["1h", "4h"],
            "exchanges": ["binance"],
            "focus_year": 2025,  # Focus on 2025 data
            "min_new_records": 10,
            "notification": {
                "enabled": True,
                "telegram_token": os.getenv('TELEGRAM_BOT_TOKEN'),
                "telegram_chat_id": os.getenv('TELEGRAM_CHAT_ID')
            }
        }

        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
            except Exception as e:
                logger.error(f"Error loading config: {e}. Using defaults.")

        return default_config

    def load_state(self) -> Dict:
        """Load monitoring state."""
        state_path = Path(self.state_file)
        state_path.parent.mkdir(exist_ok=True)

        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
            except Exception as e:
                logger.error(f"Error loading state: {e}. Using defaults.")
                state = {}
        else:
            state = {}

        # Initialize last_processed for all tables if not exists
        if 'last_processed' not in state:
            state['last_processed'] = {}
            # Start from beginning of 2025
            start_2025 = datetime(2025, 1, 1, 0, 0, 0)
            for table in self.tables:
                state['last_processed'][table] = start_2025.isoformat()

            self.save_state(state)

        return state

    def save_state(self, state: Dict = None):
        """Save monitoring state."""
        if state is None:
            state = self.state

        state_path = Path(self.state_file)
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        self.state = state

    def connect_to_database(self) -> bool:
        """Connect to database."""
        try:
            self.connection = pymysql.connect(**self.db_config)
            logger.info(f"✅ Connected to database: {self.db_config['database']}")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False

    def check_new_data(self, table: str) -> Optional[Dict]:
        """Check for new data using created_at column for efficient monitoring."""
        if not self.connection:
            if not self.connect_to_database():
                return None

        try:
            cursor = self.connection.cursor()
            table_config = self.table_configs[table]
            check_window = table_config.get('check_window', 120)  # Default 2 minutes (lighter load)

            # Get last check time dari state, fallback ke 5 menit yang lalu
            last_check_time = self.state.get('last_check_time', {}).get(table)
            if not last_check_time:
                last_check_time = datetime.now().timestamp() - check_window
            else:
                if isinstance(last_check_time, str):
                    last_check_time = datetime.fromisoformat(last_check_time).timestamp()

            # Determine column name for exchanges
            exchange_col = table_config.get('exchange_col', 'exchange')  # Default to 'exchange'
            exchanges = self.config['exchanges']

            # OPTIMIZED Query - Simple and fast
            query = f"""
            SELECT
                COUNT(*) as new_count,
                MAX(created_at) as latest_created
            FROM {table}
            WHERE created_at > DATE_SUB(NOW(), INTERVAL {check_window} SECOND)
            AND `{exchange_col}` IN %s
            LIMIT 1
            """

            cursor.execute(query, (exchanges,))
            result = cursor.fetchone()

            if result and result[0] > 0:  # If we have new records
                # Update activity tracking
                activity = self.table_activity[table]
                # Use timezone-aware datetime for consistency
                import pytz
                jakarta_tz = pytz.timezone('Asia/Jakarta')
                activity['last_data_found'] = datetime.now(jakarta_tz)
                activity['data_frequency'] = result[0]  # new_count

                # Calculate priority based on count
                new_count = result[0]
                if new_count >= table_config['urgent_threshold']:
                    priority = "URGENT"
                elif new_count >= table_config['min_new_records']:
                    priority = table_config['priority']
                else:
                    priority = "LOW"

                new_data = {
                    'table': table,
                    'new_count': new_count,
                    'latest_created': result[1] if result[1] else None,
                    'recent_count': new_count,  # All records are recent now
                    'priority': priority,
                    'detection_method': 'optimized_monitoring'
                }

                # Smart threshold checking
                should_trigger = False
                trigger_reason = ""

                # Condition 1: Minimum records met
                if result[0] >= table_config['min_new_records']:
                    should_trigger = True
                    trigger_reason = f"Minimum threshold met: {result[0]} >= {table_config['min_new_records']}"

                # Condition 2: Urgent threshold override
                elif result[0] >= table_config['urgent_threshold']:
                    should_trigger = True
                    priority = "URGENT"
                    trigger_reason = f"Urgent threshold: {result[0]} >= {table_config['urgent_threshold']}"

                # Condition 3: Recent high-frequency activity (using new_count)
                elif new_count >= table_config['urgent_threshold']:
                    should_trigger = True
                    priority = "URGENT"
                    trigger_reason = f"Urgent activity: {new_count} records in last {check_window} seconds"

                if should_trigger:
                    logger.info(f"📊 New data in {table}: {result[0]} records (Priority: {priority})")
                    logger.info(f"   Latest created: {new_data['latest_created']}")
                    logger.info(f"   Reason: {trigger_reason}")

                    cursor.close()
                    return new_data
                else:
                    logger.info(f"📝 {result[0]} new records in {table} (below threshold)")

            # Update last check time even if no data found
            # Use timezone-aware datetime for consistency
            import pytz
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            current_time = datetime.now(jakarta_tz)
            self.table_activity[table]['last_check'] = current_time

            # Update state with last check time
            if 'last_check_time' not in self.state:
                self.state['last_check_time'] = {}
            self.state['last_check_time'][table] = current_time.isoformat()
            self.save_state()

            cursor.close()
            return None

        except Exception as e:
            logger.error(f"Error checking {table}: {e}")
            return None

    def calculate_priority(self, record_count: int, seconds_since_creation: int, config: Dict) -> str:
        """Calculate priority level for new data."""
        # Very recent data (within 1 minute)
        if seconds_since_creation < 60:
            if record_count >= config['urgent_threshold']:
                return "URGENT"
            return "HIGH"

        # Recent data (within 5 minutes)
        elif seconds_since_creation < 300:
            if record_count >= config['urgent_threshold']:
                return "HIGH"
            return "MEDIUM"

        # Older data
        else:
            return config['priority']

    def trigger_training(self, new_data_list: List[Dict]) -> bool:
        """Directly trigger real-time training pipeline."""
        try:
            # Send notification first
            self.send_notification(new_data_list)

            # Update last_processed times to latest data
            self.update_last_processed(new_data_list)

            # Directly run training pipeline (always full historical data)
            logger.info(f"🚀 Starting FULL training pipeline for {len(new_data_list)} tables...")

            import subprocess
            cmd = [
                'python3', 'realtime_trainer_pipeline.py',
                '--output-dir', './output_train'
            ]

            logger.info(f"🏃 Running: {' '.join(cmd)}")
            logger.info("📊 Mode: Loading ALL historical data (client requirement)")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hours timeout
            )

            if result.returncode == 0:
                logger.info("✅ FULL training pipeline completed successfully!")
                logger.info("📈 New models saved to ./output_train/models/")
                logger.info("🔔 Mode: ALL historical data trained (client requirement)")
                if result.stdout:
                    # Log last few lines dari output
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-5:]:  # Last 5 lines
                        logger.info(f"📄 {line}")
                return True
            else:
                logger.error("❌ Training pipeline failed!")
                if result.stderr:
                    logger.error(f"📄 Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Training pipeline timed out!")
            return False
        except Exception as e:
            logger.error(f"Error triggering training: {e}")
            return False

    def update_last_processed(self, new_data_list: List[Dict]):
        """Update last_processed timestamps based on new data."""
        try:
            updated = False
            for data in new_data_list:
                table = data['table']
                if data['latest_created']:
                    # Update to the latest created_at from new data
                    self.state['last_processed'][table] = data['latest_created'].isoformat()
                    updated = True

            if updated:
                self.save_state()
                logger.info("✅ Updated last_processed timestamps")

        except Exception as e:
            logger.error(f"Error updating last_processed: {e}")

    def send_notification(self, new_data_list: List[Dict]):
        """Send notification about new data detection (no artificial limits)."""
        if not self.config['notification']['enabled']:
            return

        # Use timezone-aware datetime for consistency
        import pytz
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        current_time = datetime.now(jakarta_tz)

        try:
            total_records = sum(data['new_count'] for data in new_data_list)

            import time
            import hashlib

            # Convert to WIB (UTC+7) - current_time already has timezone
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            now_wib = datetime.now(jakarta_tz)

            # Format table names untuk lebih readable
            table_names = {
                'cg_spot_price_history': 'Spot Price',
                'cg_funding_rate_history': 'Funding Rate',
                'cg_futures_basis_history': 'Futures Basis',
                'cg_spot_aggregated_taker_volume_history': 'Taker Volume',
                'cg_long_short_global_account_ratio_history': 'Global L/S Ratio',
                'cg_long_short_top_account_ratio_history': 'Top L/S Ratio'
            }

            # Extract unique table names yang punya data baru
            tables_with_data = list(set([data['table'] for data in new_data_list]))
            readable_tables = ', '.join([table_names.get(table, table) for table in tables_with_data])

            # Create unique hash untuk avoid duplicate content
            content_hash = hashlib.md5(f"{total_records}_{readable_tables}_{now_wib.strftime('%H%M')}".encode()).hexdigest()[:8]

            # Create unique message with hash and timestamp
            message = f"""📊 New 2025 Data Detected! [#{content_hash}]

📈 Total New Records: {total_records:,}
📊 Tables: {readable_tables}
⏰ Time: {now_wib.strftime('%d-%m-%Y %H:%M:%S')} WIB

Action: FULL 6-Step Training Pipeline STARTING
Status: Complete training with ALL historical data (like manual training)
Models: Will be saved to ./output_train/models/
Duration: ~30-60 minutes (full dataset training)

Table Breakdown:
""" + '\n'.join([f"• {table_names.get(data['table'], data['table'])}: {data['new_count']:,} records ({data.get('priority', 'UNKNOWN')})"
                 for data in new_data_list]) + f"""

🤖 XGBoost Real-time Monitor | Dragon Fortune AI
🔔 Notification #{self.notification_count + 1} | ID: {content_hash}"""

            # Send Telegram notification if configured
            telegram_token = self.config['notification']['telegram_token']
            telegram_chat_id = self.config['notification']['telegram_chat_id']

            if telegram_token and telegram_chat_id:
                import requests

                url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
                payload = {
                    'chat_id': telegram_chat_id,
                    'text': message
                    # Removed parse_mode to avoid Markdown parsing errors
                }

                response = requests.post(url, json=payload, timeout=10)

                if response.status_code == 200:
                    # Update notification tracking
                    self.last_notification_time = current_time
                    self.notification_count += 1
                    logger.info(f"📱 Telegram notification sent successfully (#{self.notification_count})")

                elif response.status_code == 429:
                    # Rate limit hit!
                    self.rate_limit_hits += 1
                    error_data = response.json()
                    retry_after = error_data.get('parameters', {}).get('retry_after', 60)

                    logger.error(f"⛔ TELEGRAM RATE LIMIT HIT!")
                    logger.error(f"📊 Notification #{self.notification_count + 1} blocked")
                    logger.error(f"⏰ Retry after: {retry_after} seconds")
                    logger.error(f"📈 Total rate limit hits: {self.rate_limit_hits}")

                elif response.status_code == 400:
                    # Bad Request - likely content issue
                    error_data = response.json()
                    logger.error(f"❌ Telegram Bad Request: {error_data.get('description')}")

                else:
                    # Other API errors
                    logger.error(f"❌ Telegram API error: {response.status_code} - {response.text}")
                    if response.status_code >= 500:
                        logger.error("🔄 This is a Telegram server error, notification may arrive later")

        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    def check_all_tables(self) -> bool:
        """Check all tables for new data with time-based notification logic."""
        tables_with_new_data = []
        checked_count = 0
        skipped_count = 0
        has_data_activity = False

        # Use timezone-aware datetime to avoid timezone comparison errors
        import pytz
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        current_time = datetime.now(jakarta_tz)

        for table in self.tables:
            # Always check all tables for now
            if self.should_check_table_now(table):
                checked_count += 1
                logger.debug(f"🔍 Checking {table}")

                # Check for new data
                new_data = self.check_new_data(table)
                if new_data:
                    tables_with_new_data.append(new_data)
                    has_data_activity = True
                    self.last_data_detection_time = current_time

                    # If high priority data found, immediately check other priority tables
                    if new_data.get('priority') in ['URGENT', 'HIGH']:
                        logger.info(f"⚡ High priority data in {table}, accelerating other table checks")
                        self.accelerate_other_tables(table)

        # Log monitoring efficiency
        if checked_count > 0 or skipped_count > 0:
            logger.info(f"📊 Monitoring cycle: {checked_count} tables checked, {skipped_count} tables skipped")

        # ALWAYS trigger training if we have new data
        training_triggered = False
        if tables_with_new_data:
            # Sort by priority
            tables_with_new_data.sort(key=lambda x: (
                0 if x.get('priority') == 'URGENT' else
                1 if x.get('priority') == 'HIGH_PRIORITY' else
                2 if x.get('priority') == 'HIGH' else
                3 if x.get('priority') == 'MEDIUM' else
                4
            ))

            logger.info(f"🎯 Training trigger priority order: {[d['table'] + '(' + d.get('priority', 'UNKNOWN') + ')' for d in tables_with_new_data]}")

            success = self.trigger_training(tables_with_new_data)
            training_triggered = True
            self.save_state()

        # TIME-BASED NOTIFICATION: Send periodic notifications even without new data
        should_send_time_based_notification = False

        if self.last_notification_time:
            time_since_last_notification = (current_time - self.last_notification_time).total_seconds()

            # Send notification if it's been long enough AND we have recent data activity
            if time_since_last_notification >= self.min_notification_interval:
                if self.last_data_detection_time:
                    time_since_data = (current_time - self.last_data_detection_time).total_seconds()
                    # Only send time-based notification if data activity was recent (within 10 minutes)
                    if time_since_data <= 600:  # 10 minutes
                        should_send_time_based_notification = True
                        logger.info(f"⏰ Time-based notification: {time_since_last_notification/60:.1f} min since last, data activity: {time_since_data/60:.1f} min ago")
        else:
            # First notification if enabled
            should_send_time_based_notification = self.config.get('notification', {}).get('enabled', False)

        # Send time-based notification if conditions met
        if should_send_time_based_notification:
            try:
                total_recent_records = 0
                table_activity_summary = []

                # Get summary of recent activity
                for table in self.tables:
                    activity = self.table_activity.get(table, {})
                    if activity.get('data_frequency', 0) > 0:
                        table_activity_summary.append({
                            'table': table,
                            'records': activity['data_frequency'],
                            'last_detected': activity.get('last_data_found')
                        })
                        total_recent_records += activity['data_frequency']

                # Create time-based notification message
                import pytz
                jakarta_tz = pytz.timezone('Asia/Jakarta')
                # Make current_time timezone-aware first
                if current_time.tzinfo is None:
                    current_time = jakarta_tz.localize(current_time)
                now_wib = current_time.astimezone(jakarta_tz)

                table_names = {
                    'cg_spot_price_history': 'Spot Price',
                    'cg_funding_rate_history': 'Funding Rate',
                    'cg_futures_basis_history': 'Futures Basis',
                    'cg_spot_aggregated_taker_volume_history': 'Taker Volume',
                    'cg_long_short_global_account_ratio_history': 'Global L/S Ratio',
                    'cg_long_short_top_account_ratio_history': 'Top L/S Ratio'
                }

                readable_tables = ', '.join([table_names.get(table, table) for table in [s['table'] for s in table_activity_summary]])

                time_message = f"""📊 Real-time Market Monitor Report

📅 Time: {now_wib.strftime('%d-%m-%Y %H:%M:%S')} WIB
📈 Recent Activity: {total_recent_records} new records detected
📊 Active Tables: {readable_tables}

📋 Table Activity:
""" + '\n'.join([f"• {table_names.get(s['table'], s['table'])}: {s['records']} records"
                      for s in table_activity_summary]) + f"""

🤖 XGBoost Real-time Monitor
⏰ Periodic Report | Last training: {'Triggered' if training_triggered else 'No recent training'}"""

                # Send time-based notification
                telegram_token = self.config['notification']['telegram_token']
                telegram_chat_id = self.config['notification']['telegram_chat_id']

                if telegram_token and telegram_chat_id:
                    import requests
                    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
                    payload = {
                        'chat_id': telegram_chat_id,
                        'text': time_message
                    }

                    response = requests.post(url, json=payload, timeout=10)
                    if response.status_code == 200:
                        self.last_notification_time = current_time
                        self.notification_count += 1
                        logger.info(f"📱 Time-based notification sent successfully (#{self.notification_count})")
                    elif response.status_code == 429:
                        logger.error(f"⛔ TELEGRAM RATE LIMIT HIT during time-based notification!")
                    else:
                        logger.error(f"❌ Telegram API error during time-based notification: {response.status_code}")

            except Exception as e:
                logger.error(f"Error sending time-based notification: {e}")

        return training_triggered

    def accelerate_other_tables(self, priority_table: str):
        """Accelerate check intervals for related tables when high priority data is found."""
        for table in self.tables:
            if table != priority_table:
                activity = self.table_activity[table]
                # Reduce next check interval for related tables
                current_interval = self.get_adaptive_check_interval(table)
                activity['adaptive_interval'] = min(30, current_interval)  # Minimum 30 seconds
                logger.info(f"⚡ Accelerated {table} check interval to {activity['adaptive_interval']}s")

    def get_monitoring_efficiency_report(self) -> Dict:
        """Get monitoring efficiency statistics."""
        total_tables = len(self.tables)
        checked_now = sum(1 for table in self.tables if self.should_check_table_now(table))

        return {
            'total_tables': total_tables,
            'tables_checked': checked_now,
            'tables_skipped': total_tables - checked_now,
            'efficiency': f"{(total_tables - checked_now) / total_tables * 100:.1f}%" if total_tables > 0 else "0%",
            'adaptive_intervals': {table: self.get_adaptive_check_interval(table) for table in self.tables},
            'business_hours': self.is_business_hours()
        }

    def run_monitor(self):
        """24x daily monitoring loop with 1-hour restart cycle."""
        import pytz
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        logger.info("🚀 Starting 24x Daily Real-time Database Monitor")
        logger.info(f"📊 Monitoring {len(self.tables)} tables for {self.config['focus_year']} data")
        logger.info(f"🔔 Notifications: {'Enabled' if self.config['notification']['enabled'] else 'Disabled'}")
        logger.info(f"⏰ Current time: {datetime.now(jakarta_tz)} (Business hours: {self.is_business_hours()})")
        logger.info("🔄 Restart cycle: Every 1 hour (24x daily)")

        self.running = True
        cycle_count = 0
        hour_count = 0

        # Import timezone for consistency in the main loop
        import pytz
        jakarta_tz = pytz.timezone('Asia/Jakarta')

        while self.running:
            try:
                cycle_count += 1
                cycle_start = datetime.now(jakarta_tz)
                hour_start = datetime.now(jakarta_tz)

                # Log efficiency report every 10 cycles
                if cycle_count % 10 == 0:
                    report = self.get_monitoring_efficiency_report()
                    logger.info(f"📈 Monitoring Report #{cycle_count}:")
                    logger.info(f"   Efficiency: {report['efficiency']} tables skipped")
                    logger.info(f"   Checked: {report['tables_checked']}/{report['total_tables']}")
                    logger.info(f"   Business Hours: {report['business_hours']}")
                    logger.info(f"   Hour Cycle: #{hour_count + 1}/24")

                # Smart check all tables
                self.check_all_tables()

                # Calculate cycle time
                cycle_time = (datetime.now(jakarta_tz) - cycle_start).total_seconds()
                if cycle_time > 10:  # Warn if cycle takes too long
                    logger.warning(f"⚠️ Long monitoring cycle: {cycle_time:.1f}s")

                # 1-hour restart cycle logic
                current_time = datetime.now(jakarta_tz)
                hour_elapsed = (current_time - hour_start).total_seconds()

                if hour_elapsed >= 3600:  # 1 hour = 3600 seconds
                    hour_count += 1
                    logger.info(f"🔄 Hourly restart #{hour_count}/24 completed")
                    logger.info("⏰ Restarting monitoring cycle for fresh state")

                    # Short break before next hour cycle
                    time.sleep(5)  # 5-second break
                    continue

                # Fixed sleep for periodic checking (check every 5 minutes for new data)
                sleep_time = 300  # 5 minutes = 300 seconds
                logger.debug(f"⏰ Periodic monitoring: checking every {sleep_time}s (5 minutes)")

                time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Received interrupt - stopping monitor")
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(60)  # Wait 1 minute before retry

        self.running = False
        logger.info("🛑 24x Daily real-time monitor stopped")

    def stop(self):
        """Stop the monitor."""
        self.running = False


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Real-time database monitor for XGBoost')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Test mode - check once and exit')
    parser.add_argument('--tables', nargs='+', choices=['all', 'cg_spot_price_history', 'cg_funding_rate_history', 'cg_futures_basis_history', 'cg_spot_aggregated_taker_volume_history', 'cg_long_short_global_account_ratio_history', 'cg_long_short_top_account_ratio_history'],
                      help='Tables to monitor (default: all)')

    args = parser.parse_args()

    # Initialize monitor
    monitor = RealtimeDatabaseMonitor(config_file=args.config)

    # Override tables if specified
    if args.tables and 'all' not in args.tables:
        monitor.tables = args.tables

    if args.test:
        logger.info("🧪 Running in test mode")
        if monitor.connect_to_database():
            success = monitor.check_all_tables()
            logger.info(f"Test completed: {'Found new data' if success else 'No new data found'}")
        else:
            logger.error("❌ Database connection failed")
        return

    # Run full monitor
    try:
        monitor.run_monitor()
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()