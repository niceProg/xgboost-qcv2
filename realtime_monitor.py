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

        # Smart table configurations - based on time column for real-time detection
        self.table_configs = {
            'cg_spot_price_history': {
                'time_col': 'time',
                'key_cols': ['time', 'exchange', 'symbol', 'interval'],
                'min_new_records': 5,    # Lower threshold for real-time
                'priority': 'HIGH',
                'max_check_interval': 30,  # Check every 30 seconds
                'urgent_threshold': 20,   # Trigger immediate if >20 records
                'business_hours_only': False,
                'realtime_window': 300    # Check last 5 minutes for new data
            },
            'cg_funding_rate_history': {
                'time_col': 'time',
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'min_new_records': 3,
                'priority': 'MEDIUM',
                'max_check_interval': 60,  # Check every minute
                'urgent_threshold': 10,
                'business_hours_only': False,
                'realtime_window': 600    # Check last 10 minutes
            },
            'cg_futures_basis_history': {
                'time_col': 'time',
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'min_new_records': 3,
                'priority': 'MEDIUM',
                'max_check_interval': 60,
                'urgent_threshold': 10,
                'business_hours_only': False,
                'realtime_window': 600
            },
            'cg_spot_aggregated_taker_volume_history': {
                'time_col': 'time',
                'key_cols': ['time', 'exchange_name', 'symbol', 'interval'],
                'exchange_col': 'exchange_name',
                'min_new_records': 5,
                'priority': 'HIGH',
                'max_check_interval': 30,  # High volume - check every 30 seconds
                'urgent_threshold': 20,
                'business_hours_only': False,
                'realtime_window': 300    # Check last 5 minutes
            },
            'cg_long_short_global_account_ratio_history': {
                'time_col': 'time',
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'min_new_records': 2,
                'priority': 'LOW',
                'max_check_interval': 120,  # Check every 2 minutes
                'urgent_threshold': 5,
                'business_hours_only': True,
                'realtime_window': 900     # Check last 15 minutes
            },
            'cg_long_short_top_account_ratio_history': {
                'time_col': 'time',
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'min_new_records': 2,
                'priority': 'LOW',
                'max_check_interval': 120,
                'urgent_threshold': 5,
                'business_hours_only': True,
                'realtime_window': 900
            }
        }

        # Smart monitoring state
        self.table_activity = {}  # Track activity patterns
        self.adaptive_intervals = {}  # Dynamic check intervals

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
        now = datetime.now()
        # Major crypto market active hours (UTC)
        # 9:00 AM - 11:59 PM UTC covers most global markets
        return 9 <= now.hour <= 23

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
        """Smart logic to determine if table needs checking right now."""
        config = self.table_configs[table]
        activity = self.table_activity[table]

        # Check if enough time has passed since last check
        last_check = activity.get('last_check', datetime.min)
        adaptive_interval = self.get_adaptive_check_interval(table)

        time_since_check = (datetime.now() - last_check).total_seconds()

        if time_since_check < adaptive_interval:
            return False

        # Additional smart checks
        # Check for urgent conditions (override interval)
        if self.check_urgent_conditions(table):
            logger.info(f"🚨 Urgent condition detected for {table}")
            return True

        return True

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
        """Check for new data using smart created_at approach."""
        if not self.connection:
            if not self.connect_to_database():
                return None

        try:
            cursor = self.connection.cursor()
            table_config = self.table_configs[table]
            time_col = table_config['time_col']
            # We don't use created_col anymore, only time_col for data detection

            # Get last processed timestamp (from time column, not created_at)
            last_processed_timestamp = self.state['last_processed'][table]
            if isinstance(last_processed_timestamp, int):
                # Handle millisecond timestamp
                last_processed = datetime.fromtimestamp(last_processed_timestamp / 1000)
            else:
                # Handle ISO format string
                last_processed = datetime.fromisoformat(last_processed_timestamp)

            # Focus on 2025 data
            focus_start = datetime(self.config['focus_year'], 1, 1, 0, 0, 0)

            # Determine column name for symbols and exchanges
            symbol_col = 'symbol' if 'symbol' in table_config['key_cols'] else 'pair'
            exchange_col = table_config.get('exchange_col', 'exchange')  # Default to 'exchange'
            exchanges = self.config['exchanges']

            # Real-time query: Check for new data using time-based detection
            # Convert last_processed to milliseconds for comparison
            last_processed_ms = int(last_processed.timestamp() * 1000)
            realtime_window = table_config.get('realtime_window', 300)  # Default 5 minutes

            query = f"""
            SELECT
                COUNT(*) as count,
                MAX({time_col}) as max_time,
                MIN({time_col}) as min_time,
                COUNT(CASE WHEN {time_col} > UNIX_TIMESTAMP(DATE_SUB(NOW(), INTERVAL {realtime_window} SECOND)) * 1000 THEN 1 END) as recent_count
            FROM {table}
            WHERE {time_col} > {last_processed_ms}
            AND `{exchange_col}` IN %s
            """

            cursor.execute(query, (exchanges,))
            result = cursor.fetchone()

            if result and result[0] > 0:  # If we have new records
                # Update activity tracking
                activity = self.table_activity[table]
                activity['last_data_found'] = datetime.now()
                activity['data_frequency'] = result[3] if result[3] else 0  # recent_count

                # Calculate priority based on volume and recency
                recent_count = result[3] if result[3] else 0
                if recent_count > 0:
                    priority = "HIGH_PRIORITY"  # Recent data = high priority
                elif result[0] >= config['urgent_threshold']:
                    priority = "URGENT"
                elif result[0] >= config['min_new_records']:
                    priority = config['priority']
                else:
                    priority = "LOW"

                new_data = {
                    'table': table,
                    'new_count': result[0],
                    'min_time': result[2] if result[2] else None,
                    'max_time': result[1] if result[1] else None,
                    'recent_count': recent_count,
                    'priority': priority,
                    'last_processed': last_processed_ms,
                    'detection_method': 'time_based_realtime'
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

                # Condition 3: Recent high-frequency activity
                elif result[3] and result[3] >= table_config['urgent_threshold']:
                    should_trigger = True
                    priority = "HIGH_PRIORITY"
                    trigger_reason = f"Recent activity: {result[3]} records in last {realtime_window} seconds"

                if should_trigger:
                    logger.info(f"📊 New data in {table}: {result[0]} records (Priority: {priority})")
                    logger.info(f"   Time range: {new_data['min_time']} to {new_data['max_time']}")
                    logger.info(f"   Recent records: {new_data.get('recent_count', 0)}")
                    logger.info(f"   Reason: {trigger_reason}")

                    cursor.close()
                    return new_data
                else:
                    logger.info(f"📝 {result[0]} new records in {table} (below threshold)")

            # Update last check time even if no data found
            self.table_activity[table]['last_check'] = datetime.now()

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
        """Trigger real-time training with new data."""
        try:
            # Prepare trigger file
            trigger_file = Path('./state/realtime_trigger.json')
            trigger_info = {
                'timestamp': datetime.now().isoformat(),
                'trigger_reason': 'new_data_arrival',
                'new_data_summary': {
                    data['table']: data['new_count']
                    for data in new_data_list
                },
                'tables_with_new_data': [data['table'] for data in new_data_list]
            }

            with open(trigger_file, 'w') as f:
                json.dump(trigger_info, f, indent=2)

            logger.info(f"🚀 Triggered real-time training for {len(new_data_list)} tables")

            # Send notification
            self.send_notification(new_data_list)

            # Update last_processed times to latest data
            self.update_last_processed(new_data_list)

            return True

        except Exception as e:
            logger.error(f"Error triggering training: {e}")
            return False

    def update_last_processed(self, new_data_list: List[Dict]):
        """Update last_processed timestamps based on new data."""
        try:
            updated = False
            for data in new_data_list:
                table = data['table']
                if data['max_time']:
                    # Update to the maximum time from new data
                    self.state['last_processed'][table] = data['max_time']
                    updated = True

            if updated:
                self.save_state()
                logger.info("✅ Updated last_processed timestamps")

        except Exception as e:
            logger.error(f"Error updating last_processed: {e}")

    def send_notification(self, new_data_list: List[Dict]):
        """Send notification about new data detection."""
        if not self.config['notification']['enabled']:
            return

        try:
            total_records = sum(data['new_count'] for data in new_data_list)
            tables = ', '.join([data['table'] for data in new_data_list])

            import pytz

            # Convert to WIB (UTC+7)
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

            message = f"""
📊 New 2025 Data Detected!

📈 Total New Records: {total_records:,}
📊 Tables: {readable_tables}
⏰ Time: {now_wib.strftime('%d-%m-%Y %H:%M:%S')} WIB

Action: Real-time training triggered
Status: Model will be updated automatically
Models: Will be saved to ./output_train/models/
Performance: Check ./logs/ for detailed metrics

Table Breakdown:
""" + '\n'.join([f"• {table_names.get(data['table'], data['table'])}: {data['new_count']:,} records ({data.get('priority', 'UNKNOWN')})"
                 for data in new_data_list])

🤖 XGBoost Real-time Monitor
            """

            # Send Telegram notification if configured
            telegram_token = self.config['notification']['telegram_token']
            telegram_chat_id = self.config['notification']['telegram_chat_id']

            if telegram_token and telegram_chat_id:
                import requests
                url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
                payload = {
                    'chat_id': telegram_chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                }
                requests.post(url, json=payload)
                logger.info("📱 Telegram notification sent")

        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    def check_all_tables(self) -> bool:
        """Smart check all tables for new data using adaptive intervals."""
        tables_with_new_data = []
        checked_count = 0
        skipped_count = 0

        for table in self.tables:
            # Smart check: should we check this table now?
            if self.should_check_table_now(table):
                checked_count += 1
                logger.debug(f"🔍 Checking {table} (interval: {self.get_adaptive_check_interval(table)}s)")

                # Check for new data
                new_data = self.check_new_data(table)
                if new_data:
                    tables_with_new_data.append(new_data)

                    # If high priority data found, immediately check other priority tables
                    if new_data.get('priority') in ['URGENT', 'HIGH']:
                        logger.info(f"⚡ High priority data in {table}, accelerating other table checks")
                        self.accelerate_other_tables(table)
            else:
                skipped_count += 1
                logger.debug(f"⏭️ Skipping {table} (next check in {self.get_adaptive_check_interval(table) - (datetime.now() - self.table_activity[table]['last_check']).total_seconds():.0f}s)")

        # Log monitoring efficiency
        if checked_count > 0 or skipped_count > 0:
            logger.info(f"📊 Monitoring cycle: {checked_count} tables checked, {skipped_count} tables skipped")

        # Prioritize and trigger training if we have new data
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

            # Save state with updated check times
            self.save_state()

            return success

        # Save state for check times
        self.save_state()
        return False

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
        """Smart main monitoring loop with adaptive intervals."""
        logger.info("🚀 Starting Smart Real-time Database Monitor")
        logger.info(f"📊 Monitoring {len(self.tables)} tables for {self.config['focus_year']} data")
        logger.info(f"🔔 Notifications: {'Enabled' if self.config['notification']['enabled'] else 'Disabled'}")
        logger.info(f"⏰ Current time: {datetime.now()} (Business hours: {self.is_business_hours()})")

        self.running = True
        cycle_count = 0

        while self.running:
            try:
                cycle_count += 1
                cycle_start = datetime.now()

                # Log efficiency report every 10 cycles
                if cycle_count % 10 == 0:
                    report = self.get_monitoring_efficiency_report()
                    logger.info(f"📈 Monitoring Report #{cycle_count}:")
                    logger.info(f"   Efficiency: {report['efficiency']} tables skipped")
                    logger.info(f"   Checked: {report['tables_checked']}/{report['total_tables']}")
                    logger.info(f"   Business Hours: {report['business_hours']}")

                # Smart check all tables
                self.check_all_tables()

                # Adaptive sleep: shorter sleep when data is active, longer when quiet
                active_tables = sum(1 for t in self.tables if self.table_activity[t].get('last_data_found') and
                                  (datetime.now() - self.table_activity[t]['last_data_found']).total_seconds() < 300)

                if active_tables > 0:
                    sleep_time = 15  # Check every 15 seconds when active
                    logger.debug(f"🔄 Active period: {active_tables} tables recently active, sleeping {sleep_time}s")
                elif self.is_business_hours():
                    sleep_time = 30  # Check every 30 seconds during business hours
                    logger.debug(f"💼 Business hours: sleeping {sleep_time}s")
                else:
                    sleep_time = 60  # Check every minute during quiet hours
                    logger.debug(f"🌙 Quiet hours: sleeping {sleep_time}s")

                # Calculate cycle time
                cycle_time = (datetime.now() - cycle_start).total_seconds()
                if cycle_time > 10:  # Warn if cycle takes too long
                    logger.warning(f"⚠️ Long monitoring cycle: {cycle_time:.1f}s")

                time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Received interrupt - stopping monitor")
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(60)  # Wait 1 minute before retry

        self.running = False
        logger.info("🛑 Smart real-time monitor stopped")

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