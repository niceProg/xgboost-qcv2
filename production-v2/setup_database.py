#!/usr/bin/env python3
"""
Database Setup Script for XGBoost Real-time Trading System
Adds created_at columns and indexes for smart monitoring
"""

import os
import sys
import pymysql
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Setup database for smart real-time monitoring."""

    def __init__(self):
        self.db_config = {
            'host': os.getenv('TRADING_DB_HOST', 'localhost'),
            'port': int(os.getenv('TRADING_DB_PORT', 3306)),
            'user': os.getenv('TRADING_DB_USER'),
            'password': os.getenv('TRADING_DB_PASSWORD'),
            'database': os.getenv('TRADING_DB_NAME', 'newera'),
            'charset': 'utf8mb4'
        }

        # Tables to setup
        self.tables = [
            {
                'name': 'cg_spot_price_history',
                'time_col': 'time',
                'symbol_col': 'symbol',
                'key_cols': ['time', 'exchange', 'symbol', 'interval']
            },
            {
                'name': 'cg_funding_rate_history',
                'time_col': 'time',
                'symbol_col': 'pair',
                'key_cols': ['time', 'exchange', 'pair', 'interval']
            },
            {
                'name': 'cg_futures_basis_history',
                'time_col': 'time',
                'symbol_col': 'pair',
                'key_cols': ['time', 'exchange', 'pair', 'interval']
            },
            {
                'name': 'cg_spot_aggregated_taker_volume_history',
                'time_col': 'time',
                'symbol_col': 'symbol',
                'key_cols': ['time', 'exchange_name', 'symbol', 'interval']
            },
            {
                'name': 'cg_long_short_global_account_ratio_history',
                'time_col': 'time',
                'symbol_col': 'pair',
                'key_cols': ['time', 'exchange', 'pair', 'interval']
            },
            {
                'name': 'cg_long_short_top_account_ratio_history',
                'time_col': 'time',
                'symbol_col': 'pair',
                'key_cols': ['time', 'exchange', 'pair', 'interval']
            }
        ]

    def connect_to_database(self):
        """Connect to database."""
        try:
            conn = pymysql.connect(**self.db_config)
            logger.info(f"✅ Connected to database: {self.db_config['database']}")
            return conn
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return None

    def table_exists(self, conn, table_name: str) -> bool:
        """Check if table exists."""
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """, (self.db_config['database'], table_name))
        exists = cursor.fetchone()[0] > 0
        cursor.close()
        return exists

    def column_exists(self, conn, table_name: str, column_name: str) -> bool:
        """Check if column exists in table."""
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = %s
        """, (self.db_config['database'], table_name, column_name))
        exists = cursor.fetchone()[0] > 0
        cursor.close()
        return exists

    def create_created_at_column(self, conn, table_config):
        """Create created_at column if it doesn't exist."""
        table_name = table_config['name']
        cursor = conn.cursor()

        try:
            if not self.column_exists(conn, table_name, 'created_at'):
                logger.info(f"📝 Adding created_at column to {table_name}")

                # Add created_at column
                cursor.execute(f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """)

                # Update existing records to use time column as created_at
                logger.info(f"🔄 Updating existing records in {table_name}")
                cursor.execute(f"""
                    UPDATE {table_name}
                    SET created_at = {table_config['time_col']}
                    WHERE created_at IS NULL
                """)

                # Create index on created_at for fast queries
                logger.info(f"📊 Creating index on created_at for {table_name}")
                cursor.execute(f"""
                    CREATE INDEX idx_{table_name}_created_at ON {table_name}(created_at)
                """)

                # Create composite index for monitoring queries
                logger.info(f"📊 Creating monitoring index for {table_name}")
                key_cols = table_config['key_cols']
                index_cols = ['created_at', 'exchange'] + [col for col in key_cols if col not in ['time', 'exchange']]

                if index_cols:
                    index_name = f"idx_{table_name}_monitoring"
                    cursor.execute(f"""
                        CREATE INDEX {index_name} ON {table_name}({', '.join(index_cols)})
                    """)

                logger.info(f"✅ Setup completed for {table_name}")
                conn.commit()

            else:
                logger.info(f"✅ created_at column already exists in {table_name}")

            cursor.close()
            return True

        except Exception as e:
            logger.error(f"❌ Error setting up {table_name}: {e}")
            cursor.close()
            conn.rollback()
            return False

    def create_trigger_table(self, conn):
        """Create trigger notification table for event-driven monitoring."""
        cursor = conn.cursor()

        try:
            logger.info("📝 Creating trigger notification table")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_arrival_notifications (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    table_name VARCHAR(100) NOT NULL,
                    record_count INT NOT NULL,
                    exchange VARCHAR(50),
                    symbol VARCHAR(50),
                    priority ENUM('LOW', 'MEDIUM', 'HIGH', 'URGENT') DEFAULT 'MEDIUM',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE,
                    processed_at TIMESTAMP NULL,
                    INDEX idx_created_processed (created_at, processed),
                    INDEX idx_table_priority (table_name, priority),
                    INDEX idx_processed (processed)
                )
            """)
            conn.commit()
            logger.info("✅ Trigger notification table created")
            cursor.close()
            return True

        except Exception as e:
            logger.error(f"❌ Error creating trigger table: {e}")
            cursor.close()
            return False

    def analyze_data_patterns(self, conn):
        """Analyze existing data patterns for optimization."""
        cursor = conn.cursor()

        logger.info("📈 Analyzing data patterns for optimization")

        for table_config in self.tables:
            table_name = table_config['name']

            try:
                # Count total records
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                total_records = cursor.fetchone()[0]

                # Get date range
                cursor.execute(f"""
                    SELECT MIN({table_config['time_col']}), MAX({table_config['time_col']})
                    FROM {table_name}
                """)
                min_date, max_date = cursor.fetchone()

                # Count 2025 records
                cursor.execute(f"""
                    SELECT COUNT(*)
                    FROM {table_name}
                    WHERE YEAR({table_config['time_col']}) = 2025
                """)
                records_2025 = cursor.fetchone()[0]

                # Get recent activity (last 24 hours)
                cursor.execute(f"""
                    SELECT COUNT(*)
                    FROM {table_name}
                    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 DAY)
                """)
                recent_activity = cursor.fetchone()[0]

                logger.info(f"📊 {table_name}:")
                logger.info(f"   Total records: {total_records:,}")
                logger.info(f"   Date range: {min_date} to {max_date}")
                logger.info(f"   2025 records: {records_2025:,}")
                logger.info(f"   Last 24h: {recent_activity:,}")

            except Exception as e:
                logger.error(f"❌ Error analyzing {table_name}: {e}")

        cursor.close()

    def setup_database(self):
        """Run complete database setup."""
        logger.info("🚀 Starting database setup for smart monitoring")

        conn = self.connect_to_database()
        if not conn:
            return False

        try:
            # Create trigger notification table
            if not self.create_trigger_table(conn):
                return False

            # Setup each table
            success_count = 0
            for table_config in self.tables:
                table_name = table_config['name']

                if self.table_exists(conn, table_name):
                    if self.create_created_at_column(conn, table_config):
                        success_count += 1
                else:
                    logger.warning(f"⚠️ Table {table_name} does not exist, skipping")

            logger.info(f"📊 Setup completed: {success_count}/{len(self.tables)} tables configured")

            # Analyze data patterns
            self.analyze_data_patterns(conn)

            conn.close()
            return success_count > 0

        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            if conn:
                conn.close()
            return False

def main():
    """Main function."""
    print("🛠️ Database Setup for XGBoost Real-time Trading System")
    print("=" * 60)

    # Check environment variables
    required_vars = ['TRADING_DB_HOST', 'TRADING_DB_USER', 'TRADING_DB_PASSWORD', 'TRADING_DB_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file")
        sys.exit(1)

    # Run setup
    setup = DatabaseSetup()
    success = setup.setup_database()

    if success:
        print("\n✅ Database setup completed successfully!")
        print("\n🚀 Your smart monitoring system is now ready!")
        print("📊 Tables are configured with created_at columns and indexes")
        print("🔄 The monitor will now use efficient event-driven checking")
    else:
        print("\n❌ Database setup failed!")
        print("Please check the error messages above and fix issues")
        sys.exit(1)

if __name__ == "__main__":
    main()