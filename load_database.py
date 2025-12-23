#!/usr/bin/env python3
"""
Load data from 9 trading tables with filtering capabilities.
Extracts data based on exchange, pair, interval, time, and days parameters.

Core Training Tables (5):
- cg_futures_price_history
- cg_futures_aggregated_taker_buy_sell_volume_history
- cg_futures_aggregated_ask_bids_history
- cg_open_interest_aggregated_history
- cg_liquidation_aggregated_history

Support/Regime Filter Tables (4):
- cg_funding_rate_history
- cg_futures_basis_history
- cg_long_short_global_account_ratio_history
- cg_long_short_top_account_ratio_history
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pymysql
from sqlalchemy import create_engine
from dotenv import load_dotenv
from typing import Dict, Optional, List, Tuple

# Import our command line options handler
from command_line_options import parse_arguments, validate_arguments, DataFilter

# Import database storage
from database_storage import DatabaseStorage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseLoader:
    """Load data from 9 database tables with filtering capabilities (5 core + 4 support/regime filter)."""

    def __init__(self, data_filter: DataFilter, output_dir: str = './output_train',
                 enable_db_storage: bool = True):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize database storage
        self.enable_db_storage = enable_db_storage
        if enable_db_storage:
            self.db_storage = DatabaseStorage(storage_path=output_dir)
        else:
            self.db_storage = None

        # Database configuration from .env file (trading database)
        self.db_config = {
            'host': os.getenv('TRADING_DB_HOST'),
            'port': int(os.getenv('TRADING_DB_PORT', 3306)),
            'database': os.getenv('TRADING_DB_NAME'),
            'user': os.getenv('TRADING_DB_USER'),
            'password': os.getenv('TRADING_DB_PASSWORD')
        }

        # Define table schemas (9 tables: 5 core + 4 support/regime filter)
        self.tables = {
            # ===== CORE TRAINING TABLES =====
            'cg_futures_price_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'symbol',
                'key_columns': ['time', 'exchange', 'symbol', 'interval'],
                'data_columns': ['open', 'high', 'low', 'close', 'volume_usd']
            },
            'cg_futures_aggregated_taker_buy_sell_volume_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'symbol',
                'key_columns': ['time', 'exchange', 'symbol', 'interval'],
                'data_columns': ['aggregated_buy_volume', 'aggregated_sell_volume']
            },
            'cg_futures_aggregated_ask_bids_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'symbol',
                'key_columns': ['time', 'exchange', 'symbol', 'interval'],
                'data_columns': ['aggregated_bids_usd', 'aggregated_bids_quantity',
                               'aggregated_asks_usd', 'aggregated_asks_quantity']
            },
            'cg_open_interest_aggregated_history': {
                'time_col': 'time',
                'exchange_col': None,
                'pair_col': 'symbol',
                'key_columns': ['time', 'symbol', 'interval'],
                'data_columns': ['open', 'high', 'low', 'close']
            },
            'cg_liquidation_aggregated_history': {
                'time_col': 'time',
                'exchange_col': None,
                'pair_col': 'symbol',
                'key_columns': ['time', 'symbol', 'interval'],
                'data_columns': ['aggregated_long_liquidation_usd', 'aggregated_short_liquidation_usd']
            },
            # ===== SUPPORT / REGIME FILTER TABLES =====
            'cg_funding_rate_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'pair',
                'key_columns': ['time', 'exchange', 'pair', 'interval'],
                'data_columns': ['open', 'high', 'low', 'close']
            },
            'cg_futures_basis_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'pair',
                'key_columns': ['time', 'exchange', 'pair', 'interval'],
                'data_columns': ['open_basis', 'close_basis', 'open_change', 'close_change']
            },
            'cg_long_short_global_account_ratio_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'pair',
                'key_columns': ['time', 'exchange', 'pair', 'interval'],
                'data_columns': ['global_account_long_percent', 'global_account_short_percent',
                               'global_account_long_short_ratio']
            },
            'cg_long_short_top_account_ratio_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'pair',
                'key_columns': ['time', 'exchange', 'pair', 'interval'],
                'data_columns': ['top_account_long_percent', 'top_account_short_percent',
                               'top_account_long_short_ratio']
            }
        }

    def get_db_engine(self):
        """Establish database engine using SQLAlchemy."""
        try:
            # Create MySQL connection string
            connection_string = (
                f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            engine = create_engine(connection_string)
            logger.info("Successfully connected to MySQL database")
            return engine
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    def load_table_data(self, table_name: str) -> pd.DataFrame:
        """Load data from a specific table with filters."""
        table_info = self.tables[table_name]
        where_clause = self.data_filter.build_where_clause(table_name)

        # Build SELECT query with proper backtick quoting for column names
        columns = table_info['key_columns'] + table_info['data_columns']
        quoted_columns = [f"`{col}`" for col in columns]
        quoted_table_name = f"`{table_name}`"
        quoted_time_col = f"`{table_info['time_col']}`"

        query = f"""
        SELECT {', '.join(quoted_columns)}
        FROM {quoted_table_name}
        WHERE {where_clause}
        ORDER BY {quoted_time_col}
        """

        logger.info(f"Loading data from {table_name}...")
        logger.debug(f"Query: {query}")

        try:
            engine = self.get_db_engine()
            df = pd.read_sql_query(query, engine)
            logger.info(f"Loaded {len(df)} rows from {table_name}")
            return df
        except Exception as e:
            logger.error(f"Error loading {table_name}: {e}")
            return pd.DataFrame()

    def load_all_tables(self) -> Dict[str, pd.DataFrame]:
        """Load data from all 9 tables (5 core + 4 support/regime filter)."""
        all_data = {}

        for table_name in self.tables.keys():
            df = self.load_table_data(table_name)
            if not df.empty:
                all_data[table_name] = df
            else:
                logger.warning(f"No data loaded from {table_name}")

        return all_data

    def save_table_data(self, table_name: str, df: pd.DataFrame):
        """Save table data to parquet file."""
        if df.empty:
            logger.warning(f"No data to save for {table_name}")
            return

        filename = f"{table_name}.parquet"
        # Save to datasets/raw directory
        raw_data_dir = self.output_dir / 'datasets' / 'raw'
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        filepath = raw_data_dir / filename

        try:
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved {len(df)} rows to {filepath}")

            # Store to database if enabled
            if self.enable_db_storage and self.db_storage:
                try:
                    stored_path = self.db_storage.store_file(
                        filepath,
                        table_name=table_name,
                        feature_type='raw_data'
                    )
                    logger.info(f"File stored to database: {stored_path}")
                except Exception as db_e:
                    logger.warning(f"Failed to store to database: {db_e}")

        except Exception as e:
            logger.error(f"Error saving {table_name}: {e}")

    def load_data_from_csv_files(self, csv_dir: str) -> Dict[str, pd.DataFrame]:
        """Alternative: Load data from CSV files if database is not available."""
        csv_path = Path(csv_dir)
        all_data = {}

        # Expected CSV filenames
        csv_files = {
            # Core training tables
            'cg_futures_price_history': 'futures_price_history.csv',
            'cg_futures_aggregated_taker_buy_sell_volume_history': 'futures_taker_volume.csv',
            'cg_futures_aggregated_ask_bids_history': 'futures_ask_bids.csv',
            'cg_open_interest_aggregated_history': 'open_interest.csv',
            'cg_liquidation_aggregated_history': 'liquidation.csv',
            # Support/regime filter tables
            'cg_funding_rate_history': 'funding_rate_history.csv',
            'cg_futures_basis_history': 'futures_basis_history.csv',
            'cg_long_short_global_account_ratio_history': 'ls_global_ratio_history.csv',
            'cg_long_short_top_account_ratio_history': 'ls_top_ratio_history.csv'
        }

        for table_name, filename in csv_files.items():
            filepath = csv_path / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    df = self.apply_filters_to_dataframe(df, table_name)
                    all_data[table_name] = df
                    logger.info(f"Loaded {len(df)} rows from {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
            else:
                logger.warning(f"CSV file not found: {filepath}")

        return all_data

    def apply_filters_to_dataframe(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Apply data filters to a DataFrame."""
        if df.empty:
            return df

        table_info = self.tables[table_name]
        filtered_df = df.copy()

        # Time filter
        time_condition = self.data_filter.get_time_filter_sql(table_info['time_col'])
        if time_condition and time_condition != "1=1":
            # Parse SQL condition manually (simplified)
            if self.data_filter.days_filter:
                cutoff_time = int((datetime.now() - timedelta(days=self.data_filter.days_filter)).timestamp() * 1000)
                filtered_df = filtered_df[filtered_df[table_info['time_col']] >= cutoff_time]

            if self.data_filter.time_range:
                start_time, end_time = self.data_filter.time_range
                if start_time:
                    filtered_df = filtered_df[filtered_df[table_info['time_col']] >= start_time]
                if end_time:
                    filtered_df = filtered_df[filtered_df[table_info['time_col']] <= end_time]

        # Exchange filter (skip if table doesn't have exchange column)
        if self.data_filter.exchange_filter and table_info['exchange_col'] is not None:
            exchange_col = table_info['exchange_col']
            filtered_df = filtered_df[filtered_df[exchange_col].isin(self.data_filter.exchange_filter)]

        # Pair/Symbol filter
        if self.data_filter.pair_filter:
            pair_col = table_info['pair_col']
            filtered_df = filtered_df[filtered_df[pair_col].isin(self.data_filter.pair_filter)]
        elif self.data_filter.symbol_filter:
            pair_col = table_info['pair_col']
            filtered_df = filtered_df[filtered_df[pair_col].isin(self.data_filter.symbol_filter)]

        # Interval filter
        if self.data_filter.interval_filter:
            filtered_df = filtered_df[filtered_df['interval'].isin(self.data_filter.interval_filter)]

        return filtered_df.sort_values(table_info['time_col'])

    def validate_data_quality(self, df: pd.DataFrame, table_name: str) -> bool:
        """Validate data quality and print statistics."""
        if df.empty:
            logger.warning(f"No data to validate for {table_name}")
            return False

        logger.info(f"\n=== Data Quality Report for {table_name} ===")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Date range: {df[self.tables[table_name]['time_col']].min()} to {df[self.tables[table_name]['time_col']].max()}")

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values:\n{missing_counts[missing_counts > 0]}")
        else:
            logger.info("No missing values found")

        # Check for duplicates
        key_cols = self.tables[table_name]['key_columns']
        duplicates = df.duplicated(subset=key_cols).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        else:
            logger.info("No duplicate rows found")

        logger.info("=" * 50)
        return True

def main():
    """Main function to load all data."""
    args = parse_arguments()
    validate_arguments(args)

    # Set up logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create data filter
    data_filter = DataFilter(args)
    data_filter.print_filter_summary()

    # Enable database storage based on environment variable
    enable_db_storage = os.getenv('ENABLE_DB_STORAGE', 'true').lower() == 'true'

    # Initialize database loader
    loader = DatabaseLoader(data_filter, args.output_dir, enable_db_storage)

    # DATABASE STORAGE DISABLED per client requirement
    # Client tidak mau: xgboost_training_sessions, xgboost_features, xgboost_evaluations
    if enable_db_storage and loader.db_storage:
        logger.info("📝 Training session creation disabled per client requirement")
        loader.db_storage = None  # Disable further DB operations

    try:
        # Try to load from database first
        logger.info("Attempting to load data from database...")
        all_data = loader.load_all_tables()

        # If no data loaded, try CSV files
        if not all_data:
            logger.warning("No data loaded from database. Trying CSV files...")
            csv_dir = args.output_dir / 'csv_files'
            all_data = loader.load_data_from_csv_files(str(csv_dir))

        # Validate and save data
        if all_data:
            logger.info(f"\n=== Summary ===")
            logger.info(f"Successfully loaded data from {len(all_data)} tables:")

            for table_name, df in all_data.items():
                loader.validate_data_quality(df, table_name)
                loader.save_table_data(table_name, df)

            logger.info(f"\nAll data saved to {loader.output_dir}")

            # Update session status
            if loader.db_storage:
                try:
                    total_samples = sum(len(df) for df in all_data.values())
                    loader.db_storage.update_session_status(
                        status='data_loaded',
                        metrics={'total_samples': total_samples}
                    )
                except Exception as e:
                    logger.warning(f"Failed to update session status: {e}")

            logger.info("Ready for merge_7_tables.py")
        else:
            logger.error("No data could be loaded. Please check your database connection or CSV files.")

            # Update session status to failed
            if loader.db_storage:
                try:
                    loader.db_storage.update_session_status(
                        status='failed',
                        notes="Failed to load data"
                    )
                except Exception as e:
                    logger.warning(f"Failed to update session status: {e}")

            sys.exit(1)

    except Exception as e:
        logger.error(f"Error in data loading process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()