#!/usr/bin/env python3
"""
Load data from 9 futures trading tables with filtering capabilities.
Extracts Coinglass data based on exchange, pair, interval, time, and days parameters.
Optimized for futures trading with Open Interest, Liquidation, and Orderbook data.
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseLoaderFutures:
    """Load data from 9 futures trading tables with filtering capabilities."""

    def __init__(self, data_filter, output_dir: str = './output_train_futures', tables_filter: str = None):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        # Parse tables filter if provided (e.g., "cg_futures_price_history" or "cg_futures_price_history,cg_funding_rate_history")
        self.tables_filter = set(tables_filter.split(',')) if tables_filter else None

        # Database configuration from .env file
        self.db_config = {
            'host': os.getenv('TRADING_DB_HOST'),
            'port': int(os.getenv('TRADING_DB_PORT', 3306)),
            'database': os.getenv('TRADING_DB_NAME'),
            'user': os.getenv('TRADING_DB_USER'),
            'password': os.getenv('TRADING_DB_PASSWORD')
        }

        # Define 9 futures table schemas with accurate column mappings
        self.tables = {
            'cg_futures_price_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'symbol_col': 'symbol',           # BTCUSDT
                'base_asset_col': 'base_asset',   # BTC
                'pair_col': None,
                'key_columns': ['time', 'exchange', 'symbol', 'base_asset', 'quote_asset', 'interval'],
                'data_columns': ['open', 'high', 'low', 'close', 'volume_usd'],
                'symbol_type': 'pair',            # symbol is full pair (BTCUSDT)
                'required': True
            },
            'cg_funding_rate_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'symbol_col': None,
                'base_asset_col': None,
                'pair_col': 'pair',               # BTCUSDT
                'key_columns': ['time', 'exchange', 'pair', 'interval'],
                'data_columns': ['open', 'high', 'low', 'close'],
                'symbol_type': 'pair',
                'required': False
            },
            'cg_futures_basis_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'symbol_col': None,
                'base_asset_col': None,
                'pair_col': 'pair',               # BTCUSDT
                'key_columns': ['time', 'exchange', 'pair', 'interval'],
                'data_columns': ['open_basis', 'close_basis', 'open_change', 'close_change'],
                'symbol_type': 'pair',
                'required': False
            },
            'cg_open_interest_aggregated_history': {
                'time_col': 'time',
                'exchange_col': None,
                'symbol_col': 'symbol',           # BTC (base asset)
                'base_asset_col': None,
                'pair_col': None,
                'key_columns': ['time', 'symbol', 'interval', 'unit'],
                'data_columns': ['open', 'high', 'low', 'close'],
                'symbol_type': 'base_asset',      # symbol is base asset (BTC)
                'required': False
            },
            'cg_liquidation_aggregated_history': {
                'time_col': 'time',
                'exchange_col': None,
                'symbol_col': 'symbol',           # BTC (base asset)
                'base_asset_col': None,
                'pair_col': None,
                'key_columns': ['time', 'symbol', 'interval'],
                'data_columns': ['aggregated_long_liquidation_usd', 'aggregated_short_liquidation_usd'],
                'symbol_type': 'base_asset',
                'required': False
            },
            'cg_futures_aggregated_taker_buy_sell_volume_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'symbol_col': 'symbol',           # BTC (base asset)
                'base_asset_col': 'base_asset',   # BTC
                'pair_col': None,
                'key_columns': ['time', 'exchange', 'symbol', 'base_asset', 'interval', 'unit'],
                'data_columns': ['aggregated_buy_volume', 'aggregated_sell_volume'],
                'symbol_type': 'base_asset',
                'required': False
            },
            'cg_futures_aggregated_ask_bids_history': {
                'time_col': 'time',
                'exchange_col': 'exchange_list',
                'symbol_col': 'symbol',           # BTC (base asset)
                'base_asset_col': 'base_asset',   # BTC
                'pair_col': None,
                'key_columns': ['time', 'exchange_list', 'symbol', 'base_asset', 'interval', 'range_percent'],
                'data_columns': ['aggregated_bids_usd', 'aggregated_bids_quantity',
                               'aggregated_asks_usd', 'aggregated_asks_quantity'],
                'symbol_type': 'base_asset',
                'required': False
            },
            'cg_long_short_global_account_ratio_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'symbol_col': None,
                'base_asset_col': None,
                'pair_col': 'pair',               # BTCUSDT
                'key_columns': ['time', 'exchange', 'pair', 'interval'],
                'data_columns': ['global_account_long_percent', 'global_account_short_percent',
                               'global_account_long_short_ratio'],
                'symbol_type': 'pair',
                'required': False
            },
            'cg_long_short_top_account_ratio_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'symbol_col': None,
                'base_asset_col': None,
                'pair_col': 'pair',               # BTCUSDT
                'key_columns': ['time', 'exchange', 'pair', 'interval'],
                'data_columns': ['top_account_long_percent', 'top_account_short_percent',
                               'top_account_long_short_ratio'],
                'symbol_type': 'pair',
                'required': False
            }
        }

    def get_db_engine(self):
        """Establish database engine using SQLAlchemy."""
        try:
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

    def build_where_clause(self, table_name: str) -> str:
        """Build WHERE clause for a specific futures table."""
        conditions = []
        table_info = self.tables[table_name]

        # Time filter
        time_condition = self.data_filter.get_time_filter_sql('time')
        if time_condition:
            conditions.append(time_condition)

        # Exchange filter
        if table_info['exchange_col']:
            exchange_condition = self.data_filter.get_exchange_filter_sql(table_info['exchange_col'])
            if exchange_condition:
                conditions.append(exchange_condition)

        # Symbol/Pair filter based on symbol_type
        if table_info['symbol_type'] == 'base_asset':
            # Tables with symbol as base asset (BTC)
            # Use --symbol filter
            if self.data_filter.symbol_filter and table_info['symbol_col']:
                symbol_condition = self.data_filter.get_symbol_filter_sql(table_info['symbol_col'])
                if symbol_condition:
                    conditions.append(symbol_condition)
            elif self.data_filter.symbol_filter and table_info['base_asset_col']:
                base_asset_condition = self.data_filter.get_symbol_filter_sql(table_info['base_asset_col'])
                if base_asset_condition:
                    conditions.append(base_asset_condition)
        else:
            # Tables with pair as full pair (BTCUSDT)
            # Use --pair filter for pair column
            if self.data_filter.pair_filter and table_info['pair_col']:
                pair_condition = self.data_filter.get_pair_filter_sql(table_info['pair_col'])
                if pair_condition:
                    conditions.append(pair_condition)
            # Also check --symbol filter for symbol column (futures_price_history)
            elif self.data_filter.symbol_filter and table_info['symbol_col']:
                # For futures_price_history, --symbol BTC should match base_asset
                # We need to add base_asset filter
                if table_info['base_asset_col']:
                    base_asset_condition = self.data_filter.get_symbol_filter_sql(table_info['base_asset_col'])
                    if base_asset_condition:
                        conditions.append(base_asset_condition)

        # Interval filter
        interval_condition = self.data_filter.get_interval_filter_sql('interval')
        if interval_condition:
            conditions.append(interval_condition)

        return " AND ".join(conditions) if conditions else "1=1"

    def load_table_data(self, table_name: str) -> pd.DataFrame:
        """Load data from a specific futures table with filters."""
        table_info = self.tables[table_name]
        where_clause = self.build_where_clause(table_name)

        # Build SELECT query
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
            if table_info['required']:
                logger.error(f"Required table {table_name} failed to load")
                sys.exit(1)
            return pd.DataFrame()

    def load_all_tables(self) -> Dict[str, pd.DataFrame]:
        """Load data from futures tables (optionally filtered by tables_filter)."""
        all_data = {}

        # Filter tables if tables_filter is set
        tables_to_load = self.tables.keys() if self.tables_filter is None else self.tables_filter

        logger.info(f"Tables to load: {', '.join(sorted(tables_to_load))}")

        for table_name in tables_to_load:
            if table_name not in self.tables:
                logger.warning(f"Unknown table: {table_name}, skipping")
                continue

            info = self.tables[table_name]
            df = self.load_table_data(table_name)
            if not df.empty:
                all_data[table_name] = df
            else:
                if info['required']:
                    logger.error(f"Required table {table_name} is empty")
                    sys.exit(1)
                else:
                    logger.warning(f"Optional table {table_name} is empty, skipping")

        return all_data

    def save_table_data(self, table_name: str, df: pd.DataFrame):
        """Save table data to parquet file."""
        if df.empty:
            logger.warning(f"No data to save for {table_name}")
            return

        filename = f"{table_name}.parquet"
        raw_data_dir = self.output_dir / 'datasets' / 'raw'
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        filepath = raw_data_dir / filename

        try:
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved {len(df)} rows to {filepath}")
        except Exception as e:
            logger.error(f"Error saving {table_name}: {e}")

    def validate_data_quality(self, df: pd.DataFrame, table_name: str) -> bool:
        """Validate data quality and print statistics."""
        if df.empty:
            logger.warning(f"No data to validate for {table_name}")
            return False

        logger.info(f"\n=== Data Quality Report for {table_name} ===")
        logger.info(f"Total rows: {len(df)}")

        time_col = self.tables[table_name]['time_col']
        logger.info(f"Date range: {df[time_col].min()} to {df[time_col].max()}")

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


class DataFilter:
    """Simple data filter for futures tables."""

    def __init__(self, args):
        self.args = args
        self.exchange_filter = self._parse_comma_list(args.exchange) if args.exchange else None
        self.pair_filter = self._parse_comma_list(args.pair) if args.pair else None
        self.symbol_filter = self._parse_comma_list(args.symbol) if args.symbol else None
        self.interval_filter = self._parse_comma_list(args.interval) if args.interval else None
        self.time_range = self._parse_time_range(args.time) if args.time else None
        self.days_filter = args.days

    def _parse_comma_list(self, comma_str: str) -> List[str]:
        return [item.strip() for item in comma_str.split(',') if item.strip()]

    def _parse_time_range(self, time_str: str) -> tuple:
        try:
            parts = time_str.split(',')
            if len(parts) == 1:
                start_time = int(parts[0].strip())
                return (start_time, None)
            elif len(parts) == 2:
                start_time = int(parts[0].strip()) if parts[0].strip() else None
                end_time = int(parts[1].strip()) if parts[1].strip() else None
                return (start_time, end_time)
        except ValueError:
            print(f"Error parsing time range: {time_str}")
            sys.exit(1)

    def get_time_filter_sql(self, time_col: str = 'time') -> str:
        quoted_col = f"`{time_col}`"
        if self.days_filter:
            n_days_ago = datetime.now() - timedelta(days=self.days_filter)
            start_timestamp = int(n_days_ago.timestamp() * 1000)
            return f"{quoted_col} >= {start_timestamp}"
        if self.time_range:
            start_time, end_time = self.time_range
            conditions = []
            if start_time:
                conditions.append(f"{quoted_col} >= {start_time}")
            if end_time:
                conditions.append(f"{quoted_col} <= {end_time}")
            return " AND ".join(conditions) if conditions else None
        return None

    def get_exchange_filter_sql(self, exchange_col: str = 'exchange') -> str:
        if not self.exchange_filter:
            return None
        quoted_col = f"`{exchange_col}`"
        return f"{quoted_col} IN ({','.join([repr(ex) for ex in self.exchange_filter])})"

    def get_pair_filter_sql(self, pair_col: str = 'pair') -> str:
        if not self.pair_filter:
            return None
        quoted_col = f"`{pair_col}`"
        return f"{quoted_col} IN ({','.join([repr(pair) for pair in self.pair_filter])})"

    def get_symbol_filter_sql(self, symbol_col: str = 'symbol') -> str:
        """Get filter SQL for symbol/base_asset column (--symbol filter)."""
        if not self.symbol_filter:
            return None
        quoted_col = f"`{symbol_col}`"
        return f"{quoted_col} IN ({','.join([repr(symbol) for symbol in self.symbol_filter])})"

    def get_interval_filter_sql(self, interval_col: str = 'interval') -> str:
        if not self.interval_filter:
            return None
        quoted_col = f"`{interval_col}`"
        return f"{quoted_col} IN ({','.join([repr(interval) for interval in self.interval_filter])})"

    def print_filter_summary(self):
        print("=== Active Filters ===")
        if self.exchange_filter:
            print(f"Exchange(s): {', '.join(self.exchange_filter)}")
        if self.pair_filter:
            print(f"Pair(s): {', '.join(self.pair_filter)}")
        if self.symbol_filter:
            print(f"Symbol(s): {', '.join(self.symbol_filter)}")
        if self.interval_filter:
            print(f"Interval(s): {', '.join(self.interval_filter)}")
        if self.time_range:
            start_time, end_time = self.time_range
            if start_time:
                start_dt = datetime.fromtimestamp(start_time/1000)
                print(f"Start time: {start_dt} ({start_time})")
            if end_time:
                end_dt = datetime.fromtimestamp(end_time/1000)
                print(f"End time: {end_dt} ({end_time})")
        if self.days_filter:
            print(f"Days: {self.days_filter}")
        print("=" * 20)


def parse_arguments():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Futures XGBoost Training Pipeline - Load Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python load_database_futures.py --exchange binance --symbol BTC --interval 1h
  python load_database_futures.py --symbol BTC,ETH --interval 1h --days 30
        """
    )

    parser.add_argument('--exchange', type=str, help='Exchange name(s): binance,okx,bybit')
    parser.add_argument('--pair', type=str, help='Trading pair(s): BTCUSDT,ETHUSDT')
    parser.add_argument('--symbol', type=str, help='Symbol(s): BTC,ETH (base asset)')
    parser.add_argument('--interval', type=str, help='Time interval(s): 1m,5m,1h,4h')
    parser.add_argument('--time', type=str, help='Time range (ms): start_time,end_time')
    parser.add_argument('--days', type=int, help='Number of recent days')
    parser.add_argument('--tables', type=str, help='Specific tables to load (comma-separated): cg_futures_price_history,cg_funding_rate_history,...')
    parser.add_argument('--output-dir', type=str, default='./output_train_futures', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    return parser.parse_args()


def main():
    """Main function to load futures data (with optional table filtering)."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    data_filter = DataFilter(args)
    data_filter.print_filter_summary()

    # Get tables filter from arguments (if provided)
    tables_filter = getattr(args, 'tables', None)

    loader = DatabaseLoaderFutures(data_filter, args.output_dir, tables_filter)

    try:
        logger.info("Loading futures data from database...")
        all_data = loader.load_all_tables()

        if all_data:
            logger.info(f"\n=== Summary ===")
            logger.info(f"Successfully loaded data from {len(all_data)} tables:")

            for table_name, df in all_data.items():
                loader.validate_data_quality(df, table_name)
                loader.save_table_data(table_name, df)

            logger.info(f"\nAll data saved to {loader.output_dir}")
            logger.info("Ready for merge_futures_9_tables.py")
        else:
            logger.error("No data could be loaded.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error in data loading process: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
