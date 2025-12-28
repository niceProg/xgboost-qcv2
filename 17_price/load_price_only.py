#!/usr/bin/env python3
"""
Simple Price-Only Data Loader for 17 Price Features Training.

This script ONLY loads the price history table from database.
No 9-table merging, no complex feature engineering.
Just OHLCV data for 17 price features training.
"""

import argparse
import sys
import logging
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
# Look in parent directory (where .env file is located)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# SQLAlchemy for database connection
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SIMPLE PRICE TABLE SCHEMA (from cg_futures_price_history)
# =============================================================================
PRICE_TABLE_COLUMNS = {
    'time': 'time',
    'exchange': 'exchange',
    'symbol': 'symbol',            # e.g., 'BTCUSDT'
    'base_asset': 'base_asset',    # e.g., 'BTC'
    'quote_asset': 'quote_asset',  # e.g., 'USDT'
    'interval': 'interval',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume_usd': 'volume_usd',    # Renamed to 'volume' for feature engineering
}


class PriceOnlyLoader:
    """Load ONLY price history from database."""

    def __init__(self, output_dir: str = './output_train_17price'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Database configuration from .env file
        self.db_config = {
            'host': os.getenv('TRADING_DB_HOST'),
            'port': int(os.getenv('TRADING_DB_PORT', 3306)),
            'database': os.getenv('TRADING_DB_NAME'),
            'user': os.getenv('TRADING_DB_USER'),
            'password': os.getenv('TRADING_DB_PASSWORD')
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

    def build_where_clause(self, args) -> str:
        """Build WHERE clause for price table."""
        conditions = []

        # Time filter (escape `time` with backticks - it's a reserved keyword!)
        if args.time:
            try:
                start_time, end_time = args.time.split(',')
                conditions.append(f"`time` >= {start_time}")
                conditions.append(f"`time` <= {end_time}")
            except ValueError:
                logger.warning(f"Invalid time format: {args.time}")

        # Days filter (recent N days)
        if args.days:
            # Calculate timestamp for N days ago
            days_ago_ms = args.days * 24 * 60 * 60 * 1000
            current_time_ms = int(datetime.now().timestamp() * 1000)
            start_time = current_time_ms - days_ago_ms
            conditions.append(f"`time` >= {start_time}")

        # Exchange filter
        if args.exchange:
            exchanges = ','.join([f"'{e}'" for e in args.exchange.split(',')])
            conditions.append(f"`exchange` IN ({exchanges})")

        # Symbol filter (base_asset)
        if args.symbol:
            symbols = ','.join([f"'{s}'" for s in args.symbol.split(',')])
            conditions.append(f"`base_asset` IN ({symbols})")

        # Pair filter
        if args.pair:
            pairs = ','.join([f"'{p}'" for p in args.pair.split(',')])
            conditions.append(f"CONCAT(`base_asset`, 'USDT') IN ({pairs})")

        # Interval filter (escape `interval` - it's a reserved keyword!)
        if args.interval:
            intervals = ','.join([f"'{i}'" for i in args.interval.split(',')])
            conditions.append(f"`interval` IN ({intervals})")

        return " AND ".join(conditions) if conditions else "1=1"

    def load_price_data(self, args) -> pd.DataFrame:
        """Load price data from cg_futures_price_history table."""
        engine = self.get_db_engine()
        where_clause = self.build_where_clause(args)

        # Select columns from database (note: column is volume_usd, not volume!)
        # Need to escape reserved keywords with backticks
        columns = ['`time`', '`exchange`', '`base_asset`', '`interval`',
                   '`open`', '`high`', '`low`', '`close`', '`volume_usd`']

        query = f"""
            SELECT {', '.join(columns)}
            FROM cg_futures_price_history
            WHERE {where_clause}
            ORDER BY `time` ASC
        """

        logger.info(f"Executing query...")
        logger.info(f"WHERE clause: {where_clause}")

        try:
            df = pd.read_sql(query, engine)
            logger.info(f"Loaded {len(df)} rows from cg_futures_price_history")

            # Rename volume_usd to volume for compatibility with feature engineering
            df = df.rename(columns={'volume_usd': 'volume'})

            return df
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, output_path: str):
        """Save data to CSV file."""
        if df.empty:
            logger.warning("No data to save")
            return

        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")

    def save_to_parquet(self, df: pd.DataFrame, output_dir: str):
        """Save data to parquet file."""
        if df.empty:
            logger.warning("No data to save")
            return

        raw_data_dir = Path(output_dir) / 'datasets' / 'raw'
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        filepath = raw_data_dir / 'cg_futures_price_history.parquet'

        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")

    def print_summary(self, df: pd.DataFrame):
        """Print data summary."""
        if df.empty:
            logger.warning("No data to summarize")
            return

        logger.info("\n" + "="*50)
        logger.info("DATA SUMMARY")
        logger.info("="*50)
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")

        if 'base_asset' in df.columns:
            logger.info(f"Assets: {df['base_asset'].unique().tolist()}")

        if 'exchange' in df.columns:
            logger.info(f"Exchanges: {df['exchange'].unique().tolist()}")

        if 'interval' in df.columns:
            logger.info(f"Intervals: {df['interval'].unique().tolist()}")

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values:\n{missing[missing > 0]}")
        else:
            logger.info("No missing values")

        logger.info("="*50)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Price-Only Data Loader - 17 Price Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python load_price_only.py --symbol BTC --interval 1h --days 30
  python load_price_only.py --exchange binance --symbol BTC,ETH --interval 1h
        """
    )

    parser.add_argument('--exchange', type=str, help='Exchange: binance,okx,bybit')
    parser.add_argument('--pair', type=str, help='Trading pair: BTCUSDT,ETHUSDT')
    parser.add_argument('--symbol', type=str, help='Symbol (base asset): BTC,ETH')
    parser.add_argument('--interval', type=str, help='Interval: 1m,5m,1h,4h')
    parser.add_argument('--time', type=str, help='Time range (ms): start_time,end_time')
    parser.add_argument('--days', type=int, help='Number of recent days')
    parser.add_argument('--output-dir', type=str, default='./output_train_17price',
                       help='Output directory (default: ./output_train_17price)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("="*50)
    logger.info("PRICE-ONLY DATA LOADER")
    logger.info("="*50)
    logger.info("Loading ONLY price history table (no 9-table merge)")

    loader = PriceOnlyLoader(args.output_dir)

    try:
        # Load data
        df = loader.load_price_data(args)

        if df.empty:
            logger.error("No data loaded!")
            return 1

        # Print summary
        loader.print_summary(df)

        # Save to both parquet and CSV
        csv_path = Path(args.output_dir) / 'data.csv'
        loader.save_to_csv(df, csv_path)
        loader.save_to_parquet(df, args.output_dir)

        logger.info("\n✅ Data loading complete!")
        logger.info(f"CSV: {csv_path}")
        logger.info(f"Parquet: {args.output_dir}/datasets/raw/cg_futures_price_history.parquet")

        return 0

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
