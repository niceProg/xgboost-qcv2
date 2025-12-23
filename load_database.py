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
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv
from typing import Dict, Tuple

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
        # IMPORTANT based on your DB screenshots:
        # - taker table: exchange='Binance', symbol='BTC', base_asset='BTC', interval='1h'
        # - ask/bids table: exchange_list='Binance', symbol='BTC', base_asset='BTC', range_percent exists
        # - open interest aggregated: symbol='BTC', interval='1h', OHLC columns exist
        # - liquidation aggregated: symbol='BTC', interval='1h', liquidation columns exist
        self.tables = {
            # ===== CORE TRAINING TABLES =====
            'cg_futures_price_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'symbol',  # full pair, e.g. BTCUSDT
                'key_columns': ['time', 'exchange', 'symbol', 'interval'],
                'data_columns': ['open', 'high', 'low', 'close', 'volume_usd'],
            },

            # In DB: symbol is base (BTC) not BTCUSDT -> use base_asset for filtering; normalize to BTCUSDT after load
            'cg_futures_aggregated_taker_buy_sell_volume_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'base_asset',  # base asset column in your DB
                'key_columns': ['time', 'exchange', 'interval'],
                'data_columns': ['aggregated_buy_volume', 'aggregated_sell_volume'],
                'extra_columns': ['symbol', 'base_asset', 'unit'],
            },

            # In DB: NO exchange column, symbol (BTC), base_asset (BTC), interval, range_percent
            'cg_futures_aggregated_ask_bids_history': {
                'time_col': 'time',
                'exchange_col': None,  # No exchange column in this table
                'pair_col': 'symbol',  # symbol contains base asset (BTC)
                'key_columns': ['time', 'symbol', 'interval'],
                'data_columns': [
                    'aggregated_bids_usd', 'aggregated_bids_quantity',
                    'aggregated_asks_usd', 'aggregated_asks_quantity'
                ],
                'extra_columns': ['base_asset'],
                'pair_is_base_asset': True,  # symbol is base asset, normalize to full pair
            },

            # In DB: symbol is base asset (BTC), no exchange column
            'cg_open_interest_aggregated_history': {
                'time_col': 'time',
                'exchange_col': None,
                'pair_col': 'symbol',  # contains BTC (base), not BTCUSDT
                'key_columns': ['time', 'symbol', 'interval'],
                'data_columns': ['open', 'high', 'low', 'close'],
                'extra_columns': ['unit'],
                'pair_is_base_asset': True,  # custom flag
            },

            # FIXED based on your liquidation screenshot: symbol is base asset (BTC), no exchange column
            # -> filter by base derived from pair_filter, then normalize symbol to BTCUSDT after load
            'cg_liquidation_aggregated_history': {
                'time_col': 'time',
                'exchange_col': None,
                'pair_col': 'symbol',  # contains BTC (base), not BTCUSDT
                'key_columns': ['time', 'symbol', 'interval'],
                'data_columns': ['aggregated_long_liquidation_usd', 'aggregated_short_liquidation_usd'],
                'pair_is_base_asset': True,  # custom flag
            },

            # ===== SUPPORT / REGIME FILTER TABLES =====
            'cg_funding_rate_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'pair',
                'key_columns': ['time', 'exchange', 'pair', 'interval'],
                'data_columns': ['open', 'high', 'low', 'close'],
            },
            'cg_futures_basis_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'pair',
                'key_columns': ['time', 'exchange', 'pair', 'interval'],
                'data_columns': ['open_basis', 'close_basis', 'open_change', 'close_change'],
            },
            'cg_long_short_global_account_ratio_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'pair',
                'key_columns': ['time', 'exchange', 'pair', 'interval'],
                'data_columns': ['global_account_long_percent', 'global_account_short_percent',
                                 'global_account_long_short_ratio'],
            },
            'cg_long_short_top_account_ratio_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'pair',
                'key_columns': ['time', 'exchange', 'pair', 'interval'],
                'data_columns': ['top_account_long_percent', 'top_account_short_percent',
                                 'top_account_long_short_ratio'],
            },
        }

    def _split_pair(self, pair: str) -> Tuple[str, str]:
        """Split pair string like BTCUSDT into (base, quote)."""
        if not pair:
            return "", ""
        pair = pair.upper().replace("/", "").replace("-", "").replace("_", "")
        known_quotes = ["USDT", "USDC", "BUSD", "USD", "BTC", "ETH"]
        for q in known_quotes:
            if pair.endswith(q) and len(pair) > len(q):
                return pair[:-len(q)], q
        return pair[:-3], pair[-3:]

    def _normalize_pair_string(self, s: str) -> str:
        if s is None:
            return ""
        return str(s).upper().replace("/", "").replace("-", "").replace("_", "").strip()

    def build_where_clause_for_table(self, table_name: str, table_info: dict) -> str:
        """Build WHERE clause robustly per-table (case-insensitive exchange + base_asset handling)."""
        conditions = []

        # --- Time condition ---
        time_col = f"`{table_info['time_col']}`"
        if getattr(self.data_filter, "days_filter", None):
            cutoff = int((datetime.now() - timedelta(days=self.data_filter.days_filter)).timestamp() * 1000)
            conditions.append(f"{time_col} >= {cutoff}")

        if getattr(self.data_filter, "time_range", None):
            start_time, end_time = self.data_filter.time_range
            if start_time:
                conditions.append(f"{time_col} >= {int(start_time)}")
            if end_time:
                conditions.append(f"{time_col} <= {int(end_time)}")

        # --- Exchange condition (case-insensitive) ---
        exchange_col = table_info.get("exchange_col")
        if exchange_col and getattr(self.data_filter, "exchange_filter", None):
            ex_list = [str(x).lower().strip() for x in self.data_filter.exchange_filter]
            ex_sql = ", ".join([f"'{x}'" for x in ex_list])
            conditions.append(f"LOWER(`{exchange_col}`) IN ({ex_sql})")

        # --- Pair/Symbol condition ---
        pair_filters = getattr(self.data_filter, "pair_filter", None) or getattr(self.data_filter, "symbol_filter", None)
        if pair_filters:
            pair_filters_norm = [self._normalize_pair_string(p) for p in pair_filters]
            pair_col = table_info.get("pair_col")

            if pair_col:
                # Special: table's pair_col is actually BASE asset (BTC), not full pair
                if table_info.get("pair_is_base_asset", False):
                    bases = []
                    for p in pair_filters_norm:
                        b, _q = self._split_pair(p)
                        if b:
                            bases.append(b)
                    if bases:
                        base_sql = ", ".join([f"'{b}'" for b in sorted(set(bases))])
                        conditions.append(f"`{pair_col}` IN ({base_sql})")

                elif pair_col == "base_asset":
                    bases = []
                    for p in pair_filters_norm:
                        b, _q = self._split_pair(p)
                        if b:
                            bases.append(b)
                    if bases:
                        base_sql = ", ".join([f"'{b}'" for b in sorted(set(bases))])
                        conditions.append(f"`base_asset` IN ({base_sql})")

                else:
                    pair_sql = ", ".join([f"'{p}'" for p in sorted(set(pair_filters_norm))])
                    conditions.append(f"`{pair_col}` IN ({pair_sql})")

        # --- Interval condition ---
        if getattr(self.data_filter, "interval_filter", None):
            intervals = list(set(self.data_filter.interval_filter))
            int_sql = ", ".join([f"'{i}'" for i in intervals])
            conditions.append(f"`interval` IN ({int_sql})")

        return " AND ".join(conditions) if conditions else "1=1"

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

    def load_table_data(self, table_name: str) -> pd.DataFrame:
        """Load data from a specific table with robust filters + normalization."""
        table_info = self.tables[table_name]
        where_clause = self.build_where_clause_for_table(table_name, table_info)

        columns = []
        columns += table_info.get('key_columns', [])
        columns += table_info.get('data_columns', [])
        columns += table_info.get('extra_columns', [])

        seen = set()
        columns = [c for c in columns if not (c in seen or seen.add(c))]

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

            if df.empty:
                return df

            # Normalize exchange strings (trim)
            for ex_col in ["exchange", "exchange_list"]:
                if ex_col in df.columns:
                    df[ex_col] = df[ex_col].astype(str).str.strip()

            # Build mapping base -> requested full pair (BTC -> BTCUSDT) from CLI filters
            requested_pairs = (getattr(self.data_filter, "pair_filter", None)
                               or getattr(self.data_filter, "symbol_filter", None)
                               or [])
            requested_pairs_norm = [self._normalize_pair_string(p) for p in requested_pairs]
            base_to_pair = {}
            for p in requested_pairs_norm:
                b, q = self._split_pair(p)
                if b and q:
                    base_to_pair[b] = f"{b}{q}"

            # Normalize symbol to full pair for taker table (uses base_asset column)
            if table_name == "cg_futures_aggregated_taker_buy_sell_volume_history":
                if "base_asset" in df.columns:
                    if base_to_pair:
                        df["symbol"] = df["base_asset"].astype(str).map(base_to_pair).fillna(
                            df["base_asset"].astype(str) + "USDT"
                        )
                    else:
                        df["symbol"] = df["base_asset"].astype(str) + "USDT"
                    # Drop base_asset column after normalization
                    df = df.drop(columns=['base_asset'], errors='ignore')
                # Drop unit column (contains 'coin' string, not needed for training)
                df = df.drop(columns=['unit'], errors='ignore')

            # FIX: Ask/bids aggregated uses symbol=BTC (base), no exchange column
            if table_name == "cg_futures_aggregated_ask_bids_history":
                if "symbol" in df.columns:
                    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
                    if base_to_pair:
                        df["symbol"] = df["symbol"].map(base_to_pair).fillna(df["symbol"] + "USDT")
                    else:
                        df["symbol"] = df["symbol"] + "USDT"
                # Drop base_asset column if exists
                df = df.drop(columns=['base_asset'], errors='ignore')

            # FIX: Open interest aggregated uses symbol=BTC (base)
            if table_name == "cg_open_interest_aggregated_history":
                if "symbol" in df.columns:
                    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
                    if base_to_pair:
                        df["symbol"] = df["symbol"].map(base_to_pair).fillna(df["symbol"] + "USDT")
                    else:
                        df["symbol"] = df["symbol"] + "USDT"
                # Drop unit column (contains 'coin' string, not needed for training)
                df = df.drop(columns=['unit'], errors='ignore')

            # FIX: Liquidation aggregated uses symbol=BTC (base)
            if table_name == "cg_liquidation_aggregated_history":
                if "symbol" in df.columns:
                    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
                    if base_to_pair:
                        df["symbol"] = df["symbol"].map(base_to_pair).fillna(df["symbol"] + "USDT")
                    else:
                        df["symbol"] = df["symbol"] + "USDT"

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
        raw_data_dir = self.output_dir / 'datasets' / 'raw'
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        filepath = raw_data_dir / filename

        try:
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved {len(df)} rows to {filepath}")

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

    def validate_data_quality(self, df: pd.DataFrame, table_name: str) -> bool:
        """Validate data quality and print statistics."""
        if df.empty:
            logger.warning(f"No data to validate for {table_name}")
            return False

        logger.info(f"\n=== Data Quality Report for {table_name} ===")
        logger.info(f"Total rows: {len(df)}")
        tcol = self.tables[table_name]['time_col']
        if tcol in df.columns:
            logger.info(f"Date range: {df[tcol].min()} to {df[tcol].max()}")

        missing_counts = df.isnull().sum()
        if missing_counts.any():
            bad = missing_counts[missing_counts > 0]
            if len(bad) > 0:
                logger.warning(f"Missing values:\n{bad}")
        else:
            logger.info("No missing values found")

        key_cols = [c for c in self.tables[table_name]['key_columns'] if c in df.columns]
        duplicates = df.duplicated(subset=key_cols).sum() if key_cols else 0
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows (keys={key_cols})")
        else:
            logger.info("No duplicate rows found")

        logger.info("=" * 50)
        return True


def main():
    """Main function to load all data."""
    args = parse_arguments()
    validate_arguments(args)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    data_filter = DataFilter(args)
    data_filter.print_filter_summary()

    enable_db_storage = os.getenv('ENABLE_DB_STORAGE', 'true').lower() == 'true'
    loader = DatabaseLoader(data_filter, args.output_dir, enable_db_storage)

    # DATABASE STORAGE DISABLED per client requirement
    if enable_db_storage and loader.db_storage:
        logger.info("📝 Training session creation disabled per client requirement")
        loader.db_storage = None

    try:
        logger.info("Attempting to load data from database...")
        all_data = loader.load_all_tables()

        if all_data:
            logger.info(f"\n=== Summary ===")
            logger.info(f"Successfully loaded data from {len(all_data)} tables:")

            for table_name, df in all_data.items():
                loader.validate_data_quality(df, table_name)
                loader.save_table_data(table_name, df)

            logger.info(f"\nAll data saved to {loader.output_dir}")
            logger.info("Ready for merge_7_tables.py")
        else:
            logger.error("No data could be loaded. Please check your database connection.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error in data loading process: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
