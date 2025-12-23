#!/usr/bin/env python3
"""
Merge 9 trading tables into a unified DataFrame (FUTURES-ONLY).
Handles time alignment, missing data, and prepares data for feature engineering.

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
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our command line options handler
from command_line_options import parse_arguments, validate_arguments, DataFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TableMerger:
    """Merge 9 trading tables into a unified dataset (FUTURES-ONLY)."""

    def __init__(self, data_filter: DataFilter, output_dir: str = './output_train'):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.merged_data = None

        # Define table priority and merge strategies
        self.base_table = 'cg_futures_price_history'  # Primary table for time base
        self.tables_info = {
            # ===== CORE TRAINING TABLES =====
            'cg_futures_price_history': {
                'prefix': 'price',
                'key_cols': ['time', 'exchange', 'symbol', 'interval'],
                'data_cols': ['open', 'high', 'low', 'close', 'volume_usd'],
                'required': True
            },
            'cg_futures_aggregated_taker_buy_sell_volume_history': {
                'prefix': 'taker',
                'key_cols': ['time', 'exchange', 'symbol', 'interval'],
                'data_cols': ['aggregated_buy_volume', 'aggregated_sell_volume'],
                'required': False
            },
            'cg_futures_aggregated_ask_bids_history': {
                'prefix': 'orderbook',
                'key_cols': ['time', 'symbol', 'interval'],  # No exchange column
                'data_cols': ['aggregated_bids_usd', 'aggregated_bids_quantity',
                             'aggregated_asks_usd', 'aggregated_asks_quantity'],
                'required': False
            },
            'cg_open_interest_aggregated_history': {
                'prefix': 'oi',
                'key_cols': ['time', 'symbol', 'interval'],
                'data_cols': ['open', 'high', 'low', 'close'],
                'required': False
            },
            'cg_liquidation_aggregated_history': {
                'prefix': 'liq',
                'key_cols': ['time', 'symbol', 'interval'],
                'data_cols': ['aggregated_long_liquidation_usd', 'aggregated_short_liquidation_usd'],
                'required': False
            },
            # ===== SUPPORT / REGIME FILTER TABLES =====
            'cg_funding_rate_history': {
                'prefix': 'funding',
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'data_cols': ['open', 'high', 'low', 'close'],
                'required': False
            },
            'cg_futures_basis_history': {
                'prefix': 'basis',
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'data_cols': ['open_basis', 'close_basis', 'open_change', 'close_change'],
                'required': False
            },
            'cg_long_short_global_account_ratio_history': {
                'prefix': 'ls_global',
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'data_cols': ['global_account_long_percent', 'global_account_short_percent',
                             'global_account_long_short_ratio'],
                'required': False
            },
            'cg_long_short_top_account_ratio_history': {
                'prefix': 'ls_top',
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'data_cols': ['top_account_long_percent', 'top_account_short_percent',
                             'top_account_long_short_ratio'],
                'required': False
            }
        }

    def load_table_data(self) -> Dict[str, pd.DataFrame]:
        """Load all table data from parquet files."""
        logger.info("Loading table data from parquet files...")
        all_data = {}

        # Check for base table first (try new structure first, then root)
        raw_data_dir = self.output_dir / 'datasets' / 'raw'
        base_file = raw_data_dir / f"{self.base_table}.parquet"

        # Fallback to root directory for backward compatibility
        if not base_file.exists():
            base_file = self.output_dir / f"{self.base_table}.parquet"

        if not base_file.exists():
            logger.error(f"Base table {self.base_table} not found at {base_file}")
            logger.error(f"Checked: {raw_data_dir / f'{self.base_table}.parquet'} and {self.output_dir / f'{self.base_table}.parquet'}")
            sys.exit(1)

            # Load all tables
        for table_name, info in self.tables_info.items():
            # Try new structure first, then fallback to root
            file_path = raw_data_dir / f"{table_name}.parquet"
            if not file_path.exists():
                file_path = self.output_dir / f"{table_name}.parquet"

            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    all_data[table_name] = df
                    logger.info(f"Loaded {len(df)} rows from {table_name}")
                except Exception as e:
                    logger.error(f"Error loading {table_name}: {e}")
                    if info['required']:
                        logger.error(f"Required table {table_name} failed to load")
                        sys.exit(1)
            else:
                if info['required']:
                    logger.error(f"Required table {table_name} not found at {file_path}")
                    sys.exit(1)
                else:
                    logger.warning(f"Optional table {table_name} not found, skipping")
                    all_data[table_name] = pd.DataFrame()

        return all_data

    def standardize_column_names(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Standardize column names across tables."""
        info = self.tables_info[table_name]
        df_std = df.copy()

        # Create standardized key columns
        key_mapping = {}
        for col in info['key_cols']:
            if col == 'exchange_name' or col == 'exchange_list':
                key_mapping[col] = 'exchange'
            elif col == 'pair':
                key_mapping[col] = 'symbol'
            elif col == 'range_percent':
                # Keep range_percent as is (it's an additional key for orderbook)
                continue
            else:
                key_mapping[col] = col

        # Rename key columns
        df_std = df_std.rename(columns=key_mapping)

        # Add prefix to data columns
        data_col_mapping = {}
        for col in info['data_cols']:
            if col not in df_std.columns:
                logger.warning(f"Column {col} not found in {table_name}")
                continue
            data_col_mapping[col] = f"{info['prefix']}_{col}"

        df_std = df_std.rename(columns=data_col_mapping)

        return df_std

    def prepare_base_dataframe(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the base DataFrame with standardized columns."""
        logger.info("Preparing base DataFrame...")

        # Standardize base table
        base_std = self.standardize_column_names(base_df, self.base_table)

        # Ensure time column is datetime
        base_std['time'] = pd.to_datetime(base_std['time'], unit='ms')

        # Sort by time
        base_std = base_std.sort_values(['time', 'exchange', 'symbol', 'interval'])

        # Remove duplicates
        base_std = base_std.drop_duplicates(subset=['time', 'exchange', 'symbol', 'interval'], keep='last')

        logger.info(f"Base DataFrame prepared with {len(base_std)} rows")
        return base_std

    def merge_table(self, base_df: pd.DataFrame, table_df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Merge a single table with the base DataFrame."""
        if table_df.empty:
            logger.info(f"Skipping {table_name} (empty DataFrame)")
            return base_df

        logger.info(f"Merging {table_name}...")

        # Standardize table columns
        table_std = self.standardize_column_names(table_df, table_name)
        table_std['time'] = pd.to_datetime(table_std['time'], unit='ms')

        # Tables without exchange column (aggregated tables)
        no_exchange_tables = ['cg_open_interest_aggregated_history', 'cg_liquidation_aggregated_history']

        # Remove duplicates and define merge keys based on table type
        if table_name in no_exchange_tables:
            # These tables only have time + symbol + interval
            table_std = table_std.drop_duplicates(subset=['time', 'symbol', 'interval'], keep='last')
            merge_keys = ['time', 'symbol', 'interval']
        else:
            # Tables with exchange column
            table_std = table_std.drop_duplicates(subset=['time', 'exchange', 'symbol', 'interval'], keep='last')
            merge_keys = ['time', 'exchange', 'symbol', 'interval']

        # Merge with outer join to keep all timestamps
        merged = pd.merge(
            base_df,
            table_std,
            on=merge_keys,
            how='left',
            suffixes=('', f'_{table_name}_dup')
        )

        # Remove duplicate columns if any
        dup_cols = [col for col in merged.columns if col.endswith(f'_{table_name}_dup')]
        if dup_cols:
            merged = merged.drop(columns=dup_cols)

        logger.info(f"Merged {table_name}, result: {len(merged)} rows")
        return merged

    def merge_all_tables(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all tables into a unified DataFrame."""
        logger.info("Starting table merge process...")

        # Start with base table
        base_df = all_data[self.base_table]
        merged_df = self.prepare_base_dataframe(base_df)

        # Merge each additional table
        for table_name, table_df in all_data.items():
            if table_name == self.base_table:
                continue

            merged_df = self.merge_table(merged_df, table_df, table_name)

        # Sort final result
        merged_df = merged_df.sort_values(['time', 'exchange', 'symbol', 'interval'])

        logger.info(f"Merge completed. Final DataFrame: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        return merged_df

    def clean_merged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare merged data for feature engineering."""
        logger.info("Cleaning merged data...")

        # Remove rows with no price data (must have close price)
        df_clean = df.dropna(subset=['price_close']).copy()

        # Forward fill missing values for each symbol/exchange/interval combination
        key_cols = ['exchange', 'symbol', 'interval']

        # Sort data for proper forward fill
        df_clean = df_clean.sort_values(['time'] + key_cols)

        # Group by keys and forward fill
        for col in df_clean.columns:
            if col not in ['time'] + key_cols and df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col] = df_clean.groupby(key_cols)[col].ffill()

        # Remove rows where essential price data is still missing
        essential_cols = ['price_open', 'price_high', 'price_low', 'price_close']
        df_clean = df_clean.dropna(subset=essential_cols)

        # Log statistics
        total_rows = len(df)
        clean_rows = len(df_clean)
        logger.info(f"Data cleaning: {total_rows} -> {clean_rows} rows ({clean_rows/total_rows:.1%} retained)")

        # Report missing value statistics
        missing_stats = df_clean.isnull().sum()
        missing_cols = missing_stats[missing_stats > 0]

        if len(missing_cols) > 0:
            logger.info("Missing value statistics after cleaning:")
            for col, count in missing_cols.items():
                pct_missing = count / len(df_clean) * 100
                logger.info(f"  {col}: {count} ({pct_missing:.1f}%)")
        else:
            logger.info("No missing values after cleaning")

        return df_clean

    def validate_merged_data(self, df: pd.DataFrame) -> bool:
        """Validate merged data quality."""
        logger.info("\n=== Merged Data Validation ===")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Total columns: {len(df.columns)}")

        # Time range
        logger.info(f"Time range: {df['time'].min()} to {df['time'].max()}")

        # Unique values
        logger.info(f"Exchanges: {df['exchange'].nunique()} - {list(df['exchange'].unique())}")
        logger.info(f"Symbols: {df['symbol'].nunique()} - {list(df['symbol'].unique())}")
        logger.info(f"Intervals: {df['interval'].nunique()} - {list(df['interval'].unique())}")

        # Check for gaps in time series
        if len(df) > 1:
            time_diffs = df['time'].diff().dropna()
            median_diff = time_diffs.median()
            logger.info(f"Median time difference: {median_diff}")

            # Flag large gaps
            large_gaps = time_diffs[time_diffs > median_diff * 2]
            if len(large_gaps) > 0:
                logger.warning(f"Found {len(large_gaps)} large time gaps (> 2x median)")

        # Essential data completeness
        essential_cols = ['price_open', 'price_high', 'price_low', 'price_close']
        essential_complete = df[essential_cols].notnull().all(axis=1).sum()
        logger.info(f"Rows with complete price data: {essential_complete}/{len(df)} ({essential_complete/len(df):.1%})")

        logger.info("=" * 40)
        return True

    def save_merged_data(self, df: pd.DataFrame):
        """Save merged data to file."""
        if df.empty:
            logger.error("No data to save")
            return

        # Create datasets directory
        datasets_dir = self.output_dir / 'datasets'
        features_dir = self.output_dir / 'features'
        datasets_dir.mkdir(exist_ok=True)
        features_dir.mkdir(exist_ok=True)

        # Save as parquet to datasets directory
        output_file = datasets_dir / 'merged_9_tables.parquet'
        df.to_parquet(output_file, index=False)
        logger.info(f"Merged data saved to {output_file}")

        # Also save as CSV to datasets directory for easy inspection
        csv_file = datasets_dir / 'merged_9_tables.csv'
        df.to_csv(csv_file, index=False)
        logger.info(f"Merged data saved to {csv_file}")

        # Save column mapping to features directory
        mapping_file = features_dir / 'column_mapping.txt'
        with open(mapping_file, 'w') as f:
            f.write("Column Mapping:\n")
            f.write("=" * 20 + "\n")
            for col in sorted(df.columns):
                f.write(f"{col}\n")
        logger.info(f"Column mapping saved to {mapping_file}")

        # DATABASE STORAGE DISABLED per client requirement
        # Client tidak mau: xgboost_evaluations, xgboost_features, xgboost_training_sessions
        logger.info("📝 Merged data database storage disabled per client requirement")

def main():
    """Main function to merge all tables."""
    args = parse_arguments()
    validate_arguments(args)

    # Set up logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create data filter
    data_filter = DataFilter(args)

    # Initialize table merger
    merger = TableMerger(data_filter, args.output_dir)

    try:
        # Load all table data
        logger.info("Loading table data...")
        all_data = merger.load_table_data()

        # Merge all tables
        logger.info("Merging tables...")
        merged_data = merger.merge_all_tables(all_data)

        # Clean merged data
        logger.info("Cleaning merged data...")
        cleaned_data = merger.clean_merged_data(merged_data)

        # Validate result
        merger.validate_merged_data(cleaned_data)

        # Save merged data
        merger.save_merged_data(cleaned_data)

        logger.info("\n=== Merge Complete ===")
        logger.info("Ready for feature_engineering.py")

    except Exception as e:
        logger.error(f"Error in merge process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()