#!/usr/bin/env python3
"""
Merge 9 futures trading tables into a unified DataFrame.
Handles time alignment, missing data, and prepares data for feature engineering.
Optimized for futures trading with Open Interest, Liquidation, and Orderbook data.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TableMergerFutures:
    """Merge 9 futures trading tables into a unified dataset."""

    def __init__(self, data_filter, output_dir: str = './output_train_futures'):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.merged_data = None

        # Base table is futures price history
        self.base_table = 'cg_futures_price_history'

        # Define 9 futures table schemas with symbol_type
        self.tables_info = {
            'cg_futures_price_history': {
                'prefix': 'price',
                'symbol_type': 'pair',      # symbol = BTCUSDT, has base_asset column
                'key_cols': ['time', 'exchange', 'symbol', 'base_asset', 'quote_asset', 'interval'],
                'data_cols': ['open', 'high', 'low', 'close', 'volume_usd'],
                'required': True
            },
            'cg_funding_rate_history': {
                'prefix': 'funding',
                'symbol_type': 'pair',      # pair = BTCUSDT (no base_asset column)
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'data_cols': ['open', 'high', 'low', 'close'],
                'required': False
            },
            'cg_futures_basis_history': {
                'prefix': 'basis',
                'symbol_type': 'pair',      # pair = BTCUSDT (no base_asset column)
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'data_cols': ['open_basis', 'close_basis', 'open_change', 'close_change'],
                'required': False
            },
            'cg_open_interest_aggregated_history': {
                'prefix': 'oi',
                'symbol_type': 'base_asset', # symbol = BTC
                'key_cols': ['time', 'symbol', 'interval', 'unit'],
                'data_cols': ['open', 'high', 'low', 'close'],
                'required': False
            },
            'cg_liquidation_aggregated_history': {
                'prefix': 'liq',
                'symbol_type': 'base_asset', # symbol = BTC
                'key_cols': ['time', 'symbol', 'interval'],
                'data_cols': ['aggregated_long_liquidation_usd', 'aggregated_short_liquidation_usd'],
                'required': False
            },
            'cg_futures_aggregated_taker_buy_sell_volume_history': {
                'prefix': 'taker',
                'symbol_type': 'base_asset', # symbol = BTC, has base_asset column
                'key_cols': ['time', 'exchange', 'symbol', 'base_asset', 'interval', 'unit'],
                'data_cols': ['aggregated_buy_volume', 'aggregated_sell_volume'],
                'required': False
            },
            'cg_futures_aggregated_ask_bids_history': {
                'prefix': 'ob',
                'symbol_type': 'base_asset', # symbol = BTC, has base_asset column
                'key_cols': ['time', 'exchange_list', 'symbol', 'base_asset', 'interval', 'range_percent'],
                'data_cols': ['aggregated_bids_usd', 'aggregated_bids_quantity',
                               'aggregated_asks_usd', 'aggregated_asks_quantity'],
                'required': False
            },
            'cg_long_short_global_account_ratio_history': {
                'prefix': 'ls_global',
                'symbol_type': 'pair',      # pair = BTCUSDT (no base_asset column)
                'key_cols': ['time', 'exchange', 'pair', 'interval'],
                'data_cols': ['global_account_long_percent', 'global_account_short_percent',
                             'global_account_long_short_ratio'],
                'required': False
            },
            'cg_long_short_top_account_ratio_history': {
                'prefix': 'ls_top',
                'symbol_type': 'pair',      # pair = BTCUSDT (no base_asset column)
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

        raw_data_dir = self.output_dir / 'datasets' / 'raw'

        # Check base table first
        base_file = raw_data_dir / f"{self.base_table}.parquet"
        if not base_file.exists():
            base_file = self.output_dir / f"{self.base_table}.parquet"

        if not base_file.exists():
            logger.error(f"Base table {self.base_table} not found")
            sys.exit(1)

        # Load all tables
        for table_name, info in self.tables_info.items():
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
                    logger.error(f"Required table {table_name} not found")
                    sys.exit(1)
                else:
                    logger.warning(f"Optional table {table_name} not found, skipping")
                    all_data[table_name] = pd.DataFrame()

        return all_data

    def extract_base_asset_from_pair(self, pair: str) -> str:
        """Extract base asset from trading pair (e.g., BTCUSDT -> BTC)."""
        if not isinstance(pair, str):
            return pair
        # Common quote assets to remove
        quote_assets = ['USDT', 'USD', 'BUSD', 'USDC', 'TUSD', 'USDD', 'FDUSD', 'EUR', 'GBP']
        for quote in quote_assets:
            if pair.endswith(quote):
                return pair[:-len(quote)]
        return pair

    def standardize_column_names(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Standardize column names across tables."""
        info = self.tables_info[table_name]
        df_std = df.copy()

        # Create standardized key columns
        key_mapping = {}
        for col in info['key_cols']:
            if col == 'pair':
                # For tables with 'pair' column, rename to 'pair_original' and create 'base_asset'
                key_mapping[col] = 'pair_original'
            elif col == 'exchange_list':
                key_mapping[col] = 'exchange'
            else:
                key_mapping[col] = col

        # Rename key columns
        df_std = df_std.rename(columns=key_mapping)

        # Extract base_asset from pair for tables with symbol_type='pair'
        if info['symbol_type'] == 'pair' and 'pair_original' in df_std.columns:
            df_std['base_asset_extracted'] = df_std['pair_original'].apply(self.extract_base_asset_from_pair)
            logger.info(f"Extracted base_asset from pair for {table_name}")

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

        # Determine merge keys based on symbol_type
        table_info = self.tables_info[table_name]

        if table_info['symbol_type'] == 'base_asset':
            # Tables with symbol as base asset (BTC)
            # Merge on base_asset column (from symbol or base_asset column)
            if 'base_asset' in table_std.columns:
                # Use existing base_asset column
                pass
            else:
                # Rename symbol to base_asset for merge
                table_std = table_std.rename(columns={'symbol': 'base_asset'})

            # Remove duplicates
            dup_subset = ['time', 'base_asset', 'interval']
            if 'exchange' in table_std.columns:
                dup_subset.append('exchange')
            table_std = table_std.drop_duplicates(subset=dup_subset, keep='last')

            # Define merge keys
            merge_keys = ['time', 'base_asset', 'interval']
            if 'exchange' in base_df.columns and 'exchange' in table_std.columns:
                merge_keys.append('exchange')

        else:
            # Tables with pair as full pair (BTCUSDT)
            # Extract base_asset and merge on it
            # Rename base_asset_extracted to base_asset for matching
            if 'base_asset_extracted' in table_std.columns:
                table_std = table_std.rename(columns={'base_asset_extracted': 'base_asset_merge'})

            # Remove duplicates
            dup_subset = ['time', 'base_asset_merge', 'interval']
            if 'exchange' in table_std.columns:
                dup_subset.append('exchange')
            table_std = table_std.drop_duplicates(subset=dup_subset, keep='last')

            # Define merge keys - same length for both
            merge_keys = ['time', 'interval', 'base_asset']
            if 'exchange' in base_df.columns and 'exchange' in table_std.columns:
                merge_keys.append('exchange')

            # Rename base_asset in table_std to match with base_df
            table_std = table_std.rename(columns={'base_asset_merge': 'base_asset'})

            # Merge with left join
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

        # Merge with left join for base_asset type tables
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
        available_key_cols = [col for col in key_cols if col in df_clean.columns]

        if not available_key_cols:
            available_key_cols = ['symbol', 'interval']

        # Sort data for proper forward fill
        sort_cols = ['time'] + available_key_cols
        df_clean = df_clean.sort_values(sort_cols)

        # Group by keys and forward fill
        for col in df_clean.columns:
            if col not in ['time'] + available_key_cols and df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col] = df_clean.groupby(available_key_cols)[col].ffill()

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
        if 'exchange' in df.columns:
            logger.info(f"Exchanges: {df['exchange'].nunique()} - {list(df['exchange'].unique())}")
        logger.info(f"Symbols: {df['symbol'].nunique()} - {list(df['symbol'].unique())}")
        logger.info(f"Intervals: {df['interval'].nunique()} - {list(df['interval'].unique())}")

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

        # Create directories
        datasets_dir = self.output_dir / 'datasets'
        features_dir = self.output_dir / 'features'
        datasets_dir.mkdir(exist_ok=True)
        features_dir.mkdir(exist_ok=True)

        # Save as parquet
        output_file = datasets_dir / 'merged_futures_9_tables.parquet'
        df.to_parquet(output_file, index=False)
        logger.info(f"Merged data saved to {output_file}")

        # Also save as CSV for easy inspection
        csv_file = datasets_dir / 'merged_futures_9_tables.csv'
        df.to_csv(csv_file, index=False)
        logger.info(f"Merged data saved to {csv_file}")

        # Save column mapping
        mapping_file = features_dir / 'column_mapping_futures.txt'
        with open(mapping_file, 'w') as f:
            f.write("Column Mapping:\n")
            f.write("=" * 20 + "\n")
            for col in sorted(df.columns):
                f.write(f"{col}\n")
        logger.info(f"Column mapping saved to {mapping_file}")


class DataFilter:
    """Simple data filter for futures tables."""

    def __init__(self, args):
        self.args = args

    def print_filter_summary(self):
        print("=== Futures Merge ===")
        print(f"Output directory: {getattr(self.args, 'output_dir', './output_train_futures')}")
        print("=" * 20)


def parse_arguments():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Futures XGBoost Training Pipeline - Merge Tables"
    )
    parser.add_argument('--output-dir', type=str, default='./output_train_futures')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def main():
    """Main function to merge all futures tables."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    data_filter = DataFilter(args)
    data_filter.print_filter_summary()

    merger = TableMergerFutures(data_filter, args.output_dir)

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
        logger.info("Ready for feature_engineering_futures.py")

    except Exception as e:
        logger.error(f"Error in merge process: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
