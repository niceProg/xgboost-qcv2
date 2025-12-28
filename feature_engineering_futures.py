#!/usr/bin/env python3
"""
Feature engineering for 9 futures trading tables.
Implements features optimized for futures trading with Open Interest, Liquidation, and Orderbook data.
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


class FeatureEngineerFutures:
    """Implement feature engineering based on futures table semantics."""

    def __init__(self, data_filter, output_dir: str = './output_train_futures'):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.feature_columns = []

    def load_merged_data(self) -> pd.DataFrame:
        """Load merged data from previous step."""
        logger.info("Loading merged futures data...")

        datasets_dir = self.output_dir / 'datasets'
        merged_file = datasets_dir / 'merged_futures_9_tables.parquet'

        if not merged_file.exists():
            merged_file = self.output_dir / 'merged_futures_9_tables.parquet'

        if not merged_file.exists():
            logger.error(f"Merged data file not found: {merged_file}")
            sys.exit(1)

        try:
            df = pd.read_parquet(merged_file)
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading merged data: {e}")
            sys.exit(1)

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for cg_futures_price_history (OHLCV)."""
        logger.info("Adding price features...")

        # Build groupby keys dynamically
        groupby_cols = ['exchange', 'symbol', 'interval']
        available_groupby = [col for col in groupby_cols if col in df.columns]
        if not available_groupby:
            available_groupby = ['symbol', 'interval']

        # Price returns
        df['price_close_return_1'] = df.groupby(available_groupby)['price_close'].pct_change(1)
        df['price_close_return_5'] = df.groupby(available_groupby)['price_close'].pct_change(5)
        df['price_log_return'] = np.log(df['price_close'] / df.groupby(available_groupby)['price_close'].shift(1))

        # Rolling volatility
        df['price_rolling_vol_5'] = df.groupby(available_groupby)['price_close_return_1'].transform(lambda x: x.rolling(5).std())

        # True range and candlestick features
        df['price_true_range'] = df['price_high'] - df['price_low']
        df['price_close_mean_5'] = df.groupby(available_groupby)['price_close'].transform(lambda x: x.rolling(5).mean())
        df['price_close_std_5'] = df.groupby(available_groupby)['price_close'].transform(lambda x: x.rolling(5).std())

        # Volume features
        df['price_volume_mean_10'] = df.groupby(available_groupby)['price_volume_usd'].transform(lambda x: x.rolling(10).mean())
        volume_std_10 = df.groupby(available_groupby)['price_volume_usd'].transform(lambda x: x.rolling(10).std())
        df['price_volume_zscore'] = (df['price_volume_usd'] - df['price_volume_mean_10']) / volume_std_10
        df['price_volume_change'] = df.groupby(available_groupby)['price_volume_usd'].pct_change(1)

        # Candlestick features
        df['price_wick_upper'] = df['price_high'] - np.maximum(df['price_open'], df['price_close'])
        df['price_wick_lower'] = np.minimum(df['price_open'], df['price_close']) - df['price_low']
        df['price_body_size'] = np.abs(df['price_close'] - df['price_open'])

        self.feature_columns.extend([
            'price_close_return_1', 'price_close_return_5', 'price_log_return',
            'price_rolling_vol_5', 'price_true_range', 'price_close_mean_5',
            'price_close_std_5', 'price_volume_mean_10', 'price_volume_zscore',
            'price_volume_change', 'price_wick_upper', 'price_wick_lower', 'price_body_size'
        ])

        return df

    def add_funding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for cg_funding_rate_history."""
        logger.info("Adding funding rate features...")

        if 'funding_close' not in df.columns:
            logger.warning("Funding rate data not available")
            return df

        available_groupby = [col for col in ['exchange', 'symbol', 'interval'] if col in df.columns]
        if not available_groupby:
            available_groupby = ['symbol', 'interval']

        # Normalized funding rate
        df['funding_norm'] = df['funding_close'] / 0.01

        # Rolling statistics
        df['funding_mean_24'] = df.groupby(available_groupby)['funding_close'].transform(lambda x: x.rolling(24, min_periods=1).mean())
        df['funding_std_24'] = df.groupby(available_groupby)['funding_close'].transform(lambda x: x.rolling(24, min_periods=1).std())
        df['funding_zscore'] = (df['funding_close'] - df['funding_mean_24']) / df['funding_std_24'].replace(0, 1)

        # Extreme indicators
        df['funding_extreme_positive'] = (df['funding_close'] > 0.015).astype(int)
        df['funding_extreme_negative'] = (df['funding_close'] < -0.015).astype(int)

        self.feature_columns.extend([
            'funding_norm', 'funding_mean_24', 'funding_std_24',
            'funding_zscore', 'funding_extreme_positive', 'funding_extreme_negative'
        ])

        return df

    def add_basis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for cg_futures_basis_history."""
        logger.info("Adding basis features...")

        if 'basis_close_basis' not in df.columns:
            logger.warning("Basis data not available")
            return df

        available_groupby = [col for col in ['exchange', 'symbol', 'interval'] if col in df.columns]
        if not available_groupby:
            available_groupby = ['symbol', 'interval']

        # Basis changes
        df['basis_delta'] = df.groupby(available_groupby)['basis_close_basis'].diff()
        df['basis_drift'] = df['basis_close_basis'] - df['basis_open_basis']

        # Rolling statistics
        df['basis_mean_24'] = df.groupby(available_groupby)['basis_close_basis'].transform(lambda x: x.rolling(24, min_periods=1).mean())
        basis_std_24 = df.groupby(available_groupby)['basis_close_basis'].transform(lambda x: x.rolling(24, min_periods=1).std())
        df['basis_zscore'] = (df['basis_close_basis'] - df['basis_mean_24']) / basis_std_24.replace(0, 1)
        df['basis_volatility_24'] = basis_std_24

        self.feature_columns.extend([
            'basis_delta', 'basis_drift', 'basis_mean_24',
            'basis_zscore', 'basis_volatility_24'
        ])

        return df

    def add_open_interest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for cg_open_interest_aggregated_history."""
        logger.info("Adding open interest features...")

        if 'oi_close' not in df.columns:
            logger.warning("Open interest data not available")
            return df

        available_groupby = [col for col in ['symbol', 'interval'] if col in df.columns]

        # OI changes
        df['oi_change'] = df.groupby(available_groupby)['oi_close'].pct_change(1)
        df['oi_change_abs'] = df.groupby(available_groupby)['oi_close'].diff()

        # OI momentum
        df['oi_momentum_5'] = df['oi_close'] / df.groupby(available_groupby)['oi_close'].shift(5) - 1

        # OI z-score
        oi_mean_24 = df.groupby(available_groupby)['oi_close'].transform(lambda x: x.rolling(24, min_periods=1).mean())
        oi_std_24 = df.groupby(available_groupby)['oi_close'].transform(lambda x: x.rolling(24, min_periods=1).std())
        df['oi_zscore'] = (df['oi_close'] - oi_mean_24) / oi_std_24.replace(0, 1)

        # OI vs Price - simple ratio instead of rolling correlation (to avoid multi-index issues)
        df['oi_price_ratio'] = df['oi_close'] / (df['price_close'] * 1000000)  # Normalize

        self.feature_columns.extend([
            'oi_change', 'oi_change_abs', 'oi_momentum_5',
            'oi_zscore', 'oi_price_ratio'
        ])

        return df

    def add_liquidation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for cg_liquidation_aggregated_history."""
        logger.info("Adding liquidation features...")

        if 'liq_aggregated_long_liquidation_usd' not in df.columns:
            logger.warning("Liquidation data not available")
            return df

        available_groupby = [col for col in ['symbol', 'interval'] if col in df.columns]

        # Liquidation totals
        df['liq_total'] = df['liq_aggregated_long_liquidation_usd'] + df['liq_aggregated_short_liquidation_usd']
        df['liq_ratio'] = df['liq_aggregated_long_liquidation_usd'] / (df['liq_aggregated_short_liquidation_usd'] + 1)
        df['liq_imbalance'] = df['liq_aggregated_long_liquidation_usd'] - df['liq_aggregated_short_liquidation_usd']

        # Rolling statistics
        df['liq_total_mean_12'] = df.groupby(available_groupby)['liq_total'].transform(lambda x: x.rolling(12, min_periods=1).mean())
        df['liq_long_mean_12'] = df.groupby(available_groupby)['liq_aggregated_long_liquidation_usd'].transform(lambda x: x.rolling(12, min_periods=1).mean())
        df['liq_short_mean_12'] = df.groupby(available_groupby)['liq_aggregated_short_liquidation_usd'].transform(lambda x: x.rolling(12, min_periods=1).mean())

        # Liquidation spike detection
        df['liq_spike'] = (df['liq_total'] > df['liq_total_mean_12'] * 2).astype(int)

        # Liquidation vs price relationship
        df['liq_long_zscore'] = df['liq_aggregated_long_liquidation_usd'] / (df['liq_long_mean_12'] + 1)
        df['liq_short_zscore'] = df['liq_aggregated_short_liquidation_usd'] / (df['liq_short_mean_12'] + 1)

        self.feature_columns.extend([
            'liq_total', 'liq_ratio', 'liq_imbalance',
            'liq_total_mean_12', 'liq_long_mean_12', 'liq_short_mean_12',
            'liq_spike', 'liq_long_zscore', 'liq_short_zscore'
        ])

        return df

    def add_taker_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for cg_futures_aggregated_taker_buy_sell_volume_history."""
        logger.info("Adding taker volume features...")

        if 'taker_aggregated_buy_volume' not in df.columns:
            logger.warning("Taker volume data not available")
            return df

        available_groupby = [col for col in ['exchange', 'symbol', 'interval'] if col in df.columns]
        if not available_groupby:
            available_groupby = ['symbol', 'interval']

        # Taker imbalance metrics
        total_volume = df['taker_aggregated_buy_volume'] + df['taker_aggregated_sell_volume']
        df['taker_buy_ratio'] = df['taker_aggregated_buy_volume'] / total_volume.replace(0, 1)
        df['taker_imbalance'] = df['taker_aggregated_buy_volume'] - df['taker_aggregated_sell_volume']

        # Rolling statistics
        df['taker_buy_mean_12'] = df.groupby(available_groupby)['taker_aggregated_buy_volume'].transform(lambda x: x.rolling(12, min_periods=1).mean())
        df['taker_sell_mean_12'] = df.groupby(available_groupby)['taker_aggregated_sell_volume'].transform(lambda x: x.rolling(12, min_periods=1).mean())
        df['taker_buy_std_12'] = df.groupby(available_groupby)['taker_aggregated_buy_volume'].transform(lambda x: x.rolling(12, min_periods=1).std())
        df['taker_sell_std_12'] = df.groupby(available_groupby)['taker_aggregated_sell_volume'].transform(lambda x: x.rolling(12, min_periods=1).std())

        # Z-scores
        df['taker_buy_zscore'] = (df['taker_aggregated_buy_volume'] - df['taker_buy_mean_12']) / df['taker_buy_std_12'].replace(0, 1)
        df['taker_sell_zscore'] = (df['taker_aggregated_sell_volume'] - df['taker_sell_mean_12']) / df['taker_sell_std_12'].replace(0, 1)

        # Taker momentum
        df['taker_buy_momentum'] = df['taker_aggregated_buy_volume'] / df['taker_buy_mean_12'].replace(0, 1) - 1

        self.feature_columns.extend([
            'taker_buy_ratio', 'taker_imbalance', 'taker_buy_mean_12',
            'taker_sell_mean_12', 'taker_buy_std_12', 'taker_sell_std_12',
            'taker_buy_zscore', 'taker_sell_zscore', 'taker_buy_momentum'
        ])

        return df

    def add_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for cg_futures_aggregated_ask_bids_history."""
        logger.info("Adding orderbook features...")

        if 'ob_aggregated_bids_usd' not in df.columns:
            logger.warning("Orderbook data not available")
            return df

        available_groupby = [col for col in ['symbol', 'interval'] if col in df.columns]

        # Orderbook imbalance
        total_ob = df['ob_aggregated_bids_usd'] + df['ob_aggregated_asks_usd']
        df['ob_bid_ask_ratio'] = df['ob_aggregated_bids_usd'] / total_ob.replace(0, 1)
        df['ob_imbalance_usd'] = df['ob_aggregated_bids_usd'] - df['ob_aggregated_asks_usd']

        # Quantity imbalance
        total_qty = df['ob_aggregated_bids_quantity'] + df['ob_aggregated_asks_quantity']
        df['ob_qty_bid_ask_ratio'] = df['ob_aggregated_bids_quantity'] / total_qty.replace(0, 1)

        # Rolling statistics
        df['ob_bids_mean_12'] = df.groupby(available_groupby)['ob_aggregated_bids_usd'].transform(lambda x: x.rolling(12, min_periods=1).mean())
        df['ob_asks_mean_12'] = df.groupby(available_groupby)['ob_aggregated_asks_usd'].transform(lambda x: x.rolling(12, min_periods=1).mean())

        # Orderbook pressure
        df['ob_pressure'] = (df['ob_aggregated_bids_usd'] - df['ob_bids_mean_12']) / df['ob_bids_mean_12'].replace(0, 1)
        df['ob_pressure_asks'] = (df['ob_aggregated_asks_usd'] - df['ob_asks_mean_12']) / df['ob_asks_mean_12'].replace(0, 1)

        # Orderbook depth change
        df['ob_depth_change'] = df.groupby(available_groupby)['ob_aggregated_bids_usd'].pct_change(1)

        self.feature_columns.extend([
            'ob_bid_ask_ratio', 'ob_imbalance_usd', 'ob_qty_bid_ask_ratio',
            'ob_bids_mean_12', 'ob_asks_mean_12', 'ob_pressure',
            'ob_pressure_asks', 'ob_depth_change'
        ])

        return df

    def add_longshort_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for long/short ratio tables."""
        logger.info("Adding long/short ratio features...")

        available_groupby = [col for col in ['exchange', 'symbol', 'interval'] if col in df.columns]
        if not available_groupby:
            available_groupby = ['symbol', 'interval']

        # Global account ratios
        if 'ls_global_global_account_long_short_ratio' in df.columns:
            df['ls_global_ratio'] = df['ls_global_global_account_long_percent'] / df['ls_global_global_account_short_percent'].replace(0, 1)
            global_mean_24 = df.groupby(available_groupby)['ls_global_global_account_long_short_ratio'].transform(lambda x: x.rolling(24, min_periods=1).mean())
            global_std_24 = df.groupby(available_groupby)['ls_global_global_account_long_short_ratio'].transform(lambda x: x.rolling(24, min_periods=1).std())
            df['ls_global_zscore'] = (df['ls_global_global_account_long_short_ratio'] - global_mean_24) / global_std_24.replace(0, 1)
            df['ls_global_delta'] = df.groupby(available_groupby)['ls_global_global_account_long_short_ratio'].diff()
            df['ls_global_extreme_high'] = (df['ls_global_ratio'] > 1.5).astype(int)
            df['ls_global_extreme_low'] = (df['ls_global_ratio'] < 0.7).astype(int)

            self.feature_columns.extend([
                'ls_global_ratio', 'ls_global_zscore', 'ls_global_delta',
                'ls_global_extreme_high', 'ls_global_extreme_low'
            ])

        # Top account ratios
        if 'ls_top_top_account_long_short_ratio' in df.columns:
            df['ls_top_ratio'] = df['ls_top_top_account_long_percent'] / df['ls_top_top_account_short_percent'].replace(0, 1)
            top_mean_24 = df.groupby(available_groupby)['ls_top_top_account_long_short_ratio'].transform(lambda x: x.rolling(24, min_periods=1).mean())
            top_std_24 = df.groupby(available_groupby)['ls_top_top_account_long_short_ratio'].transform(lambda x: x.rolling(24, min_periods=1).std())
            df['ls_top_zscore'] = (df['ls_top_top_account_long_short_ratio'] - top_mean_24) / top_std_24.replace(0, 1)
            df['ls_top_delta'] = df.groupby(available_groupby)['ls_top_top_account_long_short_ratio'].diff()

            self.feature_columns.extend([
                'ls_top_ratio', 'ls_top_zscore', 'ls_top_delta'
            ])

        # Cross-table interaction: top vs global
        if 'ls_top_top_account_long_short_ratio' in df.columns and 'ls_global_global_account_long_short_ratio' in df.columns:
            df['ls_top_vs_global'] = df['ls_top_top_account_long_short_ratio'] - df['ls_global_global_account_long_short_ratio']
            self.feature_columns.append('ls_top_vs_global')

        return df

    def add_cross_table_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-table interaction features."""
        logger.info("Adding cross-table features...")

        cross_features = []

        # Funding * OI change
        if 'funding_zscore' in df.columns and 'oi_change' in df.columns:
            df['cross_funding_oi'] = df['funding_zscore'] * df['oi_change']
            cross_features.append('cross_funding_oi')

        # Funding * price return
        if 'funding_zscore' in df.columns and 'price_close_return_1' in df.columns:
            df['cross_funding_price'] = df['funding_zscore'] * df['price_close_return_1']
            cross_features.append('cross_funding_price')

        # Liquidation imbalance * price return
        if 'liq_imbalance' in df.columns and 'price_close_return_1' in df.columns:
            df['cross_liq_price'] = df['liq_imbalance'] * df['price_close_return_1'] / 1e6  # Normalize
            cross_features.append('cross_liq_price')

        # OI * taker imbalance
        if 'oi_change' in df.columns and 'taker_imbalance' in df.columns:
            df['cross_oi_taker'] = df['oi_change'] * df['taker_imbalance'] / 1e9  # Normalize
            cross_features.append('cross_oi_taker')

        # Orderbook pressure * price return
        if 'ob_pressure' in df.columns and 'price_close_return_1' in df.columns:
            df['cross_ob_price'] = df['ob_pressure'] * df['price_close_return_1']
            cross_features.append('cross_ob_price')

        # LS top vs global * price return
        if 'ls_top_vs_global' in df.columns and 'price_close_return_1' in df.columns:
            df['cross_ls_price'] = df['ls_top_vs_global'] * df['price_close_return_1']
            cross_features.append('cross_ls_price')

        # Liquidation * funding
        if 'liq_total' in df.columns and 'funding_zscore' in df.columns:
            df['cross_liq_funding'] = df['liq_total'] * df['funding_zscore'] / 1e6  # Normalize
            cross_features.append('cross_liq_funding')

        self.feature_columns.extend(cross_features)
        logger.info(f"Added {len(cross_features)} cross-table features")

        return df

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for ML."""
        logger.info("Cleaning features...")

        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Get feature columns only
        feature_cols = [col for col in self.feature_columns if col in df.columns]

        # Build groupby keys dynamically
        groupby_cols = ['exchange', 'symbol', 'interval']
        available_groupby = [col for col in groupby_cols if col in df.columns]
        if not available_groupby:
            available_groupby = ['symbol', 'interval']

        # Fill missing values with forward fill then backward fill
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                if available_groupby:
                    df[col] = df.groupby(available_groupby)[col].ffill().bfill()

        # Drop rows with missing essential features
        essential_features = ['price_close_return_1', 'price_rolling_vol_5']
        available_essential = [col for col in essential_features if col in df.columns]

        if available_essential:
            initial_rows = len(df)
            df = df.dropna(subset=available_essential)
            final_rows = len(df)
            logger.info(f"Dropped {initial_rows - final_rows} rows with missing essential features")

        # Report feature statistics
        feature_df = df[feature_cols]
        missing_counts = feature_df.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values in features after cleaning:\n{missing_counts[missing_counts > 0]}")

        logger.info(f"Final feature set: {len(feature_cols)} features")
        return df

    def validate_features(self, df: pd.DataFrame) -> bool:
        """Validate engineered features."""
        logger.info("\n=== Feature Validation ===")

        feature_cols = [col for col in self.feature_columns if col in df.columns]
        logger.info(f"Total engineered features: {len(feature_cols)}")

        if not feature_cols:
            logger.error("No features were created!")
            return False

        # Check for basic statistics
        feature_stats = df[feature_cols].describe()
        logger.info("Feature statistics summary:")
        logger.info(feature_stats.loc[['mean', 'std', 'min', 'max']].round(4))

        # Check for constant features
        constant_features = []
        for col in feature_cols:
            if df[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            logger.warning(f"Constant features found: {constant_features}")
        else:
            logger.info("No constant features found")

        logger.info("=" * 30)
        return True

    def save_features(self, df: pd.DataFrame):
        """Save engineered features."""
        if df.empty:
            logger.error("No data to save")
            return

        # Create directories
        datasets_dir = self.output_dir / 'datasets'
        features_dir = self.output_dir / 'features'
        datasets_dir.mkdir(exist_ok=True)
        features_dir.mkdir(exist_ok=True)

        # Save complete dataset with features
        output_file = datasets_dir / 'features_engineered_futures.parquet'
        df.to_parquet(output_file, index=False)
        logger.info(f"Features saved to {output_file}")

        # Save feature list
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        feature_file = features_dir / 'feature_list_futures.txt'
        with open(feature_file, 'w') as f:
            f.write("Engineered Features (Futures):\n")
            f.write("=" * 30 + "\n")
            for col in feature_cols:
                f.write(f"{col}\n")
        logger.info(f"Feature list saved to {feature_file}")

        # Save features-only dataset
        features_df = df[feature_cols].copy()
        features_file = datasets_dir / 'features_only_futures.parquet'
        features_df.to_parquet(features_file, index=False)
        logger.info(f"Features-only dataset saved to {features_file}")


class DataFilter:
    """Simple data filter."""

    def __init__(self, args):
        self.args = args


def parse_arguments():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Futures XGBoost Training Pipeline - Feature Engineering"
    )
    parser.add_argument('--output-dir', type=str, default='./output_train_futures')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def main():
    """Main function to engineer features."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    data_filter = DataFilter(args)

    engineer = FeatureEngineerFutures(data_filter, args.output_dir)

    try:
        # Load merged data
        logger.info("Loading merged data for feature engineering...")
        df = engineer.load_merged_data()

        # Add features for each table type
        df = engineer.add_price_features(df)
        df = engineer.add_funding_features(df)
        df = engineer.add_basis_features(df)
        df = engineer.add_open_interest_features(df)
        df = engineer.add_liquidation_features(df)
        df = engineer.add_taker_volume_features(df)
        df = engineer.add_orderbook_features(df)
        df = engineer.add_longshort_features(df)
        df = engineer.add_cross_table_features(df)

        # Clean features
        df = engineer.clean_features(df)

        # Validate features
        if not engineer.validate_features(df):
            sys.exit(1)

        # Save features
        engineer.save_features(df)

        logger.info("\n=== Feature Engineering Complete ===")
        logger.info("Ready for label_builder_futures.py")

    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
