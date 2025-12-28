#!/usr/bin/env python3
"""
Label builder for binary classification (0/1) based on trend accuracy for futures.
Creates labels for predicting whether the next price movement will be up (1) or down (0).
Uses closing prices at time t and t+1 for 1-bar horizon prediction.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LabelBuilderFutures:
    """Build binary classification labels for futures trend prediction."""

    def __init__(self, data_filter, output_dir: str = './output_train_futures'):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.label_col = 'target'

    def _format_timestamp(self, ts):
        """Format timestamp safely for display."""
        try:
            import pytz
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts/1000, tz=pytz.UTC)
            elif hasattr(ts, 'timestamp'):
                dt = ts.to_pydatetime().replace(tzinfo=pytz.UTC)
            else:
                dt = pd.to_datetime(ts)
                if dt.tz is None:
                    dt = dt.tz_localize(pytz.UTC)
            return dt.astimezone(jakarta_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
        except Exception:
            return str(ts)

    def load_feature_data(self) -> pd.DataFrame:
        """Load engineered features data."""
        logger.info("Loading engineered features data...")

        datasets_dir = self.output_dir / 'datasets'
        features_file = datasets_dir / 'features_engineered_futures.parquet'

        if not features_file.exists():
            features_file = self.output_dir / 'features_engineered_futures.parquet'

        if not features_file.exists():
            logger.error(f"Features file not found: {features_file}")
            sys.exit(1)

        try:
            df = pd.read_parquet(features_file)
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            sys.exit(1)

    def create_binary_labels(self, df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
        """
        Create binary labels based on next bar price movement.

        Args:
            df: DataFrame with price data
            threshold: Minimum price change to consider as movement (default: 0.0)

        Returns:
            DataFrame with binary labels (0=down, 1=up)
        """
        logger.info("Creating binary labels for trend prediction...")

        # Build groupby keys
        groupby_cols = ['exchange', 'symbol', 'interval']
        available_groupby = [col for col in groupby_cols if col in df.columns]
        if not available_groupby:
            available_groupby = ['symbol', 'interval']

        # Ensure data is sorted properly
        df = df.sort_values(available_groupby + ['time'])

        # Calculate next bar's close price for each group
        df['next_close'] = df.groupby(available_groupby)['price_close'].shift(-1)

        # Calculate price change
        df['price_change'] = (df['next_close'] - df['price_close']) / df['price_close']

        # Create binary labels
        df[self.label_col] = np.where(
            df['price_change'] > threshold,
            1,  # Bullish - price went up
            0   # Bearish - price went down or stayed same
        )

        # Remove last row of each group (no next price available)
        df = df.dropna(subset=['next_close', 'price_change', self.label_col])

        # Log label distribution
        label_counts = df[self.label_col].value_counts()
        label_pct = df[self.label_col].value_counts(normalize=True) * 100

        logger.info("Label Distribution:")
        logger.info(f"Bullish (1): {label_counts.get(1, 0)} ({label_pct.get(1, 0):.1f}%)")
        logger.info(f"Bearish (0): {label_counts.get(0, 0)} ({label_pct.get(0, 0):.1f}%)")

        # Clean up temporary columns
        df = df.drop(columns=['next_close', 'price_change'])

        return df

    def add_label_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistics about label distribution and balance."""
        logger.info("Adding label statistics...")

        # Overall label distribution
        total_samples = len(df)
        bullish_count = (df[self.label_col] == 1).sum()
        bearish_count = (df[self.label_col] == 0).sum()

        # Calculate imbalance ratio
        imbalance_ratio = bullish_count / bearish_count if bearish_count > 0 else float('inf')

        logger.info(f"Label Statistics:")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Imbalance ratio (bullish/bearish): {imbalance_ratio:.3f}")

        # Add label distribution by symbol/exchange/interval
        groupby_cols = ['exchange', 'symbol', 'interval']
        available_groupby = [col for col in groupby_cols if col in df.columns]

        if len(available_groupby) >= 2:
            group_stats = df.groupby(available_groupby)[self.label_col].agg(['count', 'mean']).reset_index()
            group_stats.columns = available_groupby + ['sample_count', 'bullish_ratio']

            logger.info("Label distribution by group (top 10):")
            top_groups = group_stats.nlargest(10, 'sample_count')
            for _, row in top_groups.iterrows():
                group_str = '/'.join([str(row[col]) for col in available_groupby])
                logger.info(f"  {group_str}: {row['sample_count']} samples, {row['bullish_ratio']:.1%} bullish")

        return df

    def validate_labels(self, df: pd.DataFrame) -> bool:
        """Validate label quality and distribution."""
        logger.info("\n=== Label Validation ===")

        if self.label_col not in df.columns:
            logger.error(f"Label column '{self.label_col}' not found")
            return False

        # Basic statistics
        total_samples = len(df)
        label_counts = df[self.label_col].value_counts()
        unique_labels = df[self.label_col].nunique()

        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Unique labels: {unique_labels}")
        logger.info(f"Label counts:\n{label_counts}")

        # Check for reasonable balance
        label_pct = df[self.label_col].value_counts(normalize=True)
        min_class_pct = label_pct.min() * 100
        max_class_pct = label_pct.max() * 100

        if min_class_pct < 20:
            logger.warning(f"Highly imbalanced dataset: min class = {min_class_pct:.1f}%")
        elif max_class_pct > 80:
            logger.warning(f"Highly imbalanced dataset: max class = {max_class_pct:.1f}%")
        else:
            logger.info("Reasonably balanced label distribution")

        # Check feature-label correlation
        feature_cols = [col for col in df.columns if col.startswith(('price_', 'funding_', 'basis_', 'oi_',
                                                                        'liq_', 'taker_', 'ob_', 'ls_', 'cross_'))]
        if feature_cols:
            correlations = df[feature_cols + [self.label_col]].corr()[self.label_col].abs().sort_values(ascending=False)
            logger.info("Top 10 feature-label correlations:")
            for feature, corr in correlations.head(11).items():
                if feature != self.label_col:
                    logger.info(f"  {feature}: {corr:.3f}")

        logger.info("=" * 30)
        return True

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for training."""
        logger.info("Preparing training data...")

        # Get feature columns (exclude metadata, label, and problematic object columns)
        exclude_cols = [
            'time', 'exchange', 'symbol', 'interval', 'base_asset', 'quote_asset', self.label_col,
            'pair_original', 'base_asset_extracted', 'base_asset_merge'
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Further filter: only keep numeric columns
        feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]

        # Ensure we have features
        if not feature_cols:
            logger.error("No feature columns found!")
            sys.exit(1)

        logger.info(f"Using {len(feature_cols)} features for training")

        # Prepare X and y
        X = df[feature_cols].copy()
        y = df[self.label_col].copy()

        # Handle any remaining NaN values in features
        initial_rows = len(X)
        X = X.fillna(0)
        y = y[X.index]

        final_rows = len(X)
        if initial_rows != final_rows:
            logger.info(f"Handled {initial_rows - final_rows} rows with missing features")

        # Remove any constant features
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            logger.warning(f"Removing {len(constant_features)} constant features: {constant_features}")
            X = X.drop(columns=constant_features)

        logger.info(f"Final training set: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

    def save_labeled_data(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
        """Save labeled dataset and training components."""
        if df.empty:
            logger.error("No data to save")
            return

        # Explicitly drop problematic columns that were created during merge
        columns_to_drop = ['pair_original', 'base_asset_extracted', 'base_asset_merge']
        df_save = df.drop(columns=[col for col in columns_to_drop if col in df.columns]).copy()

        # Keep only essential metadata + numeric features + label
        metadata_cols = ['time', 'exchange', 'symbol', 'interval', 'base_asset', 'quote_asset', self.label_col]

        # Select columns to keep: metadata (if exists) + numeric columns + datetime
        cols_to_keep = []
        for col in df_save.columns:
            if col in metadata_cols:
                cols_to_keep.append(col)
            elif pd.api.types.is_numeric_dtype(df_save[col]):
                cols_to_keep.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df_save[col]):
                cols_to_keep.append(col)

        df_save = df_save[cols_to_keep].copy()

        # Convert string columns to proper categorical/string type with NaN handling
        string_cols = ['exchange', 'symbol', 'interval', 'base_asset', 'quote_asset']
        for col in string_cols:
            if col in df_save.columns:
                # Fill NaN with empty string and convert to string type
                df_save[col] = df_save[col].fillna('').astype(str)

        # Save complete labeled dataset
        labeled_file = self.output_dir / 'labeled_data_futures.parquet'
        df_save.to_parquet(labeled_file, index=False)
        logger.info(f"Labeled dataset saved to {labeled_file}")

        # Save training features and labels separately
        X_file = self.output_dir / 'X_train_features_futures.parquet'
        X.to_parquet(X_file, index=False)
        logger.info(f"Features saved to {X_file}")

        y_file = self.output_dir / 'y_train_labels_futures.parquet'
        y.to_frame().to_parquet(y_file, index=False)
        logger.info(f"Labels saved to {y_file}")

        # Save feature list for training
        feature_list_file = self.output_dir / 'training_features_futures.txt'
        with open(feature_list_file, 'w') as f:
            f.write("Training Features (Futures):\n")
            f.write("=" * 30 + "\n")
            for feature in X.columns:
                f.write(f"{feature}\n")
        logger.info(f"Training feature list saved to {feature_list_file}")

        # Save dataset summary
        try:
            import pytz
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            current_time = datetime.now(jakarta_tz)

            dataset_dir = self.output_dir / 'datasets' / 'summary'
            dataset_dir.mkdir(parents=True, exist_ok=True)

            timestamp = current_time.strftime('%Y%m%d_%H%M%S')
            summary_file = dataset_dir / f'dataset_summary_futures_{timestamp}.txt'

            # Build exchange/symbol/interval lists
            exchanges = list(df['exchange'].unique()) if 'exchange' in df.columns else ['N/A']
            symbols = list(df['symbol'].unique()) if 'symbol' in df.columns else ['N/A']
            intervals = list(df['interval'].unique()) if 'interval' in df.columns else ['N/A']

            summary_content = f"""Futures Dataset Summary
{"=" * 30}

Training Session Info:
- Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S WIB')}
- Model Type: Futures Trading (XGBoost)

Dataset Statistics:
- Total samples: {len(df):,}
- Features: {len(X.columns)}
- Label distribution:
  * Bullish (1): {y.sum():,} ({y.mean()*100:.1f}%)
  * Bearish (0): {len(y) - y.sum():,} ({(1-y.mean())*100:.1f}%)

Time Range:
- Start: {df['time'].min()} ({self._format_timestamp(df['time'].min())})
- End: {df['time'].max()} ({self._format_timestamp(df['time'].max())})

Data Sources:
- Exchanges: {exchanges}
- Symbols: {symbols}
- Intervals: {intervals}

Feature Columns:
{chr(10).join(f"- {feature}" for feature in sorted(X.columns))}
"""

            with open(summary_file, 'w') as f:
                f.write(summary_content)

            logger.info(f"Dataset summary saved to {summary_file}")

        except Exception as e:
            logger.warning(f"Could not save dataset summary: {e}")


class DataFilter:
    """Simple data filter."""

    def __init__(self, args):
        self.args = args


def parse_arguments():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Futures XGBoost Training Pipeline - Label Builder"
    )
    parser.add_argument('--output-dir', type=str, default='./output_train_futures')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def main():
    """Main function to build labels."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    data_filter = DataFilter(args)

    builder = LabelBuilderFutures(data_filter, args.output_dir)

    try:
        # Load feature data
        logger.info("Loading feature data for label building...")
        df = builder.load_feature_data()

        # Create binary labels
        df = builder.create_binary_labels(df)

        # Add label statistics
        df = builder.add_label_statistics(df)

        # Validate labels
        if not builder.validate_labels(df):
            sys.exit(1)

        # Prepare training data
        X, y = builder.prepare_training_data(df)

        # Save labeled data
        builder.save_labeled_data(df, X, y)

        logger.info("\n=== Label Building Complete ===")
        logger.info("Ready for xgboost_trainer_futures.py")

    except Exception as e:
        logger.error(f"Error in label building: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
