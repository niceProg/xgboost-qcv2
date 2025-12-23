#!/usr/bin/env python3
"""
Label builder for binary classification (0/1) based on trend accuracy.
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

# Import our command line options handler
from command_line_options import parse_arguments, validate_arguments, DataFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LabelBuilder:
    """Build binary classification labels for trend prediction."""

    def __init__(self, data_filter: DataFilter, output_dir: str = './output_train'):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.label_col = 'target'
        # Store training parameters for database storage
        self.exchange = getattr(data_filter, 'exchange', 'N/A') if hasattr(data_filter, 'exchange') else 'N/A'
        self.pair = getattr(data_filter, 'pair', 'N/A') if hasattr(data_filter, 'pair') else 'N/A'
        self.interval = getattr(data_filter, 'interval', 'N/A') if hasattr(data_filter, 'interval') else 'N/A'

    def _format_timestamp(self, ts, jakarta_tz):
        """Format timestamp safely for display."""
        try:
            import pytz
            if isinstance(ts, (int, float)):
                # Unix timestamp in milliseconds
                dt = datetime.fromtimestamp(ts/1000, tz=pytz.UTC)
            elif hasattr(ts, 'timestamp'):
                # Pandas Timestamp object
                dt = ts.to_pydatetime().replace(tzinfo=pytz.UTC)
            else:
                # Try direct parsing
                dt = pd.to_datetime(ts)
                if dt.tz is None:
                    dt = dt.tz_localize(pytz.UTC)

            return dt.astimezone(jakarta_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
        except Exception as e:
            logger.warning(f"Could not format timestamp {ts}: {e}")
            return str(ts)

    def load_feature_data(self) -> pd.DataFrame:
        """Load engineered features data."""
        logger.info("Loading engineered features data...")

        # Try datasets directory first (new structure), then root (compatibility)
        datasets_dir = self.output_dir / 'datasets'
        features_file = datasets_dir / 'features_engineered.parquet'

        if not features_file.exists():
            # Fallback to root directory for backward compatibility
            features_file = self.output_dir / 'features_engineered.parquet'

        if not features_file.exists():
            logger.error(f"Features file not found: {features_file}")
            logger.error(f"Checked: {datasets_dir / 'features_engineered.parquet'} and {self.output_dir / 'features_engineered.parquet'}")
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
                      Can be set to avoid very small price changes due to noise

        Returns:
            DataFrame with binary labels (0=down, 1=up)
        """
        logger.info("Creating binary labels for trend prediction...")

        # Ensure data is sorted properly
        df = df.sort_values(['exchange', 'symbol', 'interval', 'time'])

        # Calculate next bar's close price for each group
        df['next_close'] = df.groupby(['exchange', 'symbol', 'interval'])['price_close'].shift(-1)

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
        group_stats = df.groupby(['exchange', 'symbol', 'interval'])[self.label_col].agg([
            'count', 'mean'
        ]).reset_index()
        group_stats.columns = ['exchange', 'symbol', 'interval', 'sample_count', 'bullish_ratio']

        logger.info("Label distribution by group (top 10):")
        top_groups = group_stats.nlargest(10, 'sample_count')
        for _, row in top_groups.iterrows():
            logger.info(f"  {row['exchange']}/{row['symbol']}/{row['interval']}: "
                       f"{row['sample_count']} samples, {row['bullish_ratio']:.1%} bullish")

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

        # Check for reasonable balance (between 20% and 80% for each class)
        label_pct = df[self.label_col].value_counts(normalize=True)
        min_class_pct = label_pct.min() * 100
        max_class_pct = label_pct.max() * 100

        if min_class_pct < 20:
            logger.warning(f"Highly imbalanced dataset: min class = {min_class_pct:.1f}%")
        elif max_class_pct > 80:
            logger.warning(f"Highly imbalanced dataset: max class = {max_class_pct:.1f}%")
        else:
            logger.info("Reasonably balanced label distribution")

        # Check for data leakage (no future information)
        # Verify that time is properly sorted
        for (exchange, symbol, interval), group in df.groupby(['exchange', 'symbol', 'interval']):
            time_diffs = group['time'].diff().dropna()
            if (time_diffs <= pd.Timedelta(0)).any():
                logger.warning(f"Time not properly sorted for {exchange}/{symbol}/{interval}")
                break
        else:
            logger.info("Time validation passed - no future leakage detected")

        # Check feature-label correlation (basic sanity check)
        feature_cols = [col for col in df.columns if col.startswith(('price_', 'funding_', 'basis_', 'taker_', 'ob_', 'ls_', 'cross_'))]
        if feature_cols:
            correlations = df[feature_cols + [self.label_col]].corr()[self.label_col].abs().sort_values(ascending=False)
            logger.info("Top 10 feature-label correlations:")
            for feature, corr in correlations.head(11).items():  # 11 because includes self-correlation
                if feature != self.label_col:
                    logger.info(f"  {feature}: {corr:.3f}")

        logger.info("=" * 30)
        return True

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for training."""
        logger.info("Preparing training data...")

        # Get feature columns (exclude metadata and label)
        exclude_cols = ['time', 'exchange', 'symbol', 'interval', self.label_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

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
        X = X.fillna(0)  # Simple imputation
        y = y[X.index]  # Align labels

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

        # Save complete labeled dataset
        labeled_file = self.output_dir / 'labeled_data.parquet'
        df.to_parquet(labeled_file, index=False)
        logger.info(f"Labeled dataset saved to {labeled_file}")

        # Save training features and labels separately
        X_file = self.output_dir / 'X_train_features.parquet'
        X.to_parquet(X_file, index=False)
        logger.info(f"Features saved to {X_file}")

        y_file = self.output_dir / 'y_train_labels.parquet'
        y.to_frame().to_parquet(y_file, index=False)
        logger.info(f"Labels saved to {y_file}")

        # Save feature list for training
        feature_list_file = self.output_dir / 'training_features.txt'
        with open(feature_list_file, 'w') as f:
            f.write("Training Features:\n")
            f.write("=" * 20 + "\n")
            for feature in X.columns:
                f.write(f"{feature}\n")
        logger.info(f"Training feature list saved to {feature_list_file}")

        # Save dataset summary with structured path
        import pytz
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        current_time = datetime.now(jakarta_tz)

        # Create structured directory
        dataset_dir = self.output_dir / 'datasets' / 'summary'
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped filename
        timestamp = current_time.strftime('%Y%m%d_%H%M%S')
        summary_file = dataset_dir / f'dataset_summary_{timestamp}.txt'

        # Generate summary content
        summary_content = f"""Dataset Summary
{"=" * 20}

Training Session Info:
- Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S WIB')}
- Exchange: {getattr(self, 'exchange', 'N/A')}
- Pair: {getattr(self, 'pair', 'N/A')}
- Interval: {getattr(self, 'interval', 'N/A')}

Dataset Statistics:
- Total samples: {len(df):,}
- Features: {len(X.columns)}
- Label distribution:
  * Bullish (1): {y.sum():,} ({y.mean()*100:.1f}%)
  * Bearish (0): {len(y) - y.sum():,} ({(1-y.mean())*100:.1f}%)

Time Range:
- Start: {df['time'].min()} ({self._format_timestamp(df['time'].min(), jakarta_tz)})
- End: {df['time'].max()} ({self._format_timestamp(df['time'].max(), jakarta_tz)})

Data Sources:
- Exchanges: {list(df['exchange'].unique())}
- Symbols: {list(df['symbol'].unique())}
- Intervals: {list(df['interval'].unique())}

Feature Columns:
{chr(10).join(f"- {feature}" for feature in sorted(X.columns))}
"""

        # Save to file
        with open(summary_file, 'w') as f:
            f.write(summary_content)

        logger.info(f"Dataset summary saved to {summary_file}")

        # Save to database using new method (xgboost_dataset_summary only - per client requirement)
        try:
            from database_storage import DatabaseStorage

            db_storage = DatabaseStorage()

            # Encode summary_content to bytes for storage
            summary_bytes = summary_content.encode('utf-8')

            # Store dataset summary with model_version and summary_data
            db_storage.store_dataset_summary(
                summary_file=summary_file.name,
                summary_data=summary_bytes,
                model_version='futures'  # V2 is futures-only
            )

            logger.info("Dataset summary saved to database")

        except Exception as e:
            logger.error(f"Error saving dataset summary to database: {e}")
            # Continue without database storage


def main():
    """Main function to build labels."""
    args = parse_arguments()
    validate_arguments(args)

    # Set up logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create data filter
    data_filter = DataFilter(args)

    # Initialize label builder
    builder = LabelBuilder(data_filter, args.output_dir)

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
        logger.info("Ready for xgboost_trainer.py")

    except Exception as e:
        logger.error(f"Error in label building: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()