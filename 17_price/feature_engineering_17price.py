#!/usr/bin/env python3
"""
Feature Engineering for 17 Price Features (QuantConnect Compatible)

This module creates features that can be computed from OHLCV data only,
making it fully compatible with QuantConnect backtesting and live trading.

Features (17 total):
- price_open, price_high, price_low, price_close, price_volume_usd
- price_close_return_1, price_close_return_5, price_log_return
- price_rolling_vol_5, price_true_range, price_close_mean_5, price_close_std_5
- price_volume_mean_10, price_volume_zscore, price_volume_change
- price_wick_upper, price_wick_lower, price_body_size
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


# =============================================================================
# FEATURE LIST - Must match training exactly
# =============================================================================
FEATURES_17 = [
    'price_open', 'price_high', 'price_low', 'price_close', 'price_volume_usd',
    'price_close_return_1', 'price_close_return_5', 'price_log_return',
    'price_rolling_vol_5', 'price_true_range', 'price_close_mean_5',
    'price_close_std_5', 'price_volume_mean_10', 'price_volume_zscore',
    'price_volume_change', 'price_wick_upper', 'price_wick_lower',
    'price_body_size',
]


class FeatureEngineer17:
    """
    Feature engineering for 17 price features only.
    Compatible with QuantConnect data availability.
    """

    def __init__(self, lookback_periods: Dict[str, int] = None):
        """
        Initialize feature engineer.

        Args:
            lookback_periods: Custom lookback periods for calculations
        """
        self.lookback_periods = lookback_periods or {
            'return_1': 2,
            'return_5': 6,
            'volatility': 5,
            'mean_5': 5,
            'volume_mean': 10,
        }

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all 17 price features to the dataframe.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            DataFrame with added features
        """
        df = df.copy()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        # Convert to float if needed
        for col in required_cols:
            df[col] = df[col].astype(float)

        # Basic features (already exist, just rename/add)
        df['price_open'] = df['open']
        df['price_high'] = df['high']
        df['price_low'] = df['low']
        df['price_close'] = df['close']
        df['price_volume_usd'] = df['close'] * df['volume']

        # Close-to-close returns
        df['price_close_return_1'] = df['close'].pct_change(1)
        df['price_close_return_5'] = df['close'].pct_change(5)
        df['price_log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Rolling volatility (5-period)
        returns = df['close'].pct_change()
        df['price_rolling_vol_5'] = returns.rolling(window=5, min_periods=1).std()

        # True Range (high - low)
        df['price_true_range'] = df['high'] - df['low']

        # Rolling mean and std (5-period)
        df['price_close_mean_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['price_close_std_5'] = df['close'].rolling(window=5, min_periods=1).std()

        # Volume features (10-period rolling)
        df['price_volume_mean_10'] = df['volume'].rolling(window=10, min_periods=1).mean()
        vol_std = df['volume'].rolling(window=10, min_periods=1).std()
        df['price_volume_zscore'] = (df['volume'] - df['price_volume_mean_10']) / vol_std.replace(0, np.nan)
        df['price_volume_change'] = df['volume'].pct_change(1)

        # Candlestick features
        df['price_wick_upper'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['price_wick_lower'] = df[['open', 'close']].min(axis=1) - df['low']
        df['price_body_size'] = (df['close'] - df['open']).abs()

        # Handle infinite and NaN values
        df = self._clean_features(df)

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite and NaN values."""
        feature_cols = FEATURES_17

        for col in feature_cols:
            if col in df.columns:
                # Replace inf with NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                # Fill NaN with 0 (or forward fill then 0)
                df[col] = df[col].ffill().fillna(0)

        return df

    def get_feature_vector(self, df: pd.DataFrame, feature_order: List[str] = None) -> np.ndarray:
        """
        Extract feature vector in the specified order.

        Args:
            df: DataFrame with features
            feature_order: List of feature names in desired order (default: FEATURES_17)

        Returns:
            Feature matrix (numpy array)
        """
        if feature_order is None:
            feature_order = FEATURES_17

        # Ensure all features exist
        df_with_features = self.add_features(df)

        # Extract features in specified order
        feature_matrix = df_with_features[feature_order].values

        # Handle any remaining NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_matrix

    def get_available_features(self) -> List[str]:
        """Return list of available feature names."""
        return FEATURES_17.copy()


# =============================================================================
# UTILITIES
# =============================================================================
def print_feature_summary(df: pd.DataFrame):
    """Print summary statistics for all 17 features."""
    print("\n" + "="*80)
    print("FEATURE SUMMARY (17 Price Features)")
    print("="*80)

    for feat in FEATURES_17:
        if feat in df.columns:
            values = df[feat].dropna()
            print(f"\n{feat}:")
            print(f"  Count: {len(values)}")
            print(f"  Mean:   {values.mean():.6f}")
            print(f"  Std:    {values.std():.6f}")
            print(f"  Min:    {values.min():.6f}")
            print(f"  Max:    {values.max():.6f}")
            print(f"  Zeros:  {(df[feat] == 0).sum()}")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering for 17 Price Features")
    print(f"Total features: {len(FEATURES_17)}")
    print("\nFeatures:")
    for i, feat in enumerate(FEATURES_17, 1):
        print(f"  {i:2d}. {feat}")
