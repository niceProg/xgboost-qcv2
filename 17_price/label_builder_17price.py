#!/usr/bin/env python3
"""
Label Builder for 17 Price Features (QuantConnect Compatible)

Creates binary classification labels for price direction prediction.
Labels are based on future price movements.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime


class LabelBuilder17:
    """
    Build labels for binary classification (buy=1, sell=0)
    using only price data.
    """

    def __init__(self,
                 forward_bars: int = 1,
                 profit_threshold: float = 0.0,
                 loss_threshold: float = 0.0):
        """
        Initialize label builder.

        Args:
            forward_bars: Number of bars to look forward for label calculation
            profit_threshold: Minimum positive return to classify as buy (default: 0 = any positive)
            loss_threshold: Maximum negative return to classify as sell (default: 0 = any negative)
        """
        self.forward_bars = forward_bars
        self.profit_threshold = profit_threshold
        self.loss_threshold = loss_threshold

    def add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary labels to dataframe.

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with added 'label' column (1 = buy, 0 = sell)
        """
        df = df.copy()

        # Calculate forward return
        df['forward_return'] = df['close'].shift(-self.forward_bars) / df['close'] - 1

        # Create binary labels
        df['label'] = 0  # Default: sell

        # Buy if forward return > profit_threshold
        df.loc[df['forward_return'] > self.profit_threshold, 'label'] = 1

        # Note: We keep sell (0) for anything <= profit_threshold

        return df

    def add_labels_with_threshold(self,
                                   df: pd.DataFrame,
                                   buy_threshold: float = 0.005,
                                   sell_threshold: float = -0.005) -> pd.DataFrame:
        """
        Add labels with threshold bands.

        Args:
            df: DataFrame with 'close' column
            buy_threshold: Forward return above this = buy (1)
            sell_threshold: Forward return below this = sell (0)
            Between thresholds = neutral (excluded or 0.5)

        Returns:
            DataFrame with labels
        """
        df = df.copy()

        # Calculate forward return
        df['forward_return'] = df['close'].shift(-self.forward_bars) / df['close'] - 1

        # Create labels
        df['label'] = 0
        df.loc[df['forward_return'] > buy_threshold, 'label'] = 1
        # Everything else = 0 (sell)

        return df

    def get_label_distribution(self, df: pd.DataFrame) -> dict:
        """Get distribution of labels."""
        if 'label' not in df.columns:
            df = self.add_labels(df)

        dist = df['label'].value_counts().to_dict()
        total = len(df)

        return {
            'buy': dist.get(1, 0),
            'sell': dist.get(0, 0),
            'total': total,
            'buy_pct': dist.get(1, 0) / total * 100 if total > 0 else 0,
            'sell_pct': dist.get(0, 0) / total * 100 if total > 0 else 0,
        }

    def create_balanced_dataset(self,
                                 df: pd.DataFrame,
                                 method: str = 'undersample') -> pd.DataFrame:
        """
        Create balanced dataset (equal buy/sell samples).

        Args:
            df: DataFrame with labels
            method: 'undersample' (remove majority) or 'oversample' (duplicate minority)

        Returns:
            Balanced DataFrame
        """
        if 'label' not in df.columns:
            df = self.add_labels(df)

        # Separate buy and sell
        buy_df = df[df['label'] == 1].copy()
        sell_df = df[df['label'] == 0].copy()

        buy_count = len(buy_df)
        sell_count = len(sell_df)

        print(f"Original - Buy: {buy_count}, Sell: {sell_count}")

        if method == 'undersample':
            # Undersample majority class
            min_count = min(buy_count, sell_count)
            buy_df = buy_df.sample(n=min_count, random_state=42) if buy_count > min_count else buy_df
            sell_df = sell_df.sample(n=min_count, random_state=42) if sell_count > min_count else sell_df

        elif method == 'oversample':
            # Oversample minority class
            max_count = max(buy_count, sell_count)
            if buy_count < max_count:
                buy_df = buy_df.sample(n=max_count, replace=True, random_state=42)
            if sell_count < max_count:
                sell_df = sell_df.sample(n=max_count, replace=True, random_state=42)

        # Combine
        balanced_df = pd.concat([buy_df, sell_df]).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Balanced - Buy: {len(balanced_df[balanced_df['label']==1])}, "
          f"Sell: {len(balanced_df[balanced_df['label']==0])}")

        return balanced_df


# =============================================================================
# DIFFERENT LABEL STRATEGIES
# =============================================================================
def create_labels_next_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple next-bar prediction.
    Label = 1 if next close > current close, else 0
    """
    df = df.copy()
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df


def create_labels_threshold(df: pd.DataFrame,
                            buy_thresh: float = 0.01,
                            sell_thresh: float = -0.01) -> pd.DataFrame:
    """
    Threshold-based labels.
    Label = 1 if forward return > buy_thresh
    Label = 0 if forward return < sell_thresh
    Exclude in-between
    """
    df = df.copy()
    df['forward_return'] = df['close'].shift(-1) / df['close'] - 1

    df['label'] = 0
    df.loc[df['forward_return'] > buy_thresh, 'label'] = 1
    # Exclude weak signals (keep as 0 or drop)

    return df


def create_labels_3bar(df: pd.DataFrame, profit_pct: float = 0.02) -> pd.DataFrame:
    """
    3-bar forward labels with profit target.

    Label = 1 if price reaches +X% within 3 bars before dropping -Y%
    Label = 0 otherwise
    """
    df = df.copy()
    df['label'] = 0

    for i in range(len(df) - 3):
        current_close = df.loc[i, 'close']
        target_price = current_close * (1 + profit_pct)

        # Check next 3 bars
        for j in range(i + 1, min(i + 4, len(df))):
            if df.loc[j, 'high'] >= target_price:
                df.loc[i, 'label'] = 1
                break

    return df


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    from feature_engineering_17price import FeatureEngineer17, FEATURES_17

    print("Label Builder for 17 Price Features")
    print("="*60)

    # Example
    print(f"\nAvailable label strategies:")
    print("  1. Next bar direction")
    print("  2. Threshold-based (e.g., >1% = buy, <-1% = sell)")
    print("  3. Multi-bar with profit target")

    print(f"\nUsing features: {len(FEATURES_17)}")
