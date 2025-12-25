#!/usr/bin/env python3
"""
Feature engineering for 9 futures trading tables based on semantic nature of each dataset.
Implements the specifications from feature_engineering.md for optimal XGBoost features.

Core Training Tables (5):
- cg_futures_price_history (OHLCV)
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

class FeatureEngineer:
    """Implement feature engineering for futures-only trading data (9 tables)."""

    def __init__(self, data_filter: DataFilter, output_dir: str = './output_train', price_only_mode: bool = False):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.feature_columns = []
        self.price_only_mode = price_only_mode

        # Log mode
        if self.price_only_mode:
            logger.info("🔵 PRICE-ONLY MODE: Only creating price features (QuantConnect compatible)")

    def load_merged_data(self) -> pd.DataFrame:
        """Load merged data from previous step, or raw price data in price-only mode."""
        logger.info("Loading data for feature engineering...")

        # In price-only mode, load raw price data directly
        if self.price_only_mode:
            raw_data_dir = self.output_dir / 'datasets' / 'raw'
            price_file = raw_data_dir / 'cg_futures_price_history.parquet'

            if not price_file.exists():
                logger.error(f"Price data file not found: {price_file}")
                logger.error("Please run load_database.py with PRICE_ONLY_MODE=true first")
                sys.exit(1)

            try:
                df = pd.read_parquet(price_file)

                # Rename raw columns to match expected naming convention
                # Raw: open, high, low, close, volume_usd
                # Expected: price_open, price_high, price_low, price_close, price_volume_usd
                column_mapping = {
                    'open': 'price_open',
                    'high': 'price_high',
                    'low': 'price_low',
                    'close': 'price_close',
                    'volume_usd': 'price_volume_usd'
                }

                df = df.rename(columns=column_mapping)
                logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns (price-only mode)")
                logger.info(f"Columns: {list(df.columns)}")
                return df
            except Exception as e:
                logger.error(f"Error loading price data: {e}")
                sys.exit(1)

        # Full mode: load merged data
        # Try datasets directory first (new structure), then root (compatibility)
        datasets_dir = self.output_dir / 'datasets'
        merged_file = datasets_dir / 'merged_9_tables.parquet'

        if not merged_file.exists():
            # Fallback to root directory for backward compatibility
            merged_file = self.output_dir / 'merged_9_tables.parquet'

        if not merged_file.exists():
            logger.error(f"Merged data file not found: {merged_file}")
            logger.error(f"Checked: {datasets_dir / 'merged_9_tables.parquet'} and {self.output_dir / 'merged_9_tables.parquet'}")
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

        # Helper function for groupby rolling with proper index alignment
        def groupby_rolling_agg(series, window, func):
            result = df.groupby(['exchange', 'symbol', 'interval'])[series].rolling(window)
            if func == 'mean':
                return result.mean().groupby(level=[0,1,2]).transform('first').reset_index(level=[0,1,2], drop=True)
            elif func == 'std':
                return result.std().groupby(level=[0,1,2]).transform('first').reset_index(level=[0,1,2], drop=True)
            return result

        # Price returns
        df['price_close_return_1'] = df.groupby(['exchange', 'symbol', 'interval'])['price_close'].pct_change(1)
        df['price_close_return_5'] = df.groupby(['exchange', 'symbol', 'interval'])['price_close'].pct_change(5)
        df['price_log_return'] = np.log(df['price_close'] / df.groupby(['exchange', 'symbol', 'interval'])['price_close'].shift(1))

        # Rolling volatility - using direct calculation to avoid index issues
        df['price_rolling_vol_5'] = df.groupby(['exchange', 'symbol', 'interval'])['price_close_return_1'].transform(lambda x: x.rolling(5).std())

        # True range and candlestick features
        df['price_true_range'] = df['price_high'] - df['price_low']
        df['price_close_mean_5'] = df.groupby(['exchange', 'symbol', 'interval'])['price_close'].transform(lambda x: x.rolling(5).mean())
        df['price_close_std_5'] = df.groupby(['exchange', 'symbol', 'interval'])['price_close'].transform(lambda x: x.rolling(5).std())

        # ========== NEW PREDICTIVE FEATURES ==========

        # RSI (Relative Strength Index) - 14 period standard
        def calculate_rsi(prices, period=14):
            """Calculate RSI for a series of prices."""
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        df['price_rsi_14'] = df.groupby(['exchange', 'symbol', 'interval'])['price_close'].transform(
            lambda x: calculate_rsi(x, period=14)
        )

        # EMA for MACD calculation
        df['price_ema_12'] = df.groupby(['exchange', 'symbol', 'interval'])['price_close'].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
        df['price_ema_26'] = df.groupby(['exchange', 'symbol', 'interval'])['price_close'].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )

        # MACD line
        df['price_macd'] = df['price_ema_12'] - df['price_ema_26']

        # MACD Signal line (9-period EMA of MACD)
        df['price_macd_signal'] = df.groupby(['exchange', 'symbol', 'interval'])['price_macd'].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )

        # MACD Histogram (MACD - Signal)
        df['price_macd_histogram'] = df['price_macd'] - df['price_macd_signal']

        # MACD Crossover signals
        df['price_macd_bullish_cross'] = (
            (df['price_macd'] > df['price_macd_signal']) &
            (df.groupby(['exchange', 'symbol', 'interval'])['price_macd'].shift(1) <=
             df.groupby(['exchange', 'symbol', 'interval'])['price_macd_signal'].shift(1))
        ).astype(int)

        df['price_macd_bearish_cross'] = (
            (df['price_macd'] < df['price_macd_signal']) &
            (df.groupby(['exchange', 'symbol', 'interval'])['price_macd'].shift(1) >=
             df.groupby(['exchange', 'symbol', 'interval'])['price_macd_signal'].shift(1))
        ).astype(int)

        # ========== END NEW PREDICTIVE FEATURES ==========

        # Volume features (with spike detection)
        df['price_volume_mean_10'] = df.groupby(['exchange', 'symbol', 'interval'])['price_volume_usd'].transform(lambda x: x.rolling(10).mean())
        volume_std_10 = df.groupby(['exchange', 'symbol', 'interval'])['price_volume_usd'].transform(lambda x: x.rolling(10).std())
        df['price_volume_zscore'] = (df['price_volume_usd'] - df['price_volume_mean_10']) / volume_std_10
        df['price_volume_change'] = df.groupby(['exchange', 'symbol', 'interval'])['price_volume_usd'].pct_change(1)

        # Volume spike detection (> 2x mean = unusual activity)
        df['price_volume_spike'] = (df['price_volume_usd'] > df['price_volume_mean_10'] * 2).astype(int)

        # Candlestick features
        df['price_wick_upper'] = df['price_high'] - np.maximum(df['price_open'], df['price_close'])
        df['price_wick_lower'] = np.minimum(df['price_open'], df['price_close']) - df['price_low']
        df['price_body_size'] = np.abs(df['price_close'] - df['price_open'])

        self.feature_columns.extend([
            'price_close_return_1', 'price_close_return_5', 'price_log_return',
            'price_rolling_vol_5', 'price_true_range', 'price_close_mean_5',
            'price_close_std_5', 'price_volume_mean_10', 'price_volume_zscore',
            'price_volume_change', 'price_wick_upper', 'price_wick_lower', 'price_body_size',
            # New predictive features
            'price_rsi_14', 'price_ema_12', 'price_ema_26', 'price_macd',
            'price_macd_signal', 'price_macd_histogram', 'price_macd_bullish_cross',
            'price_macd_bearish_cross', 'price_volume_spike'
        ])

        return df

    def add_funding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for cg_funding_rate_history."""
        logger.info("Adding funding rate features...")

        if 'funding_close' not in df.columns:
            logger.warning("Funding rate data not available")
            return df

        # Normalized funding rate
        df['funding_norm'] = df['funding_close'] / 0.01

        # Rolling statistics using transform to maintain index compatibility
        df['funding_mean_24'] = df.groupby(['exchange', 'symbol', 'interval'])['funding_close'].transform(lambda x: x.rolling(24).mean())
        df['funding_std_24'] = df.groupby(['exchange', 'symbol', 'interval'])['funding_close'].transform(lambda x: x.rolling(24).std())
        df['funding_zscore'] = (df['funding_close'] - df['funding_mean_24']) / df['funding_std_24']

        # Extreme indicators (> 1% funding rate = overextended longs/shorts)
        df['funding_extreme_positive'] = (df['funding_close'] > 0.01).astype(int)  # > 1% = overextended longs
        df['funding_extreme_negative'] = (df['funding_close'] < -0.01).astype(int)  # < -1% = overextended shorts

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

        # Basis changes
        df['basis_delta'] = df.groupby(['exchange', 'symbol', 'interval'])['basis_close_basis'].diff()
        df['basis_drift'] = df['basis_close_basis'] - df['basis_open_basis']

        # Rolling statistics using transform to maintain index compatibility
        df['basis_mean_24'] = df.groupby(['exchange', 'symbol', 'interval'])['basis_close_basis'].transform(lambda x: x.rolling(24).mean())
        basis_std_24 = df.groupby(['exchange', 'symbol', 'interval'])['basis_close_basis'].transform(lambda x: x.rolling(24).std())
        df['basis_zscore'] = (df['basis_close_basis'] - df['basis_mean_24']) / basis_std_24
        df['basis_volatility_24'] = basis_std_24

        self.feature_columns.extend([
            'basis_delta', 'basis_drift', 'basis_mean_24',
            'basis_zscore', 'basis_volatility_24'
        ])

        return df

    def add_taker_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for cg_futures_aggregated_taker_buy_sell_volume_history."""
        logger.info("Adding taker volume features...")

        if 'taker_aggregated_buy_volume' not in df.columns:
            logger.warning("Taker volume data not available")
            return df

        # Taker imbalance metrics
        total_volume = df['taker_aggregated_buy_volume'] + df['taker_aggregated_sell_volume']
        df['taker_buy_ratio'] = df['taker_aggregated_buy_volume'] / total_volume
        df['taker_imbalance'] = df['taker_aggregated_buy_volume'] - df['taker_aggregated_sell_volume']

        # Rolling statistics using transform to maintain index compatibility
        df['taker_buy_mean_12'] = df.groupby(['exchange', 'symbol', 'interval'])['taker_aggregated_buy_volume'].transform(lambda x: x.rolling(12).mean())
        df['taker_sell_mean_12'] = df.groupby(['exchange', 'symbol', 'interval'])['taker_aggregated_sell_volume'].transform(lambda x: x.rolling(12).mean())
        df['taker_buy_std_12'] = df.groupby(['exchange', 'symbol', 'interval'])['taker_aggregated_buy_volume'].transform(lambda x: x.rolling(12).std())
        df['taker_sell_std_12'] = df.groupby(['exchange', 'symbol', 'interval'])['taker_aggregated_sell_volume'].transform(lambda x: x.rolling(12).std())

        # Z-scores
        df['taker_buy_zscore'] = (df['taker_aggregated_buy_volume'] - df['taker_buy_mean_12']) / df['taker_buy_std_12']
        df['taker_sell_zscore'] = (df['taker_aggregated_sell_volume'] - df['taker_sell_mean_12']) / df['taker_sell_std_12']

        self.feature_columns.extend([
            'taker_buy_ratio', 'taker_imbalance', 'taker_buy_mean_12',
            'taker_sell_mean_12', 'taker_buy_std_12', 'taker_buy_zscore', 'taker_sell_zscore'
        ])

        return df

    def add_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for cg_futures_aggregated_ask_bids_history."""
        logger.info("Adding orderbook features...")

        if 'orderbook_aggregated_bids_usd' not in df.columns:
            logger.warning("Orderbook data not available")
            return df

        # Orderbook imbalance
        total_usd = df['orderbook_aggregated_bids_usd'] + df['orderbook_aggregated_asks_usd']
        df['orderbook_bid_ask_ratio'] = df['orderbook_aggregated_bids_usd'] / df['orderbook_aggregated_asks_usd']
        df['orderbook_imbalance_usd'] = df['orderbook_aggregated_bids_usd'] - df['orderbook_aggregated_asks_usd']

        # Quantity-based imbalance
        total_qty = df['orderbook_aggregated_bids_quantity'] + df['orderbook_aggregated_asks_quantity']
        df['orderbook_bid_ask_ratio_qty'] = df['orderbook_aggregated_bids_quantity'] / df['orderbook_aggregated_asks_quantity']
        df['orderbook_imbalance_qty'] = df['orderbook_aggregated_bids_quantity'] - df['orderbook_aggregated_asks_quantity']

        # Rolling statistics
        df['orderbook_bid_mean_12'] = df.groupby(['exchange', 'symbol', 'interval'])['orderbook_aggregated_bids_usd'].transform(lambda x: x.rolling(12).mean())
        df['orderbook_ask_mean_12'] = df.groupby(['exchange', 'symbol', 'interval'])['orderbook_aggregated_asks_usd'].transform(lambda x: x.rolling(12).mean())

        # Z-scores
        bid_std = df.groupby(['exchange', 'symbol', 'interval'])['orderbook_aggregated_bids_usd'].transform(lambda x: x.rolling(12).std())
        ask_std = df.groupby(['exchange', 'symbol', 'interval'])['orderbook_aggregated_asks_usd'].transform(lambda x: x.rolling(12).std())
        df['orderbook_bid_zscore'] = (df['orderbook_aggregated_bids_usd'] - df['orderbook_bid_mean_12']) / bid_std
        df['orderbook_ask_zscore'] = (df['orderbook_aggregated_asks_usd'] - df['orderbook_ask_mean_12']) / ask_std

        self.feature_columns.extend([
            'orderbook_bid_ask_ratio', 'orderbook_imbalance_usd',
            'orderbook_bid_ask_ratio_qty', 'orderbook_imbalance_qty',
            'orderbook_bid_mean_12', 'orderbook_ask_mean_12',
            'orderbook_bid_zscore', 'orderbook_ask_zscore'
        ])

        return df

    def add_open_interest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for cg_open_interest_aggregated_history."""
        logger.info("Adding open interest features...")

        if 'oi_close' not in df.columns:
            logger.warning("Open interest data not available")
            return df

        # OI changes
        df['oi_delta'] = df.groupby(['symbol', 'interval'])['oi_close'].diff()
        df['oi_pct_change'] = df.groupby(['symbol', 'interval'])['oi_close'].pct_change()

        # Rolling statistics
        df['oi_mean_24'] = df.groupby(['symbol', 'interval'])['oi_close'].transform(lambda x: x.rolling(24).mean())
        oi_std_24 = df.groupby(['symbol', 'interval'])['oi_close'].transform(lambda x: x.rolling(24).std())
        df['oi_zscore'] = (df['oi_close'] - df['oi_mean_24']) / oi_std_24

        # OI range features
        df['oi_range'] = df['oi_high'] - df['oi_low']
        df['oi_range_pct'] = (df['oi_high'] - df['oi_low']) / df['oi_close']

        self.feature_columns.extend([
            'oi_delta', 'oi_pct_change', 'oi_mean_24', 'oi_zscore', 'oi_range', 'oi_range_pct'
        ])

        return df

    def add_liquidation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for cg_liquidation_aggregated_history."""
        logger.info("Adding liquidation features...")

        if 'liq_aggregated_long_liquidation_usd' not in df.columns:
            logger.warning("Liquidation data not available")
            return df

        # Liquidation imbalance
        total_liq = df['liq_aggregated_long_liquidation_usd'] + df['liq_aggregated_short_liquidation_usd']
        df['liq_long_ratio'] = df['liq_aggregated_long_liquidation_usd'] / total_liq
        df['liq_imbalance'] = df['liq_aggregated_long_liquidation_usd'] - df['liq_aggregated_short_liquidation_usd']

        # Rolling statistics
        df['liq_long_mean_12'] = df.groupby(['symbol', 'interval'])['liq_aggregated_long_liquidation_usd'].transform(lambda x: x.rolling(12).mean())
        df['liq_short_mean_12'] = df.groupby(['symbol', 'interval'])['liq_aggregated_short_liquidation_usd'].transform(lambda x: x.rolling(12).mean())

        # Z-scores
        long_std = df.groupby(['symbol', 'interval'])['liq_aggregated_long_liquidation_usd'].transform(lambda x: x.rolling(12).std())
        short_std = df.groupby(['symbol', 'interval'])['liq_aggregated_short_liquidation_usd'].transform(lambda x: x.rolling(12).std())
        df['liq_long_zscore'] = (df['liq_aggregated_long_liquidation_usd'] - df['liq_long_mean_12']) / long_std
        df['liq_short_zscore'] = (df['liq_aggregated_short_liquidation_usd'] - df['liq_short_mean_12']) / short_std

        # Spike detection
        df['liq_long_spike'] = (df['liq_aggregated_long_liquidation_usd'] > df['liq_long_mean_12'] * 2).astype(int)
        df['liq_short_spike'] = (df['liq_aggregated_short_liquidation_usd'] > df['liq_short_mean_12'] * 2).astype(int)

        self.feature_columns.extend([
            'liq_long_ratio', 'liq_imbalance', 'liq_long_mean_12', 'liq_short_mean_12',
            'liq_long_zscore', 'liq_short_zscore', 'liq_long_spike', 'liq_short_spike'
        ])

        return df

    def add_longshort_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for long/short ratio tables."""
        logger.info("Adding long/short ratio features...")

        # Global account ratios
        if 'ls_global_global_account_long_short_ratio' in df.columns:
            df['ls_global_ratio'] = df['ls_global_global_account_long_percent'] / df['ls_global_global_account_short_percent']
            global_mean_24 = df.groupby(['exchange', 'symbol', 'interval'])['ls_global_global_account_long_short_ratio'].transform(lambda x: x.rolling(24).mean())
            global_std_24 = df.groupby(['exchange', 'symbol', 'interval'])['ls_global_global_account_long_short_ratio'].transform(lambda x: x.rolling(24).std())
            df['ls_global_zscore'] = (df['ls_global_global_account_long_short_ratio'] - global_mean_24) / global_std_24
            df['ls_global_delta'] = df.groupby(['exchange', 'symbol', 'interval'])['ls_global_global_account_long_short_ratio'].diff()
            df['ls_global_extreme_high'] = (df['ls_global_ratio'] > 1.5).astype(int)
            df['ls_global_extreme_low'] = (df['ls_global_ratio'] < 0.7).astype(int)

            self.feature_columns.extend([
                'ls_global_ratio', 'ls_global_zscore', 'ls_global_delta',
                'ls_global_extreme_high', 'ls_global_extreme_low'
            ])

        # Top account ratios
        if 'ls_top_top_account_long_short_ratio' in df.columns:
            df['ls_top_ratio'] = df['ls_top_top_account_long_percent'] / df['ls_top_top_account_short_percent']
            top_mean_24 = df.groupby(['exchange', 'symbol', 'interval'])['ls_top_top_account_long_short_ratio'].transform(lambda x: x.rolling(24).mean())
            top_std_24 = df.groupby(['exchange', 'symbol', 'interval'])['ls_top_top_account_long_short_ratio'].transform(lambda x: x.rolling(24).std())
            df['ls_top_zscore'] = (df['ls_top_top_account_long_short_ratio'] - top_mean_24) / top_std_24
            df['ls_top_delta'] = df.groupby(['exchange', 'symbol', 'interval'])['ls_top_top_account_long_short_ratio'].diff()

            self.feature_columns.extend([
                'ls_top_ratio', 'ls_top_zscore', 'ls_top_delta'
            ])

        # Cross-table interaction: top vs global
        if 'ls_top_top_account_long_short_ratio' in df.columns and 'ls_global_global_account_long_short_ratio' in df.columns:
            df['ls_top_vs_global'] = df['ls_top_top_account_long_short_ratio'] - df['ls_global_global_account_long_short_ratio']
            self.feature_columns.append('ls_top_vs_global')

        return df

    def add_cross_table_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-table interaction features for futures data."""
        logger.info("Adding cross-table features...")

        # Initialize cross-table features
        cross_features = []

        # OI delta * taker imbalance
        if 'oi_delta' in df.columns and 'taker_imbalance' in df.columns:
            df['cross_oi_taker'] = df['oi_delta'] * df['taker_imbalance']
            cross_features.append('cross_oi_taker')

        # OI delta * price return
        if 'oi_delta' in df.columns and 'price_close_return_1' in df.columns:
            df['cross_oi_price'] = df['oi_delta'] * df['price_close_return_1']
            cross_features.append('cross_oi_price')

        # Basis delta * taker imbalance
        if 'basis_delta' in df.columns and 'taker_imbalance' in df.columns:
            df['cross_basis_taker'] = df['basis_delta'] * df['taker_imbalance']
            cross_features.append('cross_basis_taker')

        # Funding zscore * price return
        if 'funding_zscore' in df.columns and 'price_close_return_1' in df.columns:
            df['cross_funding_price'] = df['funding_zscore'] * df['price_close_return_1']
            cross_features.append('cross_funding_price')

        # Top vs global * price return
        if 'ls_top_vs_global' in df.columns and 'price_close_return_1' in df.columns:
            df['cross_ls_price'] = df['ls_top_vs_global'] * df['price_close_return_1']
            cross_features.append('cross_ls_price')

        # Taker volume * price return
        if 'taker_buy_ratio' in df.columns and 'price_close_return_1' in df.columns:
            df['cross_taker_price'] = df['taker_buy_ratio'] * df['price_close_return_1']
            cross_features.append('cross_taker_price')

        # Taker imbalance * funding zscore
        if 'taker_imbalance' in df.columns and 'funding_zscore' in df.columns:
            df['cross_taker_funding'] = df['taker_imbalance'] * df['funding_zscore']
            cross_features.append('cross_taker_funding')

        # Orderbook imbalance * price return
        if 'orderbook_imbalance_usd' in df.columns and 'price_close_return_1' in df.columns:
            df['cross_ob_price'] = df['orderbook_imbalance_usd'] * df['price_close_return_1']
            cross_features.append('cross_ob_price')

        # Liquidation imbalance * price return
        if 'liq_imbalance' in df.columns and 'price_close_return_1' in df.columns:
            df['cross_liq_price'] = df['liq_imbalance'] * df['price_close_return_1']
            cross_features.append('cross_liq_price')

        # Liquidation spike * price return
        if 'liq_long_spike' in df.columns and 'price_close_return_1' in df.columns:
            df['cross_liq_spike_price'] = df['liq_long_spike'] * df['price_close_return_1']
            cross_features.append('cross_liq_spike_price')

        # OI zscore * funding zscore (both indicate market stress)
        if 'oi_zscore' in df.columns and 'funding_zscore' in df.columns:
            df['cross_oi_funding'] = df['oi_zscore'] * df['funding_zscore']
            cross_features.append('cross_oi_funding')

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

        # Fill missing values with forward fill then backward fill
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                # Group by symbol/exchange/interval for proper filling
                df[col] = df.groupby(['exchange', 'symbol', 'interval'])[col].ffill().bfill()

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

        # Save complete dataset with features to datasets directory
        output_file = datasets_dir / 'features_engineered.parquet'
        df.to_parquet(output_file, index=False)
        logger.info(f"Features saved to {output_file}")

        # Save feature list to features directory
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        feature_file = features_dir / 'feature_list.txt'
        with open(feature_file, 'w') as f:
            f.write("Engineered Features:\n")
            f.write("=" * 20 + "\n")
            for col in feature_cols:
                f.write(f"{col}\n")
        logger.info(f"Feature list saved to {feature_file}")

        # Save features-only dataset to datasets directory
        features_df = df[feature_cols].copy()
        features_file = datasets_dir / 'features_only.parquet'
        features_df.to_parquet(features_file, index=False)
        logger.info(f"Features-only dataset saved to {features_file}")

        # Also keep copies in root for backward compatibility
        root_output_file = self.output_dir / 'features_engineered.parquet'
        root_features_file = self.output_dir / 'features_only.parquet'
        root_feature_file = self.output_dir / 'feature_list.txt'

        try:
            df.to_parquet(root_output_file, index=False)
            features_df.to_parquet(root_features_file, index=False)
            root_feature_file.write_text(feature_file.read_text())
            logger.info("Also saved copies to root directory for compatibility")
        except Exception as e:
            logger.warning(f"Could not save root compatibility copies: {e}")

        # Duplicate line removed

        # DATABASE STORAGE DISABLED per client requirement
        # Client tidak mau: xgboost_evaluations, xgboost_features, xgboost_training_sessions
        logger.info("📝 Feature database storage disabled per client requirement")

def main():
    """Main function to engineer features."""
    args = parse_arguments()
    validate_arguments(args)

    # Set up logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check for price-only mode
    price_only_mode = os.getenv('PRICE_ONLY_MODE', 'false').lower() == 'true'

    # Create data filter
    data_filter = DataFilter(args)

    # Initialize feature engineer
    engineer = FeatureEngineer(data_filter, args.output_dir, price_only_mode)

    try:
        # Load data
        logger.info("Loading data for feature engineering...")
        df = engineer.load_merged_data()

        # Add features for each table type
        df = engineer.add_price_features(df)

        # PRICE-ONLY MODE: Skip all non-price features (QuantConnect compatibility)
        if not price_only_mode:
            # Full mode: add all futures features
            df = engineer.add_funding_features(df)
            df = engineer.add_basis_features(df)
            df = engineer.add_taker_volume_features(df)
            df = engineer.add_orderbook_features(df)
            df = engineer.add_open_interest_features(df)
            df = engineer.add_liquidation_features(df)
            df = engineer.add_longshort_features(df)
            df = engineer.add_cross_table_features(df)
        else:
            logger.info("Price-only mode: Skipping futures-specific features (funding, basis, OI, LS, taker, orderbook, liquidation)")

        # Clean features
        df = engineer.clean_features(df)

        # Validate features
        if not engineer.validate_features(df):
            sys.exit(1)

        # Save features
        engineer.save_features(df)

        logger.info("\n=== Feature Engineering Complete ===")
        logger.info("Ready for label_builder.py")

    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()