#!/usr/bin/env python3
# QuantConnect Algorithm: XGBoost Futures Trading
# Futures trend prediction using XGBoost model trained on Coinglass data

from AlgorithmImports import *
import numpy as np
import pandas as pd
from collections import deque
import requests
import base64
import pickle
from datetime import datetime, timedelta

# =============================================================================
# HARDCODED FEATURE LIST - Must match training exactly
# =============================================================================
FEATURES = [
    'price_open', 'price_high', 'price_low', 'price_close', 'price_volume_usd',
    'funding_open', 'funding_high', 'funding_low', 'funding_close',
    'basis_open_basis', 'basis_close_basis', 'basis_open_change', 'basis_close_change',
    'oi_open', 'oi_high', 'oi_low', 'oi_close',
    'liq_aggregated_long_liquidation_usd', 'liq_aggregated_short_liquidation_usd',
    'taker_aggregated_buy_volume', 'taker_aggregated_sell_volume', 'range_percent',
    'ob_aggregated_bids_usd', 'ob_aggregated_bids_quantity', 'ob_aggregated_asks_usd', 'ob_aggregated_asks_quantity',
    'ls_global_global_account_long_percent', 'ls_global_global_account_short_percent', 'ls_global_global_account_long_short_ratio',
    'ls_top_top_account_long_percent', 'ls_top_top_account_short_percent', 'ls_top_top_account_long_short_ratio',
    'price_close_return_1', 'price_close_return_5', 'price_log_return', 'price_rolling_vol_5',
    'price_true_range', 'price_close_mean_5', 'price_close_std_5', 'price_volume_mean_10',
    'price_volume_zscore', 'price_volume_change', 'price_wick_upper', 'price_wick_lower',
    'price_body_size', 'funding_norm', 'funding_mean_24', 'funding_std_24',
    'funding_zscore', 'funding_extreme_positive', 'funding_extreme_negative', 'basis_delta',
    'basis_drift', 'basis_mean_24', 'basis_zscore', 'basis_volatility_24',
    'oi_change', 'oi_change_abs', 'oi_momentum_5', 'oi_zscore', 'oi_price_ratio',
    'liq_total', 'liq_ratio', 'liq_imbalance', 'liq_total_mean_12',
    'liq_long_mean_12', 'liq_short_mean_12', 'liq_spike', 'liq_long_zscore',
    'liq_short_zscore', 'taker_buy_ratio', 'taker_imbalance', 'taker_buy_mean_12',
    'taker_sell_mean_12', 'taker_buy_std_12', 'taker_sell_std_12', 'taker_buy_zscore',
    'taker_sell_zscore', 'taker_buy_momentum', 'ob_bid_ask_ratio', 'ob_imbalance_usd',
    'ob_qty_bid_ask_ratio', 'ob_bids_mean_12', 'ob_asks_mean_12', 'ob_pressure',
    'ob_pressure_asks', 'ob_depth_change', 'ls_global_ratio', 'ls_global_zscore',
    'ls_global_delta', 'ls_global_extreme_high', 'ls_global_extreme_low', 'ls_top_ratio',
    'ls_top_zscore', 'ls_top_delta', 'ls_top_vs_global', 'cross_funding_oi',
    'cross_funding_price', 'cross_liq_price', 'cross_oi_taker', 'cross_ob_price',
    'cross_ls_price', 'cross_liq_funding'
]

# =============================================================================
# MAIN ALGORITHM
# =============================================================================
class XGBoostFuturesAlgorithm(QCAlgorithm):
    """
    XGBoost Futures Trading Algorithm

    Uses pre-trained XGBoost model to predict futures price movements.
    Model is loaded from API in live/paper mode, uses simulated predictions in backtest.
    """

    def Initialize(self):
        """Initialize algorithm parameters and load model."""

        # ===== API Configuration =====
        self.api_base_url = "https://api.dragonfortune.ai"
        self.api_timeout = 30  # seconds
        self.api_retry_count = 3

        # Model version: 'spot' or 'futures'
        self.model_version = "futures"

        # Object Store key for model
        self.object_store_key = f"xgboost_model_{self.model_version}"

        # In backtest mode, we still try to load from API to get the real model
        # Object Store will be used as cache in backtest, API as fallback in live
        self.use_api = True  # Enable API for both backtest and live mode

        self.train_start_date = None
        self.train_end_date = None

        # Only load from API if in live mode
        if self.use_api:
            self.LoadDatasetSummaryFromAPI()
        else:
            # Use default dates for backtest
            self.train_start_date = datetime(2024, 1, 1)
            self.train_end_date = datetime(2025, 12, 31)
            self.Debug("Backtest mode: Using default dates, skipping API calls")

        if self.train_start_date and self.train_end_date:
            start_dt = self.train_start_date - timedelta(days=1)
            end_dt = self.train_end_date
            self.Debug(f"[DATE] Using dataset_summary: {self.train_start_date} -> {self.train_end_date}")
        else:
            start_dt = datetime(2024, 1, 1)
            end_dt = datetime(2025, 12, 31)
            self.Debug(f"[DATE] Using fallback dates: {start_dt} -> {end_dt}")

        self.SetStartDate(start_dt.year, start_dt.month, start_dt.day)
        self.SetEndDate(end_dt.year, end_dt.month, end_dt.day)
        self.SetCash(100000)

        # ===== Symbol =====
        try:
            self.crypto = self.AddCrypto("BTCUSDT", Resolution.Hour, Market.Binance)
        except Exception:
            try:
                self.SetBrokerageModel(BrokerageName.Binance)
                self.crypto = self.AddCrypto("BTCUSDT", Resolution.Hour)
            except Exception:
                self.crypto = self.AddCrypto("BTCUSDT", Resolution.Hour)
                self.Debug("Warning: Using default market, but model expects Binance-like data")

        self.symbol = self.crypto.Symbol
        self.SetBenchmark(self.symbol)

        # ===== Feature config =====
        self.available_features = [
            "price_open", "price_high", "price_low", "price_close", "price_volume_usd",
            "price_close_return_1", "price_close_return_5", "price_log_return",
            "price_rolling_vol_5", "price_true_range", "price_close_mean_5",
            "price_close_std_5", "price_volume_mean_10", "price_volume_zscore",
            "price_volume_change", "price_wick_upper", "price_wick_lower",
            "price_body_size",
        ]

        # Offline model full feature list (fallback ordering)
        self.model_features = FEATURES

        # Model metadata
        self.model = None
        self.model_n_features = None
        self.expected_feature_order = None
        self.model_version_hash = None  # Track model version for reload detection

        # Model reload configuration
        self.model_reload_interval_days = 1  # Reload model daily
        self.last_model_reload_time = self.Time

        # Try loading model: Object Store first (fast, works in backtest), then API
        model_loaded = False

        # Try Object Store first (works in both backtest and live)
        try:
            if self.ObjectStore.ContainsKey(self.object_store_key):
                self.Debug(f"Found model in Object Store: {self.object_store_key}")
                model_loaded = self.LoadModelFromObjectStore()
        except Exception as e:
            self.Debug(f"Object Store load failed: {e}")

        # If Object Store failed, try API
        if not model_loaded and self.use_api:
            try:
                self.Debug("Object Store empty, loading from API...")
                self.LoadModelFromAPI(save_to_object_store=True)
            except Exception as e:
                self.Debug(f"API initialization failed, will use fallback: {e}")

        # If model failed to load or not in API mode, set default values
        if self.model is None:
            if self.use_api:
                self.Debug("Model not loaded from API - using default configuration")
            else:
                self.Debug("Backtest mode: Using simulated model")
            self.model_n_features = len(self.model_features)
            self.expected_feature_order = list(self.model_features)

        # ===== Rolling window =====
        self.window_size = 60
        self.price_window = deque(maxlen=self.window_size)

        # ===== Trading parameters =====
        self.prediction_buy_threshold = 0.55
        self.prediction_sell_threshold = 0.45
        self.position_size_pct = 0.80

        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.10

        # Entry/Exit tracking
        self.entry_price = None
        self.entry_time = None

        # Pending order guards
        self.pending_entry = False
        self.pending_exit = False
        self.last_exit_reason = None  # "TP" | "SL" | "PRED" | "RISK"

        # ===== Risk management =====
        self.max_drawdown_pct = 0.20
        self.high_watermark = self.Portfolio.TotalPortfolioValue
        self.cooldown_period = timedelta(days=7)
        self.trading_paused_until = None

        # Prevent same-bar re-entry
        self.last_trade_bar_time = None

        # Warmup
        self.SetWarmUp(120, Resolution.Hour)

        # Schedule daily model reload (only in live/paper mode, not backtest)
        if self.LiveMode:
            self.Schedule.On(
                self.DateRules.EveryDay(self.symbol),
                self.TimeRules.At(0, 0),  # At midnight
                self.ReloadModelIfNeeded
            )
            self.Debug("Model reload scheduled daily at midnight (live mode)")
        else:
            self.Debug("Backtest mode: Model reload scheduling skipped")

        self.Debug("XGBoost Futures Trading Algorithm initialized")

    # =========================================================
    # LOAD DATASET SUMMARY FROM API
    # =========================================================
    def LoadDatasetSummaryFromAPI(self):
        """Load dataset summary from XGBoost API."""
        for attempt in range(self.api_retry_count):
            try:
                url = f"{self.api_base_url}/api/v1/{self.model_version}/latest/dataset-summary"
                self.Debug(f"Fetching dataset summary from: {url}")

                response = requests.get(url, timeout=self.api_timeout)
                response.raise_for_status()

                data = response.json()

                if not data.get("success", False):
                    self.Debug("Dataset summary not available or success=false")
                    # Use default dates
                    self.train_start_date = datetime(2024, 1, 1)
                    self.train_end_date = datetime(2025, 12, 31)
                    return

                # Decode base64 summary data if available
                summary_data = data.get("summary_data_base64")
                if summary_data:
                    import re
                    try:
                        decoded_text = base64.b64decode(summary_data).decode('utf-8')

                        # Parse time range from the decoded text
                        m = re.search(r"Time range:\s*(.+?)\s*to\s*(.+)", decoded_text)
                        if not m:
                            self.Error("Could not parse 'Time range: ... to ...' in dataset summary")
                            # Use default dates
                            self.train_start_date = datetime(2024, 1, 1)
                            self.train_end_date = datetime(2025, 12, 31)
                            return

                        start_raw = m.group(1).strip()
                        end_raw = m.group(2).strip()
                        dt_format = "%Y-%m-%d %H:%M:%S"
                        start_dt = datetime.strptime(start_raw, dt_format)
                        end_dt = datetime.strptime(end_raw, dt_format)

                        self.train_start_date = datetime(start_dt.year, start_dt.month, start_dt.day)
                        self.train_end_date = datetime(end_dt.year, end_dt.month, end_dt.day)

                        self.Debug(f"Loaded dataset summary: {self.train_start_date} to {self.train_end_date}")
                    except Exception as decode_error:
                        self.Error(f"Error decoding dataset summary: {decode_error}")
                        # Use default dates
                        self.train_start_date = datetime(2024, 1, 1)
                        self.train_end_date = datetime(2025, 12, 31)
                else:
                    self.Debug("No dataset summary data available, using default dates")
                    self.train_start_date = datetime(2024, 1, 1)
                    self.train_end_date = datetime(2025, 12, 31)

                return

            except requests.exceptions.RequestException as e:
                self.Debug(f"API request failed (attempt {attempt + 1}/{self.api_retry_count}): {e}")
                if attempt == self.api_retry_count - 1:
                    self.Error("Failed to fetch dataset summary after all retries, using default dates")
                    self.train_start_date = datetime(2024, 1, 1)
                    self.train_end_date = datetime(2025, 12, 31)
            except Exception as e:
                self.Error(f"Error loading dataset summary from API: {e}")
                self.train_start_date = None
                self.train_end_date = None
                break

    # =========================================================
    # LOAD MODEL FROM OBJECT STORE
    # =========================================================
    def LoadModelFromObjectStore(self):
        """Load XGBoost model from QuantConnect Object Store."""
        try:
            model_bytes = self.ObjectStore.ReadBytes(self.object_store_key)
            if model_bytes is None or len(model_bytes) == 0:
                self.Debug("Object Store returned empty data")
                return False

            self.model = pickle.loads(model_bytes)

            # Try to get model metadata
            self.model_n_features = None
            self.expected_feature_order = None
            try:
                if hasattr(self.model, "n_features_in_"):
                    self.model_n_features = int(self.model.n_features_in_)

                booster = self.model.get_booster() if hasattr(self.model, "get_booster") else None
                if booster is not None and getattr(booster, "feature_names", None):
                    self.expected_feature_order = list(booster.feature_names)
                    self.model_n_features = len(self.expected_feature_order)
            except Exception as inner:
                self.Debug(f"Could not infer model feature metadata: {inner}")

            if self.expected_feature_order is None:
                self.expected_feature_order = list(self.model_features)

            if self.model_n_features is None:
                self.model_n_features = len(self.expected_feature_order)

            # Generate hash from bytes for version tracking
            self.model_version_hash = str(hash(model_bytes[:1000]))

            self.Debug(f"Successfully loaded model from Object Store (hash: {self.model_version_hash[:20]}...)")
            self.Debug(f"Model expects {self.model_n_features} features")
            return True

        except Exception as e:
            self.Error(f"Error loading model from Object Store: {e}")
            return False

    # =========================================================
    # SAVE MODEL TO OBJECT STORE
    # =========================================================
    def SaveModelToObjectStore(self):
        """Save current model to QuantConnect Object Store."""
        try:
            if self.model is None:
                self.Debug("No model to save to Object Store")
                return False

            model_bytes = pickle.dumps(self.model)
            self.ObjectStore.SaveBytes(self.object_store_key, model_bytes)
            self.Debug(f"Model saved to Object Store: {self.object_store_key}")
            return True

        except Exception as e:
            self.Error(f"Error saving model to Object Store: {e}")
            return False

    # =========================================================
    # LOAD MODEL FROM API
    # =========================================================
    def LoadModelFromAPI(self, save_to_object_store=False):
        """Load XGBoost model from API."""
        for attempt in range(self.api_retry_count):
            try:
                url = f"{self.api_base_url}/api/v1/{self.model_version}/latest/model"
                self.Debug(f"Fetching model from: {url}")

                response = requests.get(url, timeout=self.api_timeout)
                response.raise_for_status()

                data = response.json()

                if not data.get("success", False):
                    self.Error(f"Failed to get model: {data.get('message', 'Unknown error')}")
                    return

                # Get model data from base64
                model_data_b64 = data.get("model_data_base64")
                if not model_data_b64:
                    self.Error("No model data in response")
                    return

                # Get model version/hash for reload detection
                new_model_hash = data.get("model_hash") or data.get("version") or data.get("trained_at")
                if not new_model_hash:
                    # Use the model_data_base64 as a hash if no explicit version
                    new_model_hash = str(hash(model_data_b64[:1000]))  # Hash first 1000 chars

                # Check if model has changed
                if self.model_version_hash and self.model_version_hash == new_model_hash:
                    self.Debug("Model version unchanged, skipping reload")
                    return

                # Decode and load model
                try:
                    model_bytes = base64.b64decode(model_data_b64)
                    self.model = pickle.loads(model_bytes)
                    self.model_version_hash = new_model_hash
                    self.Debug(f"Successfully loaded XGBoost model from API (hash: {new_model_hash[:20]}...)")

                    # Get feature names from API response if available
                    api_feature_names = data.get("feature_names", [])
                    if api_feature_names:
                        self.expected_feature_order = list(api_feature_names)
                        self.model_n_features = len(self.expected_feature_order)
                        self.Debug(f"Using feature names from API: {self.model_n_features} features")
                    else:
                        # Try to get from model itself
                        self.model_n_features = None
                        self.expected_feature_order = None
                        try:
                            if hasattr(self.model, "n_features_in_"):
                                self.model_n_features = int(self.model.n_features_in_)

                            booster = self.model.get_booster() if hasattr(self.model, "get_booster") else None
                            if booster is not None and getattr(booster, "feature_names", None):
                                self.expected_feature_order = list(booster.feature_names)
                                self.model_n_features = len(self.expected_feature_order)
                        except Exception as inner:
                            self.Debug(f"Could not infer model feature metadata: {inner}")

                        # Fallback to default feature list
                        if self.expected_feature_order is None:
                            self.expected_feature_order = list(self.model_features)
                            self.Debug("Using fallback feature list")

                        if self.model_n_features is None:
                            self.model_n_features = len(self.expected_feature_order)

                    self.Debug(f"Model expects {self.model_n_features} features; order_len={len(self.expected_feature_order)}")

                    # Save to Object Store for future use (both backtest and live)
                    if save_to_object_store:
                        self.SaveModelToObjectStore()

                    return

                except Exception as decode_error:
                    self.Error(f"Error decoding model data: {decode_error}")
                    return

            except requests.exceptions.RequestException as e:
                self.Debug(f"API request failed (attempt {attempt + 1}/{self.api_retry_count}): {e}")
                if attempt == self.api_retry_count - 1:
                    self.Error("Failed to fetch model after all retries")
            except Exception as e:
                self.Error(f"Error loading model from API: {e}")
                break

    # =========================================================
    # RELOAD MODEL IF NEEDED (Scheduled)
    # =========================================================
    def ReloadModelIfNeeded(self):
        """Scheduled method to reload model from API periodically."""
        try:
            self.Debug(f"[MODEL RELOAD] Checking for model update at {self.Time}")
            self.LoadModelFromAPI(save_to_object_store=True)
        except Exception as e:
            self.Error(f"Error in scheduled model reload: {e}")

    # =========================================================
    # ONDATA
    # =========================================================
    def OnData(self, data):
        if self.IsWarmingUp:
            return

        if not data.ContainsKey(self.symbol):
            return

        # Prevent re-processing the same bar
        if self.last_trade_bar_time == self.Time:
            return

        bar = data[self.symbol]

        if isinstance(bar, TradeBar):
            open_ = float(bar.Open)
            high = float(bar.High)
            low = float(bar.Low)
            close = float(bar.Close)
            volume = float(bar.Volume) if bar.Volume is not None else 0.0
        elif isinstance(bar, QuoteBar):
            src = bar.Bid if bar.Bid is not None else bar.Ask
            if src is None:
                return
            open_ = float(src.Open)
            high = float(src.High)
            low = float(src.Low)
            close = float(src.Close)
            volume = 0.0
        else:
            return

        if volume <= 0:
            sec = self.Securities[self.symbol]
            if getattr(sec, "Volume", 0) and sec.Volume > 0:
                volume = float(sec.Volume)
            else:
                volume = 1_000_000.0

        # Update rolling window
        self.price_window.append({
            "time": self.Time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

        if len(self.price_window) < 30:
            return

        # Cooldown pause
        if self.trading_paused_until is not None and self.Time < self.trading_paused_until:
            return

        # If we have pending orders, do not submit more orders
        if self.pending_entry or self.pending_exit:
            return

        # Drawdown check
        if not self.CheckMaxDrawdownAndCooldown():
            self.last_trade_bar_time = self.Time
            return

        # Build features + predict
        features = self.BuildFeatures()
        if features is None:
            return

        pred = self.Predict(features)
        if pred is None:
            return

        # SL/TP first (exit has priority)
        if self.CheckStopLossTakeProfit(close):
            self.last_trade_bar_time = self.Time
            return

        # Decision based on prediction
        self.TradeLogic(pred)
        self.last_trade_bar_time = self.Time

    # =========================================================
    # FEATURES
    # =========================================================
    def BuildFeatures(self):
        try:
            df = pd.DataFrame(list(self.price_window))
            cur = df.iloc[-1]

            open_ = float(cur["open"])
            high = float(cur["high"])
            low = float(cur["low"])
            close = float(cur["close"])
            volume = float(cur["volume"])
            volume_usd = close * volume

            closes = df["close"].astype(float).values
            volumes = df["volume"].astype(float).values

            if len(closes) < 6:
                return None

            ret_1 = closes[-1] / closes[-2] - 1.0 if closes[-2] > 0 else 0.0
            ret_5 = closes[-1] / closes[-6] - 1.0 if closes[-6] > 0 else 0.0
            log_ret = np.log(closes[-1] / closes[-2]) if closes[-2] > 0 else 0.0

            returns = np.diff(closes) / closes[:-1]
            vol_5 = float(np.std(returns[-5:])) if len(returns) >= 5 else 0.0

            true_range = high - low
            mean_5 = float(np.mean(closes[-5:]))
            std_5 = float(np.std(closes[-5:]))

            vol_mean_10 = float(np.mean(volumes[-10:])) if len(volumes) >= 10 else volume
            vol_std_10 = float(np.std(volumes[-10:])) if len(volumes) >= 10 else 1.0
            vol_z = (volumes[-1] - vol_mean_10) / vol_std_10 if vol_std_10 > 0 else 0.0
            vol_change = volumes[-1] / volumes[-2] - 1.0 if len(volumes) > 1 and volumes[-2] > 0 else 0.0

            wick_up = high - max(open_, close)
            wick_low = min(open_, close) - low
            body_size = abs(close - open_)

            feat_map = {
                "price_open": open_,
                "price_high": high,
                "price_low": low,
                "price_close": close,
                "price_volume_usd": volume_usd,
                "price_close_return_1": ret_1,
                "price_close_return_5": ret_5,
                "price_log_return": log_ret,
                "price_rolling_vol_5": vol_5,
                "price_true_range": true_range,
                "price_close_mean_5": mean_5,
                "price_close_std_5": std_5,
                "price_volume_mean_10": vol_mean_10,
                "price_volume_zscore": vol_z,
                "price_volume_change": vol_change,
                "price_wick_upper": wick_up,
                "price_wick_lower": wick_low,
                "price_body_size": body_size,
            }

            order = self.expected_feature_order or self.model_features
            vec = []
            for name in order:
                v = float(feat_map.get(name, 0.0))
                if not np.isfinite(v):
                    v = 0.0
                vec.append(v)

            if len(vec) > self.model_n_features:
                vec = vec[:self.model_n_features]
            elif len(vec) < self.model_n_features:
                vec.extend([0.0] * (self.model_n_features - len(vec)))

            return np.asarray(vec, dtype=float).reshape(1, -1)
        except Exception as e:
            self.Error(f"BuildFeatures error: {e}")
            return None

    # =========================================================
    # PREDICT
    # =========================================================
    def Predict(self, feature_array):
        try:
            if self.model is None:
                # Model failed to load - this should not happen in normal operation
                self.Error("Model is None! Using neutral prediction (0.5). Check if API/Object Store is working.")
                return 0.5

            proba = self.model.predict_proba(feature_array)[0, 1]
            return float(proba)
        except Exception as e:
            self.Error(f"Predict error: {e}")
            # Return neutral prediction on error
            return 0.5

    # =========================================================
    # TRADING
    # =========================================================
    def TradeLogic(self, pred):
        qty = self.Portfolio[self.symbol].Quantity

        # ENTRY
        if pred > self.prediction_buy_threshold and qty <= 0 and not self.pending_entry and not self.pending_exit:
            self.pending_entry = True
            self.SetHoldings(self.symbol, self.position_size_pct)
            self.entry_price = float(self.Securities[self.symbol].Price)
            self.entry_time = self.Time
            self.Debug(f"BUY SetHoldings({self.position_size_pct:.0%}) pred={pred:.3f} est_entry={self.entry_price:.2f}")

        # EXIT by prediction
        elif pred < self.prediction_sell_threshold and qty > 0 and not self.pending_exit:
            self.pending_exit = True
            self.last_exit_reason = "PRED"
            self.Liquidate(self.symbol)
            self.Debug(f"EXIT (PRED) pred={pred:.3f}")

    # =========================================================
    # SL/TP
    # =========================================================
    def CheckStopLossTakeProfit(self, price):
        """
        Returns True if an exit order was submitted (so OnData should stop further actions in this bar).
        """
        try:
            if self.pending_exit:
                return True

            qty = self.Portfolio[self.symbol].Quantity
            if qty <= 0 or self.entry_price is None:
                return False

            pnl_pct = (price - self.entry_price) / self.entry_price if self.entry_price != 0 else 0.0

            if pnl_pct <= -self.stop_loss_pct:
                self.pending_exit = True
                self.last_exit_reason = "SL"
                self.Liquidate(self.symbol)
                self.Debug(f"STOP LOSS trigger pnl={pnl_pct:.2%}")
                return True

            if pnl_pct >= self.take_profit_pct:
                self.pending_exit = True
                self.last_exit_reason = "TP"
                self.Liquidate(self.symbol)
                self.Debug(f"TAKE PROFIT trigger pnl={pnl_pct:.2%}")
                return True

            return False
        except Exception as e:
            self.Error(f"CheckStopLossTakeProfit error: {e}")
            return False

    # =========================================================
    # DRAWDOWN -> COOLDOWN
    # =========================================================
    def CheckMaxDrawdownAndCooldown(self):
        pv = float(self.Portfolio.TotalPortfolioValue)
        if pv > self.high_watermark:
            self.high_watermark = pv

        dd = (self.high_watermark - pv) / self.high_watermark if self.high_watermark > 0 else 0.0

        if dd > self.max_drawdown_pct:
            # Exit and pause trading
            if self.Portfolio[self.symbol].Invested:
                self.pending_exit = True
                self.last_exit_reason = "RISK"
                self.Liquidate()
                self.Debug(f"[RISK] Liquidate due to DD={dd:.2%}")

            self.trading_paused_until = self.Time + self.cooldown_period
            self.Debug(f"[RISK] DD={dd:.2%} > {self.max_drawdown_pct:.2%}. Pause until {self.trading_paused_until}")

            # reset watermark so it can resume after cooldown
            self.high_watermark = pv
            return False

        return True

    def OnOrderEvent(self, orderEvent):
        """Handle order events."""
        try:
            if orderEvent.Status != OrderStatus.Filled:
                return

            order = self.Transactions.GetOrderById(orderEvent.OrderId)
            if order is None or order.Symbol != self.symbol:
                return

            fill_price = float(orderEvent.FillPrice)
            fill_qty = float(orderEvent.FillQuantity)

            # BUY filled
            if fill_qty > 0:
                self.entry_price = fill_price
                self.entry_time = self.Time
                self.pending_entry = False
                self.Debug(f"ENTRY FILLED @ {fill_price:.2f}")

            # SELL filled (exit)
            elif fill_qty < 0:
                reason = self.last_exit_reason or "EXIT"
                self.Debug(f"EXIT FILLED @ {fill_price:.2f} Reason: {reason}")

                # Reset after full exit
                self.entry_price = None
                self.entry_time = None
                self.pending_exit = False
                self.last_exit_reason = None

        except Exception as e:
            self.Debug(f"OnOrderEvent error: {e}")

    def OnEndOfAlgorithm(self):
        self.Debug(f"Final Portfolio Value: {self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug("NOTE: Missing non-price features are set to 0; results won't match offline full-feature backtests.")
