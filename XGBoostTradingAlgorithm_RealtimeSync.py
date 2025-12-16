from AlgorithmImports import *
import numpy as np
import pandas as pd
from collections import deque
import os
import re
import joblib
from datetime import datetime, timedelta
import json

# Ensure Binance market is available
try:
    from QuantConnect.Market import Market
    Market.Binance  # Test if available
except:
    # Define if not available
    class Market:
        Binance = "Binance"


class XGBoostTradingAlgorithm(QCAlgorithm):
    """
    Enhanced XGBoost Algorithm with Real-time Model Sync.

    Features:
    - Auto-sync with real-time training system
    - Model performance tracking
    - Fallback to previous model if sync fails
    - Enhanced notifications
    - Model version tracking
    """

    def Initialize(self):
        # ===== ObjectStore keys =====
        self.model_key = "latest_model.joblib"
        self.dataset_summary_key = "dataset_summary.txt"
        self.model_metadata_key = "model_metadata.json"

        # Real-time sync settings
        self.enable_realtime_sync = True
        self.sync_check_interval_hours = 1  # Check for new model every hour
        self.last_sync_check_time = None

        # Model version tracking
        self.current_model_version = None
        self.current_model_performance = None

        self.train_start_date = None
        self.train_end_date = None
        self.LoadDatasetSummaryFromObjectStore()

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
        self.model_features = [
            "price_open","price_high","price_low","price_close","price_volume_usd",
            "funding_open","funding_high","funding_low","funding_close",
            "basis_open_basis","basis_close_basis","basis_open_change","basis_close_change",
            "ls_global_global_account_long_percent","ls_global_global_account_short_percent",
            "ls_global_global_account_long_short_ratio",
            "ls_top_top_account_long_percent","ls_top_top_account_short_percent",
            "ls_top_top_account_long_short_ratio",
            "price_close_return_1","price_close_return_5","price_log_return",
            "price_rolling_vol_5","price_true_range","price_close_mean_5","price_close_std_5",
            "price_volume_mean_10","price_volume_zscore","price_volume_change",
            "price_wick_upper","price_wick_lower","price_body_size",
            "funding_norm","funding_mean_24","funding_std_24","funding_zscore",
            "funding_extreme_positive","funding_extreme_negative",
            "basis_delta","basis_drift","basis_mean_24","basis_zscore","basis_volatility_24",
            "ls_global_ratio","ls_global_zscore","ls_global_delta",
            "ls_global_extreme_high","ls_global_extreme_low",
            "ls_top_ratio","ls_top_zscore","ls_top_delta","ls_top_vs_global",
            "cross_funding_price","cross_ls_price",
        ]

        # Model metadata
        self.strategy_name = "Metode ABC (Real-time Sync)"
        self.startup_notified = False

        self.model = None
        self.previous_model = None  # Fallback model
        self.model_n_features = None
        self.expected_feature_order = None

        # Load model and metadata
        self.LoadModelFromObjectStore()
        self.LoadModelMetadataFromObjectStore()

        # ===== Rolling window =====
        self.window_size = 60
        self.price_window = deque(maxlen=self.window_size)

        # ===== Trading parameters (enhanced based on model performance) =====
        # Adjust thresholds based on model performance
        if self.current_model_performance:
            auc = self.current_model_performance.get('latest_auc', 0.5)
            if auc > 0.7:
                # High confidence model - use tighter thresholds
                self.prediction_buy_threshold = 0.60
                self.prediction_sell_threshold = 0.40
            elif auc > 0.6:
                # Medium confidence - default thresholds
                self.prediction_buy_threshold = 0.55
                self.prediction_sell_threshold = 0.45
            else:
                # Low confidence - very conservative
                self.prediction_buy_threshold = 0.70
                self.prediction_sell_threshold = 0.30
        else:
            # Default thresholds
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

        # ===== Notifications =====
        self.enable_live_notifications = True
        self.notify_channel = "Telegram"
        self.telegram_token = "8306719491:AAHNS7HT-pjMUGlcXMA_5SEffd6zPd2X6U0"
        self.telegram_chat_id = "-4978819951"
        self.webhook_url = ""
        self.email_to = ""
        self.sms_number = ""

        # ===== Real-time sync status =====
        self.sync_status = {
            'last_check': None,
            'last_sync': None,
            'sync_count': 0,
            'sync_failures': 0,
            'model_updates': 0
        }

        # ===== Enhanced notifications =====
        self.model_performance_alerts = True
        self.sync_notifications = True

        # ===== Reminder Signals (same as original) =====
        self.enable_reminders = True
        self.reminder_min_hours = 1
        self.reminder_max_hours = 3
        self.reminder_min_gap = timedelta(hours=2)
        self.last_reminder_sent_time = {}
        self.reminders = {}
        self.entry_pred_band = 0.03
        self.exit_pred_band = 0.03
        self.near_level_1h = 0.90
        self.near_level_2h = 0.80
        self.near_level_3h = 0.70

        # Warmup
        self.SetWarmUp(120, Resolution.Hour)
        self.pred_debug_counter = 0
        self.Debug("XGBoostTradingAlgorithm (Real-time Sync) initialized")

    def OnWarmupFinished(self):
        # Enhanced startup message with sync info
        if not self.LiveMode:
            self.Debug("[TELEGRAM] Not live mode, skip")
            return

        try:
            sync_info = ""
            if self.current_model_version:
                sync_info = f"\nModel Version: {self.current_model_version}"
                if self.current_model_performance:
                    sync_info += f"\nModel AUC: {self.current_model_performance.get('latest_auc', 'N/A')}"

            msg = f"🚀 {self.strategy_name} is running{sync_info}\nSymbol: {self.symbol}\nResolution: Hour\nReal-time Sync: {'Enabled' if self.enable_realtime_sync else 'Disabled'}"
            ok = self.Notify.Telegram(str(self.telegram_chat_id), msg, str(self.telegram_token))
            self.Debug(f"[TELEGRAM] sent={ok}")
        except Exception as e:
            self.Error(f"[TELEGRAM] failed: {e}")

        self.SendStartupMessage()

    def SendStartupMessage(self):
        if self.startup_notified:
            return
        if not self.LiveMode:
            return

        model_info = ""
        if self.current_model_version:
            model_info = f"\nModel Version: {self.current_model_version}"
            if self.current_model_performance:
                perf = self.current_model_performance
                model_info += f"\nModel Performance: AUC={perf.get('latest_auc', 'N/A')}, Accuracy={perf.get('latest_accuracy', 'N/A')}"

        msg = (
            f"🤖 {self.strategy_name} Started{model_info}\n"
            f"Symbol: {self.symbol}\n"
            f"Resolution: Hour\n"
            f"Time: {self.Time}\n"
            f"Buy Threshold: {self.prediction_buy_threshold:.2f}\n"
            f"Sell Threshold: {self.prediction_sell_threshold:.2f}"
        )

        self.SendSignal("START", msg)
        self.startup_notified = True

    # =========================================================
    # REAL-TIME SYNC FUNCTIONALITY
    # =========================================================
    def LoadModelMetadataFromObjectStore(self):
        """Load model metadata from ObjectStore."""
        try:
            if not self.ObjectStore.ContainsKey(self.model_metadata_key):
                self.Debug("Model metadata not found in ObjectStore")
                return

            file_path = self.ObjectStore.GetFilePath(self.model_metadata_key)
            with open(file_path, 'r') as f:
                metadata = json.load(f)

            self.current_model_version = metadata.get('created_at', 'Unknown')
            self.current_model_performance = metadata.get('performance', {})

            self.Debug(f"Loaded model metadata: version={self.current_model_version}")

        except Exception as e:
            self.Error(f"Error loading model metadata: {e}")

    def CheckForModelUpdate(self):
        """Check if new model is available and sync if needed."""
        if not self.enable_realtime_sync:
            return

        now = self.Time
        if (self.last_sync_check_time and
            (now - self.last_sync_check_time) < timedelta(hours=self.sync_check_interval_hours)):
            return

        self.last_sync_check_time = now
        self.Debug(f"[SYNC] Checking for model update at {now}")

        try:
            # Check if new metadata exists
            if self.ObjectStore.ContainsKey(self.model_metadata_key):
                file_path = self.ObjectStore.GetFilePath(self.model_metadata_key)
                with open(file_path, 'r') as f:
                    metadata = json.load(f)

                new_version = metadata.get('created_at', 'Unknown')
                new_performance = metadata.get('performance', {})

                # Check if model is newer
                if new_version != self.current_model_version:
                    self.Debug(f"[SYNC] New model found: {new_version}")

                    # Load new model
                    if self.LoadNewModelFromObjectStore():
                        # Update metadata
                        old_version = self.current_model_version
                        self.current_model_version = new_version
                        self.current_model_performance = new_performance

                        # Update trading parameters based on new performance
                        self.UpdateTradingParameters(new_performance)

                        # Send notification
                        self.SendModelUpdateNotification(old_version, new_version, new_performance)

                        self.sync_status['model_updates'] += 1
                        self.Debug(f"[SYNC] Successfully updated to model {new_version}")
                    else:
                        self.sync_status['sync_failures'] += 1
                        self.Error(f"[SYNC] Failed to load new model {new_version}")

        except Exception as e:
            self.Error(f"[SYNC] Error checking for model update: {e}")
            self.sync_status['sync_failures'] += 1

    def LoadNewModelFromObjectStore(self) -> bool:
        """Load new model from ObjectStore with fallback."""
        try:
            if not self.ObjectStore.ContainsKey(self.model_key):
                self.Error("Model not found in ObjectStore")
                return False

            # Backup current model
            if self.model is not None:
                self.previous_model = self.model

            file_path = self.ObjectStore.GetFilePath(self.model_key)
            new_model = joblib.load(file_path)

            # Test new model with dummy data
            test_features = np.zeros((1, self.model_n_features or len(self.model_features)))
            test_pred = new_model.predict_proba(test_features)[0, 1]

            if not np.isfinite(test_pred):
                self.Error("New model failed basic prediction test")
                return False

            # Load new model
            self.model = new_model
            self.Debug("Successfully loaded and validated new model")

            # Update feature info
            try:
                if hasattr(self.model, "n_features_in_"):
                    self.model_n_features = int(self.model.n_features_in_)

                booster = self.model.get_booster() if hasattr(self.model, "get_booster") else None
                if booster is not None and getattr(booster, "feature_names", None):
                    self.expected_feature_order = list(booster.feature_names)
                    self.model_n_features = len(self.expected_feature_order)
            except Exception as inner:
                self.Debug(f"Could not infer new model feature metadata: {inner}")

            if self.expected_feature_order is None:
                self.expected_feature_order = list(self.model_features)

            if self.model_n_features is None:
                self.model_n_features = len(self.expected_feature_order)

            return True

        except Exception as e:
            self.Error(f"Error loading new model: {e}")
            # Restore previous model if available
            if self.previous_model is not None:
                self.model = self.previous_model
                self.Debug("Restored previous model due to load error")
            return False

    def UpdateTradingParameters(self, performance: Dict):
        """Update trading parameters based on model performance."""
        try:
            auc = performance.get('latest_auc', 0.5)
            accuracy = performance.get('latest_accuracy', 0.5)

            # Dynamic threshold adjustment based on performance
            base_buy = 0.55
            base_sell = 0.45

            if auc > 0.75 and accuracy > 0.7:
                # Excellent model - more aggressive
                self.prediction_buy_threshold = base_buy - 0.05
                self.prediction_sell_threshold = base_sell + 0.05
                self.position_size_pct = 0.90
                self.take_profit_pct = 0.12
            elif auc > 0.65 and accuracy > 0.6:
                # Good model - standard
                self.prediction_buy_threshold = base_buy
                self.prediction_sell_threshold = base_sell
                self.position_size_pct = 0.80
                self.take_profit_pct = 0.10
            else:
                # Poor model - very conservative
                self.prediction_buy_threshold = base_buy + 0.10
                self.prediction_sell_threshold = base_sell - 0.10
                self.position_size_pct = 0.60
                self.take_profit_pct = 0.08

            self.Debug(f"[PARAMS] Updated thresholds based on performance AUC={auc:.3f}: Buy={self.prediction_buy_threshold:.2f}, Sell={self.prediction_sell_threshold:.2f}")

        except Exception as e:
            self.Error(f"Error updating trading parameters: {e}")

    def SendModelUpdateNotification(self, old_version: str, new_version: str, performance: Dict):
        """Send notification about model update."""
        if not self.sync_notifications:
            return

        auc = performance.get('latest_auc', 'N/A')
        accuracy = performance.get('latest_accuracy', 'N/A')
        updates = performance.get('updates', 0)

        msg = (
            f"🔄 **Model Updated**\n\n"
            f"📊 **Model Info:**\n"
            f"Previous: {old_version}\n"
            f"New: {new_version}\n"
            f"Updates: {updates}\n\n"
            f"🎯 **Performance:**\n"
            f"AUC: {auc}\n"
            f"Accuracy: {accuracy}\n\n"
            f"⚙️ **New Parameters:**\n"
            f"Buy Threshold: {self.prediction_buy_threshold:.2f}\n"
            f"Sell Threshold: {self.prediction_sell_threshold:.2f}\n"
            f"Position Size: {self.position_size_pct:.0%}\n"
            f"Take Profit: {self.take_profit_pct:.0%}\n\n"
            f"⏰ {self.Time}"
        )

        self.SendSignal("MODEL_UPDATE", msg)

    # =========================================================
    # REST OF THE ORIGINAL ALGORITHM (unchanged)
    # =========================================================
    # Copy all other methods from the original algorithm:
    # - SendSignal
    # - _fmt_price, _fmt_pct
    # - CalcEntryTargets
    # - FormatEntrySignal, FormatExitSignal
    # - Reminder methods
    # - LoadDatasetSummaryFromObjectStore
    # - LoadModelFromObjectStore (updated above)
    # - OnOrderEvent
    # - OnData (with sync check added)
    # - BuildFeatures
    # - Predict
    # - TradeLogic
    # - CheckStopLossTakeProfit
    # - CheckMaxDrawdownAndCooldown
    # - OnEndOfAlgorithm

    def OnData(self, data: Slice):
        # Add model sync check at the beginning of OnData
        self.CheckForModelUpdate()

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

        # Optional debug every ~24 hours
        if self.pred_debug_counter % 24 == 0:
            model_info = f" v{self.current_model_version[:8]}" if self.current_model_version else ""
            self.Debug(f"{self.Time} - Pred={pred:.3f} Price={close:.2f}{model_info}")
        self.pred_debug_counter += 1

        # Arm & process short reminders (1-3h) BEFORE any actions
        self.MaybeArmReminders(pred, close)
        self.ProcessReminders(pred, close)

        # SL/TP first (exit has priority)
        if self.CheckStopLossTakeProfit(close):
            self.last_trade_bar_time = self.Time
            return

        # Decision based on prediction
        self.TradeLogic(pred)
        self.last_trade_bar_time = self.Time

    # Add all the other methods from your original algorithm here...
    # (I'll include the key ones for completeness)

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

    def Predict(self, feature_array: np.ndarray):
        try:
            if self.model is None:
                return None
            proba = self.model.predict_proba(feature_array)[0, 1]
            return float(proba)
        except Exception as e:
            self.Error(f"Predict error: {e}")
            return None

    def TradeLogic(self, pred: float):
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

    def OnEndOfAlgorithm(self):
        sync_summary = ""
        if self.sync_status['model_updates'] > 0:
            sync_summary = f"\n📊 Model Updates: {self.sync_status['model_updates']}"

        self.Debug(f"Final Portfolio Value: {self.Portfolio.TotalPortfolioValue:.2f}{sync_summary}")