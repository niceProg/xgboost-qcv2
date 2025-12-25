from AlgorithmImports import *
import numpy as np
import pandas as pd
from collections import deque
import os
import re
import joblib
from datetime import datetime, timedelta

# Ensure Binance market is available
try:
    from QuantConnect.Market import Market
    Market.Binance  # Test if available
except:
    # Define if not available
    class Market:
        Binance = "Binance"


class XGBoostTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        # === CONFIG KUNCI OBJECTSTORE ===
        self.model_key = "latest_model.joblib"
        self.dataset_summary_key = "dataset_summary.txt"

        # Tempat nyimpan hasil parsing time range dari dataset_summary.txt
        self.train_start_date = None   # datetime (tanggal pertama di dataset)
        self.train_end_date = None     # datetime (tanggal terakhir di dataset)

        # 1) Baca dataset_summary.txt dulu untuk dynamic tanggal
        self.LoadDatasetSummaryFromObjectStore()

        # 2) Tentukan StartDate & EndDate sesuai dataset
        if self.train_start_date is not None and self.train_end_date is not None:
            # optional buffer 1 hari ke belakang (biar kalau nanti perlu history extra, aman)
            start_dt = self.train_start_date - timedelta(days=1)
            end_dt = self.train_end_date
            self.Debug(
                f"[DATE] Using dynamic backtest dates from dataset_summary "
                f"{self.train_start_date} -> {self.train_end_date} "
                f"(engine range: {start_dt} -> {end_dt})"
            )
        else:
            # fallback kalau summary tidak ada / gagal parse
            start_dt = datetime(2024, 1, 1)
            end_dt = datetime(2024, 12, 31)
            self.Debug(
                f"[DATE] dataset_summary not found or invalid, using fallback dates: "
                f"{start_dt} -> {end_dt}"
            )

        # === BASIC SETTINGS ===
        self.SetStartDate(start_dt.year, start_dt.month, start_dt.day)
        self.SetEndDate(end_dt.year, end_dt.month, end_dt.day)
        self.SetCash(100000)

        # === SYMBOL & RESOLUTION ===
        # Using Binance to match training data
        # Try multiple methods for Binance compatibility
        try:
            self.crypto = self.AddCrypto("BTCUSDT", Resolution.Hour, Market.Binance)
        except:
            try:
                # Alternative method 1
                self.SetBrokerageModel(BrokerageName.Binance)
                self.crypto = self.AddCrypto("BTCUSDT", Resolution.Hour)
            except:
                # Fallback to default market with Binance symbol
                self.crypto = self.AddCrypto("BTCUSDT", Resolution.Hour)
                self.Debug("Warning: Using default market, but model expects Binance data")

        self.symbol = self.crypto.Symbol
        self.SetBenchmark(self.symbol)

        # WARNING: QuantConnect doesn't provide funding rates, basis, or L/S ratio data
        # Kita hanya pakai fitur price-based yang memang ada di QC

        # === MODEL / FEATURE CONFIG ===
        # Fitur price-based yang memang bisa dihitung dari QC
        self.available_features = [
            "price_open",
            "price_high",
            "price_low",
            "price_close",
            "price_volume_usd",
            "price_close_return_1",
            "price_close_return_5",
            "price_log_return",
            "price_rolling_vol_5",
            "price_true_range",
            "price_close_mean_5",
            "price_close_std_5",
            "price_volume_mean_10",
            "price_volume_zscore",
            "price_volume_change",
            "price_wick_upper",
            "price_wick_lower",
            "price_body_size",
        ]

        # Full feature list dari model (53 fitur total)
        self.model_features = [
            "price_open",
            "price_high",
            "price_low",
            "price_close",
            "price_volume_usd",
            "funding_open",
            "funding_high",
            "funding_low",
            "funding_close",
            "basis_open_basis",
            "basis_close_basis",
            "basis_open_change",
            "basis_close_change",
            "ls_global_global_account_long_percent",
            "ls_global_global_account_short_percent",
            "ls_global_global_account_long_short_ratio",
            "ls_top_top_account_long_percent",
            "ls_top_top_account_short_percent",
            "ls_top_top_account_long_short_ratio",
            "price_close_return_1",
            "price_close_return_5",
            "price_log_return",
            "price_rolling_vol_5",
            "price_true_range",
            "price_close_mean_5",
            "price_close_std_5",
            "price_volume_mean_10",
            "price_volume_zscore",
            "price_volume_change",
            "price_wick_upper",
            "price_wick_lower",
            "price_body_size",
            "funding_norm",
            "funding_mean_24",
            "funding_std_24",
            "funding_zscore",
            "funding_extreme_positive",
            "funding_extreme_negative",
            "basis_delta",
            "basis_drift",
            "basis_mean_24",
            "basis_zscore",
            "basis_volatility_24",
            "ls_global_ratio",
            "ls_global_zscore",
            "ls_global_delta",
            "ls_global_extreme_high",
            "ls_global_extreme_low",
            "ls_top_ratio",
            "ls_top_zscore",
            "ls_top_delta",
            "ls_top_vs_global",
            "cross_funding_price",
            "cross_ls_price",
        ]

        # jumlah fitur yang diminta model (diisi setelah LoadModelFromObjectStore)
        self.model_n_features = None

        # Simpan max 30 bar terakhir untuk hitung fitur
        self.window_size = 30
        self.price_window = deque(maxlen=self.window_size)

        # === TRADING PARAMETERS ===
        self.prediction_buy_threshold = 0.55   # > 0.55 = long
        self.prediction_sell_threshold = 0.45  # < 0.45 = exit
        self.position_size_pct = 0.8           # 80% capital
        self.stop_loss_pct = 0.05              # 5% stop loss
        self.take_profit_pct = 0.10            # 10% take profit

        self.entry_price = None

        # === SIMPLE RISK ===
        self.max_drawdown_pct = 0.20
        self.high_watermark = self.Portfolio.TotalPortfolioValue

        # === LOAD MODEL DARI OBJECTSTORE ===
        self.model = None
        self.LoadModelFromObjectStore()

        # === WARMUP ===
        # warmup 100 bar 1H to ensure we have enough data for feature calculations
        self.SetWarmUp(100, Resolution.Hour)

        # Counter kecil untuk debug
        self.pred_debug_counter = 0

        self.Debug("XGBoostTradingAlgorithm initialized")

    # =========================================================
    # LOAD DATASET SUMMARY (DYNAMIC DATE RANGE)
    # =========================================================
    def LoadDatasetSummaryFromObjectStore(self):
        """
        Baca dataset_summary.txt dari ObjectStore dan parse:
        Time range: 2025-06-13 08:00:00 to 2025-12-04 13:00:00
        """
        try:
            if not self.ObjectStore.ContainsKey(self.dataset_summary_key):
                self.Debug(f"dataset_summary key not found: {self.dataset_summary_key}")
                return

            file_path = self.ObjectStore.GetFilePath(self.dataset_summary_key)
            self.Debug(f"Loading dataset_summary from ObjectStore path: {file_path}")

            with open(file_path, "r") as f:
                text = f.read()

            # Cari baris "Time range: ... to ..."
            m = re.search(r"Time range:\s*(.+?)\s*to\s*(.+)", text)
            if not m:
                self.Error("Could not find 'Time range: ... to ...' in dataset_summary.txt")
                return

            start_raw = m.group(1).strip()
            end_raw = m.group(2).strip()

            # Format contoh: 2025-06-13 08:00:00
            dt_format = "%Y-%m-%d %H:%M:%S"
            start_dt = datetime.strptime(start_raw, dt_format)
            end_dt = datetime.strptime(end_raw, dt_format)

            # Simpan hanya tanggal (jam diabaikan untuk SetStartDate/SetEndDate)
            self.train_start_date = datetime(start_dt.year, start_dt.month, start_dt.day)
            self.train_end_date = datetime(end_dt.year, end_dt.month, end_dt.day)

            self.Debug(
                f"Parsed dataset time range: {self.train_start_date} -> {self.train_end_date}"
            )

        except Exception as e:
            self.Error(f"Error loading/parsing dataset_summary.txt: {e}")
            self.train_start_date = None
            self.train_end_date = None

    # =========================================================
    # LOAD MODEL FROM OBJECTSTORE
    # =========================================================
    def LoadModelFromObjectStore(self):
        """Load XGBoost model yang disimpan di ObjectStore."""
        try:
            if not self.ObjectStore.ContainsKey(self.model_key):
                self.Error(f"ObjectStore key not found: {self.model_key}")
                self.model = None
                return

            file_path = self.ObjectStore.GetFilePath(self.model_key)
            self.Debug(f"Loading model from ObjectStore path: {file_path}")

            self.model = joblib.load(file_path)
            self.Debug("Successfully loaded XGBoost model from ObjectStore")

            # --- cari berapa banyak fitur yang diharapkan model ---
            self.model_n_features = None

            try:
                if hasattr(self.model, "n_features_in_"):
                    self.model_n_features = int(self.model.n_features_in_)
                    self.Debug(
                        f"Model expects {self.model_n_features} features (n_features_in_)"
                    )
                else:
                    booster = self.model.get_booster()
                    if (
                        hasattr(booster, "feature_names")
                        and booster.feature_names is not None
                    ):
                        self.model_n_features = len(booster.feature_names)
                        self.Debug(
                            f"Model expects {self.model_n_features} features "
                            f"(booster.feature_names)"
                        )
            except Exception as inner:
                self.Debug(f"Could not infer model feature count: {inner}")
                self.model_n_features = None

            if self.model_n_features is not None:
                self.Debug(
                    f"Local model_features={len(self.model_features)}, "
                    f"model_n_features={self.model_n_features}"
                )

        except Exception as e:
            self.Error(f"Error loading model from ObjectStore: {e}")
            self.model = None
            self.model_n_features = None

    # =========================================================
    # ONDATA
    # =========================================================
    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        if not data.ContainsKey(self.symbol):
            return

        bar = data[self.symbol]
        current_time = self.Time

        # Ambil OHLC dari TradeBar / QuoteBar
        if isinstance(bar, TradeBar):
            open_ = float(bar.Open)
            high = float(bar.High)
            low = float(bar.Low)
            close = float(bar.Close)

        elif isinstance(bar, QuoteBar):
            src = bar.Bid if bar.Bid is not None else bar.Ask
            if src is None:
                return
            open_ = float(src.Open)
            high = float(src.High)
            low = float(src.Low)
            close = float(src.Close)
        else:
            return

        # Volume: pakai volume dari security dengan proper fallback
        sec = self.Securities[self.symbol]
        if hasattr(bar, "Volume") and bar.Volume is not None and bar.Volume > 0:
            volume = float(bar.Volume)
        elif hasattr(sec, "Volume") and sec.Volume is not None and sec.Volume > 0:
            volume = float(sec.Volume)
        else:
            # Fallback: gunakan rata-rata volume recent
            if len(self.price_window) > 1:
                recent_volumes = [p["volume"] for p in list(self.price_window) if p["volume"] > 0]
                volume = float(np.mean(recent_volumes)) if recent_volumes else 1000000.0
            else:
                volume = 1000000.0  # Default fallback untuk crypto

        price_info = {
            "time": current_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        self.price_window.append(price_info)

        if len(self.price_window) < 30:  # Need enough data for feature calculation
            return

        if not self.CheckMaxDrawdown():
            return

        features = self.BuildFeatures()
        if features is None:
            return

        pred = self.Predict(features)
        if pred is None:
            return

        # debug pred secara berkala (sekitar 1x per hari)
        if self.pred_debug_counter % 24 == 0:
            self.Debug(f"{self.Time} - Pred={pred:.3f} Price={close:.2f}")
        self.pred_debug_counter += 1

        self.CheckStopLossTakeProfit(close)
        self.TradeLogic(pred, close)

    # =========================================================
    # FEATURE ENGINEERING
    # =========================================================
    def BuildFeatures(self):
        """
        Build feature vector using ONLY price-based data available in QuantConnect.

        WARNING: Ini tetap tidak akan sekuat model full karena fitur funding/basis/LS
        yang dominan di training report tidak tersedia di QC. Fitur itu diisi 0.
        """
        try:
            df = pd.DataFrame(list(self.price_window))
            current = df.iloc[-1]
            open_ = current["open"]
            high = current["high"]
            low = current["low"]
            close = current["close"]
            volume = current["volume"]
            volume_usd = close * volume

            closes = df["close"].values
            volumes = df["volume"].values

            if len(closes) < 6:
                return None

            # Price-based features
            ret_1 = closes[-1] / closes[-2] - 1
            ret_5 = closes[-1] / closes[-6] - 1
            log_ret = np.log(closes[-1] / closes[-2]) if closes[-2] > 0 else 0.0

            returns = np.diff(closes) / closes[:-1]
            vol_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0.0
            true_range = high - low
            mean_5 = np.mean(closes[-5:]) if len(closes) >= 5 else close
            std_5 = np.std(closes[-5:]) if len(closes) >= 5 else 0.0

            vol_mean_10 = np.mean(volumes[-10:]) if len(volumes) >= 10 else volume
            vol_std_10 = np.std(volumes[-10:]) if len(volumes) >= 10 else 1.0
            vol_z = (volumes[-1] - vol_mean_10) / vol_std_10 if vol_std_10 > 0 else 0.0
            vol_change = volumes[-1] / volumes[-2] - 1 if len(volumes) > 1 and volumes[-2] > 0 else 0.0

            wick_up = high - max(open_, close)
            wick_low = min(open_, close) - low
            body_size = abs(close - open_)

            # Map fitur yang tersedia
            available_feat_map = {
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

            # Build full feature vector dalam URUTAN model_features
            feature_vector = []
            for feature_name in self.model_features:
                if feature_name in available_feat_map:
                    value = float(available_feat_map[feature_name])
                    if not np.isfinite(value):
                        value = 0.0
                else:
                    # fitur funding/basis/LS dll yang tidak ada → 0
                    value = 0.0
                feature_vector.append(value)

            # Sesuaikan ke jumlah fitur yang diharapkan model
            if self.model_n_features is not None:
                if len(feature_vector) > self.model_n_features:
                    feature_vector = feature_vector[:self.model_n_features]
                elif len(feature_vector) < self.model_n_features:
                    feature_vector.extend([0.0] * (self.model_n_features - len(feature_vector)))

            # Debug logging
            if self.pred_debug_counter < 5:
                missing_features_count = len(self.model_features) - len(self.available_features)
                self.Debug(f"Using {len(self.available_features)}/{len(self.model_features)} features "
                           f"(vector len={len(feature_vector)}, expected={self.model_n_features})")
                self.Debug(f"Available features: {len(self.available_features)}, "
                           f"Missing features (set to 0): {missing_features_count}")
                self.Debug(f"Sample: close={close:.2f}, return_1={ret_1:.4f}, volume_usd={volume_usd:.0f}")

            return np.array(feature_vector).reshape(1, -1)

        except Exception as e:
            self.Error(f"Error in BuildFeatures: {e}")
            import traceback
            self.Error(traceback.format_exc())
            return None

    # =========================================================
    # PREDICTION
    # =========================================================
    def Predict(self, feature_array: np.ndarray):
        """Panggil model XGBoost untuk prediksi probabilitas naik."""
        try:
            if self.model is None:
                if self.pred_debug_counter == 0:
                    self.Error("Model is None in Predict() - no trades will be taken")
                return None

            proba = self.model.predict_proba(feature_array)[0, 1]
            return float(proba)

        except Exception as e:
            self.Error(f"Error in Predict: {e}")
            return None

    # =========================================================
    # TRADING LOGIC
    # =========================================================
    def TradeLogic(self, pred: float, price: float):
        position = self.Portfolio[self.symbol].Quantity

        # BUY (open long)
        if pred is not None and pred > self.prediction_buy_threshold and position <= 0:
            portfolio_value = self.Portfolio.TotalPortfolioValue
            target_value = portfolio_value * self.position_size_pct
            quantity = target_value / price

            self.MarketOrder(self.symbol, quantity)
            self.entry_price = price
            self.Debug(f"BUY {quantity:.6f} @ {price:.2f} (pred={pred:.3f})")

        # EXIT long
        elif pred is not None and pred < self.prediction_sell_threshold and position > 0:
            self.Liquidate(self.symbol)
            self.Debug(f"EXIT LONG @ {price:.2f} (pred={pred:.3f})")
            self.entry_price = None

    # =========================================================
    # STOP LOSS / TAKE PROFIT
    # =========================================================
    def CheckStopLossTakeProfit(self, price: float):
        try:
            position = self.Portfolio[self.symbol].Quantity
            if position <= 0 or self.entry_price is None:
                return

            pnl_pct = (price - self.entry_price) / self.entry_price

            if pnl_pct <= -self.stop_loss_pct:
                self.Debug(f"STOP LOSS triggered: pnl={pnl_pct:.2%}")
                self.Liquidate(self.symbol)
                self.entry_price = None

            elif pnl_pct >= self.take_profit_pct:
                self.Debug(f"TAKE PROFIT triggered: pnl={pnl_pct:.2%}")
                self.Liquidate(self.symbol)
                self.entry_price = None

        except Exception as e:
            self.Error(f"Error in CheckStopLossTakeProfit: {e}")

    # =========================================================
    # SIMPLE MAX DRAWDOWN CHECK
    # =========================================================
    def CheckMaxDrawdown(self):
        pv = self.Portfolio.TotalPortfolioValue
        if pv > self.high_watermark:
            self.high_watermark = pv

        dd = (self.high_watermark - pv) / self.high_watermark
        if dd > self.max_drawdown_pct:
            self.Debug(
                f"Max drawdown exceeded: {dd:.2%}, liquidating & stopping new trades"
            )
            self.Liquidate()
            return False

        return True

    def OnEndOfAlgorithm(self):
        self.Debug(f"Final Portfolio Value: {self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug("=" * 60)
        self.Debug("IMPORTANT: BACKTEST LIMITATIONS")
        self.Debug("The model was trained on full microstructure features including:")
        self.Debug("- Funding rates (not available in QuantConnect)")
        self.Debug("- Basis rates (not available in QuantConnect)")
        self.Debug("- Long/Short ratios (not available in QuantConnect)")
        self.Debug("")
        self.Debug(f"Only {len(self.available_features)}/{len(self.model_features)} features were available here;")
        self.Debug("Missing features were set to 0.")
        self.Debug("This backtest will NOT match the performance of the full-data offline model.")
        self.Debug("=" * 60)
