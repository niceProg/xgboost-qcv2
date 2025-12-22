from AlgorithmImports import *
import numpy as np
import pandas as pd
from collections import deque
import os
import re
import joblib
import base64
import hashlib
from datetime import datetime, timedelta

# Ensure Binance market is available
try:
    from QuantConnect.Market import Market
    Market.Binance  # Test if available
except:
    class Market:
        Binance = "Binance"


class XGBoostTradingAlgorithm(QCAlgorithm):
    """
    Paper/Live-ready version with customized signals + reminders + server model auto-update.

    Adds/Improves:
    - Signal Entry/Exit messages (TP/SL/RR) based on fills (OnOrderEvent)
    - Reminder signals (1-3 hours) for pre-entry/pre-exit setups
    - Telegram notify signature fixed (Notify.Telegram(chat_id, message, token))
    - Model auto-update from YOUR SERVER URLs using version + base64(joblib) (no blocking)
      * version_url: text (ver=...|trained_until=YYYY-mm-dd HH:MM:SS) or just "ver"
      * pointer_url (optional): text containing the model_b64_url for that version
      * model_b64_url: base64 text of .joblib bytes
      * sha256_url (optional): text sha256 for integrity
    - Mindset: trade at bar time using the latest READY model (typically trained_until <= bar_time - 1h)
    """

    # =========================================================
    # INITIALIZE
    # =========================================================
    def Initialize(self):
        # ===== ObjectStore keys (fallback) =====
        self.model_key = "latest_model.joblib"
        self.dataset_summary_key = "dataset_summary.txt"

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
        self.SetCash(100)

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

        # ===== Strategy meta =====
        self.strategy_name = "Metode ABC"
        self.startup_notified = False

        # ===== Model state =====
        self.model = None
        self.model_n_features = None
        self.expected_feature_order = None

        # ===== Server model auto-update (URL-based) =====
        # Put your server URLs here:
        # Example:
        #   https://domain.com/dir/latest_model.version        (text)
        #   https://domain.com/dir/latest_model.pointer        (text, optional)
        #   https://domain.com/dir/latest_model.joblib.b64     (text base64)
        #
        # Notes:
        # - version format supported:
        #     "ver=2025-12-15_13-07|trained_until=2025-12-15 12:00:00"
        #   or just:
        #     "2025-12-15_13-07"
        self.enable_server_model_update = True
        self.server_version_url = "https://ad.sygify.com/latest_model.version"
        self.server_pointer_url = ""     # optional
        self.server_model_b64_url = "https://ad.sygify.com/latest_model.joblib.b64"
        self.server_model_sha256_url = ""  # optional

        # How often to check model version (minutes). Keep >= 5 to avoid spam.
        self.model_check_interval_minutes = 10
        self.last_model_check_time = None
        self.current_model_version = None
        self.current_model_trained_until = None

        # Use "t-1" policy: for a bar at time T, accept models with trained_until <= T - 1 hour.
        # If trained_until is not present in version file, we accept immediately when version changes.
        self.require_trained_until_for_gate = False

        # Backtest safety: downloading has limits; default off in backtest
        self.enable_server_model_update_in_backtest = False

        # Local cache key for downloaded model bytes
        self.cached_model_objectstore_key = "cached_latest_model.joblib"

        # ===== Early defaults (avoid missing-attr during initial model load) =====
        # These may be overwritten later in Initialize()
        self.enable_live_notifications = True
        self.notify_channel = "telegram"  # telegram|webhook|email|sms|debug
        self.telegram_chat_id = ""
        self.telegram_token = ""
        self.webhook_url = ""
        self.email_to = ""
        self.sms_number = ""

        # Load initial model (server first, then ObjectStore fallback)
        loaded = self.TryInitialModelLoad()
        if not loaded:
            self.Debug("[MODEL] Initial load failed. Strategy will run with model=None until a model is available.")

        # Schedule periodic server model checks (live/paper only by default)
        self.Schedule.On(
            self.DateRules.EveryDay(self.symbol),
            self.TimeRules.Every(timedelta(minutes=self.model_check_interval_minutes)),
            self.ScheduledModelCheck
        )

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

        # ===== Notifications =====
        # For safety, consider moving secrets to project parameters.
        self.enable_live_notifications = True
        self.notify_channel = "Telegram"  # "Debug" | "Telegram" | "Webhook" | "Email" | "Sms"
        self.telegram_token = "8306719491:AAHNS7HT-pjMUGlcXMA_5SEffd6zPd2X6U0"
        self.telegram_chat_id = "-4978819951"
        self.webhook_url = ""
        self.email_to = ""
        self.sms_number = ""

        # ===== Reminder Signals (pre-entry/exit) =====
        # With Resolution.Hour data, reminders are naturally in 1h steps.
        self.enable_reminders = True
        self.reminder_min_hours = 1
        self.reminder_max_hours = 3

        # Anti-spam gap per reminder type
        self.reminder_min_gap = timedelta(hours=2)
        self.last_reminder_sent_time = {}  # type -> datetime

        # Active reminders keyed by name (REM_ENTRY / REM_EXIT_TP / ...)
        self.reminders = {}  # key -> dict(due, hours, tipe, reason)

        # Threshold bands for arming reminders
        self.entry_pred_band = 0.03   # arm when pred >= buy_th - band (but < buy_th)
        self.exit_pred_band = 0.03    # arm when pred <= sell_th + band (but > sell_th)

        # TP/SL progress levels for arming reminders
        self.near_level_1h = 0.90
        self.near_level_2h = 0.80
        self.near_level_3h = 0.70

        # Warmup
        self.SetWarmUp(120, Resolution.Hour)

        # Last seen price for logging
        self.last_price = None
        # OnData logging helpers
        self._reminder_fired_info = None  # (tipe, hours, reason)
        self._last_ondata_status = None
        self._last_ondata_reason = None

        self.pred_debug_counter = 0
        self.Debug("XGBoostTradingAlgorithm (server model update) initialized")

    # =========================================================
    # STARTUP NOTIFY
    # =========================================================
    def OnWarmupFinished(self):
        # Called once after WarmUp finishes
        if not self.LiveMode:
            return

        self.SendStartupMessage()

    def SendStartupMessage(self):
        if self.startup_notified:
            return
        if not self.LiveMode:
            return

        msg = (
            f"{self.strategy_name} is start running\n"
            f"symbol={self.symbol}\n"
            f"resolution=Hour\n"
            f"time={self.Time}"
        )
        self.SendSignal("START", msg)
        self.startup_notified = True

    # =========================================================
    # NOTIFICATION UTILITIES
    # =========================================================
    def SendSignal(self, title: str, message: str):
        """
        Custom signal dispatcher.
        - Always Debug() for visibility.
        - Optionally Notify.* in live/paper if enabled.
        """
        safe_title = str(title)[:120]
        safe_message = str(message)

        self.Debug(f"[SIGNAL] {safe_title}\n{safe_message}")

        if not self.LiveMode:
            return
        if not getattr(self, "enable_live_notifications", False):
            return

        try:
            ch = str(getattr(self, "notify_channel", "debug") or "debug").lower()

            if ch == "telegram":
                chat_id = str(getattr(self, "telegram_chat_id", "")).strip()
                token = str(getattr(self, "telegram_token", "")).strip()
                if chat_id and token:
                    # Correct signature: Telegram(chat_id, message, token)
                    self.Notify.Telegram(chat_id, f"{safe_title}\n{safe_message}", token)
                else:
                    self.Debug("[SIGNAL] Telegram not configured (missing chat_id/token)")
                    self.Notify.Web(self.webhook_url, f"{safe_title}\n{safe_message}")
            elif ch == "email":
                if self.email_to:
                    self.Notify.Email(self.email_to, safe_title, safe_message)
            elif ch == "sms":
                if self.sms_number:
                    self.Notify.Sms(self.sms_number, f"{safe_title} {safe_message}")
        except Exception as e:
            self.Debug(f"[SIGNAL] Notify error: {e}")

    # =========================================================
    # SERVER MODEL UPDATE (URL)
    # =========================================================
    def ScheduledModelCheck(self):
        # periodic check; do not run in backtest unless explicitly enabled
        self.MaybeRefreshModelFromServer(force=True)

    def _download_text(self, url: str):
        if not url:
            return None
        try:
            s = self.Download(url)
            if s is None:
                return None
            return str(s).strip()
        except Exception as e:
            self.Debug(f"[MODEL] Download failed url={url} err={e}")
            return None

    def _parse_version_text(self, text: str):
        """
        Supports:
          - "ver=...|trained_until=YYYY-mm-dd HH:MM:SS"
          - "trained_until=...|ver=..."
          - just "some_version_string"
        Returns (ver, trained_until_dt or None)
        """
        if text is None:
            return None, None
        t = text.strip()
        if not t:
            return None, None

        ver = None
        trained_until = None

        if "|" in t or "ver=" in t or "trained_until=" in t:
            parts = [p.strip() for p in t.split("|") if p.strip()]
            for p in parts:
                if p.lower().startswith("ver="):
                    ver = p.split("=", 1)[1].strip()
                elif p.lower().startswith("trained_until="):
                    raw = p.split("=", 1)[1].strip()
                    try:
                        trained_until = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        trained_until = None
        else:
            ver = t

        return ver, trained_until

    def _is_model_eligible_for_bar(self, trained_until: datetime, bar_time: datetime) -> bool:
        """
        "t-1" policy: for bar at time T, accept trained_until <= T - 1 hour.
        If trained_until is None:
          - if require_trained_until_for_gate: reject
          - else accept (version change triggers reload).
        """
        if trained_until is None:
            return not self.require_trained_until_for_gate

        try:
            return trained_until <= (bar_time - timedelta(hours=1))
        except Exception:
            return False

    def _load_model_from_bytes(self, raw_bytes: bytes):
        """
        Save bytes to ObjectStore then joblib.load from file path.
        """
        self.ObjectStore.SaveBytes(self.cached_model_objectstore_key, raw_bytes)
        path = self.ObjectStore.GetFilePath(self.cached_model_objectstore_key)
        return self._load_model_from_file_path(path)

    def _load_model_from_file_path(self, file_path: str):
        """
        Load joblib model and infer feature metadata.
        """
        self.model = joblib.load(file_path)
        self.Debug("[MODEL] Loaded XGBoost model")

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
            self.Debug(f"[MODEL] Could not infer feature metadata: {inner}")

        if self.expected_feature_order is None:
            self.expected_feature_order = list(self.model_features)

        if self.model_n_features is None:
            self.model_n_features = len(self.expected_feature_order)

        self.Debug(f"[MODEL] Expects {self.model_n_features} features; order_len={len(self.expected_feature_order)}")
        return True

    def TryInitialModelLoad(self) -> bool:
        # Try server (if enabled), then fallback to ObjectStore
        ok = False
        if self.enable_server_model_update and self.server_version_url and self.server_model_b64_url:
            ok = self.MaybeRefreshModelFromServer(force=True, allow_gate_bypass=True)
        if ok:
            return True

        # Fallback: ObjectStore (original behavior)
        try:
            if self.ObjectStore.ContainsKey(self.model_key):
                self.LoadModelFromObjectStore()
                return self.model is not None
        except Exception as e:
            self.Debug(f"[MODEL] ObjectStore initial load failed: {e}")
        return False

    def MaybeRefreshModelFromServer(self, force=False, allow_gate_bypass=False):
        """
        Check server version, and if changed, download model_b64, verify (optional), and reload.
        - allow_gate_bypass=True: used only at init so we can start with whatever version is available.
        """
        # Backtest safety
        if not self.LiveMode and not self.enable_server_model_update_in_backtest:
            return False
        if not self.enable_server_model_update:
            return False

        # throttle
        if not force and self.last_model_check_time is not None:
            if self.Time < self.last_model_check_time + timedelta(minutes=int(self.model_check_interval_minutes)):
                return False
        self.last_model_check_time = self.Time

        vtext = self._download_text(self.server_version_url)
        ver, trained_until = self._parse_version_text(vtext)

        if not ver:
            return False

        if self.current_model_version == ver and self.model is not None:
            return False

        # gate policy (t-1) unless bypass requested
        if not allow_gate_bypass and not self._is_model_eligible_for_bar(trained_until, self.Time):
            self.Debug(f"[MODEL] Version seen but not eligible yet ver={ver} trained_until={trained_until} bar={self.Time}")
            return False

        # pointer may provide model URL
        model_url = self.server_model_b64_url
        if self.server_pointer_url:
            ptext = self._download_text(self.server_pointer_url)
            if ptext:
                model_url = ptext.strip()

        b64 = self._download_text(model_url)
        if not b64:
            self.Debug(f"[MODEL] Missing/empty model_b64 from {model_url}")
            return False

        # decode base64
        try:
            raw_bytes = base64.b64decode(b64, validate=False)
        except Exception as e:
            self.Error(f"[MODEL] Base64 decode failed: {e}")
            return False

        # optional sha256 verify
        if self.server_model_sha256_url:
            expected = self._download_text(self.server_model_sha256_url)
            if expected:
                expected = expected.strip().lower()
                actual = hashlib.sha256(raw_bytes).hexdigest().lower()
                if expected != actual:
                    self.Error(f"[MODEL] sha256 mismatch expected={expected} actual={actual}")
                    return False

        # load and swap model
        try:
            self._load_model_from_bytes(raw_bytes)
            self.current_model_version = ver
            self.current_model_trained_until = trained_until

            self.Debug(f"[MODEL] Reloaded from server ver={ver} trained_until={trained_until}")
            if self.LiveMode:
                self.SendSignal("MODEL UPDATED", f"ver={ver}\ntrained_until={trained_until}")
            return True
        except Exception as e:
            self.Error(f"[MODEL] Load failed ver={ver}: {e}")
            if self.LiveMode:
                self.SendSignal("MODEL UPDATE FAILED", f"ver={ver}\nerr={e}")
            return False

    # =========================================================
    # LOAD DATASET SUMMARY
    # =========================================================
    def LoadDatasetSummaryFromObjectStore(self):
        try:
            if not self.ObjectStore.ContainsKey(self.dataset_summary_key):
                self.Debug(f"dataset_summary not found: {self.dataset_summary_key}")
                return

            file_path = self.ObjectStore.GetFilePath(self.dataset_summary_key)
            with open(file_path, "r") as f:
                text = f.read()

            m = re.search(r"Time range:\s*(.+?)\s*to\s*(.+)", text)
            if not m:
                self.Error("Could not parse 'Time range: ... to ...' in dataset_summary.txt")
                return

            start_raw = m.group(1).strip()
            end_raw = m.group(2).strip()
            dt_format = "%Y-%m-%d %H:%M:%S"
            start_dt = datetime.strptime(start_raw, dt_format)
            end_dt = datetime.strptime(end_raw, dt_format)

            self.train_start_date = datetime(start_dt.year, start_dt.month, start_dt.day)
            self.train_end_date = datetime(end_dt.year, end_dt.month, end_dt.day)
        except Exception as e:
            self.Error(f"Error parsing dataset_summary.txt: {e}")
            self.train_start_date = None
            self.train_end_date = None

    # =========================================================
    # LOAD MODEL (OBJECTSTORE FALLBACK)
    # =========================================================
    def LoadModelFromObjectStore(self):
        try:
            if not self.ObjectStore.ContainsKey(self.model_key):
                self.Error(f"ObjectStore key not found: {self.model_key}")
                return

            file_path = self.ObjectStore.GetFilePath(self.model_key)
            self._load_model_from_file_path(file_path)
        except Exception as e:
            self.Error(f"Error loading model from ObjectStore: {e}")
            self.model = None
            self.model_n_features = None
            self.expected_feature_order = None

    # =========================================================
    # ORDER EVENTS (Signals should be based on fills)
    # =========================================================
    def OnOrderEvent(self, orderEvent: OrderEvent):
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

                self.CancelReminder("REM_ENTRY")
                msg = self.FormatEntrySignal(self.entry_price)
                self.SendSignal("Signal Entry (Filled)", msg)

            # SELL filled (exit)
            elif fill_qty < 0:
                reason = self.last_exit_reason or "EXIT"
                if self.entry_price is not None:
                    msg = self.FormatExitSignal(self.entry_price, fill_price, reason)
                else:
                    msg = self.FormatExitSignal(0.0, fill_price, reason)

                self.SendSignal("Signal Exit (Filled)", msg)

                self.CancelAllReminders()

                self.entry_price = None
                self.entry_time = None
                self.pending_exit = False
                self.pending_entry = False
                self.last_exit_reason = None

        except Exception as e:
            self.Debug(f"OnOrderEvent error: {e}")

    # =========================================================
    # ONDATA
    # =========================================================
    def OnData(self, data: Slice):
            if self.IsWarmingUp:
                return

            # Reset per-bar reminder info for logging
            self._reminder_fired_info = None

            # Default price for logging if bar missing
            cur_price = self.last_price

            if not data.ContainsKey(self.symbol):
                # Rare in live; still log as HOLD
                self.LogOnDataLine(self.Time, cur_price, "HOLD", "No bar data for symbol")
                return

            # Prevent re-processing the same bar
            if self.last_trade_bar_time == self.Time:
                return

            # Try to refresh model (non-blocking, throttled)
            self.MaybeRefreshModelFromServer(force=False)

            bar = data[self.symbol]

            # Extract OHLCV
            if isinstance(bar, TradeBar):
                open_ = float(bar.Open)
                high = float(bar.High)
                low = float(bar.Low)
                close = float(bar.Close)
                volume = float(bar.Volume) if bar.Volume is not None else 0.0
            elif isinstance(bar, QuoteBar):
                src = bar.Bid if bar.Bid is not None else bar.Ask
                if src is None:
                    self.LogOnDataLine(self.Time, cur_price, "HOLD", "QuoteBar missing bid/ask")
                    self.last_trade_bar_time = self.Time
                    return
                open_ = float(src.Open)
                high = float(src.High)
                low = float(src.Low)
                close = float(src.Close)
                volume = 0.0
            else:
                self.LogOnDataLine(self.Time, cur_price, "HOLD", f"Unsupported bar type: {type(bar)}")
                self.last_trade_bar_time = self.Time
                return

            # fallback volume if missing
            if volume <= 0:
                sec = self.Securities[self.symbol]
                if getattr(sec, "Volume", 0) and sec.Volume > 0:
                    volume = float(sec.Volume)
                else:
                    volume = 1_000_000.0

            # update last price
            self.last_price = close
            cur_price = close

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
                self.LogOnDataLine(self.Time, cur_price, "HOLD", f"Warm window ({len(self.price_window)}/30) - waiting more bars")
                self.last_trade_bar_time = self.Time
                return

            # Cooldown pause
            if self.trading_paused_until is not None and self.Time < self.trading_paused_until:
                self.LogOnDataLine(self.Time, cur_price, "HOLD", f"COOLDOWN until {self.trading_paused_until}")
                self.last_trade_bar_time = self.Time
                return

            # If we have pending orders, do not submit more orders
            if self.pending_entry or self.pending_exit:
                self.LogOnDataLine(self.Time, cur_price, "HOLD", "Pending order fill - waiting")
                self.last_trade_bar_time = self.Time
                return

            # Drawdown check
            if not self.CheckMaxDrawdownAndCooldown():
                # CheckMaxDrawdownAndCooldown already sets pause window and liquidates
                ru = self.trading_paused_until
                self.LogOnDataLine(self.Time, cur_price, "HOLD", f"RISK cooldown triggered; paused until {ru}")
                self.last_trade_bar_time = self.Time
                return

            # Build features + predict
            features = self.BuildFeatures()
            if features is None:
                self.LogOnDataLine(self.Time, cur_price, "HOLD", "Features unavailable (insufficient history or calc error)")
                self.last_trade_bar_time = self.Time
                return

            pred = self.Predict(features)
            if pred is None:
                self.LogOnDataLine(self.Time, cur_price, "HOLD", "Prediction failed (model missing or error)")
                self.last_trade_bar_time = self.Time
                return

            # Arm & process short reminders (1-3h) BEFORE any actions
            self.MaybeArmReminders(pred, cur_price)
            self.ProcessReminders(pred, cur_price)

            # SL/TP first (exit has priority)
            if self.CheckStopLossTakeProfit(cur_price):
                rsn = self.last_exit_reason or "EXIT"
                self.LogOnDataLine(self.Time, cur_price, f"EXIT ({rsn})", f"Exit triggered by {rsn}", pred=pred)
                self.last_trade_bar_time = self.Time
                return

            # Decision based on prediction
            pre_entry = self.pending_entry
            pre_exit = self.pending_exit
            qty = self.Portfolio[self.symbol].Quantity

            self.TradeLogic(pred)

            # Decide status for logging
            status = "HOLD"
            reason = ""

            if (not pre_entry) and self.pending_entry:
                status = "ENTRY"
                reason = f"pred={pred:.3f} > buy_th={self.prediction_buy_threshold:.2f} -> submit BUY"
            elif (not pre_exit) and self.pending_exit:
                rsn = self.last_exit_reason or "PRED"
                status = f"EXIT ({rsn})"
                reason = f"pred={pred:.3f} < sell_th={self.prediction_sell_threshold:.2f} -> submit EXIT ({rsn})"
            elif self._reminder_fired_info is not None:
                tipe, hours, rsn = self._reminder_fired_info
                status = "REMINDER"
                reason = f"{tipe} | {self._reminder_label(hours)} | {rsn}"
            else:
                # HOLD reason by prediction zone
                if qty > 0 and pred >= self.prediction_sell_threshold:
                    reason = f"Holding position; pred={pred:.3f} not below sell_th={self.prediction_sell_threshold:.2f}"
                elif qty <= 0 and pred <= self.prediction_buy_threshold:
                    reason = f"No position; pred={pred:.3f} not above buy_th={self.prediction_buy_threshold:.2f}"
                else:
                    reason = f"Neutral; pred={pred:.3f}"

            # add model version info
            mv = getattr(self, "current_model_version", None)
            if mv:
                reason = f"{reason} | model_ver={mv}"

            self.LogOnDataLine(self.Time, cur_price, status, reason, pred=pred)
            self.last_trade_bar_time = self.Time
    # =========================================================
    # ONDATA LOGGING
    # =========================================================
    def LogOnDataLine(self, time_dt, price, status: str, reason: str, pred: float = None):
        try:
            t = str(time_dt)
            p = "NA" if price is None else f"{float(price):.2f}"
            s = str(status)
            r = str(reason)
            if pred is not None:
                self.Debug(f"[ONDATA] {t} | price={p} | status={s} | pred={pred:.3f} | reason={r}")
            else:
                self.Debug(f"[ONDATA] {t} | price={p} | status={s} | reason={r}")
        except Exception:
            pass

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
            n_expected = int(self.model_n_features) if self.model_n_features is not None else len(order)

            for name in order:
                v = float(feat_map.get(name, 0.0))
                if not np.isfinite(v):
                    v = 0.0
                vec.append(v)

            if len(vec) > n_expected:
                vec = vec[:n_expected]
            elif len(vec) < n_expected:
                vec.extend([0.0] * (n_expected - len(vec)))

            return np.asarray(vec, dtype=float).reshape(1, -1)
        except Exception as e:
            self.Error(f"BuildFeatures error: {e}")
            return None

    # =========================================================
    # PREDICT
    # =========================================================
    def Predict(self, feature_array: np.ndarray):
        try:
            if self.model is None:
                return None
            proba = self.model.predict_proba(feature_array)[0, 1]
            return float(proba)
        except Exception as e:
            self.Error(f"Predict error: {e}")
            return None

    # =========================================================
    # TRADING
    # =========================================================
    def TradeLogic(self, pred: float):
        qty = self.Portfolio[self.symbol].Quantity

        # ENTRY
        if pred > self.prediction_buy_threshold and qty <= 0 and not self.pending_entry and not self.pending_exit:
            self.pending_entry = True
            self.SetHoldings(self.symbol, self.position_size_pct)
            self.entry_price = float(self.Securities[self.symbol].Price)  # provisional
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
    def CheckStopLossTakeProfit(self, price: float) -> bool:
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
    def CheckMaxDrawdownAndCooldown(self) -> bool:
        pv = float(self.Portfolio.TotalPortfolioValue)
        if pv > self.high_watermark:
            self.high_watermark = pv

        dd = (self.high_watermark - pv) / self.high_watermark if self.high_watermark > 0 else 0.0

        if dd > self.max_drawdown_pct:
            if self.Portfolio[self.symbol].Invested:
                self.pending_exit = True
                self.last_exit_reason = "RISK"
                self.Liquidate()
                self.Debug(f"[RISK] Liquidate due to DD={dd:.2%}")

            self.trading_paused_until = self.Time + self.cooldown_period
            self.Debug(f"[RISK] DD={dd:.2%} > {self.max_drawdown_pct:.2%}. Pause until {self.trading_paused_until}")

            self.high_watermark = pv
            return False

        return True

    # =========================================================
    # SIGNAL FORMATTING
    # =========================================================
    def _fmt_price(self, x: float) -> str:
        try:
            return f"{float(x):,.2f}"
        except Exception:
            return str(x)

    def _fmt_pct(self, x: float) -> str:
        try:
            return f"{float(x)*100:.2f}%"
        except Exception:
            return str(x)

    def CalcEntryTargets(self, entry_price: float):
        entry = float(entry_price)
        tp_pct = float(self.take_profit_pct)
        sl_pct = float(self.stop_loss_pct)

        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 - sl_pct)

        tp_amt = tp_price - entry
        sl_amt = entry - sl_price  # positive

        rr = (tp_amt / sl_amt) if sl_amt > 0 else 0.0
        return {
            "entry": entry,
            "tp_price": tp_price,
            "tp_amt": tp_amt,
            "tp_pct": tp_pct,
            "sl_price": sl_price,
            "sl_amt": sl_amt,
            "sl_pct": sl_pct,
            "rr": rr
        }

    def FormatEntrySignal(self, entry_price: float):
        t = self.CalcEntryTargets(entry_price)
        lines = [
            "Jenis Signal: Signal Entry",
            f"Symbol: {self.symbol}",
            f"Time: {self.Time}",
            f"Price: {self._fmt_price(t['entry'])}",
            "",
            "TP:",
            f"- Price TP: {self._fmt_price(t['tp_price'])}",
            f"- Besaran TP: {self._fmt_price(t['tp_amt'])}",
            f"- Persentase TP: {self._fmt_pct(t['tp_pct'])}",
            "",
            "SL:",
            f"- Price SL: {self._fmt_price(t['sl_price'])}",
            f"- Besaran SL: {self._fmt_price(t['sl_amt'])}",
            f"- Persentase SL: {self._fmt_pct(t['sl_pct'])}",
            "",
            f"RR (Reward/Risk): {t['rr']:.2f}",
        ]
        return "\n".join(lines)

    def FormatExitSignal(self, entry_price: float, exit_price: float, reason: str):
        entry = float(entry_price) if entry_price is not None else None
        exitp = float(exit_price)

        lines = [
            "Jenis Signal: Signal Exit",
            f"Symbol: {self.symbol}",
            f"Time: {self.Time}",
        ]

        if entry is not None:
            pnl = exitp - entry
            pnl_pct = pnl / entry if entry != 0 else 0.0

            lines += [
                f"Price Entry: {self._fmt_price(entry)}",
                f"Price Exit: {self._fmt_price(exitp)}",
            ]

            r = (reason or "").upper()
            if r == "TP":
                lines += [
                    "",
                    "TP (Triggered):",
                    f"- Besaran TP: {self._fmt_price(max(pnl, 0.0))}",
                    f"- Persentase TP: {self._fmt_pct(max(pnl_pct, 0.0))}",
                ]
            elif r == "SL":
                lines += [
                    "",
                    "SL (Triggered):",
                    f"- Besaran SL: {self._fmt_price(max(-pnl, 0.0))}",
                    f"- Persentase SL: {self._fmt_pct(max(-pnl_pct, 0.0))}",
                ]

            lines += [
                "",
                f"Exit Reason: {reason}",
                f"PnL: {self._fmt_price(pnl)} ({self._fmt_pct(pnl_pct)})"
            ]
        else:
            lines += [
                f"Price Exit: {self._fmt_price(exitp)}",
                f"Exit Reason: {reason}"
            ]

        return "\n".join(lines)

    # =========================================================
    # REMINDER SIGNALS (1-3 hours ahead)
    # =========================================================
    def _reminder_label(self, hours: int) -> str:
        h = int(hours)
        if h <= 1:
            return "1 jam lagi"
        return f"{h} jam lagi"

    def ArmReminder(self, key: str, tipe: str, hours: int, reason: str):
        if not self.enable_reminders:
            return

        try:
            h = int(max(self.reminder_min_hours, min(self.reminder_max_hours, int(hours))))
        except Exception:
            h = self.reminder_min_hours

        due = self.Time + timedelta(hours=h)

        existing = self.reminders.get(key)
        if existing is None:
            self.reminders[key] = {"due": due, "hours": h, "tipe": tipe, "reason": reason}
        else:
            if due < existing.get("due", due):
                existing["due"] = due
                existing["hours"] = h
                existing["tipe"] = tipe
                existing["reason"] = reason

    def CancelReminder(self, key: str):
        if key in self.reminders:
            self.reminders.pop(key, None)

    def CancelAllReminders(self):
        self.reminders = {}

    def _can_send_reminder(self, tipe: str) -> bool:
        last = self.last_reminder_sent_time.get(tipe)
        if last is None:
            return True
        return (self.Time - last) >= self.reminder_min_gap

    def _mark_reminder_sent(self, tipe: str):
        self.last_reminder_sent_time[tipe] = self.Time

    def FormatReminderSignal(self, tipe: str, hours: int, reason: str) -> str:
        lines = [
            "Signal Reminder:",
            f"1. Tipe: {tipe}",
            f"2. Waktu: {self._reminder_label(hours)}",
            f"3. Alasan: {reason}",
            f"Time: {self.Time}",
            f"Symbol: {self.symbol}",
        ]
        return "\n".join(lines)

    def _lead_hours_by_gap(self, gap: float) -> int:
        try:
            g = float(gap)
        except Exception:
            return self.reminder_max_hours

        if g <= 0.005:
            return 1
        if g <= 0.015:
            return 2
        return 3

    def _lead_hours_by_progress(self, progress: float) -> int:
        try:
            p = float(progress)
        except Exception:
            return self.reminder_max_hours

        if p >= self.near_level_1h:
            return 1
        if p >= self.near_level_2h:
            return 2
        return 3

    def MaybeArmReminders(self, pred: float, price: float):
        if not self.enable_reminders:
            return

        qty = self.Portfolio[self.symbol].Quantity
        buy_th = float(self.prediction_buy_threshold)
        sell_th = float(self.prediction_sell_threshold)

        # Pre-ENTRY reminder (no position)
        if qty <= 0 and (buy_th - self.entry_pred_band) <= pred < buy_th:
            gap = buy_th - pred
            hours = self._lead_hours_by_gap(gap)
            reason = f"Pred mendekati Entry (pred={pred:.3f} < buy_th={buy_th:.2f})"
            self.ArmReminder("REM_ENTRY", "Entry", hours, reason)

        # Pre-EXIT reminder by prediction (has position)
        if qty > 0 and sell_th < pred <= (sell_th + self.exit_pred_band):
            gap = pred - sell_th
            hours = self._lead_hours_by_gap(gap)
            reason = f"Pred mendekati Exit (pred={pred:.3f} > sell_th={sell_th:.2f})"
            self.ArmReminder("REM_EXIT_PRED", "Exit (Pred)", hours, reason)

        # Pre-EXIT reminder by TP/SL proximity (has position)
        if qty > 0 and self.entry_price is not None and self.entry_price > 0:
            entry = float(self.entry_price)
            tp_price = entry * (1.0 + float(self.take_profit_pct))
            sl_price = entry * (1.0 - float(self.stop_loss_pct))

            tp_den = max(tp_price - entry, 1e-9)
            tp_prog = (price - entry) / tp_den
            if self.near_level_3h <= tp_prog < 1.0:
                hours = self._lead_hours_by_progress(tp_prog)
                reason = f"Harga mendekati TP (now≈{price:.2f}, TP={tp_price:.2f}, progress={tp_prog:.0%})"
                self.ArmReminder("REM_EXIT_TP", "Exit (TP)", hours, reason)
            else:
                self.CancelReminder("REM_EXIT_TP")

            sl_den = max(entry - sl_price, 1e-9)
            sl_prog = (entry - price) / sl_den
            if self.near_level_3h <= sl_prog < 1.0:
                hours = self._lead_hours_by_progress(sl_prog)
                reason = f"Harga mendekati SL (now≈{price:.2f}, SL={sl_price:.2f}, progress={sl_prog:.0%})"
                self.ArmReminder("REM_EXIT_SL", "Exit (SL)", hours, reason)
            else:
                self.CancelReminder("REM_EXIT_SL")

        if qty <= 0:
            self.CancelReminder("REM_EXIT_PRED")
            self.CancelReminder("REM_EXIT_TP")
            self.CancelReminder("REM_EXIT_SL")

    def ProcessReminders(self, pred: float, price: float):
        if not self.enable_reminders or not self.reminders:
            return

        if self.pending_entry or self.pending_exit:
            return
        if self.trading_paused_until is not None and self.Time < self.trading_paused_until:
            return

        qty = self.Portfolio[self.symbol].Quantity
        buy_th = float(self.prediction_buy_threshold)
        sell_th = float(self.prediction_sell_threshold)

        to_remove = []
        for key, r in list(self.reminders.items()):
            due = r.get("due")
            if due is None or self.Time < due:
                continue

            tipe = r.get("tipe", "Entry")
            hours = int(r.get("hours", self.reminder_min_hours))
            reason = r.get("reason", "")

            valid = True
            if key == "REM_ENTRY":
                valid = (qty <= 0) and ((buy_th - self.entry_pred_band) <= pred < buy_th)
            elif key == "REM_EXIT_PRED":
                valid = (qty > 0) and (sell_th < pred <= (sell_th + self.exit_pred_band))
            elif key == "REM_EXIT_TP":
                if qty > 0 and self.entry_price:
                    entry = float(self.entry_price)
                    tp_price = entry * (1.0 + float(self.take_profit_pct))
                    tp_den = max(tp_price - entry, 1e-9)
                    tp_prog = (price - entry) / tp_den
                    valid = self.near_level_3h <= tp_prog < 1.0
                else:
                    valid = False
            elif key == "REM_EXIT_SL":
                if qty > 0 and self.entry_price:
                    entry = float(self.entry_price)
                    sl_price = entry * (1.0 - float(self.stop_loss_pct))
                    sl_den = max(entry - sl_price, 1e-9)
                    sl_prog = (entry - price) / sl_den
                    valid = self.near_level_3h <= sl_prog < 1.0
                else:
                    valid = False

            if not valid:
                to_remove.append(key)
                continue

            if not self._can_send_reminder(tipe):
                to_remove.append(key)
                continue

            msg = self.FormatReminderSignal(tipe, hours, reason)
            # mark reminder fired for OnData logging (capture first only)
            if self._reminder_fired_info is None:
                self._reminder_fired_info = (tipe, hours, reason)
            self.SendSignal("Signal Reminder", msg)
            self._mark_reminder_sent(tipe)
            to_remove.append(key)

        for k in to_remove:
            self.CancelReminder(k)

    def OnEndOfAlgorithm(self):
        self.Debug(f"Final Portfolio Value: {self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug("NOTE: Missing non-price features are set to 0; results won't match offline full-feature backtests.")