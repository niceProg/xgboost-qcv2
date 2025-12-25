from AlgorithmImports import *
import numpy as np
import pandas as pd
from collections import deque
import os
import re
import joblib
import base64
from io import BytesIO
import hashlib
import json
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
        # ===== Training date range from dataset_summary (SERVER) =====
        # Legacy ObjectStore keys (unused - ObjectStore model/dataset loading removed)
        self.model_key = "latest_model.joblib"
        self.dataset_summary_key = "dataset_summary.txt"

        # ===== Server API config (MODEL + DATASET SUMMARY) =====
        # domain: https://namadomain.com
        # model_version: v1  -> /api/v1/...
        self.domain = "https://api.dragonfortune.ai"
        self.model_version = "v1"
        # keep compatibility for existing API calls in this script
        self.apidomain = self.domain

        # default headers (boleh ditambah Authorization dsb)
        self.api_headers = {
            "Content-Type": "application/json"
            # "Authorization": "Bearer xxx"
        }

        # Endpoints (JSON, with base64 fields)
        self.model_api_url = f"{self.domain}/api/{self.model_version}/latest/model"
        self.dataset_summary_api_url = f"{self.domain}/api/{self.model_version}/latest/dataset-summary"

        self.train_start_date = None
        self.train_end_date = None
        self.LoadDatasetSummaryFromServer()

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
        self.strategy_name = "Metode: Spot v4 - DragonFortune"
        self.strategy_id = 4
        self.startup_notified = False

        # ===== Model state =====
        self.model = None
        self.model_n_features = None
        self.expected_feature_order = None
        # ===== Server model update config =====
        # NOTE: domain/model_version + endpoints already set at the top of Initialize()
        self.enable_server_model_update = True
        self.enable_server_model_update_in_backtest = True

        # How often to check model for non-OnData triggers (unused by default)
        self.model_check_interval_minutes = 10
        self.last_model_check_time = None

        # Current server model metadata
        self.current_model_version = None
        self.current_model_created_at = None
        self.current_model_name = None

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
        # Model refresh in LIVE will be handled in OnData() each timeframe (per instructions)
        # so we don't schedule extra checks here.
        # ===== Rolling window =====
        self.window_size = 60
        self.price_window = deque(maxlen=self.window_size)

        # ===== Trading parameters =====
        self.prediction_buy_threshold = 0.63
        self.prediction_sell_threshold = 0.37
        self.position_size_pct = 0.80

        # ===== Winrate tuning (patched) =====
        self.min_hold_bars = 2  # minimum holding bars before prediction-based exit
        self.pred_smooth_n = 1   # moving average window for prediction smoothing
        self.pred_history = deque(maxlen=self.pred_smooth_n)

        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.3

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

        self._last_submit_signal_bar = None
        self._last_submit_signal_kind = None  # "ENTRY" / "EXIT"

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
        
        self.sendAPI("/logs", {"id_method":self.strategy_id, "datetime": self.Time.strftime("%Y-%m-%d %H:%M:%S"), "message": msg})

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
        # periodic check
        # PATCH: if backtest server updates are enabled, only load once at init (no periodic refresh)
        if (not self.LiveMode) and self.enable_server_model_update_in_backtest:
            return
        self.MaybeRefreshModelFromServer(force=True)

    def _download_text(self, url: str):
        if not url:
            return None
        try:
            s = self.Download(url)
            if s is None:
                return None

            
            self.Debug(f"[MODEL] Downloaded Succesfully, url={url}")
            return str(s).strip()
        except Exception as e:
            self.Debug(f"[MODEL] Download failed url={url} err={e}")
            return None

    def _download_json(self, url: str):
        if not url:
            return None
        try:
            s = self.Download(url, headers=self.api_headers)
            if s is None:
                return None
            txt = str(s).strip()
            if not txt:
                return None
            return json.loads(txt)
        except Exception as e:
            self.Debug(f"[HTTP] JSON download failed url={url} err={e}")
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

    def _load_model_from_bytes(self, model_bytes: bytes, api_feature_names=None) -> bool:
        """Load joblib model from in-memory bytes and infer feature metadata."""
        try:
            self.model = joblib.load(BytesIO(model_bytes))
            self.Debug("[MODEL] Loaded XGBoost model (bytes)")

            self.model_n_features = None
            self.expected_feature_order = None

            # Prefer API-provided feature names when available (non-empty)
            if api_feature_names and isinstance(api_feature_names, list) and len(api_feature_names) > 0:
                self.expected_feature_order = list(api_feature_names)
                self.model_n_features = len(self.expected_feature_order)

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
        except Exception as e:
            self.Error(f"[MODEL] Error loading model from bytes: {e}")
            self.model = None
            self.model_n_features = None
            self.expected_feature_order = None
            return False

    def TryInitialModelLoad(self) -> bool:
        # Server-only (no ObjectStore model load)
        if not self.enable_server_model_update:
            return False

        ok = self.MaybeRefreshModelFromServer(force=True, allow_gate_bypass=True)
        return bool(ok)

    def MaybeRefreshModelFromServer(self, force=False, allow_gate_bypass=False):
        """
        Load model/joblib from server JSON API (base64).
        - Backtest: allowed only when enable_server_model_update_in_backtest=True, and should be called once at init.
        - Live: can be called every OnData() timeframe; will reload only when model_version changes.
        """
        # Backtest safety
        if (not self.LiveMode) and (not self.enable_server_model_update_in_backtest):
            return False
        if not self.enable_server_model_update:
            return False

        # Throttle for non-forced checks (OnData can call with force=True)
        if not force and self.last_model_check_time is not None:
            if self.Time < self.last_model_check_time + timedelta(minutes=int(self.model_check_interval_minutes)):
                return False
        self.last_model_check_time = self.Time

        j = self._download_json(self.model_api_url)
        if not j or not j.get("success"):
            return False

        ver = str(j.get("model_version") or "").strip()
        if not ver:
            return False

        # No change
        if self.current_model_version == ver and self.model is not None:
            return False

        b64 = j.get("model_data_base64")
        if not b64:
            self.Error("[MODEL] Server response missing model_data_base64")
            return False

        try:
            model_bytes = base64.b64decode(b64)
        except Exception as e:
            self.Error(f"[MODEL] base64 decode failed: {e}")
            return False

        api_feature_names = j.get("feature_names") or []
        ok = self._load_model_from_bytes(model_bytes, api_feature_names)
        if not ok:
            return False

        self.current_model_version = ver
        self.current_model_created_at = j.get("created_at")
        self.current_model_name = j.get("model_name")

        # Keep dataset_summary in sync (fills self.train_start_date / self.train_end_date)
        try:
            self.LoadDatasetSummaryFromServer()
        except Exception as e:
            self.Debug(f"[DATASET] Summary load failed: {e}")

        self.Debug(f"[MODEL] Reloaded from server ver={ver} created_at={self.current_model_created_at}")
        if self.LiveMode:
            self.SendSignal("MODEL UPDATED", f"ver={ver}\ncreated_at={self.current_model_created_at}")
        return True

    def LoadDatasetSummaryFromServer(self):
        """Load dataset_summary from server API (base64), fill self.train_start_date/self.train_end_date."""
        try:
            j = self._download_json(self.dataset_summary_api_url)
            if not j or not j.get("success"):
                return

            b64 = j.get("summary_data_base64")
            if not b64:
                self.Debug("[DATASET] Server response missing summary_data_base64")
                return

            try:
                raw = base64.b64decode(b64).decode("utf-8", errors="ignore")
            except Exception as e:
                self.Error(f"[DATASET] base64 decode failed: {e}")
                return

            # Parse:
            # - Start: YYYY-mm-dd HH:MM:SS (...)
            # - End:   YYYY-mm-dd HH:MM:SS (...)
            m_start = re.search(r"^\s*-\s*Start:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})", raw, re.MULTILINE)
            m_end = re.search(r"^\s*-\s*End:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})", raw, re.MULTILINE)
            if not m_start or not m_end:
                self.Error("[DATASET] Could not parse Start/End in dataset_summary")
                return

            dt_format = "%Y-%m-%d %H:%M:%S"
            start_raw = m_start.group(1).strip()
            end_raw = m_end.group(1).strip()

            start_dt = datetime.strptime(start_raw, dt_format)
            end_dt = datetime.strptime(end_raw, dt_format)

            # Keep same behavior as existing script: store day-level dates
            self.train_start_date = datetime(start_dt.year, start_dt.month, start_dt.day)
            self.train_end_date = datetime(end_dt.year, end_dt.month, end_dt.day)

            sid = j.get("session_id")
            created_at = j.get("created_at")
            if sid:
                self.Debug(f"[DATASET] Loaded dataset_summary session_id={sid} created_at={created_at} start_date={start_dt} end_date={end_dt}")
        except Exception as e:
            self.Error(f"[DATASET] LoadDatasetSummaryFromServer error: {e}")
            self.train_start_date = None
            self.train_end_date = None

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

            balance = float(self.Portfolio.TotalPortfolioValue)

            # BUY filled
            if fill_qty > 0:
                self.entry_price = fill_price
                self.entry_time = self.Time
                self.pending_entry = False

                self.CancelReminder("REM_ENTRY")
                msg, t = self.FormatEntrySignal(self.entry_price)
                self.SendSignal("Signal Entry (Filled)", msg)

                self.sendAPI("/orders", {
                    "id_method":self.strategy_id, "datetime": self.Time.strftime("%Y-%m-%d %H:%M:%S"), 
                    "type": "entry",
                    "jenis": "buy",
                    "price": fill_price,
                    "quantity": fill_qty,
                    "balance": balance,
                    "message": msg
                    })

            # SELL filled (exit)
            elif fill_qty < 0:
                reason = self.last_exit_reason or "EXIT"
                if self.entry_price is not None:
                    msg, real_tp, real_sl = self.FormatExitSignal(self.entry_price, fill_price, reason)
                else:
                    msg, real_tp, real_sl = self.FormatExitSignal(0.0, fill_price, reason)

                self.SendSignal("Signal Exit (Filled)", msg)

                self.sendAPI("/orders", {
                    "id_method":self.strategy_id, "datetime": self.Time.strftime("%Y-%m-%d %H:%M:%S"), 
                    "type": "exit",
                    "jenis": "sell",
                    "price": fill_price,
                    "quantity": fill_qty,
                    "balance": balance,
                    "message": msg
                })

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
            # Live: check model from server every OnData() timeframe (reload only if model_version changed)
            if self.LiveMode:
                self.MaybeRefreshModelFromServer(force=True)

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

            # PATCH: prediction smoothing (moving average)
            pred_raw = float(pred)
            try:
                self.pred_history.append(pred_raw)
                pred = float(np.mean(self.pred_history))
            except Exception:
                pred = pred_raw

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
                msg = f"[ONDATA] {t} | price={p} | status={s} | pred={pred:.3f} | reason={r}"
                self.Debug(msg)
                self.sendAPI("/logs", {"id_method":self.strategy_id, "datetime": self.Time.strftime("%Y-%m-%d %H:%M:%S"), "message": msg})
            else:
                msg = f"[ONDATA] {t} | price={p} | status={s} | reason={r}"
                self.Debug(f"[ONDATA] {t} | price={p} | status={s} | reason={r}")
                self.sendAPI("/logs", {"id_method":self.strategy_id, "datetime": self.Time.strftime("%Y-%m-%d %H:%M:%S"), "message": msg})
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
            self.entry_price = float(self.Securities[self.symbol].Price)  # provisional
            self.entry_time = self.Time

            # kirim signal SUBMIT (hindari spam per bar)
            if self._last_submit_signal_bar != self.entry_time or self._last_submit_signal_kind != "ENTRY":
                self._last_submit_signal_bar = self.entry_time
                self._last_submit_signal_kind = "ENTRY"

                msg, t = self.FormatEntrySignal(self.entry_price)
                self.Debug(f"Signal Entry (Submitted): {msg}")

                self.sendAPI("/signals", {
                    "id_method": self.strategy_id,
                    "datetime": self.Time.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "entry",
                    "jenis": "buy",
                    "price_entry": self.entry_price,
                    "price_exit": 0,
                    "target_tp": t['tp_price'],
                    "target_sl": t['sl_price'],
                    "real_tp": 0,
                    "real_sl": 0,
                    "message": msg
                })

            self.SetHoldings(self.symbol, self.position_size_pct)
            self.Debug(f"BUY SetHoldings({self.position_size_pct:.0%}) pred={pred:.3f} est_entry={self.entry_price:.2f}")


        # EXIT by prediction
        elif pred < self.prediction_sell_threshold and qty > 0 and not self.pending_exit:
            # PATCH: minimum holding period before allowing prediction-based exit
            hold_ok = True
            try:
                mh = int(getattr(self, 'min_hold_bars', 0) or 0)
                if mh > 0 and getattr(self, 'entry_time', None) is not None:
                    if self.Time < (self.entry_time + timedelta(hours=mh)):
                        hold_ok = False
            except Exception:
                hold_ok = True

            if not hold_ok:
                return

            self.pending_exit = True
            self.last_exit_reason = "PRED"

            ent_price = self.entry_price
            exit_price = float(self.Securities[self.symbol].Price)  # provisional
            exit_time = self.Time

            if self._last_submit_signal_bar != exit_time or self._last_submit_signal_kind != "EXIT":
                self._last_submit_signal_bar = exit_time
                self._last_submit_signal_kind = "EXIT"

                msg, real_tp, real_sl = self.FormatExitSignal(self.entry_price or 0.0, exit_price, "PRED")
                self.Debug(f"Signal Exit (Submitted): {msg}")

                self.sendAPI("/signals", {
                    "id_method": self.strategy_id,
                    "datetime": self.Time.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "exit",
                    "jenis": "sell",
                    "price_entry": ent_price,
                    "price_exit": exit_price,
                    "target_tp": 0,
                    "target_sl": 0,
                    "real_tp": real_tp,
                    "real_sl": real_sl,
                    "message": msg
                })

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

            ent_price = self.entry_price
            pnl_pct = (price - self.entry_price) / self.entry_price if self.entry_price != 0 else 0.0

            if pnl_pct <= -self.stop_loss_pct:
                self.pending_exit = True
                self.last_exit_reason = "SL"
                self.Liquidate(self.symbol)

                msg = f"STOP LOSS trigger pnl={pnl_pct:.2%}"
                self.Debug(msg)

                self.sendAPI("/signals", {
                    "id_method": self.strategy_id,
                    "datetime": self.Time.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "exit",
                    "jenis": "sell",
                    "price_entry": ent_price,
                    "price_exit": price,
                    "target_tp": 0,
                    "target_sl": 0,
                    "real_tp": 0,
                    "real_sl": ent_price - price,
                    "message": msg
                })

                return True

            if pnl_pct >= self.take_profit_pct:
                self.pending_exit = True
                self.last_exit_reason = "TP"
                self.Liquidate(self.symbol)

                msg = f"TAKE PROFIT trigger pnl={pnl_pct:.2%}"
                self.Debug(msg)

                self.sendAPI("/signals", {
                    "id_method": self.strategy_id,
                    "datetime": self.Time.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "exit",
                    "jenis": "sell",
                    "price_entry": ent_price,
                    "price_exit": price,
                    "target_tp": 0,
                    "target_sl": 0,
                    "real_tp": price - ent_price,
                    "real_sl": 0,
                    "message": msg
                })

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
        return "\n".join(lines), t

    def FormatExitSignal(self, entry_price: float, exit_price: float, reason: str):
        entry = float(entry_price) if entry_price is not None else None
        exitp = float(exit_price)

        lines = [
            "Jenis Signal: Signal Exit",
            f"Symbol: {self.symbol}",
            f"Time: {self.Time}",
        ]

        real_tp = 0
        real_sl = 0

        if(exitp > entry):
            real_tp = exitp - entry
        else:
            real_sl = entry - exitp

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

        return "\n".join(lines), real_tp, real_sl

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
            
            self.sendAPI("/reminders", {"id_method":self.strategy_id, "datetime": self.Time.strftime("%Y-%m-%d %H:%M:%S"), "message": msg})

            self._mark_reminder_sent(tipe)
            to_remove.append(key)

        for k in to_remove:
            self.CancelReminder(k)

    def OnEndOfAlgorithm(self):
        self.Debug(f"Final Portfolio Value: {self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug("NOTE: Missing non-price features are set to 0; results won't match offline full-feature backtests.")


    def _join_url(self, base: str, path: str) -> str:
        base = (base or "").strip().rstrip("/")
        path = (path or "").strip()
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def sendAPI(self, path: str, params: dict):
        """
        Kirim POST JSON ke:
        {self.apidomain}{path}

        Contoh:
        self.sendAPI("/logs", {"id":"1"})
        self.sendAPI("/signals", {"id":"1"})
        self.sendAPI("/orders", {"id":"1"})
        self.sendAPI("/reminders", {"id":"1"})
        """
        try:
            if not getattr(self, "apidomain", None):
                self.Debug("[sendAPI] apidomain belum diset")
                return False

            url = self._join_url(self.apidomain, path)
            data = json.dumps(params if params is not None else {}, separators=(",", ":"))
            headers = dict(getattr(self, "api_headers", {}) or {})

            # Webhook QuantConnect = HTTP POST
            # PATCH: QuantConnect notification webhook
            try:
                self.Notify.Web(url, data, headers)
            except Exception:
                try:
                    self.Notify.Web(url, data)
                except Exception as e:
                    self.Debug(f"[sendAPI] Notify.Web failed: {e}")
                    return False

            # Optional log
            self.Debug(f"[sendAPI] POST {url} | payload={data}")
            return True

        except Exception as e:
            self.Debug(f"[sendAPI] ERROR: {e}")
            return False