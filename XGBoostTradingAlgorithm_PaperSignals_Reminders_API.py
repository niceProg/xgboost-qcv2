from AlgorithmImports import *
import numpy as np
import pandas as pd
from collections import deque
import os
import re
import json
import requests  # Added for API calls
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
    """
    Paper/Live-ready version dengan API Integration.

    Mengganti ObjectStore dengan FastAPI calls:
    1. Model Loading → API /models/latest
    2. Dataset Summary → API /model/metadata

    Features:
    - Real-time API integration
    - Dynamic model updates
    - Fallback mechanisms
    - Signal Entry/Exit messages
    - Pending entry/exit guards
    """

    def Initialize(self):
        # ===== API Configuration =====
        self.api_base_url = "https://test.dragonfortune.ai:8000"  # Your FastAPI server
        self.api_timeout = 10  # seconds
        self.api_retry_count = 3
        self.model_cache = None  # Cache model in memory
        self.model_last_updated = None
        self.model_update_interval = timedelta(hours=1)  # Check for updates every hour

        # Initialize model info with API
        self.InitializeModelFromAPI()

        # Set date range based on model metadata
        if self.train_start_date and self.train_end_date:
            start_dt = self.train_start_date - timedelta(days=1)
            end_dt = self.train_end_date
            self.Debug(f"[DATE] Using API dataset: {self.train_start_date} -> {self.train_end_date}")
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

        # Model features from API
        self.model_features = self.model_metadata.get('feature_names', self.available_features)
        self.model_n_features = len(self.model_features)
        self.expected_feature_order = self.model_features

        # Model metadata
        self.strategy_name = "Metode ABC (API Integration)"
        self.startup_notified = False

        # Schedule model updates
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.EveryHour(), self.CheckModelUpdate)

        # ===== Rolling window =====
        self.window_size = 60
        self.price_window = deque(maxlen=self.window_size)

        # ===== Trading parameters (dynamic from API) =====
        self.UpdateTradingParameters()

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

        # ===== API Health Tracking =====
        self.api_healthy = True
        self.api_failure_count = 0
        self.max_api_failures = 5

        # ===== Reminder Signals =====
        self.enable_reminders = True
        self.reminder_min_hours = 1
        self.reminder_max_hours = 3
        self.reminder_min_gap = timedelta(hours=2)
        self.last_reminder_sent_time = {}  # type -> datetime
        self.reminders = {}  # key -> dict(due, hours, tipe, reason)
        self.entry_pred_band = 0.03
        self.exit_pred_band = 0.03
        self.near_level_1h = 0.90
        self.near_level_2h = 0.80
        self.near_level_3h = 0.70

        # Warmup
        self.SetWarmUp(120, Resolution.Hour)
        self.pred_debug_counter = 0
        self.Debug(f"XGBoostTradingAlgorithm (API Integration) initialized - API: {self.api_base_url}")

    def InitializeModelFromAPI(self):
        """Initialize model from API instead of ObjectStore."""
        try:
            # Get model metadata from API
            metadata = self.CallAPI("GET", "/models/latest")

            if metadata:
                self.model_metadata = metadata
                self.model_n_features = metadata.get('n_features', len(self.available_features))
                self.expected_feature_order = metadata.get('feature_names', self.available_features)

                # Parse training dates
                training_info = metadata.get('training_data_info', {})
                if training_info:
                    self.train_start_date = training_info.get('start_date')
                    self.train_end_date = training_info.get('end_date')

                # Load actual model
                self.LoadModelFromAPI()

                self.Debug(f"✅ Model loaded from API: {metadata.get('name', 'Unknown')}")
            else:
                self.Error("❌ Failed to get model from API")
                self.UseFallbackModel()

        except Exception as e:
            self.Error(f"InitializeModelFromAPI error: {e}")
            self.UseFallbackModel()

    def LoadModelFromAPI(self):
        """Load XGBoost model from API."""
        try:
            # For now, we'll use signal prediction instead of loading model directly
            # Model stays on server, we just call API for predictions
            self.Debug("🔄 Using API predictions (model stays on server)")
            self.model_available = True
            return True

        except Exception as e:
            self.Error(f"LoadModelFromAPI error: {e}")
            self.model_available = False
            return False

    def UseFallbackModel(self):
        """Use fallback configuration when API fails."""
        self.Debug("⚠️ Using fallback configuration")
        self.model_available = False
        self.train_start_date = datetime(2024, 1, 1)
        self.train_end_date = datetime(2024, 12, 31)
        self.model_features = self.available_features
        self.expected_feature_order = self.available_features
        self.model_n_features = len(self.available_features)

    def CallAPI(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make API call to FastAPI server."""
        if not self.api_healthy and self.api_failure_count < self.max_api_failures:
            self.CheckAPIHealth()

        url = f"{self.api_base_url}{endpoint}"

        for attempt in range(self.api_retry_count):
            try:
                if method == "GET":
                    response = requests.get(url, timeout=self.api_timeout)
                elif method == "POST":
                    response = requests.post(url, json=data, timeout=self.api_timeout)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if response.status_code == 200:
                    return response.json()
                else:
                    self.Error(f"API Error {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                self.Error(f"API Timeout (attempt {attempt + 1}/{self.api_retry_count})")
            except requests.exceptions.ConnectionError:
                self.Error(f"API Connection Error (attempt {attempt + 1}/{self.api_retry_count})")
            except Exception as e:
                self.Error(f"API Call Error: {e}")

        self.api_failure_count += 1
        return None

    def CheckAPIHealth(self):
        """Check API health status."""
        try:
            response = self.CallAPI("GET", "/health")
            if response and response.get('api_status') == 'running':
                self.api_healthy = True
                self.api_failure_count = 0
            else:
                self.api_healthy = False
                self.api_failure_count += 1

            if self.api_failure_count >= self.max_api_failures:
                self.Error("❌ API down for too long - switching to safe mode")
                self.PauseTrading("API down")

        except Exception as e:
            self.Error(f"CheckAPIHealth error: {e}")
            self.api_healthy = False

    def CheckModelUpdate(self):
        """Check if model has been updated."""
        try:
            if not self.LiveMode:
                return  # Skip updates in backtest

            # Get current model info
            current_model = self.CallAPI("GET", "/models/latest")
            if current_model:
                new_version = current_model.get('name', '')
                current_version = getattr(self, 'current_model_version', '')

                if new_version != current_version:
                    self.Debug(f"🔄 New model detected: {new_version}")
                    self.InitializeModelFromAPI()
                    self.current_model_version = new_version

                    # Send notification
                    self.SendModelUpdateNotification(new_version)

        except Exception as e:
            self.Error(f"CheckModelUpdate error: {e}")

    def UpdateTradingParameters(self):
        """Update trading parameters based on model performance."""
        try:
            performance = self.model_metadata.get('performance', {})
            auc = performance.get('latest_auc', 0.5)
            accuracy = performance.get('latest_accuracy', 0.5)

            # Dynamic threshold adjustment
            if auc > 0.75 and accuracy > 0.7:
                # Excellent model - more aggressive
                self.prediction_buy_threshold = 0.60
                self.prediction_sell_threshold = 0.40
                self.position_size_pct = 0.90
                self.take_profit_pct = 0.12
            elif auc > 0.65 and accuracy > 0.6:
                # Good model - standard
                self.prediction_buy_threshold = 0.55
                self.prediction_sell_threshold = 0.45
                self.position_size_pct = 0.80
                self.take_profit_pct = 0.10
            else:
                # Poor model - very conservative
                self.prediction_buy_threshold = 0.70
                self.prediction_sell_threshold = 0.30
                self.position_size_pct = 0.60
                self.take_profit_pct = 0.08

            self.Debug(f"[PARAMS] Buy: {self.prediction_buy_threshold:.2f}, Sell: {self.prediction_sell_threshold:.2f}")

        except Exception as e:
            self.Error(f"UpdateTradingParameters error: {e}")

    def SendModelUpdateNotification(self, new_version: str):
        """Send notification about model update."""
        if not self.LiveMode:
            return

        try:
            performance = self.model_metadata.get('performance', {})
            auc = performance.get('latest_auc', 'N/A')
            accuracy = performance.get('latest_accuracy', 'N/A')

            msg = (
                f"🔄 **Model Updated**\n\n"
                f"New Version: {new_version}\n"
                f"AUC: {auc}\n"
                f"Accuracy: {accuracy}\n\n"
                f"New Parameters:\n"
                f"Buy Threshold: {self.prediction_buy_threshold:.2f}\n"
                f"Sell Threshold: {self.prediction_sell_threshold:.2f}\n"
                f"Position Size: {self.position_size_pct:.0%}\n"
                f"Take Profit: {self.take_profit_pct:.0%}\n\n"
                f"Time: {self.Time}"
            )

            self.SendSignal("MODEL_UPDATE", msg)

        except Exception as e:
            self.Error(f"SendModelUpdateNotification error: {e}")

    def OnWarmupFinished(self):
        if not self.LiveMode:
            self.Debug("[TELEGRAM] Not live mode, skip")
            return

        try:
            msg = f"{self.strategy_name} is running\n"
            if self.model_available:
                msg += f"API: {self.api_base_url}\n"
                msg += f"Model: {self.model_metadata.get('name', 'Unknown')}\n"

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

        api_info = ""
        if self.model_available:
            api_info = f"\nAPI: {self.api_base_url}"
            if self.model_metadata.get('performance'):
                perf = self.model_metadata['performance']
                api_info += f"\nModel AUC: {perf.get('latest_auc', 'N/A')}"

        msg = (
            f"🤖 {self.strategy_name} Started{api_info}\n"
            f"Symbol: {self.symbol}\n"
            f"Resolution: Hour\n"
            f"Buy Threshold: {self.prediction_buy_threshold:.2f}\n"
            f"Sell Threshold: {self.prediction_sell_threshold:.2f}"
        )

        self.SendSignal("START", msg)
        self.startup_notified = True

    # =========================================================
    # NOTIFICATION UTILITIES (unchanged)
    # =========================================================
    def SendSignal(self, title: str, message: str):
        safe_title = str(title)[:120]
        safe_message = str(message)

        self.Debug(f"[SIGNAL] {safe_title}\n{safe_message}")

        if not self.LiveMode:
            return

        if not self.enable_live_notifications:
            return

        try:
            ch = (self.notify_channel or "Debug").lower()

            if ch == "telegram":
                if self.telegram_token and self.telegram_chat_id:
                    self.Notify.Telegram(self.telegram_token, self.telegram_chat_id, f"{safe_title}\n{safe_message}")
        except Exception as e:
            self.Debug(f"[SIGNAL] Notify error: {e}")

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
        sl_amt = entry - sl_price

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
    # REMINDER SIGNALS (unchanged)
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

        # Pre-ENTRY reminder
        if qty <= 0 and (buy_th - self.entry_pred_band) <= pred < buy_th:
            gap = buy_th - pred
            hours = self._lead_hours_by_gap(gap)
            reason = f"Pred mendekati Entry (pred={pred:.3f} < buy_th={buy_th:.2f})"
            self.ArmReminder("REM_ENTRY", "Entry", hours, reason)

        # Pre-EXIT reminder by prediction
        if qty > 0 and sell_th < pred <= (sell_th + self.exit_pred_band):
            gap = pred - sell_th
            hours = self._lead_hours_by_gap(gap)
            reason = f"Pred mendekati Exit (pred={pred:.3f} > sell_th={sell_th:.2f})"
            self.ArmReminder("REM_EXIT_PRED", "Exit (Pred)", hours, reason)

        # Pre-EXIT reminder by TP/SL proximity
        if qty > 0 and self.entry_price is not None and self.entry_price > 0:
            entry = float(self.entry_price)
            tp_price = entry * (1.0 + float(self.take_profit_pct))
            sl_price = entry * (1.0 - float(self.stop_loss_pct))

            # TP progress
            tp_den = max(tp_price - entry, 1e-9)
            tp_prog = (price - entry) / tp_den
            if self.near_level_3h <= tp_prog < 1.0:
                hours = self._lead_hours_by_progress(tp_prog)
                reason = f"Harga mendekati TP (now≈{price:.2f}, TP={tp_price:.2f}, progress={tp_prog:.0%})"
                self.ArmReminder("REM_EXIT_TP", "Exit (TP)", hours, reason)
            else:
                self.CancelReminder("REM_EXIT_TP")

            # SL progress
            sl_den = max(entry - sl_price, 1e-9)
            sl_prog = (entry - price) / sl_den
            if self.near_level_3h <= sl_prog < 1.0:
                hours = self._lead_hours_by_progress(sl_prog)
                reason = f"Harga mendekati SL (now≈{price:.2f}, SL={sl_price:.2f}, progress={sl_prog:.0%})"
                self.ArmReminder("REM_EXIT_SL", "Exit (SL)", hours, reason)
            else:
                self.CancelReminder("REM_EXIT_SL")

        # Cancel exit reminders if no position
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
            self.SendSignal("Signal Reminder", msg)
            self._mark_reminder_sent(tipe)
            to_remove.append(key)

        for k in to_remove:
            self.CancelReminder(k)

    # =========================================================
    # ORDER EVENTS (unchanged)
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

            if fill_qty > 0:
                self.entry_price = fill_price
                self.entry_time = self.Time
                self.pending_entry = False
                self.CancelReminder("REM_ENTRY")
                msg = self.FormatEntrySignal(self.entry_price)
                self.SendSignal("Signal Entry (Filled)", msg)

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
                self.last_exit_reason = None

        except Exception as e:
            self.Debug(f"OnOrderEvent error: {e}")

    # =========================================================
    # ONDATA - MODIFIED TO USE API
    # =========================================================
    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        if not data.ContainsKey(self.symbol):
            return

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

        if self.trading_paused_until is not None and self.Time < self.trading_paused_until:
            return

        if self.pending_entry or self.pending_exit:
            return

        if not self.CheckMaxDrawdownAndCooldown():
            self.last_trade_bar_time = self.Time
            return

        # Get prediction from API instead of local model
        pred = self.GetPredictionFromAPI(close)
        if pred is None:
            return

        if self.pred_debug_counter % 24 == 0:
            api_status = "API" if self.model_available else "Fallback"
            self.Debug(f"{self.Time} - Pred={pred:.3f} Price={close:.2f} ({api_status})")
        self.pred_debug_counter += 1

        # Reminder logic (unchanged)
        self.MaybeArmReminders(pred, close)
        self.ProcessReminders(pred, close)

        # SL/TP check
        if self.CheckStopLossTakeProfit(close):
            self.last_trade_bar_time = self.Time
            return

        # Trading logic
        self.TradeLogic(pred)
        self.last_trade_bar_time = self.Time

    def GetPredictionFromAPI(self, current_price: float) -> float:
        """Get prediction from FastAPI server."""
        if not self.api_healthy or not self.model_available:
            self.Debug("API unavailable - using fallback prediction")
            return 0.5  # Neutral prediction

        try:
            # Prepare features for API
            features = self.BuildFeaturesForAPI()
            if features is None:
                return 0.5

            # Call API prediction
            request_data = {
                "features": features
            }

            response = self.CallAPI("POST", "/predict", data=request_data)

            if response:
                prediction = response.get('prediction_probability', response.get('prediction', 0.5))
                confidence = response.get('confidence', 0.5)

                self.Debug(f"API Prediction: {prediction:.3f} (Confidence: {confidence:.2f})")
                return float(prediction)
            else:
                self.Debug("API prediction failed - using fallback")
                return 0.5

        except Exception as e:
            self.Error(f"GetPredictionFromAPI error: {e}")
            return 0.5

    def BuildFeaturesForAPI(self) -> dict:
        """Build features dictionary for API call."""
        try:
            if len(self.price_window) < 6:
                return None

            df = pd.DataFrame(list(self.price_window))
            cur = df.iloc[-1]

            closes = df["close"].astype(float).values
            volumes = df["volume"].astype(float).values

            # Calculate features
            open_ = float(cur["open"])
            high = float(cur["high"])
            low = float(cur["low"])
            close = float(cur["close"])
            volume = float(cur["volume"])
            volume_usd = close * volume

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

            features = {
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

            return features

        except Exception as e:
            self.Error(f"BuildFeaturesForAPI error: {e}")
            return None

    def TradeLogic(self, pred: float):
        qty = self.Portfolio[self.symbol].Quantity

        if pred > self.prediction_buy_threshold and qty <= 0 and not self.pending_entry and not self.pending_exit:
            self.pending_entry = True
            self.SetHoldings(self.symbol, self.position_size_pct)
            self.entry_price = float(self.Securities[self.symbol].Price)
            self.entry_time = self.Time
            self.Debug(f"BUY SetHoldings({self.position_size_pct:.0%}) pred={pred:.3f} est_entry={self.entry_price:.2f}")

        elif pred < self.prediction_sell_threshold and qty > 0 and not self.pending_exit:
            self.pending_exit = True
            self.last_exit_reason = "PRED"
            self.Liquidate(self.symbol)
            self.Debug(f"EXIT (PRED) pred={pred:.3f}")

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

    def PauseTrading(self, reason: str):
        self.trading_paused_until = self.Time + timedelta(hours=1)
        msg = f"TRADING PAUSED\nReason: {reason}\nResume: {self.trading_paused_until}"
        self.SendSignal("PAUSED", msg)

    def OnEndOfAlgorithm(self):
        self.Debug(f"Final Portfolio Value: {self.Portfolio.TotalPortfolioValue:.2f}")
        if self.model_available:
            self.Debug("Using API predictions - model stays on server")
        else:
            self.Debug("Using fallback predictions")