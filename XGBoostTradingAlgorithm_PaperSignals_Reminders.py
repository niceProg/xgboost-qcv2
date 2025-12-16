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
    """
    Paper/Live-ready version with customized signals.

    Adds:
    - Signal Entry message (price, TP, SL, RR)
    - Signal Exit message (entry, exit, TP/SL realized if triggered)
    - Uses OnOrderEvent for accurate filled prices
    - Pending entry/exit guards to prevent duplicate orders
    """

    def Initialize(self):
        # ===== ObjectStore keys =====
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
        self.strategy_name = "Metode ABC"
        self.startup_notified = False

        self.model = None
        self.model_n_features = None
        self.expected_feature_order = None
        self.LoadModelFromObjectStore()

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
        # By default, signals go to Debug().
        # For paper/live notifications, set these in Initialize():
        self.enable_live_notifications = True
        self.notify_channel = "Telegram"  # "Debug" | "Telegram" | "Webhook" | "Email" | "Sms"
        self.telegram_token = "8306719491:AAHNS7HT-pjMUGlcXMA_5SEffd6zPd2X6U0"
        self.telegram_chat_id = "-4978819951"
        self.webhook_url = ""
        self.email_to = ""
        self.sms_number = ""

        # ===== Reminder Signals (pre-entry/exit) =====
        # Notes:
        # - With Resolution.Hour data, reminders are naturally in 1h steps.
        # - We keep reminders short: 1-3 hours ahead (no longer than 3 hours).
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

        self.pred_debug_counter = 0
        self.Debug("XGBoostTradingAlgorithm (paper signals) initialized")

    def OnWarmupFinished(self):
        # Dipanggil sekali setelah WarmUp selesai

        if not self.LiveMode:
            self.Debug("[TELEGRAM] Not live mode, skip")
            return

        try:
            msg = "Metode ABC is start running"
            ok = self.Notify.Telegram(str(self.telegram_chat_id), msg, str(self.telegram_token))
            self.Debug(f"[TELEGRAM] sent={ok}")
        except Exception as e:
            self.Error(f"[TELEGRAM] failed: {e}")

        self.SendStartupMessage()

    def SendStartupMessage(self):
        if self.startup_notified:
            return
        if not self.LiveMode:
            return  # biar tidak spam di backtest

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

        # Always log
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
            elif ch == "webhook":
                if self.webhook_url:
                    self.Notify.Web(self.webhook_url, f"{safe_title}\n{safe_message}")
            elif ch == "email":
                if self.email_to:
                    self.Notify.Email(self.email_to, safe_title, safe_message)
            elif ch == "sms":
                if self.sms_number:
                    self.Notify.Sms(self.sms_number, f"{safe_title} {safe_message}")
            else:
                # Debug only
                pass
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
        """
        Returns dict:
        tp_price, tp_amt, tp_pct,
        sl_price, sl_amt, sl_pct,
        rr
        """
        entry = float(entry_price)
        tp_pct = float(self.take_profit_pct)
        sl_pct = float(self.stop_loss_pct)

        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 - sl_pct)

        tp_amt = tp_price - entry
        sl_amt = entry - sl_price  # positive amount risked

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

            # Only show TP/SL section when the exit was triggered by that rule
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

            # Extra useful line (doesn't conflict with your spec)
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
        """
        Arm a reminder to be sent in N hours (1..3).
        If the reminder already exists, we only update it if the new due time is sooner.
        """
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
            # If we can remind sooner, update
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
        """
        Maps a 'gap' to lead hours (1..3). Smaller gap => sooner reminder.
        """
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
        """
        progress in [0..1], higher => closer => sooner reminder.
        """
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
        """
        Arm short reminders (1-3h) for:
        - Entry setup (pred close to buy threshold, no position)
        - Exit by pred setup (pred close to sell threshold, has position)
        - Exit by TP/SL near levels (price close to TP/SL, has position)
        """
        if not self.enable_reminders:
            return

        qty = self.Portfolio[self.symbol].Quantity
        buy_th = float(self.prediction_buy_threshold)
        sell_th = float(self.prediction_sell_threshold)

        # ---- Pre-ENTRY reminder (no position) ----
        if qty <= 0 and (buy_th - self.entry_pred_band) <= pred < buy_th:
            gap = buy_th - pred
            hours = self._lead_hours_by_gap(gap)
            reason = f"Pred mendekati Entry (pred={pred:.3f} < buy_th={buy_th:.2f})"
            self.ArmReminder("REM_ENTRY", "Entry", hours, reason)

        # ---- Pre-EXIT reminder by prediction (has position) ----
        if qty > 0 and sell_th < pred <= (sell_th + self.exit_pred_band):
            gap = pred - sell_th
            hours = self._lead_hours_by_gap(gap)
            reason = f"Pred mendekati Exit (pred={pred:.3f} > sell_th={sell_th:.2f})"
            self.ArmReminder("REM_EXIT_PRED", "Exit (Pred)", hours, reason)

        # ---- Pre-EXIT reminder by TP/SL proximity (has position) ----
        if qty > 0 and self.entry_price is not None and self.entry_price > 0:
            entry = float(self.entry_price)
            tp_price = entry * (1.0 + float(self.take_profit_pct))
            sl_price = entry * (1.0 - float(self.stop_loss_pct))

            # TP progress
            tp_den = max(tp_price - entry, 1e-9)
            tp_prog = (price - entry) / tp_den  # 0..1.. (could exceed 1)
            if self.near_level_3h <= tp_prog < 1.0:
                hours = self._lead_hours_by_progress(tp_prog)
                reason = f"Harga mendekati TP (now≈{price:.2f}, TP={tp_price:.2f}, progress={tp_prog:.0%})"
                self.ArmReminder("REM_EXIT_TP", "Exit (TP)", hours, reason)
            else:
                # Not near TP anymore -> cancel
                self.CancelReminder("REM_EXIT_TP")

            # SL progress
            sl_den = max(entry - sl_price, 1e-9)
            sl_prog = (entry - price) / sl_den  # 0..1.. (could exceed 1)
            if self.near_level_3h <= sl_prog < 1.0:
                hours = self._lead_hours_by_progress(sl_prog)
                reason = f"Harga mendekati SL (now≈{price:.2f}, SL={sl_price:.2f}, progress={sl_prog:.0%})"
                self.ArmReminder("REM_EXIT_SL", "Exit (SL)", hours, reason)
            else:
                self.CancelReminder("REM_EXIT_SL")

        # If no position, cancel exit reminders
        if qty <= 0:
            self.CancelReminder("REM_EXIT_PRED")
            self.CancelReminder("REM_EXIT_TP")
            self.CancelReminder("REM_EXIT_SL")

    def ProcessReminders(self, pred: float, price: float):
        """
        When reminder due time is reached, emit the reminder if still relevant.
        Removes reminders after sending (or if invalid).
        """
        if not self.enable_reminders or not self.reminders:
            return

        # Do not remind while we have pending orders or are paused
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

            # Still relevant?
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

            # Anti-spam per type
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
    # LOAD MODEL
    # =========================================================
    def LoadModelFromObjectStore(self):
        try:
            if not self.ObjectStore.ContainsKey(self.model_key):
                self.Error(f"ObjectStore key not found: {self.model_key}")
                return

            file_path = self.ObjectStore.GetFilePath(self.model_key)
            self.model = joblib.load(file_path)
            self.Debug("Loaded XGBoost model from ObjectStore")

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

            self.Debug(f"Model expects {self.model_n_features} features; order_len={len(self.expected_feature_order)}")
        except Exception as e:
            self.Error(f"Error loading model: {e}")
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
                # Update entry price to actual fill price
                self.entry_price = fill_price
                self.entry_time = self.Time
                self.pending_entry = False

                # Cancel any pre-entry reminders
                self.CancelReminder("REM_ENTRY")

                msg = self.FormatEntrySignal(self.entry_price)
                self.SendSignal("Signal Entry (Filled)", msg)

            # SELL filled (exit)
            elif fill_qty < 0:
                # Compute exit message using last known entry price
                reason = self.last_exit_reason or "EXIT"
                if self.entry_price is not None:
                    msg = self.FormatExitSignal(self.entry_price, fill_price, reason)
                else:
                    msg = self.FormatExitSignal(0.0, fill_price, reason)

                self.SendSignal("Signal Exit (Filled)", msg)

                # Cancel all reminders after exit
                self.CancelAllReminders()

                # Reset after full exit
                self.entry_price = None
                self.entry_time = None
                self.pending_exit = False
                self.last_exit_reason = None

        except Exception as e:
            self.Debug(f"OnOrderEvent error: {e}")

    # =========================================================
    # ONDATA
    # =========================================================
    def OnData(self, data: Slice):
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
            self.Debug(f"{self.Time} - Pred={pred:.3f} Price={close:.2f}")
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
            # Note: Fill-based entry signal will be sent in OnOrderEvent
            self.SetHoldings(self.symbol, self.position_size_pct)
            # provisional entry (will be overwritten by fill)
            self.entry_price = float(self.Securities[self.symbol].Price)
            self.entry_time = self.Time
            self.Debug(f"BUY SetHoldings({self.position_size_pct:.0%}) pred={pred:.3f} est_entry={self.entry_price:.2f}")

        # EXIT by prediction
        elif pred < self.prediction_sell_threshold and qty > 0 and not self.pending_exit:
            self.pending_exit = True
            self.last_exit_reason = "PRED"
            self.Liquidate(self.symbol)
            # Do NOT clear entry_price here; need it for exit signal on fill
            self.Debug(f"EXIT (PRED) pred={pred:.3f}")

    # =========================================================
    # SL/TP
    # =========================================================
    def CheckStopLossTakeProfit(self, price: float) -> bool:
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
    def CheckMaxDrawdownAndCooldown(self) -> bool:
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

    def OnEndOfAlgorithm(self):
        self.Debug(f"Final Portfolio Value: {self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug("NOTE: Missing non-price features are set to 0; results won't match offline full-feature backtests.")
