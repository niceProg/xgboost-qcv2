from AlgorithmImports import *
import numpy as np
import pandas as pd
from collections import deque
import base64
import json
from datetime import datetime, timedelta

# Ensure Binance market is available
try:
    from QuantConnect.Market import Market
    Market.Binance
except:
    class Market:
        Binance = "Binance"


class XGBoostFullFeatures(QCAlgorithm):
    """
    ====================================================================
    XGBOOST FULL FEATURES TRADING ALGORITHM (105 Features)
    ====================================================================

    For environments with COMPLETE futures data available.

    Features: 105 features (price + funding + basis + LS + taker + OI + liquidations + cross)
    - Training: ./simple_run.sh (without --price-only flag)
    - Model: API /api/v1/latest/model with all 105 features

    Trading Parameters:
    - Trend window: 8 hours (matches training)
    - Threshold: 0.3% (matches training)
    - TP: 2% (more room for full-feature model predictions)
    - SL: 1% (more room for full-feature model predictions)
    - Max hold: 8 hours (matches training trend_window)

    Prediction Thresholds:
    - Buy: 60% probability (higher confidence for full model)
    - Sell: 40% probability (lower confidence for exit)
    """

    def Initialize(self):
        # ============================================================
        # API CONFIGURATION
        # ============================================================
        self.api_base_url = "https://api.dragonfortune.ai"
        self.api_timeout = 30
        self.api_retry_count = 3

        # ============================================================
        # LOAD DATASET SUMMARY & DATES
        # ============================================================
        self.train_start_date = None
        self.train_end_date = None

        try:
            self.LoadDatasetSummaryFromAPI()
        except Exception as e:
            self.Debug(f"[DATE] Failed to load dataset summary: {e}")
            self.train_start_date = datetime(2024, 1, 1)
            self.train_end_date = datetime(2025, 12, 31)

        if self.train_start_date and self.train_end_date:
            start_dt = self.train_start_date - timedelta(days=1)
            end_dt = self.train_end_date
            self.Debug(f"[DATE] Using dataset: {self.train_start_date} -> {self.train_end_date}")
        else:
            start_dt = datetime(2024, 1, 1)
            end_dt = datetime(2025, 12, 31)
            self.Debug(f"[DATE] Using fallback: {start_dt} -> {end_dt}")

        self.SetStartDate(start_dt.year, start_dt.month, start_dt.day)
        self.SetEndDate(end_dt.year, end_dt.month, end_dt.day)
        self.SetCash(1000)

        # ============================================================
        # SYMBOL & BROKERAGE
        # ============================================================
        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)
        self.crypto = self.AddCrypto("BTCUSDT", Resolution.Hour, Market.Binance)
        self.symbol = self.crypto.Symbol
        self.SetBenchmark(self.symbol)

        # Leverage for futures trading
        self.leverage = 10
        self.Securities[self.symbol].SetLeverage(self.leverage)

        # ============================================================
        # FULL FEATURES (105 features)
        # ============================================================
        self.model_features_full = [
            # Price features (17)
            "price_open", "price_high", "price_low", "price_close", "price_volume_usd",
            "price_close_return_1", "price_close_return_5", "price_log_return",
            "price_rolling_vol_5", "price_true_range", "price_close_mean_5",
            "price_close_std_5", "price_volume_mean_10", "price_volume_zscore",
            "price_volume_change", "price_wick_upper", "price_wick_lower", "price_body_size",
            # Funding features (6)
            "funding_open", "funding_high", "funding_low", "funding_close",
            "funding_norm", "funding_mean_24", "funding_std_24", "funding_zscore",
            "funding_extreme_positive", "funding_extreme_negative",
            # Basis features (5)
            "basis_open_basis", "basis_close_basis", "basis_open_change", "basis_close_change",
            "basis_delta", "basis_drift", "basis_mean_24", "basis_zscore", "basis_volatility_24",
            # Long-Short Global (7)
            "ls_global_global_account_long_percent", "ls_global_global_account_short_percent",
            "ls_global_global_account_long_short_ratio",
            "ls_global_ratio", "ls_global_zscore", "ls_global_delta",
            "ls_global_extreme_high", "ls_global_extreme_low",
            # Long-Short Top (7)
            "ls_top_top_account_long_percent", "ls_top_top_account_short_percent",
            "ls_top_top_account_long_short_ratio",
            "ls_top_ratio", "ls_top_zscore", "ls_top_delta", "ls_top_vs_global",
            # Taker Volume (5)
            "taker_buy_ratio", "taker_sell_ratio", "taker_imbalance",
            "taker_volume_mean", "taker_volume_zscore",
            # Orderbook (3)
            "orderbook_imbalance_usd", "orderbook_bid_ask_spread", "orderbook_depth_ratio",
            # Open Interest (6)
            "oi_delta", "oi_pct_change", "oi_mean_24", "oi_zscore",
            "oi_range", "oi_range_pct",
            # Liquidation (4)
            "liq_long_ratio", "liq_short_ratio", "liq_imbalance", "liq_spike",
            # Cross features (10+)
            "cross_funding_price", "cross_ls_price", "cross_oi_taker",
            "cross_basis_taker", "cross_taker_price", "cross_taker_funding",
            "cross_ob_price", "cross_liq_price", "cross_liq_spike_price",
            "cross_oi_funding",
        ]
        self.model_features = self.model_features_full

        # ============================================================
        # STRATEGY METADATA
        # ============================================================
        self.strategy_name = "Full Features v1 (105 Features)"
        self.startup_notified = False

        # Model state
        self.model = None
        self.model_n_features = None
        self.expected_feature_order = None

        # ============================================================
        # LOAD MODEL FROM API
        # ============================================================
        try:
            self.Debug("[MODEL] Loading model from API...")
            self.LoadModelFromAPI()
        except Exception as e:
            self.Error(f"[MODEL] Failed to load: {e}")
            self.Error("[MODEL] No trades will be taken without a model")

        # Warning if model has fewer features
        if self.model is not None and self.model_n_features and self.model_n_features < 50:
            self.Debug(f"⚠️ WARNING: Model only has {self.model_n_features} features")
            self.Debug(f"⚠️ Expected 105 features for full model")
            self.Debug(f"⚠️ Train without --price-only flag for full features")

        # ============================================================
        # ROLLING WINDOW
        # ============================================================
        self.window_size = 60
        self.price_window = deque(maxlen=self.window_size)

        # ============================================================
        # TRADING PARAMETERS (OPTIMIZED FOR FULL MODEL)
        # ============================================================
        # Training: trend_window=8, threshold=0.3%
        # Full model has more predictive power, can use tighter thresholds

        self.prediction_buy_threshold = 0.60   # Buy when prob >= 60%
        self.prediction_sell_threshold = 0.40   # Exit when prob <= 40%
        self.position_size_pct = 0.50           # 50% capital per trade

        # TP/SL - Full model can aim for larger moves
        self.stop_loss_pct = 0.01               # 1% SL
        self.take_profit_pct = 0.02             # 2% TP

        # Max hold period - MATCHED with training trend_window
        self.max_hold_bars = 8                  # 8 hours max hold
        self.min_hold_bars = 1                  # 1 hour min hold

        # Entry/Exit tracking
        self.entry_price = None
        self.entry_time = None
        self.pending_entry = False
        self.pending_exit = False
        self.last_exit_reason = None

        # ============================================================
        # RISK MANAGEMENT
        # ============================================================
        self.max_drawdown_pct = 0.25
        self.high_watermark = self.Portfolio.TotalPortfolioValue
        self.cooldown_period = timedelta(days=1)
        self.trading_paused_until = None
        self.last_trade_bar_time = None

        # ============================================================
        # NOTIFICATIONS
        # ============================================================
        self.enable_live_notifications = True
        self.notify_channel = "Telegram"
        self.telegram_token = "YOUR_TOKEN"
        self.telegram_chat_id = "YOUR_CHAT_ID"

        # ============================================================
        # WARMUP
        # ============================================================
        self.SetWarmUp(120, Resolution.Hour)
        self.pred_debug_counter = 0
        self.Debug("[INIT] Full Features Algorithm initialized")

    # ============================================================
    # STARTUP NOTIFICATION
    # ============================================================
    def OnWarmupFinished(self):
        if not self.LiveMode:
            return
        self.SendStartupMessage()

    def SendStartupMessage(self):
        if self.startup_notified:
            return
        if not self.LiveMode:
            return

        msg = (
            f"{self.strategy_name}\n"
            f"symbol={self.symbol}\n"
            f"features={len(self.model_features)}\n"
            f"time={self.Time}"
        )
        self.SendSignal("START", msg)
        self.startup_notified = True

    # ============================================================
    # NOTIFICATION UTILS
    # ============================================================
    def SendSignal(self, title: str, message: str):
        self.Debug(f"[SIGNAL] {title}\n{message}")

        if not self.LiveMode or not self.enable_live_notifications:
            return

        try:
            ch = (self.notify_channel or "Debug").lower()
            if ch == "telegram" and self.telegram_token and self.telegram_chat_id:
                self.Notify.Telegram(self.telegram_token, self.telegram_chat_id, f"{title}\n{message}")
        except Exception as e:
            self.Debug(f"[SIGNAL] Error: {e}")

    # ============================================================
    # API - LOAD MODEL
    # ============================================================
    def _download_json(self, url: str):
        try:
            s = self.Download(url)
            if s is None:
                return None
            txt = str(s).strip()
            if not txt:
                return None
            return json.loads(txt)
        except Exception as e:
            self.Debug(f"[HTTP] Failed: {url} err={e}")
            return None

    def LoadModelFromAPI(self):
        url = f"{self.api_base_url}/api/v1/latest/model"
        self.Debug(f"[MODEL] Downloading from: {url}")

        data = self._download_json(url)
        if not data or not data.get("success"):
            self.Error("[MODEL] API request failed")
            return

        model_b64 = data.get("model_data_base64")
        if not model_b64:
            self.Error("[MODEL] No model data in response")
            return

        try:
            import pickle
            from io import BytesIO
            model_bytes = base64.b64decode(model_b64)
            self.model = pickle.load(BytesIO(model_bytes))
            self.Debug("[MODEL] Loaded successfully")

            # Get feature names from API
            api_features = data.get("feature_names", [])
            if api_features:
                self.expected_feature_order = list(api_features)
                self.model_n_features = len(self.expected_feature_order)
                self.Debug(f"[MODEL] Features from API: {self.model_n_features}")
            else:
                self.expected_feature_order = list(self.model_features)
                self.model_n_features = len(self.expected_feature_order)

            self.Debug(f"[MODEL] Total features: {self.model_n_features}")
        except Exception as e:
            self.Error(f"[MODEL] Load error: {e}")

    def LoadDatasetSummaryFromAPI(self):
        url = f"{self.api_base_url}/api/v1/latest/dataset-summary"
        data = self._download_json(url)

        if not data or not data.get("success"):
            return

        b64 = data.get("summary_data_base64")
        if not b64:
            return

        try:
            import re
            decoded = base64.b64decode(b64).decode('utf-8')

            # Parse Start: and End: lines
            m_start = re.search(r"- Start:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", decoded)
            m_end = re.search(r"- End:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", decoded)

            if m_start and m_end:
                self.train_start_date = datetime.strptime(m_start.group(1), "%Y-%m-%d")
                self.train_end_date = datetime.strptime(m_end.group(1), "%Y-%m-%d")
        except Exception as e:
            self.Debug(f"[DATE] Parse error: {e}")

    # ============================================================
    # ORDER EVENTS
    # ============================================================
    def OnOrderEvent(self, orderEvent: OrderEvent):
        if orderEvent.Status != OrderStatus.Filled:
            return

        order = self.Transactions.GetOrderById(orderEvent.OrderId)
        if order is None or order.Symbol != self.symbol:
            return

        fill_price = float(orderEvent.FillPrice)
        fill_qty = float(orderEvent.FillQuantity)

        if fill_qty > 0:  # BUY
            self.entry_price = fill_price
            self.entry_time = self.Time
            self.pending_entry = False

            msg = self.FormatEntrySignal(self.entry_price)
            self.SendSignal("ENTRY FILLED", msg)
            self.Debug(f"[BUY] {self.entry_price:.2f} pred={fill_qty:.4f}")

        elif fill_qty < 0:  # SELL
            reason = self.last_exit_reason or "EXIT"
            if self.entry_price:
                msg = self.FormatExitSignal(self.entry_price, fill_price, reason)
            else:
                msg = f"Exit at {fill_price:.2f} ({reason})"

            self.SendSignal("EXIT FILLED", msg)
            self.Debug(f"[SELL] {fill_price:.2f} reason={reason}")

            self.entry_price = None
            self.entry_time = None
            self.pending_exit = False
            self.last_exit_reason = None

    # ============================================================
    # ONDATA
    # ============================================================
    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        if not data.ContainsKey(self.symbol):
            return

        if self.last_trade_bar_time == self.Time:
            return

        bar = data[self.symbol]
        if isinstance(bar, TradeBar):
            close = float(bar.Close)
        elif isinstance(bar, QuoteBar):
            src = bar.Bid if bar.Bid else bar.Ask
            if src is None:
                return
            close = float(src.Close)
        else:
            return

        # Update rolling window
        self.price_window.append({"time": self.Time, "close": close})

        if len(self.price_window) < 30:
            return

        # Cooldown
        if self.trading_paused_until and self.Time < self.trading_paused_until:
            return

        # Pending orders
        if self.pending_entry or self.pending_exit:
            return

        # Drawdown check
        if not self.CheckMaxDrawdownAndCooldown():
            self.last_trade_bar_time = self.Time
            return

        # Features + Predict
        features = self.BuildFeatures()
        if features is None:
            return

        pred = self.Predict(features)
        if pred is None:
            return

        # Debug every 12 hours
        if self.pred_debug_counter % 12 == 0:
            self.Debug(f"[PRED] {self.Time} pred={pred:.3f} price={close:.2f}")
        self.pred_debug_counter += 1

        # SL/TP first
        if self.CheckStopLossTakeProfit(close):
            self.last_trade_bar_time = self.Time
            return

        # Max hold period
        if self.CheckMaxHoldPeriod(close):
            self.last_trade_bar_time = self.Time
            return

        # Trade logic
        self.TradeLogic(pred)
        self.last_trade_bar_time = self.Time

    # ============================================================
    # FEATURES (105 features - Price Only Fallback)
    # ============================================================
    def BuildFeatures(self):
        """
        NOTE: QuantConnect only has price data.
        Non-price features (funding, basis, LS, etc.) will be set to 0.
        For full model performance, use external environment with complete data.
        """
        try:
            df = pd.DataFrame(list(self.price_window))
            close = float(df["close"].iloc[-1])
            volume = 1_000_000.0  # Default

            closes = df["close"].astype(float).values

            if len(closes) < 6:
                return None

            # Price features (17)
            ret_1 = closes[-1] / closes[-2] - 1.0 if closes[-2] > 0 else 0.0
            ret_5 = closes[-1] / closes[-6] - 1.0 if closes[-6] > 0 else 0.0
            log_ret = np.log(closes[-1] / closes[-2]) if closes[-2] > 0 else 0.0

            return_1_series = []
            for i in range(1, min(6, len(closes))):
                if closes[i-1] > 0:
                    return_1_series.append(closes[i] / closes[i-1] - 1.0)
                else:
                    return_1_series.append(0.0)
            vol_5 = float(np.std(return_1_series[-5:])) if len(return_1_series) >= 5 else 0.0

            true_range = 0.0
            mean_5 = float(np.mean(closes[-5:]))
            std_5 = float(np.std(closes[-5:]))
            vol_mean_10 = volume
            vol_std_10 = 1.0
            vol_z = 0.0
            vol_change = 0.0
            wick_up = 0.0
            wick_low = 0.0
            body_size = 0.0

            # Feature map - price only, others default to 0
            feat_map = {name: 0.0 for name in self.model_features}

            # Fill price features
            feat_map.update({
                "price_open": close,
                "price_high": close,
                "price_low": close,
                "price_close": close,
                "price_volume_usd": close * volume,
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
            })

            # Build vector in expected order
            order = self.expected_feature_order or self.model_features
            vec = []
            for name in order:
                v = float(feat_map.get(name, 0.0))
                if not np.isfinite(v):
                    v = 0.0
                vec.append(v)

            if self.model_n_features:
                if len(vec) > self.model_n_features:
                    vec = vec[:self.model_n_features]
                elif len(vec) < self.model_n_features:
                    vec.extend([0.0] * (self.model_n_features - len(vec)))

            return np.asarray(vec, dtype=float).reshape(1, -1)
        except Exception as e:
            self.Error(f"BuildFeatures error: {e}")
            return None

    # ============================================================
    # PREDICT
    # ============================================================
    def Predict(self, feature_array):
        try:
            if self.model is None:
                if self.pred_debug_counter == 0:
                    self.Error("[MODEL] Model is None - no trades")
                return None
            proba = self.model.predict_proba(feature_array)[0, 1]
            return float(proba)
        except Exception as e:
            self.Error(f"Predict error: {e}")
            return None

    # ============================================================
    # TRADING LOGIC
    # ============================================================
    def TradeLogic(self, pred: float):
        qty = self.Portfolio[self.symbol].Quantity
        buy_th = float(self.prediction_buy_threshold)
        sell_th = float(self.prediction_sell_threshold)

        # ENTRY
        if pred > buy_th and qty <= 0 and not self.pending_entry and not self.pending_exit:
            self.pending_entry = True
            self.SetHoldings(self.symbol, self.position_size_pct)
            self.entry_price = float(self.Securities[self.symbol].Price)
            self.entry_time = self.Time
            self.Debug(f"[BUY] pred={pred:.3f} > {buy_th:.2f}")

        # EXIT
        elif pred < sell_th and qty > 0 and not self.pending_exit:
            # Check min hold
            if self.entry_time:
                min_hold = timedelta(hours=self.min_hold_bars)
                if self.Time < (self.entry_time + min_hold):
                    return

            self.pending_exit = True
            self.last_exit_reason = "PRED"
            self.Liquidate(self.symbol)
            self.Debug(f"[EXIT] pred={pred:.3f} < {sell_th:.2f}")

    # ============================================================
    # SL/TP CHECK
    # ============================================================
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
                self.Debug(f"[SL] pnl={pnl_pct:.2%}")
                return True

            if pnl_pct >= self.take_profit_pct:
                self.pending_exit = True
                self.last_exit_reason = "TP"
                self.Liquidate(self.symbol)
                self.Debug(f"[TP] pnl={pnl_pct:.2%}")
                return True

            return False
        except Exception as e:
            self.Error(f"SL/TP error: {e}")
            return False

    # ============================================================
    # MAX HOLD PERIOD CHECK
    # ============================================================
    def CheckMaxHoldPeriod(self, price: float) -> bool:
        try:
            if self.pending_exit:
                return True

            qty = self.Portfolio[self.symbol].Quantity
            if qty <= 0 or self.entry_time is None or self.entry_price is None:
                return False

            max_hold_hours = getattr(self, 'max_hold_bars', 8)
            hold_duration = self.Time - self.entry_time

            if hold_duration >= timedelta(hours=max_hold_hours):
                self.pending_exit = True
                self.last_exit_reason = "TIME"
                self.Liquidate(self.symbol)

                if self.entry_price != 0:
                    pnl_pct = (price - self.entry_price) / self.entry_price
                else:
                    pnl_pct = 0.0
                self.Debug(f"[TIME] {max_hold_hours}h held={hold_duration} pnl={pnl_pct:.2%}")
                return True

            return False
        except Exception as e:
            self.Error(f"MaxHold error: {e}")
            return False

    # ============================================================
    # DRAWDOWN CHECK
    # ============================================================
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
                self.Debug(f"[RISK] DD={dd:.2%}")

            self.trading_paused_until = self.Time + self.cooldown_period
            self.Debug(f"[RISK] Pause until {self.trading_paused_until}")
            self.high_watermark = pv
            return False

        return True

    # ============================================================
    # SIGNAL FORMATTING
    # ============================================================
    def FormatEntrySignal(self, entry_price: float):
        tp = entry_price * (1 + self.take_profit_pct)
        sl = entry_price * (1 - self.stop_loss_pct)
        rr = (tp - entry_price) / (entry_price - sl) if sl < entry_price else 0

        return (
            f"ENTRY SIGNAL\n"
            f"Price: {entry_price:.2f}\n"
            f"TP: {tp:.2f} ({self.take_profit_pct:.1%})\n"
            f"SL: {sl:.2f} ({self.stop_loss_pct:.1%})\n"
            f"RR: {rr:.2f}"
        )

    def FormatExitSignal(self, entry_price: float, exit_price: float, reason: str):
        if entry_price:
            pnl = exit_price - entry_price
            pnl_pct = pnl / entry_price if entry_price != 0 else 0
            return (
                f"EXIT SIGNAL ({reason})\n"
                f"Entry: {entry_price:.2f}\n"
                f"Exit: {exit_price:.2f}\n"
                f"PnL: {pnl:.2f} ({pnl_pct:.1%})"
            )
        return f"EXIT SIGNAL ({reason})\nPrice: {exit_price:.2f}"

    def OnEndOfAlgorithm(self):
        self.Debug(f"Final PV: {self.Portfolio.TotalPortfolioValue:.2f}")
