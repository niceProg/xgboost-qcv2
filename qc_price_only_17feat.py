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


class XGBoostPriceOnly(QCAlgorithm):
    """
    ====================================================================
    XGBOOST PRICE-ONLY TRADING ALGORITHM (17 Features)
    ====================================================================

    Designed for QuantConnect with ONLY price data available.

    Features: 17 price features (OHLCV + derivatives)
    - Training: ./simple_run.sh --price-only
    - Model: API /api/v1/latest/model with feature_names containing price_ only

    Trading Parameters:
    - Trend window: 8 hours (matches training)
    - Threshold: 0.3% (matches training)
    - TP: 0.5% (slightly above training threshold)
    - SL: 0.2% (slightly below training threshold)
    - Max hold: 8 hours (matches training trend_window)

    Prediction Thresholds:
    - Buy: 35% probability
    - Sell: 30% probability
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
        # PRICE-ONLY FEATURES (17 features)
        # ============================================================
        self.model_features_price_only = [
            "price_open", "price_high", "price_low", "price_close", "price_volume_usd",
            "price_close_return_1", "price_close_return_5", "price_log_return",
            "price_rolling_vol_5", "price_true_range", "price_close_mean_5",
            "price_close_std_5", "price_volume_mean_10", "price_volume_zscore",
            "price_volume_change", "price_wick_upper", "price_wick_lower",
            "price_body_size",
        ]
        self.model_features = self.model_features_price_only

        # ============================================================
        # STRATEGY METADATA
        # ============================================================
        self.strategy_name = "Price-Only v1 (17 Features)"
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

        # Warning if model expects more features than available
        if self.model is not None and self.model_n_features and self.model_n_features > 20:
            self.Debug(f"⚠️ WARNING: Model expects {self.model_n_features} features")
            self.Debug(f"⚠️ QuantConnect only has {len(self.model_features)} price features")
            self.Debug(f"⚠️ Feature mismatch! Train with --price-only flag")

        # ============================================================
        # ROLLING WINDOW
        # ============================================================
        self.window_size = 60
        self.price_window = deque(maxlen=self.window_size)

        # ============================================================
        # TRADING PARAMETERS (MATCHED WITH TRAINING)
        # ============================================================
        # Training: trend_window=8, threshold=0.3%
        # Model predicts: "Will price move >= 0.3% in 8 hours?"

        self.prediction_buy_threshold = 0.35   # Buy when prob >= 35%
        self.prediction_sell_threshold = 0.30   # Exit when prob <= 30%
        self.position_size_pct = 0.50           # 50% capital per trade

        # TP/SL - MATCHED with training threshold (0.3%)
        self.stop_loss_pct = 0.002              # 0.2% SL
        self.take_profit_pct = 0.005            # 0.5% TP

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
        self.Debug("[INIT] Price-Only Algorithm initialized")

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
        url = f"{self.api_base_url}/api/v1/futures/latest/model"
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
                # Filter to only include features we can actually compute (price-only)
                self.expected_feature_order = [f for f in api_features if f in self.model_features]
                self.model_n_features = len(self.expected_feature_order)
                self.Debug(f"[MODEL] API features: {len(api_features)}, Filtered to: {self.model_n_features}")
                if len(api_features) != self.model_n_features:
                    skipped = set(api_features) - set(self.expected_feature_order)
                    self.Debug(f"[MODEL] Skipped features: {list(skipped)}")
            else:
                self.expected_feature_order = list(self.model_features)
                self.model_n_features = len(self.expected_feature_order)

            self.Debug(f"[MODEL] Total features: {self.model_n_features}")
        except Exception as e:
            self.Error(f"[MODEL] Load error: {e}")

    def LoadDatasetSummaryFromAPI(self):
        url = f"{self.api_base_url}/api/v1/futures/latest/dataset-summary"
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
            open_ = float(bar.Open)
            high = float(bar.High)
            low = float(bar.Low)
            close = float(bar.Close)
            volume = float(bar.Volume) if bar.Volume is not None else 0.0
        elif isinstance(bar, QuoteBar):
            src = bar.Bid if bar.Bid else bar.Ask
            if src is None:
                return
            open_ = float(src.Open)
            high = float(src.High)
            low = float(src.Low)
            close = float(src.Close)
            volume = 0.0
        else:
            return

        # Update rolling window with full OHLCV
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
    # FEATURES (17 price features)
    # ============================================================
    def BuildFeatures(self):
        try:
            df = pd.DataFrame(list(self.price_window))

            # Extract OHLCV from rolling window
            open_vals = df["open"].astype(float).values if "open" in df.columns else None
            high_vals = df["high"].astype(float).values if "high" in df.columns else None
            low_vals = df["low"].astype(float).values if "low" in df.columns else None
            closes = df["close"].astype(float).values
            volumes = df["volume"].astype(float).values if "volume" in df.columns else None

            if len(closes) < 6:
                return None

            close = float(closes[-1])
            volume = float(volumes[-1]) if volumes is not None and len(volumes) > 0 else 1_000_000.0

            # Returns
            ret_1 = closes[-1] / closes[-2] - 1.0 if closes[-2] > 0 else 0.0
            ret_5 = closes[-1] / closes[-6] - 1.0 if closes[-6] > 0 else 0.0
            log_ret = np.log(closes[-1] / closes[-2]) if closes[-2] > 0 else 0.0

            # Rolling vol
            return_1_series = []
            for i in range(1, min(6, len(closes))):
                if closes[i-1] > 0:
                    return_1_series.append(closes[i] / closes[i-1] - 1.0)
                else:
                    return_1_series.append(0.0)
            vol_5 = float(np.std(return_1_series[-5:])) if len(return_1_series) >= 5 else 0.0

            # True Range (using OHLC if available)
            if high_vals is not None and low_vals is not None and len(high_vals) >= 2:
                prev_close = closes[-2] if len(closes) >= 2 else close
                true_range = float(max(
                    high_vals[-1] - low_vals[-1],
                    abs(high_vals[-1] - prev_close),
                    abs(low_vals[-1] - prev_close)
                ))
            else:
                true_range = 0.0

            # Close mean/std
            mean_5 = float(np.mean(closes[-5:]))
            std_5 = float(np.std(closes[-5:]))

            # Volume features (if available)
            if volumes is not None and len(volumes) >= 10:
                vol_mean_10 = float(np.mean(volumes[-10:]))
                vol_std_10 = float(np.std(volumes[-10:])) if len(volumes) >= 10 else 1.0
                vol_z = (volume - vol_mean_10) / vol_std_10 if vol_std_10 > 0 else 0.0
                vol_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] > 0 else 0.0
            else:
                vol_mean_10 = volume
                vol_std_10 = 1.0
                vol_z = 0.0
                vol_change = 0.0

            # Candlestick features (if OHLC available)
            if open_vals is not None and high_vals is not None and low_vals is not None:
                curr_open = float(open_vals[-1])
                curr_high = float(high_vals[-1])
                curr_low = float(low_vals[-1])
                body_size = abs(close - curr_open)
                wick_up = curr_high - max(close, curr_open)
                wick_low = min(close, curr_open) - curr_low
            else:
                body_size = 0.0
                wick_up = 0.0
                wick_low = 0.0

            feat_map = {
                "price_open": float(open_vals[-1]) if open_vals is not None else close,
                "price_high": float(high_vals[-1]) if high_vals is not None else close,
                "price_low": float(low_vals[-1]) if low_vals is not None else close,
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
            }

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
