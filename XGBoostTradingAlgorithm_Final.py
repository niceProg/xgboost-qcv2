from AlgorithmImports import *
import numpy as np
import pandas as pd
from collections import deque
import requests
from datetime import datetime, timedelta

# Ensure Binance market is available
try:
    from QuantConnect.Market import Market
    Market.Binance
except:
    class Market:
        Binance = "Binance"


class XGBoostTradingAlgorithm(QCAlgorithm):
    """
    Final XGBoost Algorithm with Clean API Integration.
    No ObjectStore, no model loading - pure API calls.
    """

    def Initialize(self):
        # ===== API Configuration =====
        # Update this to your domain
        self.api_base_url = "https://test.dragonfortune.ai:8000"
        self.api_timeout = 10
        self.api_retry_count = 3

        # Set date range for backtesting
        self.SetStartDate(2025, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(100000)

        # ===== Symbol =====
        self.symbol_str = "BTCUSDT"
        try:
            self.crypto = self.AddCrypto(self.symbol_str, Resolution.Hour, Market.Binance)
        except Exception:
            try:
                self.SetBrokerageModel(BrokerageName.Binance)
                self.crypto = self.AddCrypto(self.symbol_str, Resolution.Hour)
            except Exception:
                self.crypto = self.AddCrypto(self.symbol_str, Resolution.Hour)

        self.symbol = self.crypto.Symbol
        self.SetBenchmark(self.symbol)

        # ===== Trading Parameters =====
        self.prediction_buy_threshold = 0.55
        self.prediction_sell_threshold = 0.45
        self.position_size_pct = 0.80
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.10

        # ===== State Management =====
        self.window_size = 60
        self.price_window = deque(maxlen=self.window_size)

        # Entry/Exit tracking
        self.entry_price = None
        self.entry_time = None
        self.pending_entry = False
        self.pending_exit = False
        self.last_exit_reason = None

        # Risk management
        self.max_drawdown_pct = 0.20
        self.high_watermark = self.Portfolio.TotalPortfolioValue

        # API health tracking
        self.api_healthy = True
        self.api_failure_count = 0
        self.max_api_failures = 5

        # Notifications
        self.enable_live_notifications = True
        self.telegram_token = "8306719491:AAHNS7HT-pjMUGlcXMA_5SEffd6zPd2X6U0"
        self.telegram_chat_id = "-4978819951"

        # Model tracking
        self.last_model_check = None
        self.model_check_interval = timedelta(hours=1)

        # Signal tracking
        self.last_signal = None
        self.last_signal_time = None

        # Start model check schedule
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.EveryHour(), self.CheckModelStatus)

        # Set warmup
        self.SetWarmUp(60, Resolution.Hour)
        self.Debug("XGBoostTradingAlgorithm (Final API Integration) initialized")

    def CheckModelStatus(self):
        """Check if model is available and healthy."""
        if (self.last_model_check and
            (self.Time - self.last_model_check) < self.model_check_interval):
            return

        self.last_model_check = self.Time

        try:
            response = self.CallAPI("GET", "/health")
            if response and response.get('model_available'):
                self.api_healthy = True
                self.api_failure_count = 0
                self.Debug("✅ API and Model healthy")
            else:
                self.api_healthy = False
                self.api_failure_count += 1
                self.Error(f"❌ Model not available (Failure #{self.api_failure_count})")

        except Exception as e:
            self.Error(f"Model status check error: {e}")
            self.api_failure_count += 1

    def CallAPI(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make API call to XGBoost server."""
        if not self.api_healthy and self.api_failure_count < self.max_api_failures:
            # Try to recover
            self.CheckModelStatus()

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

    def OnData(self, data: Slice):
        """Main data handler with API integration."""
        if self.IsWarmingUp:
            return

        if not data.ContainsKey(self.symbol):
            return

        bar = data[self.symbol]
        if not isinstance(bar, TradeBar):
            return

        current_price = float(bar.Close)
        volume = float(bar.Volume or 0)

        # Update price window
        self.price_window.append({
            "time": self.Time,
            "price": current_price,
            "volume": volume
        })

        if len(self.price_window) < 30:
            return

        # Get trading signal from API
        signal_data = self.GetTradingSignal()

        if not signal_data.get("success"):
            self.Error(f"Failed to get signal: {signal_data.get('error', 'Unknown')}")
            return

        signal = signal_data.get("signal", "HOLD")
        confidence = signal_data.get("confidence", 0)
        recommendation = signal_data.get("recommendation", {})

        # Log signal
        if self.pred_debug_counter % 6 == 0:
            self.Debug(f"{self.Time} - Signal: {signal} (Conf: {confidence:.2f})")
        self.pred_debug_counter += 1

        # Execute trading logic
        self.ExecuteSignal(signal, confidence, signal_data, current_price)

        # Update signal tracking
        self.last_signal = signal
        self.last_signal_time = self.Time

    def GetTradingSignal(self) -> dict:
        """Get trading signal from API."""
        try:
            request_data = {
                "exchange": "binance",
                "symbol": self.symbol_str,
                "interval": "1h"
            }

            response = self.CallAPI("POST", "/signal", data=request_data)

            if response:
                return {
                    "success": True,
                    "signal": response.get('signal', 'HOLD'),
                    "confidence": response.get('confidence', 0),
                    "recommendation": response.get('recommendation', {})
                }
            else:
                return {"success": False, "error": "API call failed"}

        except Exception as e:
            self.Error(f"GetTradingSignal Error: {e}")
            return {"success": False, "error": str(e)}

    def ExecuteSignal(self, signal: str, confidence: float, signal_data: dict, current_price: float):
        """Execute trading signal."""
        try:
            qty = self.Portfolio[self.symbol].Quantity

            # SL/TP first (priority)
            if self.CheckStopLossTakeProfit(current_price):
                return

            # Entry signals
            if signal == "BUY" and qty <= 0 and not self.pending_entry and not self.pending_exit:
                if confidence > 0.6:  # Minimum confidence for entry
                    self.ExecuteBuy(confidence, signal_data, current_price)

            # Exit signals
            elif signal == "SELL" and qty > 0 and not self.pending_exit:
                if confidence > 0.6:  # Minimum confidence for exit
                    self.ExecuteSell(confidence, signal_data, current_price)

            # HOLD signals
            elif signal == "HOLD":
                self.Debug(f"HOLD signal (Conf: {confidence:.2f})")

        except Exception as e:
            self.Error(f"ExecuteSignal Error: {e}")

    def ExecuteBuy(self, confidence: float, signal_data: dict, current_price: float):
        """Execute buy order."""
        try:
            self.pending_entry = True

            # Get position size from recommendation
            recommendation = signal_data.get('recommendation', {})
            api_position_size = recommendation.get('position_size', self.position_size_pct)

            # Use API recommendation if reasonable
            if 0 < api_position_size <= 1:
                position_size = api_position_size
            else:
                position_size = self.position_size_pct

            # Execute trade
            self.SetHoldings(self.symbol, position_size)

            # Track entry
            self.entry_price = float(self.Securities[self.symbol].Price)
            self.entry_time = self.Time

            # Send notification
            msg = self.FormatBuyMessage(confidence, signal_data, current_price, position_size)
            self.SendSignal("BUY", msg)

            self.Debug(f"BUY executed - Size: {position_size:.1%}, Conf: {confidence:.2f}")

        except Exception as e:
            self.Error(f"ExecuteBuy Error: {e}")
            self.pending_entry = False

    def ExecuteSell(self, confidence: float, signal_data: dict, current_price: float):
        """Execute sell order."""
        try:
            self.pending_exit = True
            self.last_exit_reason = "API"

            # Close position
            self.Liquidate(self.symbol)

            # Calculate and send PnL
            if self.entry_price:
                pnl = current_price - self.entry_price
                pnl_pct = pnl / self.entry_price if self.entry_price > 0 else 0
            else:
                pnl = 0
                pnl_pct = 0

            # Send notification
            msg = self.FormatSellMessage(confidence, signal_data, current_price, pnl, pnl_pct)
            self.SendSignal("SELL", msg)

            # Reset tracking
            self.entry_price = None
            self.entry_time = None
            self.pending_exit = False

            self.Debug(f"SELL executed - PnL: {pnl_pct:.2%}, Conf: {confidence:.2f}")

        except Exception as e:
            self.Error(f"ExecuteSell Error: {e}")
            self.pending_exit = False

    def CheckStopLossTakeProfit(self, price: float) -> bool:
        """Check and execute SL/TP."""
        if self.pending_exit or not self.entry_price:
            return False

        try:
            pnl_pct = (price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0

            # Stop Loss
            if pnl_pct <= -self.stop_loss_pct:
                self.pending_exit = True
                self.last_exit_reason = "SL"
                self.Liquidate(self.symbol)

                msg = f"STOP LOSS TRIGGERED\nEntry: {self.entry_price:,.2f}\nExit: {price:,.2f}\nPnL: {pnl_pct:.2%}"
                self.SendSignal("STOP LOSS", msg)

                self.entry_price = None
                self.entry_time = None
                self.pending_exit = False
                return True

            # Take Profit
            if pnl_pct >= self.take_profit_pct:
                self.pending_exit = True
                self.last_exit_reason = "TP"
                self.Liquidate(self.symbol)

                msg = f"TAKE PROFIT TRIGGERED\nEntry: {self.entry_price:,.2f}\nExit: {price:,.2f}\nPnL: {pnl_pct:.2%}"
                self.SendSignal("TAKE PROFIT", msg)

                self.entry_price = None
                self.entry_time = None
                self.pending_exit = False
                return True

        except Exception as e:
            self.Error(f"CheckStopLossTakeProfit Error: {e}")
            return False

    def CheckMaxDrawdownAndCooldown(self) -> bool:
        """Check drawdown and implement cooldown."""
        pv = float(self.Portfolio.TotalPortfolioValue)
        if pv > self.high_watermark:
            self.high_watermark = pv

        dd = (self.high_watermark - pv) / self.high_watermark if self.high_watermark > 0 else 0

        if dd > self.max_drawdown_pct:
            # Exit and pause
            if self.Portfolio[self.symbol].Invested:
                self.pending_exit = True
                self.last_exit_reason = "RISK"
                self.Liquidate()

            self.trading_paused_until = self.Time + timedelta(days=7)

            msg = f"RISK MANAGEMENT - Drawdown {dd:.2%} > {self.max_drawdown_pct:.2%}\nTrading paused for 7 days"
            self.SendSignal("RISK PAUSE", msg)

            self.high_watermark = pv
            return False

        return True

    # ===== NOTIFICATION METHODS =====
    def SendSignal(self, title: str, message: str):
        """Send signal notification."""
        self.Debug(f"[SIGNAL] {title}\n{message}")

        if not self.LiveMode or not self.enable_live_notifications:
            return

        try:
            if self.telegram_token and self.telegram_chat_id:
                self.Notify.Telegram(self.telegram_token, self.telegram_chat_id, f"{title}\n{message}")
        except Exception as e:
            self.Error(f"SendSignal Error: {e}")

    def _fmt_price(self, x: float) -> str:
        return f"{float(x):,.2f}"

    def _fmt_pct(self, x: float) -> str:
        return f"{float(x)*100:.2f}%"

    def FormatBuyMessage(self, confidence: float, signal_data: dict, price: float, position_size: float) -> str:
        """Format buy signal message."""
        recommendation = signal_data.get('recommendation', {})

        lines = [
            "🟢 **BUY SIGNAL**",
            f"Symbol: {self.symbol_str}",
            f"Price: ${self._fmt_price(price)}",
            f"Confidence: {confidence:.1%}",
            f"Position Size: {position_size:.1%}",
            ""
        ]

        if recommendation:
            lines.extend([
                "📋 **Recommendation:**",
                f"Risk Level: {recommendation.get('risk_level', 'Unknown')}",
                f"Stop Loss: ${self._fmt_price(recommendation.get('stop_loss', 0))}",
                f"Take Profit: ${self._fmt_price(recommendation.get('take_profit', 0))}",
                ""
            ])

        lines.extend([
            f"Time: {self.Time}",
            f"Source: XGBoost API v2"
        ])

        return "\n".join(lines)

    def FormatSellMessage(self, confidence: float, signal_data: dict, price: float, pnl: float, pnl_pct: float) -> str:
        """Format sell signal message."""
        lines = [
            "🔴 **SELL SIGNAL**",
            f"Symbol: {self.symbol_str}",
            f"Price: ${self._fmt_price(price)}",
            f"Confidence: {confidence:.1%}",
            ""
        ]

        if self.entry_price:
            lines.extend([
                f"Entry Price: ${self._fmt_price(self.entry_price)}",
                f"PnL: ${self._fmt_price(pnl)} ({self._fmt_pct(pnl_pct)})",
                ""
            ])

        lines.extend([
            f"Reason: {self.last_exit_reason or 'API Signal'}",
            f"Time: {self.Time}",
            f"Source: XGBoost API v2"
        ])

        return "\n".join(lines)

    def OnEndOfAlgorithm(self):
        """Algorithm end."""
        final_msg = (
            f"Algorithm Completed\n"
            f"Final Portfolio: ${self.Portfolio.TotalPortfolioValue:,.2f}\n"
            f"API Health: {'Healthy' if self.api_healthy else 'Unhealthy'}\n"
            f"API Failures: {self.api_failure_count}\n"
            f"Final Signal: {self.last_signal or 'None'}"
        )

        self.Debug(final_msg)

        if self.LiveMode:
            self.Notify.Telegram(self.telegram_token, self.telegram_chat_id, final_msg)