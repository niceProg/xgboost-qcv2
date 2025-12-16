from AlgorithmImports import *
import numpy as np
import pandas as pd
from collections import deque
import json
import requests
import time
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
    XGBoost Algorithm dengan Real-time API Integration.

    Menggunakan FastAPI XGBoost server untuk real-time predictions.
    No local model loading, no ObjectStore dependency.
    """

    def Initialize(self):
        # ===== API Configuration =====
        self.api_base_url = "https://your-server.com:8000"  # Ganti dengan server Anda
        self.api_timeout = 10  # seconds
        self.api_retry_count = 3
        self.api_last_call_time = None
        self.api_call_interval = timedelta(minutes=1)  # Rate limiting

        # ===== Trading Configuration =====
        self.strategy_name = "Metode ABC (API)"
        self.symbol_str = "BTCUSDT"
        self.exchange = "binance"
        self.interval = "1h"

        # Add crypto
        try:
            self.crypto = self.AddCrypto(self.symbol_str, Resolution.Hour, Market.Binance)
        except Exception:
            try:
                self.SetBrokerageModel(BrokerageName.Binance)
                self.crypto = self.AddCrypto(self.symbol_str, Resolution.Hour)
            except Exception:
                self.crypto = self.AddCrypto(self.symbol_str, Resolution.Hour)
                self.Debug("Warning: Using default market")

        self.symbol = self.crypto.Symbol
        self.SetBenchmark(self.symbol)

        # Set date range
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(100000)

        # ===== Trading Parameters (Dynamic from API) =====
        self.prediction_buy_threshold = 0.55
        self.prediction_sell_threshold = 0.45
        self.position_size_pct = 0.80
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.10

        # ===== State Management =====
        self.window_size = 20  # Smaller window since API does feature engineering
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
        self.cooldown_period = timedelta(days=7)
        self.trading_paused_until = None

        # API health tracking
        self.api_healthy = True
        self.api_last_health_check = None
        self.api_health_check_interval = timedelta(hours=1)
        self.api_failure_count = 0
        self.max_api_failures = 5

        # Signal tracking
        self.last_signal = None
        self.last_signal_time = None
        self.signal_cooldown = timedelta(hours=2)  # Avoid signal spam

        # Notifications
        self.enable_live_notifications = True
        self.telegram_token = "8306719491:AAHNS7HT-pjMUGlcXMA_5SEffd6zPd2X6U0"
        self.telegram_chat_id = "-4978819951"

        # Start API health monitor
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.EveryHour(), self.CheckAPIHealth)

        # Set warmup
        self.SetWarmUp(60, Resolution.Hour)

        self.Debug(f"XGBoost API Algorithm initialized - API: {self.api_base_url}")

    def OnWarmupFinished(self):
        """Initialize after warmup."""
        if self.LiveMode:
            # Initial API health check
            self.CheckAPIHealth()

            # Send startup message
            msg = f"🚀 {self.strategy_name} Started\\nSymbol: {self.symbol_str}\\nAPI: {self.api_base_url}\\nMode: Real-time API"
            self.SendTelegramNotification(msg)

    def CheckAPIHealth(self):
        """Periodic API health check."""
        try:
            if (self.api_last_health_check and
                (self.Time - self.api_last_health_check) < self.api_health_check_interval):
                return

            self.api_last_health_check = self.Time

            # Call API status endpoint
            response = self.CallAPI("GET", "/status", timeout=5)

            if response and response.get('api_status') == 'running':
                self.api_healthy = True
                self.api_failure_count = 0

                # Update trading parameters from API if available
                if response.get('model_available'):
                    self.Debug(f"✅ API Healthy - Model Available")
                else:
                    self.Debug("⚠️ API Healthy - No Model Available")

            else:
                self.api_healthy = False
                self.api_failure_count += 1
                self.Error(f"❌ API Unhealthy (Failure #{self.api_failure_count})")

                if self.api_failure_count >= self.max_api_failures:
                    self.PauseTrading("API down for too long")

        except Exception as e:
            self.Error(f"API Health Check Error: {e}")
            self.api_failure_count += 1

    def CallAPI(self, method: str, endpoint: str, data: dict = None, timeout: int = None) -> dict:
        """Generic API call method with retry logic."""
        if not self.api_healthy and self.api_failure_count < self.max_api_failures:
            # Try to recover
            self.CheckAPIHealth()

        if timeout is None:
            timeout = self.api_timeout

        url = f"{self.api_base_url}{endpoint}"

        for attempt in range(self.api_retry_count):
            try:
                if method == "GET":
                    response = requests.get(url, timeout=timeout)
                elif method == "POST":
                    response = requests.post(url, json=data, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if response.status_code == 200:
                    return response.json()
                else:
                    self.Error(f"API Error {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                self.Error(f"API Timeout (attempt {attempt + 1}/{self.api_retry_count})")
                if attempt < self.api_retry_count - 1:
                    time.sleep(1)
                    continue

            except requests.exceptions.ConnectionError:
                self.Error(f"API Connection Error (attempt {attempt + 1}/{self.api_retry_count})")
                if attempt < self.api_retry_count - 1:
                    time.sleep(2)
                    continue

            except Exception as e:
                self.Error(f"API Call Error: {e}")
                break

        # All attempts failed
        self.api_failure_count += 1
        return None

    def GetTradingSignal(self) -> dict:
        """Get trading signal from API."""
        try:
            # Prepare request
            request_data = {
                "exchange": self.exchange,
                "symbol": self.symbol_str,
                "interval": self.interval,
                "threshold": 0.5  # API will use its own threshold
            }

            # Call API
            response = self.CallAPI("POST", "/signal", data=request_data)

            if response:
                return {
                    "success": True,
                    "signal": response.get('signal', 'HOLD'),
                    "confidence": response.get('confidence', 0),
                    "price": response.get('price', 0),
                    "prediction_probability": response.get('prediction_probability', 0),
                    "recommendation": response.get('recommendation', {})
                }
            else:
                return {"success": False, "error": "API call failed"}

        except Exception as e:
            self.Error(f"GetTradingSignal Error: {e}")
            return {"success": False, "error": str(e)}

    def GetMarketData(self) -> dict:
        """Get current market data from API."""
        try:
            response = self.CallAPI("GET", f"/market/{self.exchange}/{self.symbol_str}/{self.interval}")

            if response:
                return response
            else:
                return {}

        except Exception as e:
            self.Error(f"GetMarketData Error: {e}")
            return {}

    def OnData(self, data: Slice):
        """Main data handler with API integration."""
        if self.IsWarmingUp:
            return

        if not data.ContainsKey(self.symbol):
            return

        # Rate limiting
        if (self.api_last_call_time and
            (self.Time - self.api_last_call_time) < self.api_call_interval):
            return

        self.api_last_call_time = self.Time

        # Check if trading is paused
        if self.trading_paused_until and self.Time < self.trading_paused_until:
            return

        # Check API health
        if not self.api_healthy:
            self.Debug("⚠️ API unhealthy - skipping trading")
            return

        # Get current bar data
        bar = data[self.symbol]
        if not isinstance(bar, TradeBar):
            return

        current_price = float(bar.Close)

        # Update price window for SL/TP calculation
        self.price_window.append({
            "time": self.Time,
            "price": current_price,
            "volume": float(bar.Volume or 0)
        })

        # Get trading signal from API
        signal_data = self.GetTradingSignal()

        if not signal_data.get("success"):
            self.Error(f"Failed to get signal: {signal_data.get('error', 'Unknown')}")
            return

        signal = signal_data.get("signal", "HOLD")
        confidence = signal_data.get("confidence", 0)
        api_price = signal_data.get("price", 0)
        recommendation = signal_data.get("recommendation", {})

        # Log signal
        if self.pred_debug_counter % 6 == 0:  # Every 6 hours
            self.Debug(f"{self.Time} - Signal: {signal} (Conf: {confidence:.2f}, Price: {api_price})")
        self.pred_debug_counter += 1

        # Update trading parameters from API recommendation
        if recommendation:
            self.UpdateParametersFromRecommendation(recommendation)

        # Signal cooldown to avoid spam
        if (signal == self.last_signal and
            self.last_signal_time and
            (self.Time - self.last_signal_time) < self.signal_cooldown):
            return

        # Execute trading logic
        self.ExecuteSignal(signal, confidence, signal_data, current_price)

        # Update signal tracking
        self.last_signal = signal
        self.last_signal_time = self.Time

    def UpdateParametersFromRecommendation(self, recommendation: dict):
        """Update trading parameters based on API recommendation."""
        try:
            # Update thresholds if provided
            api_thresholds = recommendation.get('api_thresholds', {})
            if api_thresholds:
                self.prediction_buy_threshold = api_thresholds.get('buy_threshold', self.prediction_buy_threshold)
                self.prediction_sell_threshold = api_thresholds.get('sell_threshold', self.prediction_sell_threshold)

            # Update position sizing
            suggested_size = recommendation.get('suggested_position_size', 0)
            if suggested_size != 0:
                self.position_size_pct = min(abs(suggested_size), 0.95)  # Max 95%

            # Update SL/TP
            if self.entry_price and recommendation.get('stop_loss'):
                self.stop_loss_pct = abs(recommendation['stop_loss'] - self.entry_price) / self.entry_price

            if self.entry_price and recommendation.get('take_profit'):
                self.take_profit_pct = abs(recommendation['take_profit'] - self.entry_price) / self.entry_price

        except Exception as e:
            self.Error(f"UpdateParameters Error: {e}")

    def ExecuteSignal(self, signal: str, confidence: float, signal_data: dict, current_price: float):
        """Execute trading signal."""
        try:
            qty = self.Portfolio[self.symbol].Quantity

            # Check SL/TP first (priority)
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

            # HOLD signals - no action
            elif signal == "HOLD":
                self.Debug(f"HOLD signal (Conf: {confidence:.2f})")

        except Exception as e:
            self.Error(f"ExecuteSignal Error: {e}")

    def ExecuteBuy(self, confidence: float, signal_data: dict, current_price: float):
        """Execute buy order with API recommendation."""
        try:
            self.pending_entry = True

            # Get position size from API recommendation or default
            recommendation = signal_data.get('recommendation', {})
            api_position_size = recommendation.get('suggested_position_size', self.position_size_pct)

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
        """Check and execute SL/TP if triggered."""
        if self.pending_exit or not self.entry_price:
            return False

        try:
            pnl_pct = (price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0

            # Stop Loss
            if pnl_pct <= -self.stop_loss_pct:
                self.pending_exit = True
                self.last_exit_reason = "SL"
                self.Liquidate(self.symbol)

                msg = f"STOP LOSS TRIGGERED\\nEntry: {self.entry_price:,.2f}\\nExit: {price:,.2f}\\nPnL: {pnl_pct:.2%}"
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

                msg = f"TAKE PROFIT TRIGGERED\\nEntry: {self.entry_price:,.2f}\\nExit: {price:,.2f}\\nPnL: {pnl_pct:.2%}"
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
        try:
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

                self.trading_paused_until = self.Time + self.cooldown_period

                msg = f"RISK MANAGEMENT - Drawdown {dd:.2%} > {self.max_drawdown_pct:.2%}\\nTrading paused until {self.trading_paused_until}"
                self.SendSignal("RISK PAUSE", msg)

                self.high_watermark = pv
                return False

        except Exception as e:
            self.Error(f"CheckMaxDrawdownAndCooldown Error: {e}")

        return True

    def PauseTrading(self, reason: str):
        """Pause trading due to API issues or other reasons."""
        self.trading_paused_until = self.Time + timedelta(hours=1)
        msg = f"TRADING PAUSED\\nReason: {reason}\\nResume: {self.trading_paused_until}"
        self.SendSignal("PAUSED", msg)

    # =========================================================
    # NOTIFICATION METHODS
    # =========================================================
    def SendSignal(self, title: str, message: str):
        """Send signal notification."""
        self.Debug(f"[SIGNAL] {title}\\n{message}")

        if not self.LiveMode or not self.enable_live_notifications:
            return

        try:
            if self.telegram_token and self.telegram_chat_id:
                self.Notify.Telegram(self.telegram_token, self.telegram_chat_id, f"{title}\\n{message}")
        except Exception as e:
            self.Error(f"SendSignal Error: {e}")

    def SendTelegramNotification(self, message: str):
        """Send Telegram notification."""
        if self.LiveMode and self.telegram_token and self.telegram_chat_id:
            try:
                self.Notify.Telegram(self.telegram_token, self.telegram_chat_id, message)
            except Exception as e:
                self.Error(f"Telegram Error: {e}")

    def FormatBuyMessage(self, confidence: float, signal_data: dict, price: float, position_size: float) -> str:
        """Format buy signal message."""
        recommendation = signal_data.get('recommendation', {})

        lines = [
            "🟢 **BUY SIGNAL**",
            f"Symbol: {self.symbol_str}",
            f"Price: ${price:,.2f}",
            f"Confidence: {confidence:.1%}",
            f"Position Size: {position_size:.1%}",
            ""
        ]

        if recommendation:
            lines.extend([
                "📋 **Recommendation:**",
                f"Risk Level: {recommendation.get('risk_level', 'Unknown')}",
                f"Holding Period: {recommendation.get('holding_period', 'Unknown')}",
                ""
            ])

            if recommendation.get('stop_loss'):
                lines.append(f"Stop Loss: ${recommendation['stop_loss']:,.2f}")
            if recommendation.get('take_profit'):
                lines.append(f"Take Profit: ${recommendation['take_profit']:,.2f}")

        lines.extend([
            f"Time: {self.Time}",
            f"Source: XGBoost API"
        ])

        return "\\n".join(lines)

    def FormatSellMessage(self, confidence: float, signal_data: dict, price: float, pnl: float, pnl_pct: float) -> str:
        """Format sell signal message."""
        lines = [
            "🔴 **SELL SIGNAL**",
            f"Symbol: {self.symbol_str}",
            f"Price: ${price:,.2f}",
            f"Confidence: {confidence:.1%}",
            ""
        ]

        if self.entry_price:
            lines.extend([
                f"Entry Price: ${self.entry_price:,.2f}",
                f"PnL: ${pnl:,.2f} ({pnl_pct:.2%})",
                ""
            ])

        lines.extend([
            f"Reason: {self.last_exit_reason or 'API Signal'}",
            f"Time: {self.Time}",
            f"Source: XGBoost API"
        ])

        return "\\n".join(lines)

    def OnEndOfAlgorithm(self):
        """Algorithm end."""
        final_msg = (
            f"Algorithm Completed\\n"
            f"Final Portfolio: ${self.Portfolio.TotalPortfolioValue:,.2f}\\n"
            f"API Health: {'Healthy' if self.api_healthy else 'Unhealthy'}\\n"
            f"API Failures: {self.api_failure_count}"
        )
        self.Debug(final_msg)

        if self.LiveMode:
            self.SendTelegramNotification(final_msg)