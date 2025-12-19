#!/usr/bin/env python3
"""
Model evaluation with trading performance metrics + trade logs + leverage (margin simulation).

Outputs (in output_dir):
- rekening_koran.csv        : Saldo = EQUITY (cash + locked_margin + unrealized_pnl)
- rekening_koran_cash.csv   : Saldo = CASH after each event (can drop, but equity won't)
- trade_events.csv          : BUY/SELL events with qty, cash_after, equity_after, leverage, margin_used
- trades.csv                : paired trades with pnl

Leverage model (simple, transparent):
- On BUY: lock margin = cash * MARGIN_FRACTION (or less if cash smaller), notional = margin * LEVERAGE
- Qty = notional / price
- Fees are applied to notional (buy and sell)
- On SELL: release margin + realized PnL back to cash (minus fee)
- No interest/funding is modeled.

Important:
- If you previously saw Saldo=0 on BUY, you were likely viewing the CASH statement.
  Use rekening_koran.csv (EQUITY) to avoid the misleading "0 cash" look.
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Import our command line options handler (from your project)
from command_line_options import parse_arguments, validate_arguments, DataFilter

load_dotenv()


import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained model with trading metrics."""

    def __init__(self, data_filter: DataFilter, output_dir: str = "./output_train"):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.model = None

    def load_model_and_data(self) -> pd.DataFrame:
        logger.info("Loading model and test data...")

        # Try models directory first (new structure), then root (compatibility)
        models_dir = self.output_dir / 'models'
        model_path = models_dir / "latest_model.joblib"

        if not model_path.exists():
            # Fallback to root directory for backward compatibility
            model_path = self.output_dir / "latest_model.joblib"

        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            logger.error(f"Checked: {models_dir / 'latest_model.joblib'} and {self.output_dir / 'latest_model.joblib'}")
            sys.exit(1)

        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Try datasets directory first, then root
        datasets_dir = self.output_dir / 'datasets'
        labeled_file = datasets_dir / "labeled_data.parquet"

        if not labeled_file.exists():
            # Fallback to root directory for backward compatibility
            labeled_file = self.output_dir / "labeled_data.parquet"

        if not labeled_file.exists():
            logger.error(f"Labeled data not found: {labeled_file}")
            logger.error(f"Checked: {datasets_dir / 'labeled_data.parquet'} and {self.output_dir / 'labeled_data.parquet'}")
            sys.exit(1)

        df = pd.read_parquet(labeled_file)
        logger.info(f"Loaded {len(df)} samples from labeled data")
        return df

    def create_trading_signals(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        logger.info("Creating trading signals...")

        training_features_file = self.output_dir / "training_features.txt"
        if not training_features_file.exists():
            logger.error(f"Training features file not found: {training_features_file}")
            sys.exit(1)

        with open(training_features_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines()]

        feature_cols = [ln for ln in lines if ln and ln not in ("Training Features:", "====================")]

        available_features = [c for c in feature_cols if c in df.columns]
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            logger.warning(f"Missing features in data: {missing_features}")

        if not available_features:
            logger.error("No training features found in data")
            sys.exit(1)

        X = df[available_features].fillna(0)
        df["model_prediction"] = self.model.predict_proba(X)[:, 1]
        df["trading_signal"] = (df["model_prediction"] > threshold).astype(int)

        counts = df["trading_signal"].value_counts()
        pct = df["trading_signal"].value_counts(normalize=True) * 100
        logger.info("Trading Signal Distribution:")
        logger.info(f"Buy signals (1): {counts.get(1,0)} ({pct.get(1,0):.1f}%)")
        logger.info(f"Neutral/Sell signals (0): {counts.get(0,0)} ({pct.get(0,0):.1f}%)")

        return df

    def calculate_returns(self, df: pd.DataFrame, exposure_mult: float = 1.0) -> pd.DataFrame:
        """Calculate returns based on trading signals and actual price movements."""
        logger.info("Calculating trading returns...")

        # Ensure group keys exist & non-null (otherwise groupby drops NaN keys)
        for col in ["exchange", "symbol", "interval"]:
            if col not in df.columns:
                df[col] = "UNKNOWN"
            df[col] = df[col].fillna("UNKNOWN")

        df = df.sort_values(["exchange", "symbol", "interval", "time"]).copy()
        df["next_close"] = df.groupby(["exchange", "symbol", "interval"])["price_close"].shift(-1)
        df["actual_return"] = (df["next_close"] - df["price_close"]) / df["price_close"]

        # Return on equity approximation with exposure multiplier (e.g., leverage * margin_fraction)
        df["strategy_return"] = df["trading_signal"] * df["actual_return"] * float(exposure_mult)

        df = df.dropna(subset=["next_close", "actual_return", "strategy_return"])

        df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()

        df["benchmark_return"] = df["actual_return"]
        df["cumulative_benchmark"] = (1 + df["benchmark_return"]).cumprod()

        logger.info(f"Calculated returns for {len(df)} trading periods")
        return df

    # ----------------------------
    # Leverage / margin simulation
    # ----------------------------
    def build_trade_logs_with_leverage(
        self,
        df: pd.DataFrame,
        initial_cash: float,
        leverage: float,
        margin_fraction: float,
        fee_rate: float = 0.0,
        slippage_rate: float = 0.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build:
          - events_df: BUY/SELL events
          - trades_df: paired trades
          - statement_df: rekening koran rows with both cash and equity after each event

        SaldoEquity = cash + locked_margin + unrealized_pnl
        Saldo (cash column) is just cash after event (for rekening_koran_cash.csv)
        """

        d = df.copy()

        if "time" not in d.columns or "price_close" not in d.columns or "trading_signal" not in d.columns:
            raise ValueError("df must contain columns: time, price_close, trading_signal")

        for col in ["exchange", "symbol", "interval"]:
            if col not in d.columns:
                d[col] = "UNKNOWN"
            d[col] = d[col].fillna("UNKNOWN")

        # Ensure time is datetime if possible (nice CSV)
        if not np.issubdtype(d["time"].dtype, np.datetime64):
            try:
                d["time"] = pd.to_datetime(d["time"])
            except Exception:
                pass

        d = d.sort_values(["exchange", "symbol", "interval", "time"]).copy()

        event_rows = []
        trade_rows = []
        statement_rows = []

        for (ex, sym, itv), g in d.groupby(["exchange", "symbol", "interval"], sort=False):
            g = g.sort_values("time").reset_index(drop=True)

            # Initial row like a bank statement
            seq = 0
            statement_rows.append({
                "exchange": ex, "symbol": sym, "interval": itv,
                "_seq": seq,
                "TimeStamp": "-", "Signal": "-", "Price": "-", "Quantity": "-",
                "Saldo": float(initial_cash),          # cash
                "SaldoEquity": float(initial_cash),    # equity
            })
            seq += 1

            cash = float(initial_cash)
            locked_margin = 0.0
            qty = 0.0
            entry_price = None
            entry_time = None
            entry_margin = 0.0

            prev_sig = g["trading_signal"].shift(1).fillna(0).astype(int)
            sig = g["trading_signal"].fillna(0).astype(int)

            buy_idx = g.index[(prev_sig == 0) & (sig == 1)].tolist()
            sell_idx = g.index[(prev_sig == 1) & (sig == 0)].tolist()
            if len(buy_idx) > len(sell_idx):
                sell_idx.append(g.index[-1])

            logger.info(f"[{ex} {sym} {itv}] transitions: buy={len(buy_idx)} sell={len(sell_idx)} rows={len(g)}")
            logger.info(f"[{ex} {sym} {itv}] leverage={leverage} margin_fraction={margin_fraction} initial_cash={initial_cash}")

            for bi, si in zip(buy_idx, sell_idx):
                # BUY
                t_buy = g.loc[bi, "time"]
                mid_buy = float(g.loc[bi, "price_close"])
                buy_price = mid_buy * (1.0 + float(slippage_rate))

                # Margin to lock this trade
                margin_to_lock = max(0.0, min(cash, cash * float(margin_fraction)))
                if margin_to_lock <= 0 or buy_price <= 0 or leverage <= 0:
                    continue

                notional = margin_to_lock * float(leverage)
                qty = notional / buy_price

                buy_fee = notional * float(fee_rate)
                required_cash = margin_to_lock + buy_fee

                if cash < required_cash or qty <= 0:
                    continue

                cash -= required_cash
                locked_margin = margin_to_lock
                entry_margin = margin_to_lock
                entry_price = buy_price
                entry_time = t_buy

                equity_after = cash + locked_margin  # at entry pnl=0

                event_rows.append({
                    "exchange": ex, "symbol": sym, "interval": itv,
                    "time": t_buy, "signal": "buy",
                    "price": buy_price, "quantity": qty,
                    "leverage": float(leverage),
                    "margin_used": margin_to_lock,
                    "notional": notional,
                    "fee": buy_fee,
                    "cash_after": cash,
                    "equity_after": equity_after,
                })

                statement_rows.append({
                    "exchange": ex, "symbol": sym, "interval": itv,
                    "_seq": seq,
                    "TimeStamp": t_buy, "Signal": "Buy",
                    "Price": buy_price, "Quantity": qty,
                    "Saldo": cash,                 # cash after buy
                    "SaldoEquity": equity_after,   # equity after buy
                })
                seq += 1

                # SELL
                t_sell = g.loc[si, "time"]
                mid_sell = float(g.loc[si, "price_close"])
                sell_price = mid_sell * (1.0 - float(slippage_rate))

                if entry_price is None or qty <= 0:
                    continue

                pnl = (sell_price - entry_price) * qty
                sell_notional = sell_price * qty
                sell_fee = sell_notional * float(fee_rate)

                # Release margin + realized pnl (minus fee)
                cash += (entry_margin + pnl - sell_fee)
                locked_margin = 0.0

                equity_after = cash  # after close, no position

                event_rows.append({
                    "exchange": ex, "symbol": sym, "interval": itv,
                    "time": t_sell, "signal": "sell",
                    "price": sell_price, "quantity": qty,
                    "leverage": float(leverage),
                    "margin_used": entry_margin,
                    "notional": sell_notional,
                    "fee": sell_fee,
                    "cash_after": cash,
                    "equity_after": equity_after,
                })

                statement_rows.append({
                    "exchange": ex, "symbol": sym, "interval": itv,
                    "_seq": seq,
                    "TimeStamp": t_sell, "Signal": "Sell",
                    "Price": sell_price, "Quantity": qty,
                    "Saldo": cash,
                    "SaldoEquity": equity_after,
                })
                seq += 1

                pnl_pct_on_margin = (pnl / entry_margin) if entry_margin != 0 else 0.0

                trade_rows.append({
                    "exchange": ex, "symbol": sym, "interval": itv,
                    "side": "long",
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": t_sell,
                    "exit_price": sell_price,
                    "quantity": qty,
                    "leverage": float(leverage),
                    "margin_used": entry_margin,
                    "pnl": pnl,
                    "pnl_pct_on_margin": pnl_pct_on_margin,
                })

                # reset position
                qty = 0.0
                entry_price = None
                entry_time = None
                entry_margin = 0.0

        events_df = pd.DataFrame(event_rows)
        trades_df = pd.DataFrame(trade_rows)
        stmt_df = pd.DataFrame(statement_rows)

        if not events_df.empty:
            events_df = events_df.sort_values(["exchange", "symbol", "interval", "time"]).reset_index(drop=True)
        if not trades_df.empty:
            trades_df = trades_df.sort_values(["exchange", "symbol", "interval", "entry_time"]).reset_index(drop=True)
        if not stmt_df.empty:
            stmt_df = stmt_df.sort_values(["exchange", "symbol", "interval", "_seq"]).reset_index(drop=True)

        return events_df, trades_df, stmt_df

    def save_trade_logs_with_leverage(
        self,
        df_with_signals: pd.DataFrame,
        initial_cash: float,
        leverage: float,
        margin_fraction: float,
        fee_rate: float,
        slippage_rate: float,
    ) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        events_df, trades_df, stmt_df = self.build_trade_logs_with_leverage(
            df=df_with_signals,
            initial_cash=initial_cash,
            leverage=leverage,
            margin_fraction=margin_fraction,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )

        events_path = self.output_dir / "trade_events.csv"
        trades_path = self.output_dir / "trades.csv"
        stmt_equity_path = self.output_dir / "rekening_koran.csv"
        stmt_cash_path = self.output_dir / "rekening_koran_cash.csv"

        events_df.to_csv(events_path, index=False)
        trades_df.to_csv(trades_path, index=False)

        if stmt_df.empty:
            stmt_df = pd.DataFrame([{
                "TimeStamp": "-", "Signal": "-", "Price": "-", "Quantity": "-",
                "Saldo": float(initial_cash), "SaldoEquity": float(initial_cash)
            }])

        # Cash statement
        stmt_df[["TimeStamp", "Signal", "Price", "Quantity", "Saldo"]].to_csv(stmt_cash_path, index=False)

        # Equity statement
        equity_df = stmt_df[["TimeStamp", "Signal", "Price", "Quantity", "SaldoEquity"]].rename(columns={"SaldoEquity": "Saldo"})
        equity_df.to_csv(stmt_equity_path, index=False)

        logger.info(f"Saved trade events to: {events_path} (rows={len(events_df)})")
        logger.info(f"Saved paired trades to: {trades_path} (rows={len(trades_df)})")
        logger.info(f"Saved account statement (EQUITY) to: {stmt_equity_path} (rows={len(equity_df)})")
        logger.info(f"Saved account statement (CASH) to: {stmt_cash_path} (rows={len(stmt_df)})")

        return {
            "trade_events_csv": str(events_path),
            "trades_csv": str(trades_path),
            "rekening_koran_equity_csv": str(stmt_equity_path),
            "rekening_koran_cash_csv": str(stmt_cash_path),
        }

    # ----------------------------
    # Performance metrics & plots
    # ----------------------------
    def calculate_performance_metrics(self, df: pd.DataFrame) -> dict:
        logger.info("Calculating performance metrics...")
        if df.empty:
            logger.error("No data for performance calculation")
            return {}

        total_return = df["cumulative_return"].iloc[-1] - 1
        benchmark_return = df["cumulative_benchmark"].iloc[-1] - 1

        time_span = (df["time"].max() - df["time"].min()).days
        if time_span == 0:
            time_span = 1

        cagr = (df["cumulative_return"].iloc[-1] ** (365 / time_span)) - 1
        benchmark_cagr = (df["cumulative_benchmark"].iloc[-1] ** (365 / time_span)) - 1

        peak = df["cumulative_return"].expanding().max()
        drawdown = (df["cumulative_return"] - peak) / peak
        max_drawdown = drawdown.min()

        benchmark_peak = df["cumulative_benchmark"].expanding().max()
        benchmark_drawdown = (df["cumulative_benchmark"] - benchmark_peak) / benchmark_peak
        benchmark_max_drawdown = benchmark_drawdown.min()

        strategy_returns = df["strategy_return"].dropna()
        if len(strategy_returns) > 1 and strategy_returns.std() > 0:
            sharpe_ratio = (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0.0

        winning_trades = (df["strategy_return"] > 0).sum()
        total_trades = (df["trading_signal"] == 1).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        avg_win = df[df["strategy_return"] > 0]["strategy_return"].mean() if winning_trades > 0 else 0.0
        avg_loss = df[df["strategy_return"] < 0]["strategy_return"].mean() if (total_trades - winning_trades) > 0 else 0.0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        metrics = {
            "total_return": float(total_return),
            "benchmark_return": float(benchmark_return),
            "cagr": float(cagr),
            "benchmark_cagr": float(benchmark_cagr),
            "max_drawdown": float(max_drawdown),
            "benchmark_max_drawdown": float(benchmark_max_drawdown),
            "sharpe_ratio": float(sharpe_ratio),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else float("inf"),
            "total_trades": int(total_trades),
            "winning_trades": int(winning_trades),
            "losing_trades": int(total_trades - winning_trades),
            "avg_win": float(avg_win) if not np.isnan(avg_win) else 0.0,
            "avg_loss": float(avg_loss) if not np.isnan(avg_loss) else 0.0,
            "trading_days": int(time_span),
        }

        logger.info("Performance Summary:")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v}")

        return metrics

    def create_performance_plots(self, df: pd.DataFrame, metrics: dict):
        logger.info("Creating performance plots...")
        if df.empty:
            logger.warning("No data for plotting")
            return

        plt.style.use("default")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.plot(df["time"], df["cumulative_return"], label="Strategy", linewidth=2)
        ax1.plot(df["time"], df["cumulative_benchmark"], label="Benchmark", linewidth=2)
        ax1.set_title("Cumulative Returns")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Return")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        peak = df["cumulative_return"].expanding().max()
        drawdown = (df["cumulative_return"] - peak) / peak
        ax2.fill_between(df["time"], drawdown, 0, alpha=0.3)
        ax2.set_title("Drawdown")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Drawdown")
        ax2.grid(True, alpha=0.3)

        returns = df["strategy_return"].dropna()
        if len(returns) > 0:
            ax3.hist(returns, bins=50, alpha=0.7)
        ax3.set_title("Returns Distribution")
        ax3.set_xlabel("Return")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)

        metrics_data = [
            ["Total Return", f"{metrics.get('total_return', 0):.2%}"],
            ["CAGR", f"{metrics.get('cagr', 0):.2%}"],
            ["Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}"],
            ["Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"],
            ["Win Rate", f"{metrics.get('win_rate', 0):.2%}"],
            ["Total Trades", f"{metrics.get('total_trades', 0)}"],
        ]
        table = ax4.table(cellText=metrics_data, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title("Performance Summary", pad=20)
        ax4.axis("off")

        plt.tight_layout()
        out_path = self.output_dir / "performance_analysis.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info(f"Performance plots saved to {out_path}")

    def generate_detailed_report(self, df: pd.DataFrame, metrics: dict) -> dict:
        logger.info("Generating detailed report...")
        if df.empty:
            return {"strategy_summary": {}, "benchmark_comparison": {}, "risk_metrics": {}}

        daily_returns = df["strategy_return"].dropna()
        report = {
            "strategy_summary": {
                "trading_period": {
                    "start_date": df["time"].min().isoformat() if hasattr(df["time"].min(), "isoformat") else str(df["time"].min()),
                    "end_date": df["time"].max().isoformat() if hasattr(df["time"].max(), "isoformat") else str(df["time"].max()),
                    "total_days": metrics.get("trading_days", 0),
                },
                "performance_metrics": {
                    "total_return": metrics.get("total_return", 0),
                    "annualized_return": metrics.get("cagr", 0),
                    "max_drawdown": metrics.get("max_drawdown", 0),
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                    "win_rate": metrics.get("win_rate", 0),
                    "profit_factor": metrics.get("profit_factor", 0),
                },
                "trading_activity": {
                    "total_trades": metrics.get("total_trades", 0),
                    "winning_trades": metrics.get("winning_trades", 0),
                    "losing_trades": metrics.get("losing_trades", 0),
                    "avg_win": metrics.get("avg_win", 0),
                    "avg_loss": metrics.get("avg_loss", 0),
                },
            },
            "benchmark_comparison": {
                "benchmark_return": metrics.get("benchmark_return", 0),
                "benchmark_cagr": metrics.get("benchmark_cagr", 0),
                "benchmark_max_drawdown": metrics.get("benchmark_max_drawdown", 0),
                "outperformance": metrics.get("cagr", 0) - metrics.get("benchmark_cagr", 0),
            },
            "risk_metrics": {
                "volatility": float(daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 1 else 0.0,
                "var_95": float(daily_returns.quantile(0.05)) if len(daily_returns) > 0 else 0.0,
                "skewness": float(daily_returns.skew()) if len(daily_returns) > 2 else 0.0,
                "kurtosis": float(daily_returns.kurtosis()) if len(daily_returns) > 2 else 0.0,
            },
        }
        return report

    def save_evaluation_results(self, df: pd.DataFrame, metrics: dict, report: dict):
        logger.info("Saving evaluation results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        metrics_file = self.output_dir / f"performance_metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        report_file = self.output_dir / f"performance_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        results_file = self.output_dir / "trading_results.parquet"
        df.to_parquet(results_file, index=False)

        logger.info(f"Saved metrics to {metrics_file}")
        logger.info(f"Saved report to {report_file}")
        logger.info(f"Saved trading results to {results_file}")

        # Save to database if enabled
        if os.getenv('ENABLE_DB_STORAGE', 'true').lower() == 'true':
            self.save_evaluation_to_database(metrics, report, timestamp)
            # DATABASE STORAGE DISABLED per client requirement
            # self.update_session_trading_metrics(metrics, report)

    def save_evaluation_to_database(self, metrics: dict, report: dict, timestamp: str):
        """Save evaluation results to xgboostqc database."""
        # DATABASE STORAGE DISABLED per client requirement
        # Client tidak mau: xgboost_evaluations, xgboost_features, xgboost_training_sessions
        logger.info("📝 Evaluation database storage disabled per client requirement")

    # METHOD DISABLED per client requirement
    def update_session_trading_metrics(self, metrics: dict, report: dict):
        """Update training session with trading performance metrics."""
        logger.info("❌ Session trading metrics update disabled per client requirement")


def main():
    args = parse_arguments()
    validate_arguments(args)

    data_filter = DataFilter(args)
    evaluator = ModelEvaluator(data_filter, args.output_dir)

    # Config via ENV (so you don't need to modify CLI parser)
    threshold = float(os.getenv("THRESHOLD", "0.5"))  # Default threshold of 0.5
    initial_cash = float(os.getenv("INITIAL_CASH", "1000"))  # Default initial cash

    # Leverage settings
    leverage = float(os.getenv("LEVERAGE", "10"))                   # e.g. 5x, default 10x
    margin_fraction = float(os.getenv("MARGIN_FRACTION", "0.2"))   # use 20% of cash as margin per trade
    # Trading friction
    fee_rate = float(os.getenv("FEE_RATE", "0.0004"))              # example taker fee
    slippage_rate = float(os.getenv("SLIPPAGE_RATE", "0"))

    # Return exposure multiplier approximation (for CAGR/drawdown metrics)
    exposure_mult = leverage * margin_fraction

    try:
        logger.info("Loading model and data for evaluation...")
        df = evaluator.load_model_and_data()

        df = evaluator.create_trading_signals(df, threshold=threshold)

        evaluator.save_trade_logs_with_leverage(
            df_with_signals=df,
            initial_cash=initial_cash,
            leverage=leverage,
            margin_fraction=margin_fraction,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )

        df = evaluator.calculate_returns(df, exposure_mult=exposure_mult)

        metrics = evaluator.calculate_performance_metrics(df)
        evaluator.create_performance_plots(df, metrics)
        report = evaluator.generate_detailed_report(df, metrics)
        evaluator.save_evaluation_results(df, metrics, report)

        logger.info("\n=== Model Evaluation Complete ===")
        logger.info("Generated files:")
        logger.info(f"  - {evaluator.output_dir / 'rekening_koran.csv'} (EQUITY)")
        logger.info(f"  - {evaluator.output_dir / 'rekening_koran_cash.csv'} (CASH)")
        logger.info(f"  - {evaluator.output_dir / 'trade_events.csv'}")
        logger.info(f"  - {evaluator.output_dir / 'trades.csv'}")

        logger.info("\nLeverage settings used:")
        logger.info(f"  LEVERAGE={leverage}")
        logger.info(f"  MARGIN_FRACTION={margin_fraction}")
        logger.info(f"  INITIAL_CASH={initial_cash}")
        logger.info(f"  FEE_RATE={fee_rate}")
        logger.info(f"  THRESHOLD={threshold}")

    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
