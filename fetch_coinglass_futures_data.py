#!/usr/bin/env python3
"""
Fetch Coinglass Futures Price Data for QuantConnect

This script fetches BTC futures price data from Coinglass API
and converts it to QuantConnect LEAN CSV format.

Usage:
    python fetch_coinglass_futures_data.py

Output:
    data/quantconnect/BINANCE_BTCUSDT_FUTURES_hour.csv
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path

# Coinglass API Configuration
COINGLASS_API_BASE = "https://open-api-v4.coinglass.com"
EXCHANGE = "Binance"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"  # 1 hour

# Output directory
OUTPUT_DIR = Path("./data/quantconnect")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_coinglass_futures_data(start_date, end_date):
    """
    Fetch futures price data from Coinglass API.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        DataFrame with price data
    """
    print(f"Fetching Coinglass futures data...")
    print(f"Exchange: {EXCHANGE}")
    print(f"Symbol: {SYMBOL}")
    print(f"Interval: {INTERVAL}")
    print(f"Date range: {start_date} to {end_date}")

    all_data = []
    current_date = pd.to_datetime(start_date)

    end_dt = pd.to_datetime(end_date)

    while current_date <= end_dt:
        date_str = current_date.strftime("%Y-%m-%d")

        try:
            # Coinglass API: /api/futures/price/history
            url = f"{COINGLASS_API_BASE}/api/futures/price/history"
            params = {
                "exchange": EXCHANGE,
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startDate": date_str,
                "endDate": date_str
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get("success") and data.get("data"):
                # Parse data
                for item in data["data"]:
                    all_data.append({
                        "time": pd.to_datetime(item["time"], unit="ms"),
                        "open": float(item["open"]),
                        "high": float(item["high"]),
                        "low": float(item["low"]),
                        "close": float(item["close"]),
                        "volume": float(item.get("volume", 0)),
                        "volumeCcy": float(item.get("volumeCcy", 0))
                    })
                print(f"✓ Fetched {date_str}: {len(data['data'])} records")
            else:
                print(f"⚠ No data for {date_str}")

        except Exception as e:
            print(f"✗ Error fetching {date_str}: {e}")

        # Move to next day
        current_date += timedelta(days=1)

    if not all_data:
        print("No data fetched!")
        return None

    df = pd.DataFrame(all_data)
    df = df.sort_values("time").drop_duplicates(subset=["time"])

    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")

    return df

def convert_to_lean_csv(df):
    """
    Convert DataFrame to QuantConnect LEAN CSV format.

    LEAN CSV format for crypto:
    date,time,open,high,low,close,volume
    20240101,0000,45000.0,45500.0,44800.0,45200.0,1000.0
    """
    print("\nConverting to LEAN CSV format...")

    # Create LEAN format columns
    lean_df = pd.DataFrame()
    lean_df["date"] = df["time"].dt.strftime("%Y%m%d")
    lean_df["time"] = df["time"].dt.strftime("%H%M")
    lean_df["open"] = df["open"]
    lean_df["high"] = df["high"]
    lean_df["low"] = df["low"]
    lean_df["close"] = df["close"]
    lean_df["volume"] = df["volume"]

    # Output filename
    output_file = OUTPUT_DIR / f"{EXCHANGE}_{SYMBOL}_FUTURES_hour.csv"

    # Save to CSV (no header for LEAN format)
    lean_df.to_csv(output_file, index=False, header=False)

    print(f"✓ Saved to: {output_file}")
    print(f"Total rows: {len(lean_df)}")

    return output_file

def main():
    """Main function."""
    print("=" * 60)
    print("Coinglass Futures Data Fetcher for QuantConnect")
    print("=" * 60)
    print()

    # Set date range (adjust as needed)
    start_date = "2024-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Fetch data
    df = fetch_coinglass_futures_data(start_date, end_date)

    if df is not None:
        # Convert to LEAN CSV
        output_file = convert_to_lean_csv(df)

        print()
        print("=" * 60)
        print("Done!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Upload the CSV file to QuantConnect")
        print(f"   File location: {output_file}")
        print("2. In QuantConnect Lab, go to Data > Custom Data")
        print("3. Upload the file and configure as crypto data")
        print()
    else:
        print("Failed to fetch data!")

if __name__ == "__main__":
    main()
