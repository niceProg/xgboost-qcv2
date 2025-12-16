#!/usr/bin/env python3
"""Debug database connection and data availability."""

import os
import pymysql
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def debug_database():
    """Debug database connection and check data."""

    # Database config
    config = {
        'host': os.getenv('TRADING_DB_HOST'),
        'port': int(os.getenv('TRADING_DB_PORT', 3306)),
        'database': os.getenv('TRADING_DB_NAME'),
        'user': os.getenv('TRADING_DB_USER'),
        'password': os.getenv('TRADING_DB_PASSWORD')
    }

    # Print config being used
    print(f"\nUsing database: {config['database']}")
    print(f"Using user: {config['user']}")
    print(f"Using host: {config['host']}\n")

    print("Database Config:")
    for k, v in config.items():
        if k == 'password':
            print(f"  {k}: {'*' * len(str(v))}")
        else:
            print(f"  {k}: {v}")
    print()

    try:
        # Connect
        conn = pymysql.connect(**config)
        print("✅ Database connection successful!")

        cursor = conn.cursor()

        # Check tables
        tables = [
            'cg_spot_price_history',
            'cg_funding_rate_history',
            'cg_futures_basis_history',
            'cg_long_short_global_account_ratio_history',
            'cg_long_short_top_account_ratio_history'
        ]

        for table in tables:
            print(f"\n=== {table} ===")

            # Check if table exists
            cursor.execute(f"SHOW TABLES LIKE '{table}'")
            if not cursor.fetchone():
                print(f"❌ Table {table} does not exist!")
                continue

            # Check data count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            total_count = cursor.fetchone()[0]
            print(f"Total rows: {total_count}")

            if total_count > 0:
                # Check recent data
                if 'price' in table:
                    cursor.execute(f"""
                        SELECT exchange, symbol, interval,
                               COUNT(*) as count,
                               MIN(time) as oldest,
                               MAX(time) as newest
                        FROM {table}
                        WHERE symbol LIKE '%BTC%'
                        GROUP BY exchange, symbol, interval
                        ORDER BY newest DESC
                        LIMIT 5
                    """)
                else:
                    cursor.execute(f"""
                        SELECT exchange, pair, interval,
                               COUNT(*) as count,
                               MIN(time) as oldest,
                               MAX(time) as newest
                        FROM {table}
                        WHERE pair LIKE '%BTC%'
                        GROUP BY exchange, pair, interval
                        ORDER BY newest DESC
                        LIMIT 5
                    """)

                results = cursor.fetchall()
                if results:
                    print("Recent BTC data:")
                    for row in results:
                        print(f"  {row[0]}/{row[1]}/{row[2]}: {row[3]} rows")
                        print(f"    Time range: {row[4]} to {row[5]}")
                else:
                    print("❌ No BTC data found")

            # Check today's data
            today_start = int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
            today_end = int(datetime.now().timestamp() * 1000)

            if 'price' in table:
                cursor.execute(f"""
                    SELECT COUNT(*)
                    FROM {table}
                    WHERE time >= {today_start} AND time <= {today_end}
                    AND symbol = 'BTCUSDT' AND exchange = 'binance'
                """)
            else:
                cursor.execute(f"""
                    SELECT COUNT(*)
                    FROM {table}
                    WHERE time >= {today_start} AND time <= {today_end}
                    AND pair = 'BTCUSDT' AND exchange = 'binance'
                """)

            today_count = cursor.fetchone()[0]
            print(f"Today's BTCUSDT data: {today_count} rows")

        # Test specific query
        print("\n=== Testing Daily Mode Query ===")
        trading_hours_start = datetime.now().replace(hour=7, minute=0, second=0, microsecond=0)
        trading_hours_end = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)

        start_timestamp = int(trading_hours_start.timestamp() * 1000)
        end_timestamp = int(trading_hours_end.timestamp() * 1000)

        print(f"Querying data between {start_timestamp} and {end_timestamp}")
        print(f"Time range: {trading_hours_start} to {trading_hours_end}")

        cursor.execute(f"""
            SELECT COUNT(*), MIN(time), MAX(time)
            FROM cg_spot_price_history
            WHERE time >= {start_timestamp} AND time <= {end_timestamp}
            AND exchange = 'binance' AND symbol = 'BTCUSDT'
        """)

        result = cursor.fetchone()
        print(f"Found {result[0]} rows")
        if result[0] > 0:
            print(f"Data time range: {result[1]} to {result[2]}")

        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_database()