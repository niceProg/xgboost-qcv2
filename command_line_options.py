#!/usr/bin/env python3
"""
Command-line options handler for XGBoost trading pipeline.
Provides filtering capabilities for exchange, pair, interval, time, and days.
"""

import argparse
from typing import Optional, List
import sys
from datetime import datetime, timedelta

def parse_arguments():
    """Parse command-line arguments for the XGBoost pipeline."""
    parser = argparse.ArgumentParser(
        description="XGBoost Trading Pipeline with Database Filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python load_database.py --exchange binance --pair BTCUSDT --interval 1h
  python load_database.py --exchange binance,okx --pair BTCUSDT --interval 1h,4h --days 30
  python load_database.py --time 1700000000000,1701000000000
  python load_database.py --initial --exchange binance --pair BTCUSDT --interval 1h
  python load_database.py --daily --exchange binance --pair BTCUSDT --interval 1h
        """
    )

    # Exchange filtering
    parser.add_argument(
        '--exchange',
        type=str,
        help='Exchange name(s). Use comma for multiple: binance,okx,bybit'
    )

    # Pair/Symbol filtering
    parser.add_argument(
        '--pair',
        type=str,
        help='Trading pair(s). Use comma for multiple: BTCUSDT,ETHUSDT'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        help='Trading symbol(s). Use comma for multiple: BTC,ETH'
    )

    # Interval filtering
    parser.add_argument(
        '--interval',
        type=str,
        help='Time interval(s). Use comma for multiple: 1m,5m,1h,4h'
    )

    # Time range filtering
    parser.add_argument(
        '--time',
        type=str,
        help='Time range as Unix timestamps (ms): start_time,end_time'
    )

    parser.add_argument(
        '--minutes',
        type=int,
        help='Number of recent minutes to include for real-time updates'
    )

    parser.add_argument(
        '--days',
        type=int,
        help='Number of recent days to include'
    )

  
    # Mode selection
    parser.add_argument(
        '--initial',
        action='store_true',
        help='Load initial historical data from 2024 onwards (use with specific time range if needed)'
    )

    parser.add_argument(
        '--daily',
        action='store_true',
        help='Load current day data only (from 00:00 today to now)'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output_train',
        help='Output directory for processed data (default: ./output_train)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()

class DataFilter:
    """Handles data filtering based on command-line arguments."""

    def __init__(self, args):
        self.args = args
        self.exchange_filter = self._parse_comma_list(args.exchange) if args.exchange else None
        self.pair_filter = self._parse_comma_list(args.pair) if args.pair else None
        self.symbol_filter = self._parse_comma_list(args.symbol) if args.symbol else None
        self.interval_filter = self._parse_comma_list(args.interval) if args.interval else None
        self.time_range = self._parse_time_range(args.time) if args.time else None
        self.days_filter = args.days
        self.minutes_filter = args.minutes  # New: minutes filter for real-time
        self.initial_mode = args.initial if hasattr(args, 'initial') else False
        self.daily_mode = args.daily if hasattr(args, 'daily') else False

        # Process minutes filter into time_range
        if self.minutes_filter and not self.time_range:
            self.time_range = self._get_minutes_time_range(self.minutes_filter)

    def _parse_comma_list(self, comma_str: str) -> List[str]:
        """Parse comma-separated list into list of strings."""
        return [item.strip() for item in comma_str.split(',') if item.strip()]

  
    def get_mode_time_filter(self):
        """Get time filter based on mode flags."""
        if self.daily_mode:
            # For daily mode, load data from start of day
            import datetime
            today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = datetime.datetime.now()
            return (int(today_start.timestamp() * 1000), int(today_end.timestamp() * 1000))
        elif self.initial_mode:
            # For initial mode, load data from 2024 onwards
            import datetime
            start_dt = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
            end_dt = datetime.datetime.now(datetime.timezone.utc)
            return (int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000))
        return None

    def _get_minutes_time_range(self, minutes: int) -> tuple:
        """Get time range for the last N minutes for real-time updates."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        start_time = now - datetime.timedelta(minutes=minutes)
        return (int(start_time.timestamp() * 1000), int(now.timestamp() * 1000))

    def _parse_time_range(self, time_str: str) -> tuple:
        """Parse time range string into (start_time, end_time) tuple."""
        try:
            parts = time_str.split(',')
            if len(parts) == 1:
                start_time = int(parts[0].strip())
                return (start_time, None)
            elif len(parts) == 2:
                start_time = int(parts[0].strip()) if parts[0].strip() else None
                end_time = int(parts[1].strip()) if parts[1].strip() else None
                return (start_time, end_time)
            else:
                raise ValueError("Invalid time range format")
        except ValueError as e:
            print(f"Error parsing time range: {e}")
            print("Expected format: start_time,end_time or just start_time")
            sys.exit(1)

    def get_time_filter_sql(self, time_col: str = 'time') -> str:
        """Generate SQL WHERE clause for time filtering."""
        import datetime

        # Quote the column name with backticks to handle reserved keywords
        quoted_col = f"`{time_col}`"

        # Check for daily/initial mode first
        time_range = self.get_mode_time_filter()
        if time_range:
            start_time, end_time = time_range
            conditions = []
            if start_time:
                conditions.append(f"{quoted_col} >= {start_time}")
            if end_time:
                conditions.append(f"{quoted_col} <= {end_time}")
            return " AND ".join(conditions) if conditions else None

        if self.days_filter:
            # Calculate timestamp for N days ago
            now = datetime.datetime.now()
            n_days_ago = now - datetime.timedelta(days=self.days_filter)
            start_timestamp = int(n_days_ago.timestamp() * 1000)
            return f"{quoted_col} >= {start_timestamp}"

        if self.time_range:
            start_time, end_time = self.time_range
            conditions = []
            if start_time:
                conditions.append(f"{quoted_col} >= {start_time}")
            if end_time:
                conditions.append(f"{quoted_col} <= {end_time}")
            return " AND ".join(conditions) if conditions else None

        return None

    def get_exchange_filter_sql(self, exchange_col: str = 'exchange') -> str:
        """Generate SQL WHERE clause for exchange filtering."""
        if not self.exchange_filter:
            return None

        # Quote the column name with backticks to handle reserved keywords
        quoted_col = f"`{exchange_col}`"
        return f"{quoted_col} IN ({','.join([repr(ex) for ex in self.exchange_filter])})"

    def get_pair_filter_sql(self, pair_col: str = 'pair') -> str:
        """Generate SQL WHERE clause for pair/symbol filtering."""
        if not self.pair_filter and not self.symbol_filter:
            return None

        # Quote the column name with backticks to handle reserved keywords
        quoted_col = f"`{pair_col}`"

        if self.pair_filter:
            return f"{quoted_col} IN ({','.join([repr(pair) for pair in self.pair_filter])})"
        elif self.symbol_filter:
            return f"{quoted_col} IN ({','.join([repr(symbol) for symbol in self.symbol_filter])})"

        return None

    def get_interval_filter_sql(self, interval_col: str = 'interval') -> str:
        """Generate SQL WHERE clause for interval filtering."""
        if not self.interval_filter:
            return None

        # Quote the column name with backticks to handle reserved keywords
        quoted_col = f"`{interval_col}`"
        return f"{quoted_col} IN ({','.join([repr(interval) for interval in self.interval_filter])})"

    def build_where_clause(self, table_name: str) -> str:
        """Build complete WHERE clause for a specific table."""
        conditions = []

        # Tables without exchange column (aggregated tables)
        no_exchange_tables = ['cg_open_interest_aggregated_history', 'cg_liquidation_aggregated_history']

        # Tables with exchange_list column
        exchange_list_tables = ['cg_futures_aggregated_taker_buy_sell_volume_history',
                               'cg_futures_aggregated_ask_bids_history']

        # Tables with symbol column (vs pair column)
        symbol_tables = ['cg_futures_price_history',
                        'cg_futures_aggregated_taker_buy_sell_volume_history',
                        'cg_futures_aggregated_ask_bids_history',
                        'cg_open_interest_aggregated_history',
                        'cg_liquidation_aggregated_history']

        # Determine column names based on table
        if table_name in no_exchange_tables:
            exchange_col = None  # No exchange filter for these tables
        elif table_name in exchange_list_tables:
            exchange_col = 'exchange_list'
        else:
            exchange_col = 'exchange'

        pair_col = 'symbol' if table_name in symbol_tables else 'pair'

        # Add conditions
        time_condition = self.get_time_filter_sql()
        if time_condition:
            conditions.append(time_condition)

        # Only add exchange condition if table has exchange column
        if exchange_col is not None:
            exchange_condition = self.get_exchange_filter_sql(exchange_col)
            if exchange_condition:
                conditions.append(exchange_condition)

        pair_condition = self.get_pair_filter_sql(pair_col)
        if pair_condition:
            conditions.append(pair_condition)

        interval_condition = self.get_interval_filter_sql()
        if interval_condition:
            conditions.append(interval_condition)

        return " AND ".join(conditions) if conditions else "1=1"

    def print_filter_summary(self):
        """Print summary of active filters."""
        print("=== Active Filters ===")
        if self.initial_mode:
            print(f"Mode: initial (from 2024 onwards)")
        elif self.daily_mode:
            print(f"Mode: daily (current day only)")
        if self.exchange_filter:
            print(f"Exchange(s): {', '.join(self.exchange_filter)}")
        if self.pair_filter:
            print(f"Pair(s): {', '.join(self.pair_filter)}")
        if self.symbol_filter:
            print(f"Symbol(s): {', '.join(self.symbol_filter)}")
        if self.interval_filter:
            print(f"Interval(s): {', '.join(self.interval_filter)}")

        # Show time range based on filter
        if self.minutes_filter:
            print(f"Time Filter: Last {self.minutes_filter} minutes (Real-time)")
        elif self.time_range:
            start_time, end_time = self.time_range
            if start_time:
                import datetime
                start_dt = datetime.datetime.fromtimestamp(start_time/1000)
                print(f"Start time: {start_dt} ({start_time})")
            if end_time:
                end_dt = datetime.datetime.fromtimestamp(end_time/1000)
                print(f"End time: {end_dt} ({end_time})")
        else:
            # Show time range based on mode
            time_range = self.get_daily_time_filter()
            if time_range:
                start_time, end_time = time_range
                import datetime
                start_dt = datetime.datetime.fromtimestamp(start_time/1000)
                end_dt = datetime.datetime.fromtimestamp(end_time/1000)
                print(f"Time Range: {start_dt} to {end_dt}")
        if self.days_filter:
            print(f"Days: {self.days_filter}")
        if self.minutes_filter:
            print(f"Minutes: {self.minutes_filter}")
        print("=" * 20)

def validate_arguments(args):
    """Validate command-line arguments."""
    # Check for mutual exclusive options
    if args.pair and args.symbol:
        print("Error: Cannot specify both --pair and --symbol")
        sys.exit(1)

    # Validate time format
    if args.time:
        try:
            parts = args.time.split(',')
            for part in parts:
                if part.strip():
                    int(part.strip())
        except ValueError:
            print("Error: Invalid time format. Use Unix timestamps in milliseconds.")
            sys.exit(1)

    # Validate days
    if args.days and args.days < 1:
        print("Error: Days must be a positive integer")
        sys.exit(1)

if __name__ == "__main__":
    # Example usage
    args = parse_arguments()
    validate_arguments(args)

    data_filter = DataFilter(args)
    data_filter.print_filter_summary()

    # Print WHERE clauses for each table
    tables = [
        # Core training tables
        'cg_futures_price_history',
        'cg_futures_aggregated_taker_buy_sell_volume_history',
        'cg_futures_aggregated_ask_bids_history',
        'cg_open_interest_aggregated_history',
        'cg_liquidation_aggregated_history',
        # Support/regime filter tables
        'cg_funding_rate_history',
        'cg_futures_basis_history',
        'cg_long_short_global_account_ratio_history',
        'cg_long_short_top_account_ratio_history'
    ]

    print("\n=== SQL WHERE Clauses ===")
    for table in tables:
        where_clause = data_filter.build_where_clause(table)
        print(f"{table}: {where_clause}")