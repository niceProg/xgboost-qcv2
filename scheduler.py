#!/usr/bin/env python3
"""
Scheduler for XGBoost trading pipeline.
Automatically runs the daily pipeline during trading hours (7:00 - 16:00).
"""

import os
import sys
import time
import logging
import schedule
from datetime import datetime, timedelta
import pytz

# Import pipeline runner
from run_daily_pipeline import DailyPipelineRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineScheduler:
    """Scheduler for automated daily pipeline execution."""

    def __init__(self):
        self.exchange = os.getenv('EXCHANGE', 'binance')
        self.pair = os.getenv('PAIR', 'BTCUSDT')
        self.interval = os.getenv('INTERVAL', '1h')
        self.trading_hours = os.getenv('TRADING_HOURS', '00:00-09:00')  # 7:00-16:00 WIB in UTC
        self.timezone = os.getenv('TIMEZONE', 'UTC')

        # Parse trading hours
        start_str, end_str = self.trading_hours.split('-')
        self.start_hour = int(start_str.split(':')[0])
        self.start_min = int(start_str.split(':')[1])
        self.end_hour = int(end_str.split(':')[0])
        self.end_min = int(end_str.split(':')[1])

        # Get timezone object
        self.tz = pytz.timezone(self.timezone) if self.timezone != 'UTC' else None

        logger.info(f"Scheduler initialized for {self.exchange}/{self.pair}/{self.interval}")
        logger.info(f"Trading hours: {self.trading_hours} ({self.timezone})")

    def is_trading_day(self):
        """Check if today is a trading day (weekday)."""
        now = datetime.now(self.tz) if self.tz else datetime.now()
        # Only run on weekdays (Monday to Friday)
        return now.weekday() < 5

    def run_pipeline(self):
        """Run the daily pipeline."""
        if not self.is_trading_day():
            logger.info("Today is not a trading day (weekday). Skipping.")
            return

        logger.info("Starting scheduled daily pipeline run...")
        runner = DailyPipelineRunner()
        success = runner.run(mode='daily')

        if success:
            logger.info("Scheduled pipeline completed successfully")
        else:
            logger.error("Scheduled pipeline failed")

    def setup_schedule(self):
        """Setup the schedule for pipeline runs."""
        # Schedule to run at the start of trading hours
        schedule.every().monday.at(f"{self.start_hour:02d}:{self.start_min:02d}").do(self.run_pipeline)
        schedule.every().tuesday.at(f"{self.start_hour:02d}:{self.start_min:02d}").do(self.run_pipeline)
        schedule.every().wednesday.at(f"{self.start_hour:02d}:{self.start_min:02d}").do(self.run_pipeline)
        schedule.every().thursday.at(f"{self.start_hour:02d}:{self.start_min:02d}").do(self.run_pipeline)
        schedule.every().friday.at(f"{self.start_hour:02d}:{self.start_min:02d}").do(self.run_pipeline)

        # Also run every hour during trading hours to update
        for hour in range(self.start_hour, self.end_hour + 1):
            if hour < self.end_hour or (hour == self.end_hour and self.end_min > 0):
                for minute in [0, 30]:  # Run twice per hour
                    time_str = f"{hour:02d}:{minute:02d}"
                    # Skip times before start
                    if hour < self.start_hour or (hour == self.start_hour and minute < self.start_min):
                        continue
                    # Skip times after end
                    if hour > self.end_hour or (hour == self.end_hour and minute > self.end_min):
                        continue

                    schedule.every().day.at(time_str).do(self.check_and_run)

        logger.info(f"Schedule setup complete. First run at {self.start_hour:02d}:{self.start_min:02d}")

    def check_and_run(self):
        """Check if within trading hours and run if necessary."""
        now = datetime.now(self.tz) if self.tz else datetime.now()

        # Check if within trading hours
        current_time = now.time()
        start_time = now.replace(hour=self.start_hour, minute=self.start_min, second=0, microsecond=0).time()
        end_time = now.replace(hour=self.end_hour, minute=self.end_min, second=0, microsecond=0).time()

        if start_time <= current_time <= end_time and self.is_trading_day():
            logger.info(f"Within trading hours, running pipeline update at {now}")
            runner = DailyPipelineRunner()
            runner.run(mode='daily')
        else:
            logger.debug(f"Outside trading hours or weekend at {now}")

    def run_once(self):
        """Run the pipeline once immediately (for testing)."""
        logger.info("Running pipeline immediately...")
        self.run_pipeline()

    def status(self):
        """Show current scheduler status."""
        now = datetime.now(self.tz) if self.tz else datetime.now()
        jobs = schedule.get_jobs()

        print(f"\n=== Scheduler Status ===")
        print(f"Current time: {now}")
        print(f"Trading hours: {self.trading_hours} ({self.timezone})")
        print(f"Is trading day: {self.is_trading_day()}")
        print(f"\nScheduled jobs ({len(jobs)}):")

        for job in jobs:
            next_run = job.next_run
            if next_run:
                next_run_str = next_run.strftime('%Y-%m-%d %H:%M:%S')
                print(f"  - {job.job_func.__name__}: {next_run_str}")

        print("=" * 30)

    def run(self):
        """Main scheduler loop."""
        self.setup_schedule()

        logger.info("Scheduler started. Press Ctrl+C to stop.")
        logger.info("Type 'status' to see current status, 'run' to run immediately, 'quit' to exit.")

        try:
            while True:
                # Check for user input
                try:
                    import select
                    import sys

                    # Check if there's input waiting
                    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                        cmd = sys.stdin.readline().strip().lower()
                        if cmd == 'quit':
                            break
                        elif cmd == 'status':
                            self.status()
                        elif cmd == 'run':
                            self.run_once()
                        elif cmd == 'help':
                            print("Commands: status, run, quit")
                except:
                    pass

                # Run scheduled jobs
                schedule.run_pending()

                # Sleep for a short interval
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\nScheduler stopped by user")

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='XGBoost Trading Pipeline Scheduler')
    parser.add_argument(
        '--run-once',
        action='store_true',
        help='Run pipeline once and exit'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show scheduler status and exit'
    )

    args = parser.parse_args()

    # Create scheduler
    scheduler = PipelineScheduler()

    if args.status:
        scheduler.status()
        return

    if args.run_once:
        scheduler.run_once()
        return

    # Run scheduler
    scheduler.run()

if __name__ == "__main__":
    main()