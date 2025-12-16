#!/usr/bin/env python3
"""
Real-time Monitor untuk XGBoost Trading Model.
Monitor new data di database dan trigger real-time processing.
"""

import os
import sys
import time
import logging
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path
import asyncio
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s',
    handlers=[
        logging.FileHandler('/var/log/realtime_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealtimeMonitor:
    """Monitor real-time data arrival untuk semua 6 tabel."""

    def __init__(self, config_file: str = 'realtime_config.json'):
        self.config_file = config_file
        self.config = self.load_config()

        # Database config
        self.db_config = {
            'host': os.getenv('TRADING_DB_HOST'),
            'port': int(os.getenv('TRADING_DB_PORT', 3306)),
            'database': os.getenv('TRADING_DB_NAME'),
            'user': os.getenv('TRADING_DB_USER'),
            'password': os.getenv('TRADING_DB_PASSWORD')
        }

        # Table configurations
        self.tables = {
            'cg_spot_price_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'symbol',
                'check_interval': 60,  # seconds
                'batch_size': 1000
            },
            'cg_funding_rate_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'pair',
                'check_interval': 300,  # 5 minutes
                'batch_size': 500
            },
            'cg_futures_basis_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'pair',
                'check_interval': 300,
                'batch_size': 500
            },
            'cg_spot_aggregated_taker_volume_history': {
                'time_col': 'time',
                'exchange_col': 'exchange_name',
                'pair_col': 'symbol',
                'check_interval': 60,
                'batch_size': 1000
            },
            'cg_long_short_global_account_ratio_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'pair',
                'check_interval': 600,  # 10 minutes
                'batch_size': 200
            },
            'cg_long_short_top_account_ratio_history': {
                'time_col': 'time',
                'exchange_col': 'exchange',
                'pair_col': 'pair',
                'check_interval': 600,
                'batch_size': 200
            }
        }

        # Track last processed timestamp for each table
        self.last_processed = self.load_last_processed()

        # Monitor status
        self.running = False
        self.threads = []

        # Processing queue
        self.processing_queue = asyncio.Queue()

        # Initialize database connection
        self.engine = self.create_db_connection()

    def load_config(self) -> Dict:
        """Load konfigurasi real-time."""
        default_config = {
            "monitor_pairs": ["BTCUSDT", "ETHUSDT"],
            "monitor_intervals": ["1h", "4h"],
            "exchanges": ["binance"],
            "processing": {
                "batch_delay": 30,  # seconds
                "max_batch_size": 5000,
                "min_new_records": 10  # minimum records to trigger processing
            },
            "notifications": {
                "enabled": True,
                "webhook_url": os.getenv("WEBHOOK_URL"),
                "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
                "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID")
            },
            "model_update": {
                "enabled": True,
                "update_frequency": "hourly",  # hourly, daily, on_new_data
                "min_samples_for_update": 100,
                "performance_threshold": 0.6
            }
        }

        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge dengan defaults
                for key in default_config:
                    if key not in user_config:
                        user_config[key] = default_config[key]
                return user_config
            except Exception as e:
                logger.error(f"Error loading config: {e}. Using defaults.")
                return default_config
        else:
            # Create default config
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config

    def load_last_processed(self) -> Dict:
        """Load last processed timestamps."""
        file_path = Path('./state/last_processed.json')
        file_path.parent.mkdir(exist_ok=True)

        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            # Initialize with current time
            default = {table: datetime.now().isoformat() for table in self.tables.keys()}
            with open(file_path, 'w') as f:
                json.dump(default, f, indent=2)
            return default

    def save_last_processed(self):
        """Save last processed timestamps."""
        file_path = Path('./state/last_processed.json')
        with open(file_path, 'w') as f:
            json.dump(self.last_processed, f, indent=2)

    def create_db_connection(self):
        """Create database connection."""
        try:
            connection_string = f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            engine = create_engine(connection_string)
            return engine
        except Exception as e:
            logger.error(f"Failed to create DB connection: {e}")
            return None

    def get_new_data(self, table: str, since_time: datetime) -> Optional[pd.DataFrame]:
        """Get new data dari table sejak timestamp tertentu."""
        if not self.engine:
            logger.error("No database connection")
            return None

        table_config = self.tables[table]
        time_col = table_config['time_col']

        try:
            # Query untuk new data
            query = f"""
            SELECT * FROM {table}
            WHERE {time_col} > '{since_time}'
            AND exchange IN %(exchanges)s
            AND {'symbol' if 'symbol' in table_config else 'pair'} IN %(pairs)s
            ORDER BY {time_col} ASC
            LIMIT %(batch_size)s
            """

            df = pd.read_sql(
                query,
                self.engine,
                params={
                    'exchanges': tuple(self.config['exchanges']),
                    'pairs': tuple(self.config['monitor_pairs']),
                    'batch_size': table_config['batch_size']
                }
            )

            if not df.empty:
                logger.info(f"Found {len(df)} new records in {table}")
                return df
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting new data from {table}: {e}")
            return None

    def monitor_table(self, table: str):
        """Monitor thread untuk satu table."""
        logger.info(f"Starting monitor for {table}")

        while self.running:
            try:
                # Get last processed time
                last_time = datetime.fromisoformat(self.last_processed[table])

                # Check for new data
                new_data = self.get_new_data(table, last_time)

                if new_data is not None and len(new_data) > 0:
                    logger.info(f"📊 New data in {table}: {len(new_data)} records")

                    # Add to processing queue
                    asyncio.run(
                        self.processing_queue.put({
                            'table': table,
                            'data': new_data,
                            'timestamp': datetime.now().isoformat()
                        })
                    )

                    # Update last processed timestamp
                    max_time = new_data[self.tables[table]['time_col']].max()
                    self.last_processed[table] = max_time.isoformat()
                    self.save_last_processed()

                    # Send notification
                    self.send_notification(f"📈 New {table} data: {len(new_data)} records")

                # Sleep sesuai interval
                time.sleep(self.tables[table]['check_interval'])

            except Exception as e:
                logger.error(f"Error in monitor for {table}: {e}")
                time.sleep(30)  # Wait before retry

    async def process_data_queue(self):
        """Process data dari queue."""
        logger.info("Starting data processor")

        batch_data = {}
        last_batch_time = datetime.now()

        while self.running:
            try:
                # Wait for data or timeout
                try:
                    item = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=self.config['processing']['batch_delay']
                    )

                    table = item['table']
                    if table not in batch_data:
                        batch_data[table] = []

                    batch_data[table].append(item['data'])

                    # Check if we should process batch
                    total_records = sum(len(df) for dfs in batch_data.values() for df in dfs)
                    time_since_batch = (datetime.now() - last_batch_time).total_seconds()

                    if (total_records >= self.config['processing']['max_batch_size'] or
                        time_since_batch >= self.config['processing']['batch_delay'] * 3):

                        logger.info(f"Processing batch: {total_records} total records")
                        await self.process_batch(batch_data)
                        batch_data = {}
                        last_batch_time = datetime.now()

                except asyncio.TimeoutError:
                    # Process any pending data
                    if batch_data:
                        logger.info("Processing timeout batch")
                        await self.process_batch(batch_data)
                        batch_data = {}
                        last_batch_time = datetime.now()

            except Exception as e:
                logger.error(f"Error in data processor: {e}")
                await asyncio.sleep(10)

    async def process_batch(self, batch_data: Dict):
        """Process batch data - trigger real-time training."""
        try:
            # Combine data for each table
            combined_data = {}
            for table, dfs in batch_data.items():
                if dfs:
                    combined_data[table] = pd.concat(dfs, ignore_index=True)

            # Save new data
            self.save_new_data(combined_data)

            # Trigger real-time pipeline
            if self.config['model_update']['enabled']:
                await self.trigger_realtime_pipeline(combined_data)

            # Generate predictions if we have enough data
            if self.should_generate_predictions(combined_data):
                await self.generate_predictions(combined_data)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")

    def save_new_data(self, combined_data: Dict):
        """Save new data untuk real-time processing."""
        output_dir = Path('./realtime_data')
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for table, df in combined_data.items():
            if not df.empty:
                file_path = output_dir / f"{table}_{timestamp}.parquet"
                df.to_parquet(file_path, index=False)
                logger.info(f"Saved {len(df)} records to {file_path}")

    async def trigger_realtime_pipeline(self, new_data: Dict):
        """Trigger real-time training pipeline."""
        logger.info("🚀 Triggering real-time training pipeline")

        # Save trigger file
        trigger_file = Path('./realtime_trigger.json')
        trigger_info = {
            'timestamp': datetime.now().isoformat(),
            'new_data_summary': {table: len(df) for table, df in new_data.items()},
            'trigger_reason': 'new_data_arrival'
        }

        with open(trigger_file, 'w') as f:
            json.dump(trigger_info, f, indent=2)

        # Launch realtime trainer
        try:
            # Run realtime trainer as subprocess
            import subprocess

            cmd = [
                'python', 'realtime_trainer.py',
                '--trigger-file', str(trigger_file),
                '--mode', 'incremental'
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            logger.info(f"Started realtime trainer (PID: {process.pid})")

        except Exception as e:
            logger.error(f"Failed to trigger realtime pipeline: {e}")

    def should_generate_predictions(self, new_data: Dict) -> bool:
        """Check if we should generate predictions."""
        # Check if we have new price data
        if 'cg_spot_price_history' in new_data:
            return len(new_data['cg_spot_price_history']) >= self.config['processing']['min_new_records']
        return False

    async def generate_predictions(self, new_data: Dict):
        """Generate real-time predictions."""
        logger.info("🎯 Generating real-time predictions")

        try:
            # Call prediction API
            import requests

            # Prepare features from new data
            if 'cg_spot_price_history' in new_data:
                price_data = new_data['cg_spot_price_history'].iloc[-1]  # Latest

                # Create simple features (in real implementation, use full feature engineering)
                features = {
                    'price_close': price_data.get('close', 0),
                    'volume_usd': price_data.get('volume_usd', 0),
                    'exchange': price_data.get('exchange', 'binance'),
                    'symbol': price_data.get('symbol', 'BTCUSDT')
                }

                # Call prediction API
                response = requests.post(
                    "http://localhost:8000/api/v1/predict",
                    json={"features": features},
                    timeout=10
                )

                if response.status_code == 200:
                    prediction = response.json()
                    signal = "BUY" if prediction.get('prediction', 0) == 1 else "HOLD"
                    confidence = prediction.get('confidence', 0)

                    logger.info(f"🎯 Prediction: {signal} (confidence: {confidence:.2f})")

                    # Send signal notification
                    if signal == "BUY" and confidence > 0.7:
                        await self.send_signal_notification(features, prediction)
                else:
                    logger.error(f"Prediction API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")

    async def send_signal_notification(self, features: Dict, prediction: Dict):
        """Send trading signal notification."""
        if not self.config['notifications']['enabled']:
            return

        message = f"""
🚀 **TRADING SIGNAL DETECTED!**

📊 Asset: {features.get('symbol', 'Unknown')}
💰 Price: ${features.get('price_close', 0):,.2f}
📈 Signal: BUY
🎯 Confidence: {prediction.get('confidence', 0):.2%}
📊 Prediction Score: {prediction.get('prediction_probability', 0):.3f}

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔥 Source: XGBoost Real-Time Model
        """

        # Send to webhook
        webhook_url = self.config['notifications'].get('webhook_url')
        if webhook_url:
            try:
                import requests
                requests.post(webhook_url, json={"text": message})
            except Exception as e:
                logger.error(f"Failed to send webhook: {e}")

        # Send to Telegram
        telegram_bot_token = self.config['notifications'].get('telegram_bot_token')
        telegram_chat_id = self.config['notifications'].get('telegram_chat_id')

        if telegram_bot_token and telegram_chat_id:
            try:
                import requests
                url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
                requests.post(url, json={
                    'chat_id': telegram_chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                })
            except Exception as e:
                logger.error(f"Failed to send Telegram: {e}")

    def send_notification(self, message: str):
        """Send general notification."""
        if not self.config['notifications']['enabled']:
            return

        logger.info(f"📢 {message}")

        # Send to webhook if configured
        webhook_url = self.config['notifications'].get('webhook_url')
        if webhook_url:
            try:
                import requests
                requests.post(webhook_url, json={"text": f"🤖 XGBoost Monitor: {message}"})
            except Exception as e:
                logger.error(f"Failed to send webhook: {e}")

    def start(self):
        """Start monitoring semua tables."""
        logger.info("🚀 Starting Real-time Monitor")

        self.running = True

        # Start monitor thread for each table
        for table in self.tables.keys():
            thread = threading.Thread(
                target=self.monitor_table,
                args=(table,),
                name=f"Monitor-{table}"
            )
            thread.start()
            self.threads.append(thread)

        # Start async processor
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run processor in background
        processor_thread = threading.Thread(
            target=self.run_async_processor,
            name="AsyncProcessor"
        )
        processor_thread.start()
        self.threads.append(processor_thread)

        logger.info(f"✅ Started {len(self.threads)} monitoring threads")

        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            self.stop()

    def run_async_processor(self):
        """Run async processor in thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.process_data_queue())

    def stop(self):
        """Stop monitoring."""
        logger.info("🛑 Stopping Real-time Monitor")
        self.running = False

        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=5)

        logger.info("✅ Monitor stopped")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Real-time XGBoost monitor')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Test mode - run once and exit')

    args = parser.parse_args()

    monitor = RealtimeMonitor(config_file=args.config or 'realtime_config.json')

    if args.test:
        logger.info("Running in test mode")
        # Test dengan mengambil data terbaru
        for table in monitor.tables.keys():
            last_time = datetime.fromisoformat(monitor.last_processed[table]) - timedelta(hours=1)
            new_data = monitor.get_new_data(table, last_time)
            if new_data is not None:
                logger.info(f"Test {table}: Found {len(new_data)} records")
    else:
        monitor.start()


if __name__ == "__main__":
    main()