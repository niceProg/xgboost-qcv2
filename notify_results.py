#!/usr/bin/env python3
"""
Optional script to send notifications after pipeline completion.
This is referenced in the systemd service but is optional.
"""

import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def send_notification():
    """Send notification about pipeline results."""
    # Check if webhook URL is configured
    webhook_url = os.getenv('WEBHOOK_URL')
    if not webhook_url:
        logger.info("No webhook URL configured, skipping notification")
        return

    # Load latest results
    output_dir = os.getenv('OUTPUT_DIR', './output_train')
    results_dir = os.path.join(output_dir, 'daily_runs')

    if not os.path.exists(results_dir):
        logger.warning("No results directory found")
        return

    # Get latest result file
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not result_files:
        logger.warning("No result files found")
        return

    latest_file = max(result_files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))

    try:
        with open(os.path.join(results_dir, latest_file), 'r') as f:
            results = json.load(f)

        # Prepare message
        status = "✅ Success" if results['success'] else "❌ Failed"
        duration = results.get('duration_seconds', 0) / 60

        message = f"""
🤖 XGBoost Daily Pipeline Report
{status}
Duration: {duration:.1f} minutes
Pairs processed: {len(results['pairs_processed'])}
        """

        # Send webhook (implementation depends on your notification system)
        logger.info("Notification message prepared")
        # Implement your notification logic here

    except Exception as e:
        logger.error(f"Failed to send notification: {e}")

if __name__ == "__main__":
    send_notification()