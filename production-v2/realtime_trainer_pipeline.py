#!/usr/bin/env python3
"""
Real-time XGBoost Trainer using CORE PIPELINE
Integrates with 6 core training files for proper incremental training
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Add parent directory to path for importing core modules
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/realtime_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealtimeTrainerPipeline:
    """
    Real-time trainer that uses the CORE 6-step training pipeline
    This properly integrates with the existing training infrastructure
    """

    def __init__(self, output_dir: str = '/app/output_train'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)

        # State directory for tracking
        self.state_dir = Path('/app/state')
        self.state_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"🚀 Real-time trainer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🤖 Using CORE pipeline (6 steps)")

    def load_trigger_data(self) -> Optional[Dict]:
        """Load training trigger data"""
        trigger_file = self.state_dir / 'realtime_trigger.json'

        if trigger_file.exists():
            try:
                with open(trigger_file, 'r') as f:
                    trigger_data = json.load(f)
                logger.info(f"📋 Loaded trigger: {trigger_data.get('trigger_reason')}")
                return trigger_data
            except Exception as e:
                logger.error(f"❌ Error loading trigger: {e}")
                return None
        else:
            logger.info("📋 No trigger file found")
            return None

    def get_last_training_time(self) -> Optional[datetime]:
        """Get the last time training was completed"""
        status_file = self.state_dir / 'last_training_status.json'

        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)
                last_time = status.get('last_training_time')
                if last_time:
                    return datetime.fromisoformat(last_time)
            except Exception as e:
                logger.error(f"❌ Error reading last training time: {e}")

        return None

    def run_core_pipeline(self, incremental: bool = True) -> bool:
        """
        Run the CORE 6-step training pipeline
        This is the SAME pipeline used in simple_run.sh but for new data only
        """

        logger.info("🔥 Starting CORE training pipeline")
        logger.info("📋 Step 1: load_database.py")

        # Core pipeline configuration
        exchange = os.getenv('EXCHANGE', 'binance')
        pair = os.getenv('PAIR', 'BTCUSDT,ETHUSDT')
        interval = os.getenv('INTERVAL', '1h,4h')

        core_scripts = [
            ("load_database.py", "Load Database (Market Data)"),
            ("merge_7_tables.py", "Merge Tables"),
            ("feature_engineering.py", "Feature Engineering"),
            ("label_builder.py", "Label Building"),
            ("xgboost_trainer.py", "Model Training"),
            ("model_evaluation_with_leverage.py", "Model Evaluation")
        ]

        # Change to parent directory where core scripts are located
        parent_dir = Path(__file__).parent.parent
        original_cwd = os.getcwd()

        try:
            os.chdir(parent_dir)

            for step_num, (script, description) in enumerate(core_scripts, 1):
                logger.info(f"🔧 Step {step_num}: {description}")

                # Build command with absolute paths
                cmd = [
                    "python3", str(Path(parent_dir) / script),
                    "--exchange", exchange,
                    "--pair", pair,
                    "--interval", interval,
                    "--output-dir", str(self.output_dir.absolute())
                ]

                # Add incremental mode for specific scripts
                if incremental and script in ['load_database.py', 'xgboost_trainer.py']:
                    cmd.extend(["--incremental"])

                logger.info(f"🏃 Running: {' '.join(cmd)}")

                # Execute core script
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout per step
                )

                if result.returncode == 0:
                    logger.info(f"✅ Step {step_num} completed successfully")
                    if result.stdout:
                        logger.info(f"📄 Output: {result.stdout[-500:]}")  # Last 500 chars
                else:
                    logger.error(f"❌ Step {step_num} failed!")
                    logger.error(f"📄 Error: {result.stderr}")
                    return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Pipeline step timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            return False
        finally:
            os.chdir(original_cwd)

        logger.info("🎉 CORE pipeline completed successfully!")
        return True

    def validate_model_output(self) -> bool:
        """Validate that models were created"""
        try:
            model_files = list(self.models_dir.glob("*.joblib"))

            if len(model_files) > 0:
                logger.info(f"✅ Found {len(model_files)} model files:")
                for model_file in model_files:
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    logger.info(f"   📄 {model_file.name} ({size_mb:.1f} MB)")
                return True
            else:
                logger.error("❌ No model files found after training!")
                return False

        except Exception as e:
            logger.error(f"❌ Error validating models: {e}")
            return False

    def update_training_status(self, success: bool, trigger_data: Optional[Dict] = None):
        """Update training status and log results"""
        status = {
            "last_training_time": datetime.now().isoformat(),
            "training_successful": success,
            "pipeline_type": "CORE_6_STEP",
            "trigger_reason": trigger_data.get('trigger_reason', 'automatic') if trigger_data else 'automatic',
            "tables_with_new_data": trigger_data.get('tables_with_new_data', []) if trigger_data else []
        }

        # Save training status
        status_file = self.state_dir / 'last_training_status.json'
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

        # Clean trigger file if exists
        trigger_file = self.state_dir / 'realtime_trigger.json'
        if trigger_file.exists():
            trigger_file.unlink()
            logger.info("🧹 Cleaned trigger file")

        if success:
            logger.info("✅ Training status updated successfully")
        else:
            logger.error("❌ Training failed - status updated")

    def log_training_result(self, success: bool, trigger_data: Optional[Dict] = None):
        """Log training result to file and console"""
        try:
            result = {
                "timestamp": datetime.now().isoformat(),
                "training_successful": success,
                "pipeline_type": "CORE_6_STEP",
                "trigger_reason": trigger_data.get('trigger_reason', 'automatic') if trigger_data else 'automatic',
                "models_created": len(list(self.models_dir.glob('*.joblib'))) if success else 0,
                "tables_with_new_data": trigger_data.get('tables_with_new_data', []) if trigger_data else []
            }

            # Save to log file
            log_file = self.state_dir / 'training_results.log'
            with open(log_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

            # Console logging
            if success:
                logger.info(f"🎉 XGBoost Training Completed Successfully!")
                logger.info(f"📊 Pipeline: CORE 6-Step")
                logger.info(f"📁 Models: {result['models_created']} files created")
            else:
                logger.error(f"❌ XGBoost Training Failed!")
                logger.error(f"⚠️ Please check logs for details")

        except Exception as e:
            logger.error(f"❌ Error logging training result: {e}")

    def run_training(self, mode: str = 'incremental') -> bool:
        """
        Main training execution method
        Integrates with CORE 6-step pipeline
        """
        logger.info(f"🚀 Starting real-time training (mode: {mode})")

        # Load trigger data if available
        trigger_data = self.load_trigger_data()

        # Real-time check: Always run if there's trigger data OR check for new data
        if mode == 'incremental':
            last_training = self.get_last_training_time()

            # Always proceed if trigger data exists
            if trigger_data:
                logger.info(f"🚀 Training triggered by new data: {trigger_data.get('tables_with_new_data', [])}")

            # If no trigger, check if enough time has passed for periodic check
            elif last_training:
                time_since_last = datetime.now() - last_training
                # Allow more frequent checks - every 15 minutes instead of 1 hour
                if time_since_last < timedelta(minutes=15):
                    logger.info(f"⏭️ Skipping frequent check - last run {time_since_last} ago (next check in 15 min)")
                    return True
                else:
                    logger.info(f"🔄 Periodic check - last run {time_since_last} ago")
            else:
                logger.info("🆕 No previous training found - running initial check")

        try:
            # Run CORE pipeline (this integrates with 6 core files)
            success = self.run_core_pipeline(incremental=(mode == 'incremental'))

            if success:
                # Validate models were created
                success = self.validate_model_output()

                if success:
                    logger.info("🎉 Real-time training completed successfully!")
                else:
                    logger.error("❌ Training completed but no models found!")

            # Update status
            self.update_training_status(success, trigger_data)

            # Log training result
            self.log_training_result(success, trigger_data)

            return success

        except Exception as e:
            logger.error(f"❌ Real-time training failed: {e}")
            self.update_training_status(False, trigger_data)
            self.log_training_result(False, trigger_data)
            return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Real-time XGBoost Trainer with CORE Pipeline')
    parser.add_argument('--mode', choices=['incremental', 'full'], default='incremental',
                      help='Training mode (default: incremental)')
    parser.add_argument('--output-dir', default='/app/output_train',
                      help='Output directory (default: /app/output_train)')

    args = parser.parse_args()

    logger.info("🤖 Real-time XGBoost Trainer (CORE Pipeline Integration)")
    logger.info("=" * 60)

    # Initialize trainer
    trainer = RealtimeTrainerPipeline(args.output_dir)

    # Run training
    success = trainer.run_training(args.mode)

    if success:
        logger.info("🎉 Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()