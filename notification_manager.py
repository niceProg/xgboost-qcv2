#!/usr/bin/env python3
"""
Notification manager untuk model update alerts.
Memberikan instruksi jelas ketika model siap di-upload.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelUpdateNotifier:
    """Send notifications when model is ready for upload."""

    def __init__(self):
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.webhook_url = os.getenv("WEBHOOK_URL")

    def send_model_ready_notification(self, model_info: dict):
        """Send notification bahwa model siap di-upload."""

        # Create API access instructions
        instructions = """
🔧 **API Access Instructions:**

1. **Health Check:**
   ```bash
   curl https://api.dragonfortune.ai/health
   ```

2. **Get Latest Model:**
   ```bash
   curl https://api.dragonfortune.ai/api/v1/latest/model
   ```

3. **Get Dataset Summary:**
   ```bash
   curl https://api.dragonfortune.ai/api/v1/latest/dataset-summary
   ```

4. **API Documentation:**
   📖 https://api.dragonfortune.ai/docs

🤖 **Dragon Fortune AI API is ready for QuantConnect integration!**
"""

        message = f"""
🚀 **🐉 DRAGON FORTUNE AI - MODEL UPDATE READY**

📊 **Model Information:**
• Session ID: {model_info.get('session_id', 'Unknown')}
• Accuracy: {model_info.get('accuracy', 'N/A')}
• AUC Score: {model_info.get('auc', 'N/A')}
• Features Count: {model_info.get('features', 'N/A')}
• Model Size: {model_info.get('size_mb', 'N/A')}MB

🕐 **Update Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S WIB')}

🌐 **API Endpoints:**
• Health: https://api.dragonfortune.ai/health
• Latest Model: https://api.dragonfortune.ai/api/v1/latest/model
• Dataset Summary: https://api.dragonfortune.ai/api/v1/latest/dataset-summary
• All Sessions: https://api.dragonfortune.ai/api/v1/sessions
• Documentation: https://api.dragonfortune.ai/docs

{instructions}

🔥 **Dragon Fortune AI Trading System is LIVE!**
📱 Telegram: @DragonFortuneAI
🌐 Website: https://dragonfortune.ai
"""

        # Send to Telegram
        if self.telegram_token and self.telegram_chat_id:
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                    json={
                        'chat_id': self.telegram_chat_id,
                        'text': message,
                        'parse_mode': 'Markdown'
                    }
                )
                if response.status_code == 200:
                    logger.info("✅ Telegram notification sent")
                else:
                    logger.error(f"❌ Telegram failed: {response.status_code}")
            except Exception as e:
                logger.error(f"❌ Telegram error: {e}")

        # Send to webhook
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json={"text": message})
                logger.info("✅ Webhook notification sent")
            except Exception as e:
                logger.error(f"❌ Webhook error: {e}")

        # Log locally
        logger.info("🚀 Model ready notification sent")
        print(message)

    def send_telegram_message(self, message: str):
        """Send custom message to Telegram."""
        if self.telegram_token and self.telegram_chat_id:
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                    json={
                        'chat_id': self.telegram_chat_id,
                        'text': message,
                        'parse_mode': 'Markdown'
                    }
                )
                if response.status_code == 200:
                    logger.info("✅ Telegram message sent")
                    return True
                else:
                    logger.error(f"❌ Telegram failed: {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"❌ Telegram error: {e}")
                return False
        else:
            logger.warning("❌ Telegram not configured")
            return False

    def send_upload_reminder(self):
        """Send reminder jika model belum di-upload."""

        message = f"""
⏰ **UPLOAD REMINDER**

⚠️ Model update siap tapi belum di-upload!

🕐 **Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
⏰ **Trading Session:** Approaching

📋 **Quick Upload:**
```bash
cd /home/yumna/Working/dragonfortune/xgboost-qc
./upload_to_qc.sh
```

🔄 **Auto-sync will be paused until upload completed**
"""

        if self.telegram_token and self.telegram_chat_id:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                    json={
                        'chat_id': self.telegram_chat_id,
                        'text': message,
                        'parse_mode': 'Markdown'
                    }
                )
            except Exception:
                pass

def test_notification():
    """Test notification system."""
    notifier = ModelUpdateNotifier()

    test_model_info = {
        'version': '2024-12-17_v1.2',
        'auc': 0.75,
        'accuracy': 0.68,
        'features': 50,
        'size_mb': 12.5
    }

    notifier.send_model_ready_notification(test_model_info)

if __name__ == "__main__":
    test_notification()