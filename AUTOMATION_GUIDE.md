# 🤖 XGBoost Automation Setup Guide

## 📋 Overview

XGBoost Automation system includes:
- **Automatic Training** - Training berdasarkan schedule dan data baru
- **Telegram Notifications** - Real-time alerts untuk model updates
- **API Integration** - Model otomatis available di https://api.dragonfortune.ai

---

## 🚀 Quick Setup

### Step 1: Run Setup Script
```bash
./setup_automation.sh
```

### Step 2: Configure Telegram
Edit `.env` file:
```env
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_CHAT_ID=123456789
```

### Step 3: Start Automation
```bash
# Option 1: Start scheduler
python3 automation_scheduler.py

# Option 2: Run with systemd (recommended)
sudo cp /tmp/xgboost-monitor.service /etc/systemd/system/
sudo systemctl enable xgboost-monitor
sudo systemctl start xgboost-monitor
```

---

## 📱 Telegram Bot Setup

### 1. Create Telegram Bot
1. Message @BotFather on Telegram
2. Send: `/newbot`
3. Bot name: `Dragon Fortune AI Bot`
4. Bot username: `dragonfortune_ai_bot`
5. Copy the **BOT TOKEN**

### 2. Get Chat ID
1. Message your bot: `/start`
2. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
3. Copy the **chat_id** number

### 3. Test Connection
```bash
python3 -c "from notification_manager import ModelUpdateNotifier; ModelUpdateNotifier().send_telegram_message('🤖 Test message')"
```

---

## ⚙️ Configuration Options

### Environment Variables (.env)
```env
# Database Configuration
DB_HOST=103.150.81.86
DB_PORT=3306
DB_NAME=xgboostqc
DB_USER=xgboostqc
DB_PASSWORD=your_password

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Training Configuration
AUTO_TRAIN_ENABLED=true          # Enable/disable auto training
TRAINING_SCHEDULE=0 */6 * * *     # Every 6 hours (cron format)
MIN_NEW_RECORDS=100               # Minimum new records to trigger training
```

### Training Schedule Examples
```env
# Every 6 hours
TRAINING_SCHEDULE=0 */6 * * *

# Every 4 hours
TRAINING_SCHEDULE=0 */4 * * *

# Daily at 2 AM
TRAINING_SCHEDULE=0 2 * * *

# Weekdays at 9 AM and 3 PM
TRAINING_SCHEDULE=0 9,15 * * 1-5
```

---

## 🔧 Automation Components

### 1. Realtime Monitor (`realtime_monitor.py`)
- **Purpose**: Monitor database untuk data baru
- **Trigger**: New data arrival
- **Action**: Start training pipeline

### 2. Training Pipeline (`realtime_trainer_pipeline.py`)
- **Purpose**: Run 6-step training process
- **Process**: Load → Merge → Feature → Label → Train → Evaluate
- **Output**: Model di API + Notification

### 3. Notification Manager (`notification_manager.py`)
- **Purpose**: Send Telegram notifications
- **Triggers**: Model ready, errors, reminders
- **Content**: API endpoints, model info, instructions

### 4. Automation Scheduler (`automation_scheduler.py`)
- **Purpose**: Schedule-based training
- **Schedule**: Based on TRAINING_SCHEDULE
- **Features**: Automatic retry, error handling

---

## 🧪 Testing Automation

### Test Training Pipeline
```bash
# Run manual training test
python3 realtime_trainer_pipeline.py --test

# Check logs
tail -f logs/realtime_trainer.log
```

### Test Notifications
```bash
# Test Telegram connection
python3 -c "
from notification_manager import ModelUpdateNotifier
notifier = ModelUpdateNotifier()
notifier.send_telegram_message('🤖 XGBoost System Online!')
"

# Test model notification
python3 notification_manager.py
```

### Test API Integration
```bash
# Test API endpoints
curl https://api.dragonfortune.ai/health
curl https://api.dragonfortune.ai/api/v1/sessions
```

---

## 📊 Monitoring & Logs

### Log Files Location
```
logs/
├── realtime_monitor.log      # Database monitoring
├── realtime_trainer.log      # Training process
├── automation_scheduler.log  # Scheduler activity
└── notification.log          # Telegram notifications
```

### Monitor Commands
```bash
# View all logs
tail -f logs/*.log

# Check systemd service
sudo systemctl status xgboost-monitor
sudo journalctl -u xgboost-monitor -f

# Check Docker containers
docker ps | grep xgboost-api
docker logs xgboost-api
```

---

## 🔄 Workflow Diagram

```
New Data Arrival → Realtime Monitor → Training Pipeline
                                                ↓
API Update ← Model Storage ← Evaluation ← Training
                                                ↓
                                        Telegram Notification
```

## 📱 Notification Types

### 1. Model Ready Notification
- 🚀 Model training completed
- 📊 Model performance metrics
- 🌐 API endpoints information
- 🔗 QuantConnect integration guide

### 2. System Status Updates
- 🟢 System online/offline
- ⚠️ Warnings and errors
- 🔄 Training progress updates

### 3. Error Notifications
- ❌ Training failures
- 🔌 Database connection issues
- 🤖 API deployment problems

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Telegram Not Working
```bash
# Check token and chat_id
curl https://api.telegram.org/bot<YOUR_TOKEN>/getMe

# Check chat updates
curl https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
```

#### 2. Training Not Triggered
```bash
# Check monitor logs
tail -f logs/realtime_monitor.log

# Check database connection
python3 -c "
from realtime_monitor import RealtimeDatabaseMonitor
monitor = RealtimeDatabaseMonitor()
print('Database connection:', monitor.check_database_connection())
"
```

#### 3. API Not Updating
```bash
# Check API container
docker ps | grep xgboost-api

# Check API logs
docker logs xgboost-api

# Restart API if needed
docker-compose -f docker-compose.api.yml restart
```

### Recovery Commands
```bash
# Restart automation
sudo systemctl restart xgboost-monitor

# Clear stuck training
rm -f state/training.lock

# Reset scheduler
pkill -f automation_scheduler.py
python3 automation_scheduler.py &
```

---

## 📈 Performance Monitoring

### Key Metrics to Track
- Training frequency (should follow schedule)
- Model performance (accuracy, AUC)
- API response times
- Database query performance
- Telegram delivery rates

### Monitoring Commands
```bash
# Training frequency
grep "Training completed" logs/realtime_trainer.log | wc -l

# Model performance
grep -E "(accuracy|auc)" logs/realtime_trainer.log | tail -5

# API health
curl -w "@curl-format.txt" https://api.dragonfortune.ai/health
```

---

## ✅ Success Checklist

- [ ] Telegram bot created and configured
- [ ] Environment variables set in `.env`
- [ ] Automation setup script executed
- [ ] Systemd service installed and running
- [ ] Test training completed successfully
- [ ] Telegram notifications received
- [ ] API integration verified
- [ ] Logs monitoring setup
- [ ] Error handling tested

---

## 🎯 Expected Results

Once fully configured, you should receive:

1. **Setup Confirmation**: "🤖 XGBoost Automation System Online!"
2. **Training Notifications**: Every 6 hours with model updates
3. **API Availability**: New models automatically available via API
4. **Error Alerts**: Immediate notifications for any issues
5. **Performance Reports**: Regular updates on model performance

---

## 📞 Support

For issues or questions:
- 📱 Telegram: @DragonFortuneAI
- 🌐 API: https://api.dragonfortune.ai/docs
- 📊 Dashboard: https://dragonfortune.ai

**🔥 Happy Automated Trading!**