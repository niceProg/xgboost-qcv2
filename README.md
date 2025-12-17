# XGBoost Trading System - Production V2

## 🎯 Dual Purpose System

### **1. Incremental Training Engine**
- 📊 **Monitor**: Cek database untuk data baru (2025+)
- 🧠 **Train**: Incremental XGBoost training (hanya data baru)
- 🔄 **Update**: Model otomatis updated tanpa training ulang dari 2024

### **2. Universal FastAPI Server**
- 🌐 **Open API**: Untuk siapapun (QuantConnect, external apps, web, mobile)
- 📚 **Complete Documentation**: Auto-generated OpenAPI docs
- 🔌 **Comprehensive Routes**: Full CRUD dan utility endpoints

**Historical training tetap pakai:** `simple_run.sh` di parent folder (hanya sekali untuk initial model)

## 🚀 Quick Start - Real-time System

### Prerequisites:
- ✅ Historical training sudah dijalankan (`simple_run.sh`)
- ✅ Model files sudah ada di `../output_train/`
- ✅ Database credentials sudah di setup di `.env`

### Step 1: Configure Environment
```bash
# Copy template
cp .env.example .env

# Edit dengan credentials kamu
nano .env
```

### Step 2: Deploy Real-time System
```bash
# One-click deployment
./deploy.sh
```

### Step 3: Check Status
```bash
./status.sh
```

## 📁 Folder Structure (Clean & Focused)

```
production-v2/
├── README.md                      # This file
├── .env.example                   # Environment template
├── deploy.sh                      # One-click real-time deployment ⭐
├── setup_database.py              # Database setup untuk monitoring
├── requirements.txt               # Python dependencies
├──
├── 🔥 Core Real-time Components:
│   ├── realtime_monitor.py        # Smart event-driven monitor
│   ├── realtime_trainer.py        # Incremental model trainer
│   └── quantconnect_api.py        # FastAPI server
├──
├── 📄 QuantConnect Integration:
│   └── XGBoostTradingAlgorithm_Final.py  # Updated algorithm
├──
├── 🐳 Docker Files:
│   ├── docker-compose.yml         # Service orchestration
│   ├── Dockerfile.api             # API server container
│   ├── Dockerfile.monitor         # Monitor container
│   └── Dockerfile.trainer         # Trainer container
├──
└── 🔧 Management Scripts:
    ├── status.sh                  # System status checker
    ├── stop.sh                    # Stop all services
    ├── restart.sh                 # Restart services
    ├── trigger_training.sh        # Manual training trigger
    └── test_api.sh                # Test API endpoints
```

## 🔄 System Workflow

### Pre-requisite (Historical):
```bash
# Run this ONCE from parent folder
cd /home/yumna/Working/dragonfortune/xgboost-qc
./simple_run.sh
```

### Real-time (Continuous):
```bash
# Run this AFTER historical training
cd production-v2
./deploy.sh
```

### Data Flow:
```
New 2025 Data → Database → Real-time Monitor → Real-time Trainer → Updated Model → QuantConnect API → Trading Algorithm
```

## 📊 API Endpoints

### Health Check
```bash
GET /health
Response: {"status": "healthy", "model_available": true}
```

### Generate Trading Signal
```bash
POST /signal
{
  "exchange": "binance",
  "symbol": "BTCUSDT",
  "interval": "1h"
}
Response: {
  "signal": "BUY",
  "confidence": 0.85,
  "recommendation": {...}
}
```

### Model Prediction
```bash
POST /predict
{
  "features": {
    "price_close": 42000.0,
    "volume_usd": 1000000.0
  }
}
Response: {
  "prediction": 1,
  "confidence": 0.78
}
```

## 🔧 QuantConnect Integration

Algorithm sudah updated untuk pakai API:
```python
# XGBoostTradingAlgorithm_Final.py
self.api_base_url = "https://test.dragonfortune.ai:8000"
signal_data = self.GetTradingSignal()  # API call
```

Tidak ada ObjectStore lagi - murni API calls!

## ⚡ Key Features

1. **Smart Event-driven Monitoring**: Uses `created_at` timestamps
2. **Adaptive Intervals**: 15s active, 60s normal, 300s quiet
3. **Priority Processing**: URGENT, HIGH, MEDIUM, LOW
4. **Zero Manual Intervention**: Fully automated
5. **Professional API**: Clean endpoints for QuantConnect
6. **Resource Efficient**: Smart database queries

## 🛠️ Management Commands

```bash
# System operations
./deploy.sh          # Deploy/start system
./status.sh          # Check all services
./stop.sh            # Stop all services
./restart.sh         # Restart services

# Training operations
./trigger_training.sh # Manual training trigger
./test_api.sh         # Test API endpoints

# Logs
docker-compose logs -f quantconnect-api      # API logs
docker-compose logs -f realtime-monitor      # Monitor logs
docker-compose logs -f realtime-trainer      # Trainer logs
```

## 📊 System Components

### 1. Real-time Monitor (`realtime_monitor.py`)
- Monitor 6 database tables untuk new 2025 data
- Smart adaptive checking based on activity patterns
- Trigger training automatically saat data cukup

### 2. Real-time Trainer (`realtime_trainer.py`)
- Incremental XGBoost training
- Performance validation
- Automatic model deployment

### 3. FastAPI Server (`quantconnect_api.py`)
- Real-time predictions
- Trading signal generation
- Health checks and status

### 4. QuantConnect Algorithm
- Replace ObjectStore dengan API calls
- Real-time trading decisions
- Built-in risk management

## 🔐 Environment Variables

```bash
# Trading Database (newera) - Market Data Source
TRADING_DB_HOST=localhost
TRADING_DB_PORT=3306
TRADING_DB_USER=your_db_user
TRADING_DB_PASSWORD=your_db_password
TRADING_DB_NAME=newera

# Results Database (xgboostqc) - Storage
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=xgboostqc

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DOMAIN=test.dragonfortune.ai

# QuantConnect Integration
QUANTCONNECT_CORS_ORIGIN=https://www.quantconnect.com

# Optional: Notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 🚨 Production Considerations

- **SSL/TLS**: Configure HTTPS for production
- **Firewall**: Restrict API access
- **Rate Limiting**: Implement API rate limits
- **Monitoring**: Set up alerts and monitoring
- **Backups**: Regular model and database backups

## 📞 Support

- **Status Check**: `./status.sh`
- **API Test**: `./test_api.sh`
- **Logs**: `docker-compose logs -f`
- **Configuration**: Edit `.env` file

---

**🚀 Production-ready real-time trading system!**