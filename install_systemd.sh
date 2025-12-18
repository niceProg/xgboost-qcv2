#!/bin/bash

# XGBoost Real-time Systemd Service Installer
# One-click installation for production deployment

echo "🚀 XGBoost Real-time Monitor - Systemd Service Installer"
echo "======================================================"

# Get current user and directory
CURRENT_USER=$(whoami)
CURRENT_DIR=$(pwd)
SERVICE_NAME="xgboost-realtime"

echo ""
echo "📋 Installation Details:"
echo "   User: $CURRENT_USER"
echo "   Directory: $CURRENT_DIR"
echo "   Service Name: $SERVICE_NAME"
echo ""

# Check if running as root for system installation
if [[ $EUID -eq 0 ]]; then
    echo "⚠️  Running as root detected"
    echo "   Current user will be: $CURRENT_USER"
    read -p "❓ Continue with $CURRENT_USER as service user? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Installation cancelled"
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "$CURRENT_DIR/.xgboost-qc" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Creating virtual environment..."

    python3 -m venv .xgboost-qc
    source .xgboost-qc/bin/activate

    # Install dependencies
    if [ -f "requirements.txt" ]; then
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
    else
        echo "📦 Installing basic dependencies..."
        pip install pymysql python-dotenv schedule requests pytz
    fi
else
    echo "✅ Virtual environment found"
fi

# Check .env file
if [ ! -f "$CURRENT_DIR/.env" ]; then
    echo "⚠️  .env file not found!"
    echo "   Creating template .env file..."
    cat > .env << EOF
# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Database Configuration
TRADING_DB_HOST=localhost
TRADING_DB_PORT=3306
TRADING_DB_USER=your_db_user
TRADING_DB_PASSWORD=your_db_password
TRADING_DB_NAME=newera

# Additional Configuration
EXCHANGE=binance
PAIR=BTCUSDT
INTERVAL=1h
EOF
    echo ""
    echo "📝 Please edit .env file with your credentials:"
    echo "   nano .env"
    echo ""
    read -p "❓ Continue after editing .env? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Installation cancelled"
        exit 1
    fi
else
    echo "✅ .env file found"
fi

# Check required Python scripts
REQUIRED_SCRIPTS=("realtime_monitor.py" "realtime_trainer_pipeline.py")
for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ ! -f "$CURRENT_DIR/$script" ]; then
        echo "❌ Required script not found: $script"
        echo "   Please ensure all required files are in: $CURRENT_DIR"
        exit 1
    fi
done
echo "✅ All required scripts found"

# Create directories
echo "📁 Creating directories..."
mkdir -p logs state output_train/models
chmod 755 logs state output_train

# Create systemd service file
echo "🔧 Creating systemd service..."
cat > $SERVICE_NAME.service << EOF
[Unit]
Description=XGBoost Real-time Trading System Monitor
After=network.target mysql.service
Wants=network.target

[Service]
Type=simple
User=$CURRENT_USER
Group=$CURRENT_USER
WorkingDirectory=$CURRENT_DIR
Environment=PATH=$CURRENT_DIR/.xgboost-qc/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONPATH=$CURRENT_DIR
EnvironmentFile=-$CURRENT_DIR/.env
ExecStart=/bin/bash -c "source .xgboost-qc/bin/activate && python realtime_monitor.py"
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always
RestartSec=10
StandardOutput=append:$CURRENT_DIR/logs/systemd_stdout.log
StandardError=append:$CURRENT_DIR/logs/systemd_stderr.log

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$CURRENT_DIR/logs $CURRENT_DIR/state $CURRENT_DIR/output_train

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Service file created: $SERVICE_NAME.service"

# Install to systemd
echo ""
echo "📦 Installing to systemd..."

if [[ $EUID -eq 0 ]]; then
    # Running as root
    cp $SERVICE_NAME.service /etc/systemd/system/
    systemctl daemon-reload
    echo "✅ Service installed to /etc/systemd/system/"
else
    # Need sudo
    echo "⚠️  Sudo required for system installation"
    read -p "❓ Install service system-wide? (requires sudo) (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo cp $SERVICE_NAME.service /etc/systemd/system/
        sudo systemctl daemon-reload
        echo "✅ Service installed system-wide"
        SYSTEMD_SUDO=true
    else
        echo "⚠️  Service not installed. You can manually install later:"
        echo "   sudo cp $SERVICE_NAME.service /etc/systemd/system/"
        echo "   sudo systemctl daemon-reload"
        exit 0
    fi
fi

# Enable and optionally start service
echo ""
echo "⚙️  Configuring service..."
if [[ $SYSTEMD_SUDO == true ]] || [[ $EUID -eq 0 ]]; then
    sudo systemctl enable $SERVICE_NAME
    echo "✅ Service enabled to start on boot"

    echo ""
    read -p "❓ Start service now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo systemctl start $SERVICE_NAME
        sleep 2
        if sudo systemctl is-active --quiet $SERVICE_NAME; then
            echo "✅ Service started successfully!"
        else
            echo "❌ Service failed to start!"
            sudo systemctl status $SERVICE_NAME
            exit 1
        fi
    fi
else
    echo "⚠️  Run these commands to complete installation:"
    echo "   sudo systemctl enable $SERVICE_NAME"
    echo "   sudo systemctl start $SERVICE_NAME"
fi

# Clean up temporary service file
rm -f $SERVICE_NAME.service

echo ""
echo "🎉 Installation Complete!"
echo "========================="
echo ""
echo "📋 Service Commands:"
echo "   sudo systemctl start $SERVICE_NAME      # Start service"
echo "   sudo systemctl stop $SERVICE_NAME       # Stop service"
echo "   sudo systemctl restart $SERVICE_NAME    # Restart service"
echo "   sudo systemctl status $SERVICE_NAME     # Check status"
echo "   sudo systemctl enable $SERVICE_NAME     # Enable on boot"
echo "   sudo systemctl disable $SERVICE_NAME    # Disable on boot"
echo ""
echo "📋 Log Commands:"
echo "   journalctl -u $SERVICE_NAME -f          # Follow service logs"
echo "   tail -f $CURRENT_DIR/logs/realtime_monitor.log  # Monitor logs"
echo "   tail -f $CURRENT_DIR/logs/realtime_trainer.log   # Trainer logs"
echo ""
echo "📁 Important Directories:"
echo "   Logs:      $CURRENT_DIR/logs/"
echo "   State:     $CURRENT_DIR/state/"
echo "   Models:    $CURRENT_DIR/output_train/models/"
echo ""
echo "⚡ Service is now running and will:"
echo "   ✓ Check database every 1 minute"
echo "   ✓ Trigger training if ≥10 new records found"
echo "   ✓ Send Telegram notifications"
echo "   ✓ Run 6-step CORE pipeline"
echo "   ✓ Auto-restart if crashes"
echo ""

if [[ $SYSTEMD_SUDO == true ]] || [[ $EUID -eq 0 ]]; then
    if sudo systemctl is-active --quiet $SERVICE_NAME; then
        echo "🟢 Service Status: RUNNING"
        echo ""
        echo "📊 Current status:"
        sudo systemctl status $SERVICE_NAME --no-pager -l
    else
        echo "🔴 Service Status: NOT RUNNING"
        echo ""
        echo "❌ Check status with: sudo systemctl status $SERVICE_NAME"
    fi
fi

echo "✨ Done! Your XGBoost real-time system is ready!"