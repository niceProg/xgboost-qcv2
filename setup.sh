#!/bin/bash

# Setup script for XGBoost Trading Pipeline
# This script sets up both daily runner and API server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}XGBoost Trading Pipeline Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please don't run this script as root. Use a regular user with sudo access."
    exit 1
fi

# Get current user
USER_NAME=$(whoami)
PROJECT_DIR=$(pwd)

print_status "Setting up for user: $USER_NAME"
print_status "Project directory: $PROJECT_DIR"

# 1. Check Python version
echo -e "\n${YELLOW}Step 1: Checking Python installation${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Found Python $PYTHON_VERSION"
else
    print_error "Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# 2. Install system dependencies
echo -e "\n${YELLOW}Step 2: Installing system dependencies${NC}"
sudo apt update
sudo apt install -y python3-pip python3-venv curl

# 3. Create and activate virtual environment
echo -e "\n${YELLOW}Step 3: Setting up Python virtual environment${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Created virtual environment"
fi

# Activate venv for subsequent commands
source venv/bin/activate

# 4. Install Python dependencies
echo -e "\n${YELLOW}Step 4: Installing Python dependencies${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_status "Installed requirements.txt"
fi

if [ -f "requirements_db.txt" ]; then
    pip install -r requirements_db.txt
    print_status "Installed requirements_db.txt"
fi

# 5. Setup environment file
echo -e "\n${YELLOW}Step 5: Setting up environment configuration${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_status "Created .env from .env.example"
        print_warning "Please edit .env file with your database credentials"
    else
        cat > .env << EOF
# Database Configuration
TRADING_DB_HOST=localhost
TRADING_DB_PORT=3306
TRADING_DB_NAME=trading_data
TRADING_DB_USER=your_username
TRADING_DB_PASSWORD=your_password

# Training Configuration
OUTPUT_DIR=./output_train
ENABLE_DB_STORAGE=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
EOF
        print_status "Created default .env file"
        print_warning "Please update .env with your actual configuration"
    fi
else
    print_status ".env file already exists"
fi

# 6. Create necessary directories
echo -e "\n${YELLOW}Step 6: Creating necessary directories${NC}"
mkdir -p output_train logs
print_status "Created directories: output_train, logs"

# 7. Setup systemd service for daily runner
echo -e "\n${YELLOW}Step 7: Setting up systemd service for daily runs${NC}"
if [ -f "xgboost-daily.service" ]; then
    # Update service file with actual paths
    sed -i "s|/home/ubuntu|$HOME|g" xgboost-daily.service
    sed -i "s|User=ubuntu|User=$USER_NAME|g" xgboost-daily.service
    sed -i "s|Group=ubuntu|Group=$USER_NAME|g" xgboost-daily.service

    sudo cp xgboost-daily.service /etc/systemd/system/
    sudo cp xgboost-daily.timer /etc/systemd/system/

    sudo systemctl daemon-reload
    print_status "Installed systemd service files"
else
    print_warning "xgboost-daily.service not found, skipping systemd setup"
fi

# 8. Check Docker installation
echo -e "\n${YELLOW}Step 8: Checking Docker installation${NC}"
if command -v docker &> /dev/null; then
    print_status "Docker is installed"

    # Add user to docker group if not already
    if ! groups $USER_NAME | grep -q "docker"; then
        print_warning "Adding $USER_NAME to docker group. You may need to log out and log back in."
        sudo usermod -aG docker $USER_NAME
    fi

    # Check docker-compose
    if command -v docker-compose &> /dev/null; then
        print_status "Docker Compose is installed"
    else
        print_warning "Docker Compose not found. Installing..."
        sudo apt install -y docker-compose
    fi
else
    print_warning "Docker is not installed. API server deployment requires Docker."
    print_warning "To install Docker, run: curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh"
fi

# 9. Setup log rotation
echo -e "\n${YELLOW}Step 9: Setting up log rotation${NC}"
sudo tee /etc/logrotate.d/xgboost-pipeline > /dev/null << EOF
$PROJECT_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF
print_status "Configured log rotation"

# 10. Create startup scripts
echo -e "\n${YELLOW}Step 10: Creating convenience scripts${NC}"

# Script to start API server
cat > start_api.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export API_HOST=0.0.0.0
export API_PORT=8000
uvicorn api_server:app --host $API_HOST --port $API_PORT --reload
EOF
chmod +x start_api.sh
print_status "Created start_api.sh"

# Script to run daily pipeline
cat > run_daily.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python daily_runner.py
EOF
chmod +x run_daily.sh
print_status "Created run_daily.sh"

# 11. Final instructions
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${YELLOW}Next Steps:${NC}\n"

echo -e "1. ${GREEN}Edit configuration:${NC}"
echo -e "   nano .env\n"

echo -e "2. ${GREEN}Test the pipeline:${NC}"
echo -e "   python daily_runner.py --dry-run\n"

echo -e "3. ${GREEN}Enable daily runs:${NC}"
echo -e "   sudo systemctl enable xgboost-daily.timer"
echo -e "   sudo systemctl start xgboost-daily.timer\n"

echo -e "4. ${GREEN}Start API server (option A):${NC}"
echo -e "   ./start_api.sh\n"

echo -e "5. ${GREEN}Start API server (option B - Docker):${NC}"
echo -e "   docker-compose -f docker-compose.api.yml up -d\n"

echo -e "6. ${GREEN}Check API health:${NC}"
echo -e "   curl http://localhost:8000/api/v1/health\n"

echo -e "${YELLOW}Useful Commands:${NC}"
echo -e "- Check timer status: systemctl status xgboost-daily.timer"
echo -e "- View logs: sudo journalctl -u xgboost-daily.service -f"
echo -e "- API docs: http://localhost:8000/docs"
echo -e "- Stop API: docker-compose -f docker-compose.api.yml down\n"

echo -e "${YELLOW}Notes:${NC}"
echo -e "- If you were added to docker group, you may need to: newgrp docker"
echo -e "- The timer runs weekdays at 09:30 UTC (16:30 WIB)"
echo -e "- All outputs are saved in: $PROJECT_DIR/output_train"
echo -e "- Logs are saved in: $PROJECT_DIR/logs\n"

echo -e "${GREEN}Happy trading!${NC}\n"