#!/bin/bash

echo "🚀 Deploying XGBoost FastAPI Production Server"
echo "==============================================="

# Check requirements
echo "🔍 Checking requirements..."

# Check domain
if [ -z "$DOMAIN" ]; then
    echo "❌ DOMAIN environment variable not set"
    echo "Set: export DOMAIN=test.dragonfortune.ai"
    exit 1
fi

# Check database credentials
required_vars=("TRADING_DB_HOST" "TRADING_DB_USER" "TRADING_DB_PASSWORD" "TRADING_DB_NAME")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "❌ Missing environment variables: ${missing_vars[*]}"
    exit 1
fi

# Create production directories
echo "📁 Creating production directories..."
mkdir -p logs/nginx logs/api logs/cron
mkdir -p nginx ssl redis
mkdir -p state/monitoring

# Update nginx config
echo "🔧 Configuring Nginx..."
cat > nginx/prod.conf << EOF
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api_limit:10m rate=20r/m;
    limit_req_zone \$binary_remote_addr zone=health_limit:10m rate=60r/m;

    # Upstream API server
    upstream xgboost_api {
        server xgboost_api_prod:8000;
        keepalive 32;
    }

    # HTTP redirect to HTTPS
    server {
        listen 80;
        server_name $DOMAIN;
        return 301 https://\$server_name\$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name $DOMAIN;

        # SSL Configuration
        ssl_certificate /etc/ssl/cert.pem;
        ssl_certificate_key /etc/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # CORS for QuantConnect
        add_header Access-Control-Allow-Origin "https://www.quantconnect.com";
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
        add_header Access-Control-Allow-Headers "Content-Type, Authorization";

        # API endpoints with rate limiting
        location / {
            limit_req zone=api_limit burst=10 nodelay;

            proxy_pass http://xgboost_api;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;

            # Timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;

            # Enable keep-alive
            proxy_http_version 1.1;
            proxy_set_header Connection "";

            # Buffer settings
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
        }

        # Health check endpoint (no rate limiting)
        location /health {
            limit_req zone=health_limit burst=20 nodelay;
            proxy_pass http://xgboost_api/health;
            access_log off;
        }

        # CORS preflight handling
        location = /options {
            if (\$request_method = 'OPTIONS') {
                add_header Access-Control-Allow-Origin "https://www.quantconnect.com";
                add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
                add_header Access-Control-Allow-Headers "Content-Type, Authorization";
                add_header Access-Control-Max-Age 1728000;
                add_header Content-Length 0;
                add_header Content-Type text/plain;
                return 204;
            }
        }
    }
}
EOF

# Create Redis config
echo "📦 Configuring Redis..."
cat > redis/redis.conf << EOF
# Redis Configuration
bind 0.0.0.0
port 6379
timeout 0
tcp-keepalive 300
daemonize no

# Memory
maxmemory 256mb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Logging
loglevel notice
logfile /var/log/redis/redis.log

# Security
protected-mode no
requirepass your_redis_password

# Performance
tcp-backlog 511
databases 16
EOF

# Create logrotate config
echo "📋 Setting up log rotation..."
cat > logrotate.conf << EOF
# API logs
/var/log/api/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 appuser appuser
    postrotate
        docker kill -s USR1 xgboost_api_prod
    endscript
}

# Monitor logs
/var/log/cron/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644 appuser appuser
}

# Nginx logs
/var/log/nginx/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 nginx nginx
    postrotate
        docker exec xgboost_nginx_prod nginx -s reload
    endscript
}
EOF

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose-prod-api.yml down

# Build and start services
echo "🔨 Building and starting production services..."
docker-compose -f docker-compose-prod-api.yml up --build -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 60

# Health checks
echo "🏥 Performing health checks..."

# Check API health
if curl -f https://$DOMAIN/health > /dev/null 2>&1; then
    echo "✅ API health check passed"
else
    echo "❌ API health check failed"
    docker logs xgboost_api_prod
fi

# Test signal endpoint
echo "📊 Testing signal generation..."
response=$(curl -s -X POST https://$DOMAIN/signal \
    -H 'Content-Type: application/json' \
    -d '{"symbol": "BTCUSDT", "threshold": 0.5}')

if echo "$response" | grep -q "signal"; then
    echo "✅ Signal endpoint working"
    echo "📈 Sample signal: $response"
else
    echo "❌ Signal endpoint failed"
fi

# Setup SSL certificate (if needed)
if [ ! -f "ssl/cert.pem" ]; then
    echo "🔐 SSL certificate not found"
    echo "Options:"
    echo "1. Let's Encrypt: certbot certonly --nginx -d $DOMAIN"
    echo "2. Self-signed: openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ssl/key.pem -out ssl/cert.pem"
    echo "3. Upload your existing certificates to ssl/"
fi

# Display deployment information
echo ""
echo "🎉 Production API Server Deployed!"
echo "=================================="
echo ""
echo "🌐 API Endpoints:"
echo "HTTPS: https://$DOMAIN"
echo "Health: https://$DOMAIN/health"
echo "Signal: https://$DOMAIN/signal"
echo ""
echo "📊 Monitoring:"
echo "Docker stats: docker stats"
echo "View logs: docker-compose -f docker-compose-prod-api.yml logs -f"
echo "API logs: docker logs -f xgboost_api_prod"
echo "Monitor logs: docker logs -f realtime_monitor_prod"
echo ""
echo "🔧 Management:"
echo "Restart: docker-compose -f docker-compose-prod-api.yml restart"
echo "Stop: docker-compose -f docker-compose-prod-api.yml down"
echo "Update: docker-compose -f docker-compose-prod-api.yml pull && docker-compose -f docker-compose-prod-api.yml up -d"
echo ""
echo "📋 Environment Variables Set:"
echo "- DOMAIN=$DOMAIN"
echo "- TRADING_DB_HOST=$TRADING_DB_HOST"
echo "- TRADING_DB_NAME=$TRADING_DB_NAME"
echo ""
echo "✅ Production server is ready for QuantConnect integration!"

# Create deployment info file
cat > deployment-info.json << EOF
{
  "deployment": {
    "domain": "$DOMAIN",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "api_url": "https://$DOMAIN",
    "status": "active"
  },
  "endpoints": {
    "health": "/health",
    "signal": "/signal",
    "predict": "/predict",
    "status": "/status"
  },
  "containers": {
    "api": "xgboost_api_prod",
    "monitor": "realtime_monitor_prod",
    "nginx": "xgboost_nginx_prod",
    "redis": "xgboost_redis_prod"
  }
}
EOF

echo ""
echo "📄 Deployment info saved to deployment-info.json"