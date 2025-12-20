# SSL/HTTPS Setup Guide for api.dragonfortune.ai

This guide will help you set up SSL/HTTPS for your XGBoost API domain using aPanel.

## Prerequisites

- aPanel installed on your server
- Domain `api.dragonfortune.ai` pointed to your server IP
- Docker and Docker Compose installed
- XGBoost API project deployed

## Step 1: Domain Configuration

### 1.1 DNS Configuration
Ensure your domain `api.dragonfortune.ai` is properly configured:

```bash
# Check DNS resolution
nslookup api.dragonfortune.ai
# Should return your server IP
```

### 1.2 Firewall Configuration
Open necessary ports:

```bash
# Open port 80 (for SSL verification)
sudo ufw allow 80/tcp
# Open port 443 (for HTTPS)
sudo ufw allow 443/tcp
# Open port 5000 (for API)
sudo ufw allow 5000/tcp
```

## Step 2: Install SSL Certificate in aPanel

### 2.1 Access aPanel SSL Module

1. Log in to your aPanel dashboard
2. Navigate to **Websites** → **SSL**
3. Click on your domain `api.dragonfortune.ai`

### 2.2 Install Let's Encrypt Certificate

1. Click **Install Certificate** or **SSL Management**
2. Choose **Let's Encrypt** as certificate provider
3. Ensure these settings are configured:
   - Domain: `api.dragonfortune.ai`
   - Force HTTPS: ✅ Enabled
   - Auto Renewal: ✅ Enabled
4. Click **Install/Issue Certificate**

### 2.3 Alternative: Manual SSL Installation

If Let's Encrypt isn't available in aPanel:

1. Generate SSL certificate using Certbot:

```bash
# Install certbot if not installed
sudo apt update
sudo apt install certbot

# Generate certificate for nginx
sudo certbot certonly --nginx -d api.dragonfortune.ai

# Or for standalone (if nginx not configured yet)
sudo certbot certonly --standalone -d api.dragonfortune.ai
```

2. Certificate files will be located at:
   - Private Key: `/etc/letsencrypt/live/api.dragonfortune.ai/privkey.pem`
   - Certificate: `/etc/letsencrypt/live/api.dragonfortune.ai/fullchain.pem`

## Step 3: Configure Nginx Reverse Proxy

Create Nginx configuration to handle HTTPS and proxy to Docker container:

### 3.1 Create Nginx Configuration File

```bash
sudo nano /etc/nginx/sites-available/api.dragonfortune.ai
```

Add this configuration:

```nginx
server {
    listen 80;
    server_name api.dragonfortune.ai;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.dragonfortune.ai;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/api.dragonfortune.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.dragonfortune.ai/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Logging
    access_log /var/log/nginx/api.dragonfortune.ai.access.log;
    error_log /var/log/nginx/api.dragonfortune.ai.error.log;

    # Proxy to Docker Container
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Health Check (optional)
    location /health {
        proxy_pass http://127.0.0.1:5000/health;
        access_log off;
    }
}
```

### 3.2 Enable the Site

```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/api.dragonfortune.ai /etc/nginx/sites-enabled/

# Test Nginx configuration
sudo nginx -t

# If successful, reload Nginx
sudo systemctl reload nginx
```

## Step 4: Update Environment Configuration

Update your `.env.production` file for HTTPS:

```bash
nano /home/yumna/Working/dragonfortune/xgboost-qc/.env.production
```

Ensure it includes:

```env
# Production Environment Variables
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=false

# Database Configuration
DB_HOST=103.150.81.86
DB_PORT=3306
DB_NAME=xgboostqc
DB_USER=xgboostqc
DB_PASSWORD=6SPxBDwXH6WyxpfT

# CORS Configuration - HTTPS Only
ALLOWED_ORIGINS=https://api.dragonfortune.ai,https://app.dragonfortune.ai

# Optional: Add if behind proxy
PROXIED_REQUESTS=true
```

## Step 5: Update Docker Configuration for Production

Update your `docker-compose.api.yml` for production deployment:

```yaml
version: '3.8'

services:
  xgboost-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: xgboost-api
    ports:
      - "127.0.0.1:5000:5000"  # Only listen on localhost
    env_file:
      - .env.production
    environment:
      - DB_HOST=${DB_HOST:-103.150.81.86}
      - DB_PORT=${DB_PORT:-3306}
      - DB_NAME=${DB_NAME:-xgboostqc}
      - DB_USER=${DB_USER:-xgboostqc}
      - DB_PASSWORD=${DB_PASSWORD:-6SPxBDwXH6WyxpfT}
      - PROXIED_REQUESTS=true
    volumes:
      - ./output_train:/app/output_train
    networks:
      - xgboost-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  xgboost-network:
    driver: bridge
```

## Step 6: Deploy and Test

### 6.1 Deploy the API

```bash
# Navigate to project directory
cd /home/yumna/Working/dragonfortune/xgboost-qc

# Build and deploy
docker-compose -f docker-compose.api.yml down
docker-compose -f docker-compose.api.yml build
docker-compose -f docker-compose.api.yml up -d
```

### 6.2 Test HTTPS Endpoints

```bash
# Test health endpoint
curl -X GET "https://api.dragonfortune.ai/health"

# Test API with verbose output
curl -v -X GET "https://api.dragonfortune.ai/api/v1/sessions"

# Test with certificate validation
curl --cacert /etc/ssl/certs/ca-certificates.crt -X GET "https://api.dragonfortune.ai/health"
```

### 6.3 Verify SSL Certificate

```bash
# Check SSL certificate
openssl s_client -connect api.dragonfortune.ai:443 -servername api.dragonfortune.ai

# Check SSL certificate details
curl -vI https://api.dragonfortune.ai

# Check SSL rating (optional)
sudo apt install ssllabs-scan
ssllabs-scan api.dragonfortune.ai
```

## Step 7: Update QuantConnect Integration

Update your `quantconnect_integration_example.py` to use HTTPS:

```python
# Update this line
API_BASE_URL = "https://api.dragonfortune.ai"
```

## Step 8: Monitoring and Maintenance

### 8.1 SSL Certificate Auto-Renewal

Let's Encrypt certificates auto-renew, but you can test:

```bash
# Test renewal process
sudo certbot renew --dry-run

# Set up cron job for auto-renewal (if not already)
sudo crontab -e
# Add this line:
# 0 12 * * * /usr/bin/certbot renew --quiet
```

### 8.2 Monitoring

Monitor your API:

```bash
# Check Nginx logs
sudo tail -f /var/log/nginx/api.dragonfortune.ai.access.log
sudo tail -f /var/log/nginx/api.dragonfortune.ai.error.log

# Check Docker logs
docker-compose -f docker-compose.api.yml logs -f

# Check SSL certificate expiration
sudo certbot certificates
```

## Troubleshooting

### Common Issues:

1. **Certificate not found**
   - Ensure DNS is properly configured
   - Check that domain points to correct IP

2. **Nginx errors**
   - Check Nginx configuration: `sudo nginx -t`
   - Check logs: `sudo journalctl -u nginx`

3. **Proxy connection errors**
   - Ensure Docker container is running
   - Check if port 5000 is accessible from localhost

4. **CORS errors**
   - Update ALLOWED_ORIGINS in `.env.production`
   - Restart Docker container after changes

5. **SSL verification errors**
   - Wait a few minutes after DNS changes
   - Check certificate chain: `openssl s_client -connect api.dragonfortune.ai:443`

### Commands for Debugging:

```bash
# Check if API container is running
docker ps | grep xgboost-api

# Check port binding
netstat -tlnp | grep :5000

# Test direct container access
curl http://localhost:5000/health

# Test through Nginx proxy
curl http://api.dragonfortune.ai/health

# Test HTTPS
curl https://api.dragonfortune.ai/health
```

## Security Recommendations

1. **Firewall Configuration**
   - Only open necessary ports (80, 443)
   - Use UFW or iptables for additional security

2. **Rate Limiting**
   - Add rate limiting to Nginx configuration

3. **API Security**
   - Consider adding API key authentication
   - Implement request rate limiting in FastAPI

4. **Regular Updates**
   - Keep Nginx, Docker, and certificates updated
   - Monitor security advisories

## Success Checklist

- [ ] DNS points to correct server IP
- [ ] SSL certificate installed and valid
- [ ] HTTP redirects to HTTPS
- [ ] Nginx proxy working correctly
- [ ] Docker container accessible through proxy
- [ ] API endpoints accessible via HTTPS
- [ ] CORS properly configured for Dragon Fortune domains
- [ ] QuantConnect integration working with HTTPS
- [ ] Auto-renewal configured for SSL certificate

Once completed, your XGBoost API will be accessible at:
- **API Base URL**: `https://api.dragonfortune.ai`
- **API Documentation**: `https://api.dragonfortune.ai/docs`
- **Health Check**: `https://api.dragonfortune.ai/health`