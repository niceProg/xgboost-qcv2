#!/bin/bash

# XGBoost API HTTPS Setup Script
# This script sets up HTTPS for api.dragonfortune.ai domain

set -e

echo "🔒 XGBoost API HTTPS Setup Script"
echo "=================================="

# Configuration
DOMAIN="api.dragonfortune.ai"
PROJECT_DIR="/home/yumna/Working/dragonfortune/xgboost-qc"
NGINX_CONFIG="/etc/nginx/sites-available/$DOMAIN"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script needs to be run with sudo privileges"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check if aPanel is installed
    if [ -d "/usr/local/aapanel" ]; then
        print_success "aPanel detected"
    else
        print_info "aPanel not detected, proceeding with manual setup"
    fi

    # Check if domain resolves to this server
    SERVER_IP=$(curl -s ifconfig.me)
    DOMAIN_IP=$(nslookup $DOMAIN | grep -A1 'Name:' | tail -1 | awk '{print $2}')

    if [ "$SERVER_IP" = "$DOMAIN_IP" ]; then
        print_success "Domain $DOMAIN resolves to server IP: $SERVER_IP"
    else
        print_error "Domain $DOMAIN does not resolve to server IP"
        print_info "Server IP: $SERVER_IP"
        print_info "Domain IP: $DOMAIN_IP"
        print_error "Please configure DNS correctly before proceeding"
        exit 1
    fi

    # Check if docker is installed
    if command -v docker &> /dev/null; then
        print_success "Docker is installed"
    else
        print_error "Docker is not installed"
        exit 1
    fi

    # Check if nginx is installed
    if command -v nginx &> /dev/null; then
        print_success "Nginx is installed"
    else
        print_error "Nginx is not installed"
        print_info "Installing Nginx..."
        apt update && apt install -y nginx
    fi
}

# Install SSL certificate
install_ssl() {
    print_info "Installing SSL certificate for $DOMAIN..."

    # Install certbot if not installed
    if ! command -v certbot &> /dev/null; then
        print_info "Installing Certbot..."
        apt update
        apt install -y certbot python3-certbot-nginx
    fi

    # Get SSL certificate
    if certbot certificates | grep -q "$DOMAIN"; then
        print_success "SSL certificate already exists for $DOMAIN"
    else
        print_info "Obtaining new SSL certificate..."
        certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@dragonfortune.ai --redirect
        print_success "SSL certificate obtained for $DOMAIN"
    fi
}

# Create Nginx configuration
create_nginx_config() {
    print_info "Creating Nginx configuration..."

    # Backup existing config if it exists
    if [ -f "$NGINX_CONFIG" ]; then
        cp "$NGINX_CONFIG" "$NGINX_CONFIG.backup.$(date +%Y%m%d%H%M%S)"
        print_info "Backed up existing Nginx configuration"
    fi

    # Create new configuration
    cat > "$NGINX_CONFIG" << EOF
server {
    listen 80;
    server_name $DOMAIN;

    # Redirect HTTP to HTTPS
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name $DOMAIN;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # CORS Headers (public access)
    add_header Access-Control-Allow-Origin * always;
    add_header Access-Control-Allow-Methods "GET, POST, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Content-Type, Authorization" always;

    # Security Headers
    add_header X-Frame-Options SAMEORIGIN;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Logging
    access_log /var/log/nginx/$DOMAIN.access.log;
    error_log /var/log/nginx/$DOMAIN.error.log;

    # Proxy to Docker Container
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Health Check
    location /health {
        proxy_pass http://127.0.0.1:5000/health;
        access_log off;
    }
}
EOF

    print_success "Nginx configuration created"
}

# Enable Nginx site
enable_nginx_site() {
    print_info "Enabling Nginx site..."

    # Enable the site
    ln -sf "$NGINX_CONFIG" "/etc/nginx/sites-enabled/"

    # Test configuration
    nginx -t

    if [ $? -eq 0 ]; then
        print_success "Nginx configuration is valid"
        systemctl reload nginx
        print_success "Nginx reloaded"
    else
        print_error "Nginx configuration error"
        exit 1
    fi
}

# Update Docker configuration
update_docker_config() {
    print_info "Updating Docker configuration..."

    # Update docker-compose.api.yml to only bind to localhost
    if [ -f "$PROJECT_DIR/docker-compose.api.yml" ]; then
        sed -i 's/- "5000:5000"/- "127.0.0.1:5000:5000"/' "$PROJECT_DIR/docker-compose.api.yml"
        print_success "Updated Docker Compose to bind to localhost"
    fi
}

# Deploy API
deploy_api() {
    print_info "Deploying XGBoost API..."

    cd "$PROJECT_DIR"

    # Stop existing container
    docker-compose -f docker-compose.api.yml down 2>/dev/null || true

    # Build and start
    docker-compose -f docker-compose.api.yml build
    docker-compose -f docker-compose.api.yml up -d

    # Wait for container to start
    sleep 10

    print_success "XGBoost API deployed"
}

# Test configuration
test_configuration() {
    print_info "Testing HTTPS configuration..."

    # Wait a bit for services to start
    sleep 5

    # Test HTTP to HTTPS redirect
    if curl -s -I http://$DOMAIN | grep -q "301 Moved Permanently"; then
        print_success "HTTP to HTTPS redirect working"
    else
        print_error "HTTP to HTTPS redirect not working"
    fi

    # Test HTTPS access
    if curl -s -k https://$DOMAIN/health | grep -q "OK"; then
        print_success "HTTPS API endpoint working"
    else
        print_error "HTTPS API endpoint not accessible"
    fi

    # Test SSL certificate
    if openssl s_client -connect $DOMAIN:443 -servername $DOMAIN </dev/null 2>/dev/null | grep -q "Verification: OK"; then
        print_success "SSL certificate is valid"
    else
        print_info "SSL certificate might need more time to propagate"
    fi
}

# Setup auto-renewal
setup_auto_renewal() {
    print_info "Setting up SSL certificate auto-renewal..."

    # Test renewal
    certbot renew --dry-run

    # Add cron job if not exists
    if ! crontab -l 2>/dev/null | grep -q "certbot renew"; then
        (crontab -l 2>/dev/null; echo "0 12 * * * /usr/bin/certbot renew --quiet") | crontab -
        print_success "Auto-renewal cron job added"
    else
        print_success "Auto-renewal already configured"
    fi
}

# Display final information
display_info() {
    echo ""
    print_success "🎉 HTTPS setup completed successfully!"
    echo ""
    echo "📍 API Endpoints:"
    echo "   • HTTPS API: https://$DOMAIN"
    echo "   • Health: https://$DOMAIN/health"
    echo "   • Docs: https://$DOMAIN/docs"
    echo ""
    echo "🔧 Useful Commands:"
    echo "   • View API logs: docker-compose -f $PROJECT_DIR/docker-compose.api.yml logs -f"
    echo "   • View Nginx logs: tail -f /var/log/nginx/$DOMAIN.access.log"
    echo "   • Restart API: docker-compose -f $PROJECT_DIR/docker-compose.api.yml restart"
    echo "   • Test SSL: openssl s_client -connect $DOMAIN:443"
    echo ""
    echo "📚 QuantConnect Integration:"
    echo "   Update API_BASE_URL in quantconnect_integration_example.py to:"
    echo "   API_BASE_URL = \"https://$DOMAIN\""
    echo ""
}

# Main execution
main() {
    check_root
    check_prerequisites
    install_ssl
    create_nginx_config
    enable_nginx_site
    update_docker_config
    deploy_api
    test_configuration
    setup_auto_renewal
    display_info
}

# Run main function
main "$@"