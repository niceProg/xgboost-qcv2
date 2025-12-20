#!/bin/bash

# Test HTTPS Configuration Script
# This script tests all HTTPS endpoints and configurations

DOMAIN="api.dragonfortune.ai"

echo "🧪 Testing HTTPS Configuration for $DOMAIN"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "   ${GREEN}✅ PASSED${NC}"
    else
        echo -e "   ${RED}❌ FAILED${NC}"
    fi
}

echo ""
echo "1. DNS Resolution Test"
echo "----------------------"
if nslookup $DOMAIN > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓ DNS resolves correctly${NC}"
    SERVER_IP=$(curl -s ifconfig.me)
    DOMAIN_IP=$(nslookup $DOMAIN | grep -A1 'Name:' | tail -1 | awk '{print $2}')
    echo "   Server IP: $SERVER_IP"
    echo "   Domain IP: $DOMAIN_IP"
else
    echo -e "   ${RED}✗ DNS resolution failed${NC}"
fi

echo ""
echo "2. HTTP to HTTPS Redirect"
echo "-------------------------"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://$DOMAIN)
if [ "$HTTP_CODE" = "301" ]; then
    echo -e "   ${GREEN}✓ Redirecting HTTP to HTTPS (301)${NC}"
else
    echo -e "   ${RED}✗ Not redirecting (Status: $HTTP_CODE)${NC}"
fi

echo ""
echo "3. SSL Certificate Test"
echo "------------------------"
if openssl s_client -connect $DOMAIN:443 -servername $DOMAIN </dev/null 2>/dev/null | grep -q "Verify return code: 0 (ok)"; then
    echo -e "   ${GREEN}✓ SSL Certificate is valid${NC}"

    # Show certificate info
    EXPIRY_DATE=$(echo | openssl s_client -connect $DOMAIN:443 -servername $DOMAIN 2>/dev/null | openssl x509 -noout -enddate | cut -d= -f2)
    echo "   Expires: $EXPIRY_DATE"
else
    echo -e "   ${RED}✗ SSL Certificate validation failed${NC}"
fi

echo ""
echo "4. API Endpoints Test"
echo "---------------------"
echo -e "${YELLOW}Testing health endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s -k https://$DOMAIN/health)
if echo "$HEALTH_RESPONSE" | grep -q "OK"; then
    echo -e "   ${GREEN}✓ /health endpoint working${NC}"
else
    echo -e "   ${RED}✗ /health endpoint failed${NC}"
    echo "   Response: $HEALTH_RESPONSE"
fi

echo -e "${YELLOW}Testing sessions endpoint...${NC}"
SESSIONS_RESPONSE=$(curl -s -k https://$DOMAIN/api/v1/sessions)
if echo "$SESSIONS_RESPONSE" | grep -q "total_sessions"; then
    echo -e "   ${GREEN}✓ /api/v1/sessions endpoint working${NC}"
else
    echo -e "   ${RED}✗ /api/v1/sessions endpoint failed${NC}"
fi

echo -e "${YELLOW}Testing docs endpoint...${NC}"
DOCS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" https://$DOMAIN/docs)
if [ "$DOCS_RESPONSE" = "200" ]; then
    echo -e "   ${GREEN}✓ /docs endpoint accessible${NC}"
else
    echo -e "   ${RED}✗ /docs endpoint not accessible (Status: $DOCS_RESPONSE)${NC}"
fi

echo ""
echo "5. Security Headers Test"
echo "------------------------"
echo -e "${YELLOW}Checking security headers...${NC}"
HEADERS=$(curl -s -I https://$DOMAIN)

echo -n "   X-Frame-Options: "
if echo "$HEADERS" | grep -qi "X-Frame-Options: DENY"; then
    echo -e "${GREEN}✓ Present${NC}"
else
    echo -e "${YELLOW}⚠ Not present${NC}"
fi

echo -n "   X-Content-Type-Options: "
if echo "$HEADERS" | grep -qi "X-Content-Type-Options: nosniff"; then
    echo -e "${GREEN}✓ Present${NC}"
else
    echo -e "${YELLOW}⚠ Not present${NC}"
fi

echo -n "   Strict-Transport-Security: "
if echo "$HEADERS" | grep -qi "Strict-Transport-Security"; then
    echo -e "${GREEN}✓ Present${NC}"
else
    echo -e "${YELLOW}⚠ Not present${NC}"
fi

echo ""
echo "6. Docker Container Status"
echo "--------------------------"
if docker ps | grep -q "xgboost-api"; then
    echo -e "   ${GREEN}✓ xgboost-api container is running${NC}"
    CONTAINER_ID=$(docker ps | grep "xgboost-api" | awk '{print $1}')
    echo "   Container ID: $CONTAINER_ID"
else
    echo -e "   ${RED}✗ xgboost-api container is not running${NC}"
fi

echo ""
echo "7. QuantConnect Integration Test"
echo "--------------------------------"
echo -e "${YELLOW}Testing Python import...${NC}"
cd /home/yumna/Working/dragonfortune/xgboost-qc
if python3 -c "from quantconnect_integration_example import XGBoostQuantConnectAPI; api = XGBoostQuantConnectAPI(api_base_url='https://$DOMAIN'); print('✓ API client initialized successfully')" 2>/dev/null; then
    echo -e "   ${GREEN}✓ QuantConnect integration ready${NC}"
else
    echo -e "   ${RED}✗ QuantConnect integration error${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ HTTPS Test Complete!${NC}"
echo ""
echo "📊 Summary:"
echo "   • Domain: https://$DOMAIN"
echo "   • API Base: https://$DOMAIN/api/v1"
echo "   • Documentation: https://$DOMAIN/docs"
echo ""
echo "🔗 Quick Links:"
echo "   • Health Check: curl https://$DOMAIN/health"
echo "   • Latest Model: curl https://$DOMAIN/api/v1/latest/model"
echo "   • All Sessions: curl https://$DOMAIN/api/v1/sessions"