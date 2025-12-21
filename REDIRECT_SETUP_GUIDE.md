# 🔄 IP:Port ke Domain Redirect Setup Guide

## 🎯 Goal:
- **API hanya accessible via domain** - `https://api.dragonfortune.ai`
- **IP:Port access redirect ke domain** - `http://IP:8000` → `https://api.dragonfortune.ai`
- **Security enforced** - Tidak ada direct IP access tanpa redirect

---

## 🔧 Step 1: Deploy API dengan Port Mapping

```bash
cd /www/wwwroot/api.dragonfortune.ai
./deploy_api.sh
```

Hasil:
- ✅ Container jalan di port 8000 (eksternal)
- ✅ API accessible via `http://IP:8000` (sementara)
- ✅ Siap untuk redirect

---

## 🔧 Step 2: Setup Nginx Redirect

### 2.1 Gunakan `nginx_with_redirect.conf`
Copy content dari `nginx_with_redirect.conf` ke Nginx config di aPanel.

### 2.2 Atau manual add redirect server:
Add ini di atas existing server block:

```nginx
# Default server untuk IP:Port redirect
server {
    listen 8000 default_server;
    listen [::]:8000 default_server;
    server_name _;

    # Redirect semua IP:Port access ke domain HTTPS
    return 301 https://api.dragonfortune.ai$request_uri;
}
```

---

## 🧪 Step 3: Test Redirects

### 3.1 Test IP:Port (should redirect):
```bash
# Test IP access - harus redirect ke domain
curl -v http://YOUR_SERVER_IP:8000/health
# Expected: 301 redirect ke https://api.dragonfortune.ai/health
```

### 3.2 Test Domain (should work):
```bash
# Test domain access - langsung work
curl https://api.dragonfortune.ai/health
# Expected: 200 OK dengan response dari API
```

### 3.3 Test HTTP (should redirect):
```bash
# Test HTTP - harus redirect ke HTTPS
curl -v http://api.dragonfortune.ai/health
# Expected: 301 redirect ke https://api.dragonfortune.ai/health
```

---

## ✅ Expected Results:

| Access Method | Result |
|---------------|---------|
| `http://IP:8000` | 301 → `https://api.dragonfortune.ai` |
| `https://IP:8000` | Connection error (SSL cert untuk domain) |
| `http://api.dragonfortune.ai` | 301 → `https://api.dragonfortune.ai` |
| `https://api.dragonfortune.ai` | ✅ 200 OK (API Response) |

---

## 🚨 Troubleshooting:

### 1. IP:Port masih accessible (no redirect)
```bash
# Check Nginx config
nginx -t
# Check if default_server block active
grep -r "default_server" /etc/nginx/sites-enabled/
```

### 2. Domain 403 Forbidden
```bash
# Check Nginx error logs
tail -f /www/wwwlogs/api.dragonfortune.ai.error.log
# Check Docker logs
docker logs xgboost-api
```

### 3. Redirect loop
- Pastikan hanya ada satu default_server untuk port 8000
- Check bahwa api.dragonfortune.ai tidak juga defined sebagai default_server

---

## 🔒 Security Features:

- ✅ **Force HTTPS** - HTTP auto redirect
- ✅ **Domain only** - IP:Port redirect ke domain
- ✅ **SSL only** - Tidak ada HTTP direct access
- ✅ **Professional** - Hanya domain resmi yang accessible

---

## 🎉 Success When:

```bash
# IP:Port redirect
curl http://IP:8000/health
# Output: 301 Moved Permanently → https://api.dragonfortune.ai/health

# Domain work
curl https://api.dragonfortune.ai/health
# Output: {"status": "ok", "message": "XGBoost API is running!"}
```

Semua access akan redirect ke domain HTTPS profesional! 🚀