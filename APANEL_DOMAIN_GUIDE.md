# 🚀 aPanel Domain Setup Guide untuk api.dragonfortune.ai

## 📋 Prasyarat:
- ✅ aPanel sudah terinstall
- ✅ Domain `api.dragonfortune.ai` sudah diarahkan ke IP server
- ✅ Docker dan Docker Compose sudah terinstall
- ✅ Project XGBoost API sudah diupload ke server

---

## 🔧 Step 1: Setup Domain di aPanel

### 1.1 Tambah Website Domain
1. Login ke aPanel
2. Pilih menu **Website**
3. Klik **Add Site**
4. Pilih **Domain Configuration**
5. Masukkan: `api.dragonfortune.ai`
6. Klik **Submit**

### 1.2 Setup SSL Certificate
1. Di halaman website, klik **SSL**
2. Pilih domain `api.dragonfortune.ai`
3. Klik **Apply for a Certificate**
4. Pilih **Let's Encrypt**
5. Centang:
   - ✅ Force HTTPS
   - ✅ Auto Renewal
6. Klik **Apply**

---

## 📁 Step 2: Upload Project Files

### 2.1 Upload Project ke Server
```bash
# Upload project folder ke /www/wwwroot/
scp -r xgboost-qc/ root@your-server:/www/wwwroot/api.dragonfortune.ai/
```

### 2.2 Atur Permissions
```bash
chmod -R 755 /www/wwwroot/api.dragonfortune.ai
chown -R www:www /www/wwwroot/api.dragonfortune.ai
```

---

## 🐳 Step 3: Deploy API dengan Docker

### 3.1 Masuk ke Directory Project
```bash
cd /www/wwwroot/api.dragonfortune.ai
```

### 3.2 Jalankan Deploy Script
```bash
./deploy_api.sh
```

### 3.3 Verify Container Running
```bash
docker ps | grep xgboost-api
```

---

## ⚙️ Step 4: Update Nginx Configuration

### 4.1 Edit Nginx Config di aPanel
1. Pilih **Website** → **api.dragonfortune.ai**
2. Klik **Settings**
3. Klik **Config File**
4. Ganti semua content dengan konfigurasi dari file `nginx_api.conf`

### 4.2 Atau Copy Manual
Copy content dari `nginx_api.conf` ke Nginx config di aPanel.

### 4.3 Reload Nginx
```bash
nginx -t
nginx -s reload
```

---

## 🔍 Step 5: Testing & Verification

### 5.1 Test Internal Connection
```bash
# Test dari dalam server
curl http://localhost:8000/health
```

### 5.2 Test Domain Connection
```bash
# Test via domain
curl https://api.dragonfortune.ai/health
```

### 5.3 Test API Endpoints
```bash
# Test health endpoint
curl https://api.dragonfortune.ai/health

# Test model endpoint
curl https://api.dragonfortune.ai/api/v1/sessions

# Test documentation
curl https://api.dragonfortune.ai/docs
```

---

## 🚨 Step 6: Troubleshooting

### 6.1 Cek Container Status
```bash
docker ps -a
docker logs xgboost-api
```

### 6.2 Cek Nginx Error Log
```bash
tail -f /www/wwwlogs/api.dragonfortune.ai.error.log
```

### 6.3 Cek API Error Log
```bash
docker-compose -f docker-compose.api.yml logs -f
```

### 6.4 Restart Services
```bash
# Restart API container
docker-compose -f docker-compose.api.yml restart

# Restart Nginx
systemctl restart nginx
```

---

## ✅ Success Checklist:

- [ ] Domain pointing ke server IP
- [ ] SSL Certificate installed
- [ ] API container running on port 8000
- [ ] Nginx proxy configuration updated
- [ ] HTTPS redirect working
- [ ] API endpoints accessible via domain

## 🎯 Expected Results:

Setelah selesai, API akan accessible di:
- **Health**: https://api.dragonfortune.ai/health
- **API Base**: https://api.dragonfortune.ai/api/v1
- **Documentation**: https://api.dragonfortune.ai/docs

## 🔐 Security Notes:

- ✅ Only HTTPS access (HTTP redirect ke HTTPS)
- ✅ No direct port access (hanya via Nginx proxy)
- ✅ SSL certificate auto-renewal
- ✅ CORS enabled untuk public access

---

## 📞 Quick Commands:

```bash
# Check API status
curl https://api.dragonfortune.ai/health

# View logs
docker-compose -f /www/wwwroot/api.dragonfortune.ai/docker-compose.api.yml logs -f

# Restart API
cd /www/wwwroot/api.dragonfortune.ai && docker-compose -f docker-compose.api.yml restart

# Test SSL certificate
curl -I https://api.dragonfortune.ai
```

Selamat mencoba! 🎉