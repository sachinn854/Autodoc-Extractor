# 🚀 Render Deployment Guide - Autodoc Extractor

## Quick Deploy to Render (5 Minutes)

### Step 1: Prepare Your Repository

1. **Push all changes to GitHub:**
```bash
git add .
git commit -m "🚀 Ready for Render deployment"
git push origin main
```

### Step 2: Deploy on Render

1. **Go to [render.com](https://render.com)** and sign up/login with GitHub

2. **Create New Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect the `Dockerfile`

3. **Configure Service:**
   ```
   Name: autodoc-extractor
   Region: Oregon (US West)
   Branch: main
   Root Directory: (leave empty)
   Environment: Docker
   Dockerfile Path: ./Dockerfile
   Docker Build Context Directory: .
   ```

4. **Set Environment Variables:**
   ```bash
   # Required
   PORT=8001
   PYTHONUNBUFFERED=1
   HUB_HOME=/app/backend/models
   DATABASE_URL=sqlite:///./data/autodoc.db
   RENDER=true
   
   # Generate a secure secret key
   SECRET_KEY=your-generated-secret-key-here
   
   # CORS (allow all for now)
   CORS_ORIGINS=*
   
   # Optional: Email verification
   SMTP_EMAIL=your-email@gmail.com
   SMTP_PASSWORD=your-gmail-app-password
   ```

5. **Add Persistent Disk:**
   - Click "Add Disk"
   - Name: `autodoc-data`
   - Mount Path: `/app/backend/data`
   - Size: 2 GB

6. **Deploy:**
   - Click "Create Web Service"
   - Wait 10-15 minutes for first build (downloads AI models)

### Step 3: Verify Deployment

1. **Check Health:**
   ```bash
   curl https://your-app-name.onrender.com/health
   ```

2. **Expected Response:**
   ```json
   {
     "status": "healthy",
     "service": "autodoc-extractor",
     "version": "2.0.0",
     "ocr_engine": "ready",
     "environment": "true"
   }
   ```

## 🔧 Environment Variables Explained

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | ✅ Yes | 8001 | Render assigns this automatically |
| `SECRET_KEY` | ✅ Yes | - | JWT token secret (generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`) |
| `DATABASE_URL` | ✅ Yes | sqlite:///./data/autodoc.db | Database connection string |
| `HUB_HOME` | ✅ Yes | /app/backend/models | PaddleOCR model cache directory |
| `CORS_ORIGINS` | ⚠️ Recommended | * | Comma-separated allowed origins |
| `SMTP_EMAIL` | ❌ Optional | - | Gmail for email verification |
| `SMTP_PASSWORD` | ❌ Optional | - | Gmail app password |
| `RENDER` | ✅ Yes | true | Flag for Render-specific config |

## 📊 Performance Expectations

### First Deployment
- **Build Time:** 10-15 minutes
  - Installing system dependencies: 2-3 min
  - Installing Python packages: 3-5 min
  - Downloading PaddleOCR models: 5-7 min
  - Building frontend: 2-3 min

### Runtime Performance
- **First Request:** 30-60 seconds (model initialization)
- **Subsequent Requests:** 5-15 seconds per document
- **Memory Usage:** ~1.5 GB (Starter plan sufficient)
- **Storage:** ~500 MB for models + database

## 🐛 Troubleshooting

### Build Fails

**Issue:** "Model download timeout"
```bash
# Solution: Models are now preloaded during build
# If still failing, check Render build logs
```

**Issue:** "Out of memory during build"
```bash
# Solution: Upgrade to higher Render plan
# Or reduce model size in Dockerfile
```

### Runtime Issues

**Issue:** "Upload timeout"
```bash
# Check: Is OCR engine initialized?
curl https://your-app.onrender.com/health

# Expected: "ocr_engine": "ready"
```

**Issue:** "Database locked"
```bash
# Solution: Ensure persistent disk is mounted
# Check: /app/backend/data exists and is writable
```

**Issue:** "CORS errors"
```bash
# Solution: Update CORS_ORIGINS environment variable
CORS_ORIGINS=https://your-frontend-domain.com,https://another-domain.com
```

## 🔐 Security Checklist

- [ ] Generate strong `SECRET_KEY` (32+ characters)
- [ ] Set specific `CORS_ORIGINS` (not `*` in production)
- [ ] Enable HTTPS (Render provides free SSL)
- [ ] Set up email verification (optional but recommended)
- [ ] Review Render logs regularly
- [ ] Enable Render's DDoS protection

## 📈 Scaling

### Free Tier Limitations
- Spins down after 15 minutes of inactivity
- First request after spin-down: 30-60 seconds
- 750 hours/month free

### Upgrade to Paid Plan
- Always-on service (no spin-down)
- Faster CPU and more memory
- Better for production use

### Horizontal Scaling
```yaml
# In render.yaml, add:
numInstances: 2  # Run 2 instances
```

## 🔄 Continuous Deployment

Render automatically deploys when you push to GitHub:

```bash
# Make changes
git add .
git commit -m "Update feature"
git push origin main

# Render will automatically:
# 1. Detect the push
# 2. Build new Docker image
# 3. Deploy with zero downtime
```

## 📝 Monitoring

### View Logs
```bash
# In Render dashboard:
# Your Service → Logs → Live Logs
```

### Set Up Alerts
```bash
# In Render dashboard:
# Your Service → Settings → Notifications
# Add email/Slack for deployment failures
```

## 🎯 Production Checklist

- [ ] Repository pushed to GitHub
- [ ] Environment variables configured
- [ ] Persistent disk added (2GB)
- [ ] Health check endpoint working
- [ ] First upload tested successfully
- [ ] Email verification working (if enabled)
- [ ] CORS configured for your domain
- [ ] Monitoring/alerts set up

## 🆘 Support

- **Render Docs:** https://render.com/docs
- **Project Issues:** https://github.com/your-repo/issues
- **Render Community:** https://community.render.com

## 🎉 Success!

Your Autodoc Extractor is now live on Render! 

**Your App URL:** `https://your-app-name.onrender.com`

Test it:
1. Go to your app URL
2. Sign up for an account
3. Upload a receipt/invoice
4. Watch the magic happen! ✨
