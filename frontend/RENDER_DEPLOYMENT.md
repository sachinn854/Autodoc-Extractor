# 🚀 Frontend Deployment Guide (Render)

## ✅ Pre-deployment Checklist
- [x] Backend deployed on Railway: `https://autodoc-backend-production-45d1.up.railway.app`
- [x] Frontend API URL configured: `NEXT_PUBLIC_API_URL`
- [x] Next.js standalone output enabled
- [x] Docker configuration optimized

## 🎯 Deployment Steps

### 1. Push Frontend Code
```bash
git add .
git commit -m "Frontend ready for Render deployment"
git push origin main
```

### 2. Create Render Web Service
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New"** → **"Web Service"**
3. Connect your frontend repository
4. Configure:
   - **Name**: `autodoc-frontend`
   - **Environment**: `Docker`
   - **Region**: `Oregon (US West)` (same as backend)
   - **Branch**: `main`
   - **Root Directory**: `frontend`

### 3. Environment Variables
Add in Render Environment section:
```
NEXT_PUBLIC_API_URL=https://autodoc-backend-production-45d1.up.railway.app
NODE_ENV=production
```

### 4. Build Settings
Render will auto-detect Docker, but verify:
- **Build Command**: `docker build -t frontend .`
- **Start Command**: `docker run -p $PORT:3000 frontend`

### 5. Deploy & Test
1. Click **"Create Web Service"**
2. Wait for build completion (~5-10 minutes)
3. Test your frontend URL
4. Verify API connectivity

## 🔧 Backend CORS Update (if needed)
If you get CORS errors, update backend `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-render-frontend-url.onrender.com",
        "http://localhost:3000"  # for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🎉 Final Architecture
```
Frontend (Render) → Backend (Railway) → Database
     ↓                    ↓
  Static Files        API + ML Processing
```

## 🚨 Troubleshooting
- **Build fails**: Check Docker logs in Render
- **API errors**: Verify `NEXT_PUBLIC_API_URL` environment variable
- **CORS issues**: Update backend allowed origins
- **Slow loading**: Check Railway backend health at `/health`

## 📱 Test Checklist
- [ ] Frontend loads successfully
- [ ] Login/Signup works
- [ ] File upload works
- [ ] Document processing works
- [ ] Results display correctly
- [ ] Download functionality works