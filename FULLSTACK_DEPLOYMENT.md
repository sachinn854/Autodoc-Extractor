# 🚀 Full-Stack Deployment Guide

## 🎯 **Single Container Approach**
Frontend (Next.js) + Backend (FastAPI) in one Docker container
- **No CORS issues** ✅
- **Single deployment** ✅  
- **Same origin API calls** ✅

## 🏗️ **Architecture**
```
Single Container (Render)
├── FastAPI Backend (Port 8001)
│   ├── API endpoints (/upload, /process, etc.)
│   ├── Authentication (/auth/*)
│   └── Static file serving (/)
└── Next.js Frontend (Static Export)
    ├── Built to /static folder
    └── Served by FastAPI
```

## 📁 **Project Structure**
```
autodoc-extractor/
├── Dockerfile                 # Root Dockerfile (builds both)
├── backend/
│   ├── app/
│   ├── requirements.txt
│   └── Dockerfile            # Individual backend (not used)
├── frontend/
│   ├── src/
│   ├── package.json
│   └── Dockerfile            # Individual frontend (not used)
└── FULLSTACK_DEPLOYMENT.md   # This guide
```

## 🔧 **Key Changes Made**

### **1. Root Dockerfile**
- Multi-stage build
- Stage 1: Build Next.js frontend → static export
- Stage 2: Python backend + serve frontend static files

### **2. Frontend Changes**
- `next.config.js`: `output: 'export'` for static generation
- `api.ts`: `API_BASE_URL = ''` for same-origin calls
- `package.json`: Added `build-export` script

### **3. Backend Changes**
- `main.py`: Mount static files at root `/`
- Serves both API and frontend from same port

## 🚀 **Deployment Steps**

### **1. Push to Git**
```bash
git add .
git commit -m "Full-stack single container deployment"
git push origin main
```

### **2. Deploy on Render**
1. **New Web Service**
2. **Connect Repository** (root directory)
3. **Environment**: `Docker`
4. **Build Command**: Auto-detected
5. **Start Command**: Auto-detected

### **3. Environment Variables**
```
PORT=10000                    # Render auto-sets
PYTHONUNBUFFERED=1           # Already in Dockerfile
```

**Optional (for email features):**
```
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

## 🎯 **URL Structure**
```
https://your-app.onrender.com/           # Frontend (Next.js)
https://your-app.onrender.com/docs       # API Documentation
https://your-app.onrender.com/health     # Health Check
https://your-app.onrender.com/upload     # API Endpoints
```

## ✅ **Benefits**
- **Single deployment** - No separate frontend/backend
- **No CORS issues** - Same origin requests
- **Simplified routing** - One URL for everything
- **Cost effective** - One service instead of two
- **Easy maintenance** - Single container to manage

## 🧪 **Testing**
1. **Health Check**: `/health`
2. **API Docs**: `/docs`
3. **Frontend**: `/` (root)
4. **Upload Flow**: Frontend → API (same origin)

## 🚨 **Troubleshooting**
- **Build fails**: Check Docker logs in Render
- **Frontend not loading**: Check static files mount in logs
- **API not working**: Verify `/health` endpoint
- **Slow startup**: First deploy takes longer (model downloads)

## 🎉 **Expected Result**
Single URL serves complete application:
- Frontend UI at root
- API endpoints at `/api/*`
- No CORS configuration needed
- Seamless user experience