# Autodoc Extractor - Docker Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Docker installed
- Docker Compose installed

### Local Development with Docker

```bash
# Build and start all services
docker-compose up --build

# Run in background (detached mode)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### Access Applications
- Frontend: http://localhost:3000
- Backend API: http://localhost:8001
- API Docs: http://localhost:8001/docs

## 📦 Production Deployment

### Option 1: Docker Hub (Recommended)

```bash
# 1. Build images
docker-compose build

# 2. Tag images
docker tag autodoc-extractor-backend:latest YOUR_DOCKERHUB_USERNAME/autodoc-backend:latest
docker tag autodoc-extractor-frontend:latest YOUR_DOCKERHUB_USERNAME/autodoc-frontend:latest

# 3. Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/autodoc-backend:latest
docker push YOUR_DOCKERHUB_USERNAME/autodoc-frontend:latest

# 4. On production server, pull and run
docker pull YOUR_DOCKERHUB_USERNAME/autodoc-backend:latest
docker pull YOUR_DOCKERHUB_USERNAME/autodoc-frontend:latest
docker-compose up -d
```

### Option 2: Deploy to Cloud Platforms

#### Railway.app
1. Connect GitHub repo
2. Add `railway.json` config
3. Deploy automatically on push

#### Render.com
1. Connect GitHub repo
2. Select "Docker" as environment
3. Deploy from `docker-compose.yml`

#### DigitalOcean App Platform
1. Connect GitHub repo
2. Detect Docker configuration
3. Deploy with one click

#### AWS ECS / Azure Container Instances
1. Push images to container registry (ECR/ACR)
2. Create task definitions
3. Deploy as services

### Option 3: VPS/Cloud Server (Ubuntu)

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Clone repository
git clone https://github.com/sachinn854/Autodoc-Extractor.git
cd Autodoc-Extractor

# Create production environment file
cp .env.example .env
# Edit .env with production values

# Download models (if needed)
# Place yolov8n.pt in backend/models/

# Build and run
docker-compose up -d

# Set up nginx reverse proxy (optional)
sudo apt install nginx
# Configure nginx to proxy port 80/443 to port 3000
```

## 🔧 Configuration

### Environment Variables

Create `.env` file in project root:

```env
# Backend
DATABASE_URL=sqlite:///./data/autodoc.db
SECRET_KEY=your-secret-key-here-change-in-production
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

### Production Checklist

- [ ] Change `SECRET_KEY` in backend
- [ ] Update `CORS_ORIGINS` with production domain
- [ ] Set up SSL/HTTPS (Let's Encrypt)
- [ ] Configure production database (PostgreSQL recommended)
- [ ] Set up backup for `backend/data/`
- [ ] Configure logging and monitoring
- [ ] Set up domain name and DNS
- [ ] Enable firewall rules
- [ ] Set up CI/CD pipeline

## 🐛 Troubleshooting

### View logs
```bash
docker-compose logs backend
docker-compose logs frontend
```

### Restart specific service
```bash
docker-compose restart backend
```

### Rebuild after code changes
```bash
docker-compose up --build
```

### Clean everything
```bash
docker-compose down -v
docker system prune -a
```

## 📊 Monitoring

### Check container health
```bash
docker-compose ps
```

### Resource usage
```bash
docker stats
```

## 🔐 Security Notes

1. Never commit `.env` file
2. Use secrets management for production
3. Enable HTTPS in production
4. Regularly update base images
5. Scan images for vulnerabilities:
   ```bash
   docker scan autodoc-extractor-backend
   ```

## 🎯 Scaling

For high traffic, use:
- Load balancer (nginx/Traefik)
- Multiple backend replicas
- Separate database service (PostgreSQL)
- Redis for caching
- S3 for file storage

Example scaling:
```bash
docker-compose up --scale backend=3
```
