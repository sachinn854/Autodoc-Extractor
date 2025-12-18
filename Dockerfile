# ============================================
# Stage 1: Build Backend
# ============================================
FROM python:3.11-slim AS backend-build

WORKDIR /backend

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/app ./app
COPY backend/models ./models
COPY backend/migrate_db.py ./
COPY backend/.env.example ./
COPY yolov8n.pt ./


# ============================================
# Stage 2: Build Frontend
# ============================================
FROM node:18-alpine AS frontend-build

WORKDIR /frontend

# Install dependencies
COPY frontend/package*.json ./
RUN npm install

# Copy frontend code and build
COPY frontend/ .
RUN npm run build


# ============================================
# Stage 3: Production Image (Render)
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ✅ Install Python deps again (SAFE way)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY --from=backend-build /backend /app/backend

# Copy additional backend files
COPY backend/migrate_db.py /app/backend/
COPY backend/.env.example /app/backend/
COPY yolov8n.pt /app/backend/

# 🚀 Model download setup (skip for Railway to avoid timeout)
ENV HUB_HOME=/app/backend/models
# Note: Models will be downloaded on first OCR request to avoid Railway build timeout

# Copy frontend build (Next.js)
COPY --from=frontend-build /frontend/.next /app/frontend/.next
COPY --from=frontend-build /frontend/package.json /app/frontend/package.json
COPY --from=frontend-build /frontend/next.config.js /app/frontend/next.config.js

# Create required directories
RUN mkdir -p /app/backend/data /app/backend/tmp /app/backend/models /app/backend/models/yolo_config

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/backend
ENV HUB_HOME=/app/backend/models
ENV DATABASE_URL=sqlite:///./data/autodoc.db
ENV YOLO_CONFIG_DIR=/app/backend/models/yolo_config

# (EXPOSE optional for Render, but keep generic)
EXPOSE 8001

# ❌ Healthcheck removed (Render-safe)

# Start backend on Render dynamic PORT
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
