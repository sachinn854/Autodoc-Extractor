# Multi-stage Dockerfile for Full-Stack Autodoc Extractor
# Frontend (Next.js) + Backend (FastAPI) - FastAPI serves frontend

# Stage 1: Build Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy frontend package files
COPY frontend/package*.json ./
RUN npm ci

# Copy frontend source and build + export
COPY frontend/ ./
RUN npm run build-export

# Stage 2: Production with FastAPI serving frontend
FROM python:3.11-slim AS production

# Set working directory
WORKDIR /app

# Install system dependencies for backend
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application
COPY backend/app ./app
COPY backend/models ./models

# Copy built frontend static files
COPY --from=frontend-builder /app/frontend/out ./static

# Create necessary directories
RUN mkdir -p tmp/uploads tmp/preprocessed tmp/results data models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HUB_HOME=/app/models

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8001}/health || exit 1

# Run FastAPI (which will serve both API and frontend)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]