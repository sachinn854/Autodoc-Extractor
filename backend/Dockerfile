# Autodoc Extractor Backend - Production Dockerfile
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
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

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY models ./models
COPY .env.example ./
COPY ../yolov8n.pt ./

# 🚀 CRITICAL FIX: Preload PaddleOCR models during build (prevents Render timeout)
RUN python -c "\
import os\
os.environ['HUB_HOME'] = '/app/models'\
from paddleocr import PaddleOCR\
print('🔄 Downloading PaddleOCR models...')\
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)\
print('✅ PaddleOCR models downloaded successfully!')\
"

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
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]