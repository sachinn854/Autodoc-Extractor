# AutoDoc Extractor Backend - Hugging Face Spaces Deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies optimized for Hugging Face Spaces
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies with optimizations for Hugging Face
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code
COPY app ./app

# Create models directory (will be populated at runtime)
RUN mkdir -p models

# Create necessary directories with proper permissions
RUN mkdir -p tmp/uploads tmp/preprocessed tmp/results data models && \
    chmod -R 755 tmp data models

# Set environment variables for Hugging Face Spaces
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HUB_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Create non-root user for security (Hugging Face best practice)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Expose port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Health check optimized for Hugging Face
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application with Hugging Face Spaces compatible settings
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]