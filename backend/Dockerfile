# AutoDoc Extractor Backend - Heavy Models for High Accuracy
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for heavy ML models
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    cmake \
    pkg-config \
    # Image processing libraries
    libopencv-dev \
    python3-opencv \
    # OCR dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # PDF processing
    poppler-utils \
    # Graphics libraries for ML models
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Additional ML dependencies
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    # Network tools
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies with heavy ML models (optimized for HF Spaces)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Install PyTorch with CPU support (smaller than CUDA version)
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    # Install PaddlePaddle and PaddleOCR in separate steps to avoid memory issues
    pip install --no-cache-dir paddlepaddle && \
    pip cache purge && \
    pip install --no-cache-dir paddleocr && \
    pip cache purge && \
    # Install YOLO
    pip install --no-cache-dir ultralytics && \
    pip cache purge && \
    # Install remaining requirements
    pip install --no-cache-dir -r requirements.txt && \
    # Final cleanup
    pip cache purge

# Copy application code
COPY app ./app

# Create models directory (will be populated at runtime)
RUN mkdir -p models

# Create necessary directories with proper permissions
RUN mkdir -p tmp/uploads tmp/preprocessed tmp/results data models && \
    chmod -R 755 tmp data models

# Download YOLO model at build time (with fallback)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "YOLO download will happen at runtime"

# Set environment variables for heavy models (optimized for HF Spaces)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HUB_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
# Memory optimization for HF Spaces
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# Create non-root user for security
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Expose port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Health check with longer timeout for heavy models
HEALTHCHECK --interval=60s --timeout=60s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application with single worker (heavy models need more memory per worker)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]