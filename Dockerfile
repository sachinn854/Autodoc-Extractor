# ============================================
# Stage 1: Build Backend
# ============================================
FROM python:3.11-slim AS backend-build

WORKDIR /backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/app ./app
COPY backend/models ./models


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

# Copy frontend build (Next.js)
COPY --from=frontend-build /frontend/.next /app/frontend/.next
COPY --from=frontend-build /frontend/public /app/frontend/public
COPY --from=frontend-build /frontend/package.json /app/frontend/package.json
COPY --from=frontend-build /frontend/next.config.js /app/frontend/next.config.js

# Create required directories
RUN mkdir -p /app/backend/data /app/backend/tmp

# (EXPOSE optional for Render, but keep generic)
EXPOSE 8001

# ❌ Healthcheck removed (Render-safe)

# Start backend on Render dynamic PORT
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
