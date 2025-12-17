# Multi-stage build for both backend and frontend
FROM python:3.11-slim as backend-build

# Install system dependencies for backend
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set backend working directory
WORKDIR /app/backend

# Copy backend requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Frontend build stage
FROM node:18-alpine as frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Final production image
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for serving frontend
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g serve

WORKDIR /app

# Copy backend from build stage
COPY --from=backend-build /app/backend ./backend
COPY --from=backend-build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-build /usr/local/bin /usr/local/bin

# Copy frontend build from build stage
COPY --from=frontend-build /app/frontend/.next ./frontend/.next
COPY --from=frontend-build /app/frontend/public ./frontend/public
COPY --from=frontend-build /app/frontend/package*.json ./frontend/
COPY --from=frontend-build /app/frontend/next.config.js ./frontend/

# Install frontend production dependencies
WORKDIR /app/frontend
RUN npm ci --only=production

# Create startup script
WORKDIR /app
RUN echo '#!/bin/bash\n\
cd /app/backend && uvicorn app.main:app --host 0.0.0.0 --port 8001 &\n\
cd /app/frontend && npm start -- -p 3000 &\n\
wait' > /app/start.sh && chmod +x /app/start.sh

# Expose ports
EXPOSE 8001 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["/app/start.sh"]
