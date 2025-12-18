#!/bin/bash

# Test Docker Build Script for Autodoc Extractor
# This script tests if the Docker image builds successfully

echo "🚀 Testing Docker build for Render deployment..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Build the image
echo "📦 Step 1: Building Docker image..."
if docker build -t autodoc-extractor-test .; then
    echo -e "${GREEN}✅ Docker build successful!${NC}"
else
    echo -e "${RED}❌ Docker build failed!${NC}"
    exit 1
fi

echo ""

# Step 2: Test run the container
echo "🧪 Step 2: Testing container startup..."
CONTAINER_ID=$(docker run -d -p 8001:8001 \
    -e PORT=8001 \
    -e PYTHONUNBUFFERED=1 \
    -e HUB_HOME=/app/backend/models \
    -e DATABASE_URL=sqlite:///./data/autodoc.db \
    autodoc-extractor-test)

if [ -z "$CONTAINER_ID" ]; then
    echo -e "${RED}❌ Failed to start container!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Container started: $CONTAINER_ID${NC}"
echo ""

# Step 3: Wait for service to be ready
echo "⏳ Step 3: Waiting for service to be ready (30 seconds)..."
sleep 30

# Step 4: Test health endpoint
echo "🏥 Step 4: Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8001/health)

if [ -z "$HEALTH_RESPONSE" ]; then
    echo -e "${RED}❌ Health check failed - no response${NC}"
    docker logs $CONTAINER_ID
    docker stop $CONTAINER_ID
    docker rm $CONTAINER_ID
    exit 1
fi

echo -e "${GREEN}✅ Health check response:${NC}"
echo "$HEALTH_RESPONSE" | python -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE"
echo ""

# Step 5: Check if OCR engine is ready
if echo "$HEALTH_RESPONSE" | grep -q '"ocr_engine".*"ready"'; then
    echo -e "${GREEN}✅ OCR engine is ready!${NC}"
else
    echo -e "${YELLOW}⚠️  OCR engine status: $(echo $HEALTH_RESPONSE | grep -o '"ocr_engine"[^,]*')${NC}"
fi

echo ""

# Step 6: Cleanup
echo "🧹 Step 6: Cleaning up..."
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

echo ""
echo -e "${GREEN}🎉 All tests passed! Ready for Render deployment!${NC}"
echo ""
echo "Next steps:"
echo "1. git add ."
echo "2. git commit -m '🚀 Ready for Render deployment'"
echo "3. git push origin main"
echo "4. Deploy on Render.com"
