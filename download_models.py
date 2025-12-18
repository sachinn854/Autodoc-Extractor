#!/usr/bin/env python3
"""
Download PaddleOCR models during Docker build
This prevents timeout issues on Render deployment
"""

import os
import sys

def download_paddleocr_models():
    """Download PaddleOCR models for production deployment"""
    
    # Set model cache directory
    os.environ['HUB_HOME'] = '/app/backend/models'
    
    print('🔄 Downloading PaddleOCR models for Render deployment...')
    
    try:
        # Import and initialize PaddleOCR
        from paddleocr import PaddleOCR
        
        # Initialize with common settings
        ocr = PaddleOCR(
            use_angle_cls=True, 
            lang='en', 
            show_log=False,
            use_gpu=False  # Render doesn't have GPU
        )
        
        print('✅ PaddleOCR models downloaded successfully!')
        print(f'📁 Models cached in: {os.environ.get("HUB_HOME", "default location")}')
        
        return True
        
    except ImportError as e:
        print(f'❌ PaddleOCR import failed: {e}')
        print('Make sure paddleocr is installed: pip install paddleocr')
        return False
        
    except Exception as e:
        print(f'⚠️ Model download failed: {e}')
        print('Models will be downloaded on first request instead')
        return False

if __name__ == '__main__':
    success = download_paddleocr_models()
    sys.exit(0 if success else 1)