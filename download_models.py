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
    models_dir = '/app/backend/models'
    os.environ['HUB_HOME'] = models_dir
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    print('🔄 Downloading PaddleOCR models for Render deployment...')
    print(f'📁 Models directory: {models_dir}')
    
    try:
        # Check if PaddleOCR is available
        try:
            import paddleocr
            print(f'✅ PaddleOCR version: {paddleocr.__version__}')
        except ImportError as e:
            print(f'❌ PaddleOCR not available: {e}')
            return False
        
        # Import and initialize PaddleOCR with minimal settings first
        from paddleocr import PaddleOCR
        
        print('🔄 Initializing PaddleOCR with minimal settings...')
        
        # Try with most basic settings first
        ocr = PaddleOCR(
            use_angle_cls=False,  # Disable angle classification for faster init
            lang='en', 
            show_log=False,
            use_gpu=False,  # Render doesn't have GPU
            enable_mkldnn=False,  # Disable Intel optimization
            use_tensorrt=False,   # Disable TensorRT
            warmup=False         # Skip warmup
        )
        
        print('✅ PaddleOCR initialized successfully!')
        print(f'📁 Models cached in: {models_dir}')
        
        # Test with a small dummy image to ensure models are loaded
        import numpy as np
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        print('🧪 Testing OCR with dummy image...')
        result = ocr.ocr(test_img, cls=False)
        print('✅ OCR test completed successfully!')
        
        return True
        
    except ImportError as e:
        print(f'❌ PaddleOCR import failed: {e}')
        print('📦 Make sure paddleocr is installed: pip install paddleocr')
        return False
        
    except Exception as e:
        print(f'⚠️ Model download/initialization failed: {e}')
        print(f'🔍 Error type: {type(e).__name__}')
        print('📝 Models will be downloaded on first request instead')
        return False

if __name__ == '__main__':
    try:
        success = download_paddleocr_models()
        if success:
            print('✅ Model download completed successfully')
            sys.exit(0)
        else:
            print('⚠️ Model download failed, but continuing build...')
            sys.exit(0)  # Don't fail the build, models will download at runtime
    except Exception as e:
        print(f'❌ Unexpected error in model download: {e}')
        print('⚠️ Continuing build without preloaded models...')
        sys.exit(0)  # Don't fail the build