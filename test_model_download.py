#!/usr/bin/env python3
"""
Test model download script locally
"""

import os
import sys
import subprocess

def test_model_download():
    """Test the model download script"""
    
    print("🧪 Testing model download script...")
    
    # Set environment
    os.environ['HUB_HOME'] = './test_models'
    
    try:
        # Run the download script
        result = subprocess.run([
            sys.executable, 'download_models.py'
        ], capture_output=True, text=True, timeout=300)
        
        print("📤 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("📤 STDERR:")
            print(result.stderr)
        
        print(f"📊 Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Model download test PASSED")
        else:
            print("❌ Model download test FAILED")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Model download test TIMED OUT (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == '__main__':
    success = test_model_download()
    sys.exit(0 if success else 1)