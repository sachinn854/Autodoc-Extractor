"""
OCR Engine module using PaddleOCR for high-accuracy document text extraction.
Handles initialization, text extraction, bounding box processing, and result formatting.
"""

import os
import json
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global OCR engine cache (singleton pattern)
_ocr_engine_cache = {}


def get_ocr_engine(lang: str = "en", job_id: str = None):
    """
    Returns a PaddleOCR instance for high-accuracy text extraction.
    Uses singleton pattern for efficient memory usage.
    
    Args:
        lang: Language code for OCR (default: "en")
        job_id: Job ID for status updates
        
    Returns:
        PaddleOCR instance
        
    Raises:
        Exception: If PaddleOCR initialization fails
    """
    global _ocr_engine_cache
    
    # Check if engine already cached
    if lang in _ocr_engine_cache:
        logger.info(f"✅ Using cached PaddleOCR engine for language: {lang}")
        return _ocr_engine_cache[lang]
    
    logger.info(f"🔄 Initializing PaddleOCR engine for language: {lang}")
    
    # Update job status
    if job_id:
        from app.main import update_job_status
        update_job_status(job_id, "processing", "🔄 Loading OCR engine (PaddleOCR)...")
    
    from paddleocr import PaddleOCR
    
    # Initialize PaddleOCR v3.3.2 with correct parameters
    ocr_engine = PaddleOCR(
        use_textline_orientation=True,  # Updated parameter name in v3.3.2
        lang=lang                       # Language setting
        # Note: show_log parameter removed in PaddleOCR 3.3.2
    )
    
    logger.info("✅ PaddleOCR engine initialized successfully")
    
    # Test OCR engine
    logger.info("Testing PaddleOCR engine...")
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
    test_result = ocr_engine.predict(test_img)  # Updated API in v3.3.2
    logger.info(f"✅ PaddleOCR test passed, result type: {type(test_result)}")
    
    # Cache the OCR engine
    _ocr_engine_cache[lang] = ocr_engine
    
    # Update job status
    if job_id:
        update_job_status(job_id, "processing", "✅ OCR engine ready, proceeding with text extraction...")
    
    logger.info(f"✅ PaddleOCR engine initialized and cached for language: {lang}")
    return ocr_engine


def run_ocr_on_image(image_path: str, lang: str = "en", job_id: str = None) -> List[Dict]:
    """
    Run OCR on a single image and return structured results
    
    Args:
        image_path: Path to the image file
        lang: Language code for OCR
        job_id: Job ID for status updates
        
    Returns:
        List of OCR results with text and bounding boxes
    """
    logger.info(f"🔍 Running PaddleOCR on image: {image_path}")
    
    # Get PaddleOCR engine
    ocr_engine = get_ocr_engine(lang, job_id)
    
    # Multi-pass OCR for better detection
    all_tokens = []
    
    # Pass 1: Standard OCR
    logger.info("🔄 OCR Pass 1: Standard detection")
    results = ocr_engine.predict(image_path)  # Updated API in v3.3.2
    tokens_pass1 = _extract_tokens_from_results(results, "Pass1")
    all_tokens.extend(tokens_pass1)
    
    # Pass 2: Enhanced preprocessing for faint text
    logger.info("🔄 OCR Pass 2: Enhanced preprocessing for faint text")
    tokens_pass2 = []
    enhanced_image = _enhance_image_for_ocr(image_path)
    if enhanced_image is not None:
        results_enhanced = ocr_engine.predict(enhanced_image)  # Updated API in v3.3.2
        tokens_pass2 = _extract_tokens_from_results(results_enhanced, "Pass2")
        all_tokens.extend(tokens_pass2)
    
    # Remove duplicates based on text and position
    unique_tokens = _deduplicate_tokens(all_tokens)
    
    logger.info(f"✅ PaddleOCR completed: {len(unique_tokens)} unique tokens extracted")
    logger.info(f"   Pass 1: {len(tokens_pass1)} tokens, Pass 2: {len(tokens_pass2)} tokens")
    
    return unique_tokens


def _extract_tokens_from_results(results, pass_name: str) -> List[Dict]:
    """Extract tokens from PaddleOCR v3.3.2 results"""
    tokens = []
    
    # Handle new PaddleOCR 3.3.2 result format
    if isinstance(results, list) and len(results) > 0:
        for page_result in results:
            if isinstance(page_result, dict):
                # New format: extract from rec_texts, rec_scores, and rec_polys
                rec_texts = page_result.get('rec_texts', [])
                rec_scores = page_result.get('rec_scores', [])
                rec_polys = page_result.get('rec_polys', [])
                
                # Combine texts, scores, and polygons
                for i, text in enumerate(rec_texts):
                    if i < len(rec_scores) and i < len(rec_polys):
                        confidence = rec_scores[i]
                        poly = rec_polys[i]
                        
                        # Convert polygon to bbox [x1, y1, x2, y2]
                        if len(poly) >= 4:
                            x_coords = [float(point[0]) for point in poly]
                            y_coords = [float(point[1]) for point in poly]
                            flat_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                        else:
                            flat_bbox = [0.0, 0.0, 0.0, 0.0]  # Default bbox
                        
                        tokens.append({
                            "text": str(text),
                            "bbox": flat_bbox,
                            "confidence": float(confidence),
                            "source": pass_name
                        })
            else:
                # Fallback: try old format for compatibility
                if page_result is None:
                    continue
                for item in page_result:
                    if len(item) >= 2:
                        bbox, (text, confidence) = item[0], item[1]
                        
                        # Convert bbox to flat format [x1, y1, x2, y2]
                        if len(bbox) == 4 and len(bbox[0]) == 2:
                            x_coords = [float(point[0]) for point in bbox]
                            y_coords = [float(point[1]) for point in bbox]
                            flat_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                        else:
                            flat_bbox = [float(x) for x in bbox] if bbox else [0.0, 0.0, 0.0, 0.0]
                        
                        tokens.append({
                            "text": str(text),
                            "bbox": flat_bbox,
                            "confidence": float(confidence),
                            "source": pass_name
                        })
    
    return tokens


def _enhance_image_for_ocr(image_path: str):
    """Enhance image specifically for better OCR of faint text"""
    try:
        import cv2
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply gamma correction to brighten dark text
        gamma = 1.2
        enhanced = np.power(enhanced / 255.0, gamma) * 255.0
        enhanced = enhanced.astype(np.uint8)
        
        # Apply slight Gaussian blur to smooth noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Convert back to 3-channel for PaddleOCR
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced_bgr
        
    except Exception as e:
        logger.warning(f"⚠️ Image enhancement failed: {e}")
        return None


def _deduplicate_tokens(tokens: List[Dict]) -> List[Dict]:
    """Remove duplicate tokens based on text and position"""
    unique_tokens = []
    seen_tokens = set()
    
    for token in tokens:
        # Create a key based on text and approximate position
        text = token['text'].strip()
        bbox = token['bbox']
        x_center = (bbox[0] + bbox[2]) // 2
        y_center = (bbox[1] + bbox[3]) // 2
        
        # Round position to nearest 10 pixels to handle slight variations
        key = (text, x_center // 10, y_center // 10)
        
        if key not in seen_tokens:
            seen_tokens.add(key)
            unique_tokens.append(token)
    
    return unique_tokens


def process_document_ocr(job_id: str, image_paths: List[str], lang: str = "en") -> Dict:
    """
    Process multiple images with OCR and return structured results
    
    Args:
        job_id: Job identifier
        image_paths: List of image file paths
        lang: Language code for OCR
        
    Returns:
        Dictionary with OCR results for all pages
    """
    logger.info(f"Starting complete OCR processing for job: {job_id}")
    
    try:
        logger.info("Getting OCR engine...")
        ocr_engine = get_ocr_engine(lang, job_id)
        
        pages_data = []
        total_tokens = 0
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing page {i+1}/{len(image_paths)}: {image_path}")
            
            # Run OCR on image
            tokens = run_ocr_on_image(image_path, lang, job_id)
            
            page_data = {
                "page_index": i,
                "image_path": image_path,
                "tokens": tokens
            }
            
            pages_data.append(page_data)
            total_tokens += len(tokens)
            
            logger.info(f"Page {i+1} completed: {len(tokens)} tokens")
        
        # Prepare final results
        ocr_results = {
            "job_id": job_id,
            "timestamp": str(pd.Timestamp.now()),
            "ocr_engine": "PaddleOCR",
            "pages": pages_data,
            "total_pages": len(pages_data),
            "total_tokens": total_tokens
        }
        
        logger.info(f"✅ OCR processing completed: {len(pages_data)} pages, {total_tokens} tokens")
        return ocr_results
        
    except Exception as e:
        logger.error(f"❌ OCR processing failed: {e}")
        raise


def save_ocr_output(job_id: str, ocr_results: Dict, output_dir: Path = None):
    """
    Save OCR results to JSON file
    
    Args:
        job_id: Job identifier
        ocr_results: OCR results dictionary
        output_dir: Output directory (optional)
    """
    try:
        if output_dir is None:
            script_dir = Path(__file__).parent.parent
            output_dir = script_dir / "tmp" / "results" / job_id
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "ocr.json"
        
        # Custom JSON encoder to handle numpy types
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            if hasattr(obj, 'dtype'):
                if 'int' in str(obj.dtype):
                    return int(obj)
                elif 'float' in str(obj.dtype):
                    return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            return obj
        
        # Convert all numpy types in the results
        def clean_results(data):
            if isinstance(data, dict):
                return {k: clean_results(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_results(item) for item in data]
            else:
                return convert_numpy_types(data)
        
        cleaned_results = clean_results(ocr_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ OCR results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save OCR results: {e}")
        # Save error info to OCR file for debugging
        try:
            error_results = {
                "job_id": job_id,
                "pages": [],
                "total_pages": 0,
                "total_tokens": 0,
                "error": str(e),
                "status": "ocr_failed_continuing"
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(error_results, f, indent=2, ensure_ascii=False)
        except:
            pass


# Import pandas for timestamp (if available)
try:
    import pandas as pd
except ImportError:
    # Fallback to datetime if pandas not available
    from datetime import datetime
    class pd:
        @staticmethod
        def Timestamp():
            class MockTimestamp:
                def now():
                    return datetime.now().isoformat()
            return MockTimestamp
        Timestamp = Timestamp()