"""
OCR Engine module using Tesseract for document text extraction.
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
    Returns a Tesseract OCR instance (lightweight alternative to PaddleOCR)
    Uses singleton pattern for efficient memory usage.
    
    Args:
        lang: Language code for OCR (default: "en")
        job_id: Job ID for status updates
        
    Returns:
        Tesseract OCR function
        
    Raises:
        Exception: If Tesseract OCR initialization fails
    """
    global _ocr_engine_cache
    
    # Check if engine already cached
    if lang in _ocr_engine_cache:
        logger.info(f"✅ Using cached Tesseract OCR engine for language: {lang}")
        return _ocr_engine_cache[lang]
    
    logger.info(f"🔄 Initializing Tesseract OCR engine for language: {lang}")
    
    # Update job status
    if job_id:
        from app.main import update_job_status
        update_job_status(job_id, "processing", "🔄 Loading OCR engine (Tesseract)...")
    
    try:
        import pytesseract
        
        # Set Tesseract path explicitly for Windows
        if os.name == 'nt':  # Windows
            tesseract_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Tesseract-OCR\tesseract.exe"
            ]
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info(f"✅ Set Tesseract path: {path}")
                    break
        
        # Simple Tesseract wrapper function to match PaddleOCR interface
        def tesseract_ocr(image_path):
            """Tesseract OCR wrapper to match PaddleOCR interface"""
            try:
                logger.info(f"🔍 Tesseract processing: {image_path}")
                
                # Read image
                if isinstance(image_path, str):
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.error(f"❌ Could not read image: {image_path}")
                        return [[]]
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    logger.info(f"✅ Image loaded: {image.shape}")
                else:
                    pil_image = Image.fromarray(image_path)
                
                # Get OCR data with bounding boxes
                logger.info("🔄 Running Tesseract OCR...")
                data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
                logger.info(f"📊 Tesseract raw results: {len(data['text'])} text elements")
                
                # Convert to PaddleOCR-like format: [[[bbox], (text, confidence)], ...]
                results = []
                valid_count = 0
                for i in range(len(data['text'])):
                    conf = int(data['conf'][i])
                    text = data['text'][i].strip()
                    
                    if conf > 30 and text:  # Confidence threshold
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                        confidence = float(conf) / 100.0
                        results.append([bbox, (text, confidence)])
                        valid_count += 1
                        
                        if valid_count <= 5:  # Log first 5 for debugging
                            logger.info(f"  📝 Text: '{text}' (conf: {conf}%)")
                
                logger.info(f"✅ Tesseract extracted {len(results)} valid tokens")
                return [results] if results else [[]]
                
            except Exception as e:
                logger.error(f"❌ Tesseract OCR failed: {e}")
                logger.error(f"📍 Error details: {traceback.format_exc()}")
                return [[]]
        
        # Test OCR engine
        logger.info("Testing Tesseract OCR engine...")
        import numpy as np
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
        test_result = tesseract_ocr(test_img)
        logger.info(f"✅ Tesseract OCR test passed, result type: {type(test_result)}")
        
        # Cache the OCR function
        _ocr_engine_cache[lang] = tesseract_ocr
        
        # Update job status
        if job_id:
            update_job_status(job_id, "processing", "✅ OCR engine ready, proceeding with text extraction...")
        
        logger.info(f"✅ Tesseract OCR engine initialized and cached for language: {lang}")
        return tesseract_ocr
        
    except Exception as e:
        error_msg = f"Failed to initialize Tesseract OCR: {str(e)}"
        logger.error(error_msg)
        
        if job_id:
            update_job_status(job_id, "failed", error=error_msg)
        
        raise Exception(error_msg)


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
    try:
        logger.info(f"🔍 Running OCR on image: {image_path}")
        
        # Get OCR engine
        ocr_engine = get_ocr_engine(lang, job_id)
        
        # Run OCR
        results = ocr_engine(image_path)
        
        # Convert results to structured format
        tokens = []
        for page_result in results:
            for item in page_result:
                if len(item) >= 2:
                    bbox, (text, confidence) = item[0], item[1]
                    
                    # Convert bbox to flat format [x1, y1, x2, y2]
                    if len(bbox) == 4 and len(bbox[0]) == 2:
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        flat_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    else:
                        flat_bbox = bbox
                    
                    tokens.append({
                        "text": text,
                        "bbox": flat_bbox,
                        "confidence": confidence
                    })
        
        logger.info(f"✅ OCR completed: {len(tokens)} tokens extracted")
        return tokens
        
    except Exception as e:
        logger.error(f"❌ OCR failed on {image_path}: {e}")
        return []


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
            "ocr_engine": "Tesseract",
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
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ OCR results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save OCR results: {e}")


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