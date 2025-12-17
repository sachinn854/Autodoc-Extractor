"""
OCR Engine module using PaddleOCR for document text extraction.
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
    Returns a PaddleOCR instance initialized for the given language.
    Uses singleton pattern for efficient memory usage.
    
    Args:
        lang: Language code for OCR (default: "en")
        job_id: Job ID for status updates during model download
        
    Returns:
        PaddleOCR instance
        
    Raises:
        Exception: If PaddleOCR initialization fails
    """
    global _ocr_engine_cache
    
    # Check if engine already cached
    if lang in _ocr_engine_cache:
        logger.info(f"Using cached OCR engine for language: {lang}")
        return _ocr_engine_cache[lang]
    
    logger.info(f"Initializing PaddleOCR engine for language: {lang}")
    
    # Update job status to indicate model download
    if job_id:
        from app.main import update_job_status
        update_job_status(job_id, "processing", "Downloading OCR models (first time setup, may take 2-3 minutes)...")
    
    try:
        import os
        # Disable model download progress to reduce verbosity
        os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        
        from paddleocr import PaddleOCR
        
        # Initialize PaddleOCR with progressive fallback approaches
        init_attempts = [
            # Modern PaddleOCR 3.x+ approach
            {'use_textline_orientation': True, 'lang': lang},
            # Legacy approach with angle classification
            {'use_angle_cls': True, 'lang': lang},
            # Basic approach without angle features
            {'lang': lang}
        ]
        
        ocr = None
        for i, params in enumerate(init_attempts, 1):
            try:
                logger.info(f"OCR initialization attempt {i}: {params}")
                ocr = PaddleOCR(**params)
                logger.info(f"OCR initialized successfully on attempt {i}")
                logger.info(f"OCR type: {type(ocr)}")
                logger.info(f"OCR has predict: {hasattr(ocr, 'predict')}")
                logger.info(f"OCR has ocr: {hasattr(ocr, 'ocr')}")
                break
            except Exception as e:
                logger.error(f"OCR attempt {i} failed: {e}")
                logger.error(f"Exception details: {traceback.format_exc()}")
                if i == len(init_attempts):
                    raise Exception(f"All OCR initialization attempts failed: {e}")
        
        if ocr is None:
            raise Exception("Failed to initialize PaddleOCR with any configuration")
        
        # Verify OCR engine is working
        try:
            logger.info("Testing OCR engine with a simple verification...")
            # Create a minimal test to verify OCR is responsive
            import numpy as np
            test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
            if hasattr(ocr, 'predict'):
                logger.info("Using predict method for verification...")
                test_result = ocr.predict(test_img)
                logger.info(f"✅ OCR predict verification passed, result type: {type(test_result)}")
            else:
                logger.info("Using ocr method for verification...")  
                test_result = ocr.ocr(test_img)
                logger.info(f"✅ OCR method verification passed, result type: {type(test_result)}")
            
        except Exception as test_error:
            logger.error(f"❌ OCR engine verification failed: {test_error}")
            logger.error(f"Test error traceback: {traceback.format_exc()}")
            raise Exception(f"OCR engine failed post-initialization verification: {test_error}")

        # Cache the engine
        _ocr_engine_cache[lang] = ocr
        
        # Update job status to indicate model download complete
        if job_id:
            update_job_status(job_id, "processing", "OCR models downloaded, proceeding with text extraction...")
        
        logger.info(f"✅ PaddleOCR engine verified and cached for language: {lang}")
        return ocr
        
    except ImportError:
        logger.error("PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")
        raise Exception("PaddleOCR not available. Please install paddlepaddle and paddleocr.")
    except Exception as e:
        logger.error(f"Failed to initialize PaddleOCR: {e}")
        raise Exception(f"OCR engine initialization failed: {e}")


def run_ocr_on_image(image_path: str, ocr_engine) -> List[dict]:
    """
    Runs OCR on a single processed image and extracts word-level tokens.
    
    Args:
        image_path: Path to the processed image file
        ocr_engine: PaddleOCR instance from get_ocr_engine()
        
    Returns:
        List of token dictionaries:
        [
          {
            "text": "extracted_text",
            "bbox": [x1, y1, x2, y2],  # pixel coordinates
            "confidence": 0.95
          },
          ...
        ]
        
    Raises:
        Exception: If OCR processing fails
    """
    logger.info(f"Running OCR on image: {image_path}")
    
    try:
        # Verify image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Run PaddleOCR with fallback approaches
        try:
            # Try predict method (new PaddleOCR API)
            if hasattr(ocr_engine, 'predict'):
                logger.info("Using new PaddleOCR predict() API")
                result = ocr_engine.predict(image_path)
            else:
                # Fallback to old OCR method
                logger.info("Using legacy PaddleOCR ocr() API")
                result = ocr_engine.ocr(image_path)
            
            logger.info(f"OCR processing successful, result type: {type(result)}, length: {len(result) if result else 0}")
            
            # Debug: Log detailed result structure
            if result:
                logger.info(f"OCR Result details:")
                logger.info(f"  - Type: {type(result)}")
                logger.info(f"  - Length: {len(result)}")
                if len(result) > 0:
                    logger.info(f"  - First element type: {type(result[0])}")
                    if hasattr(result[0], 'keys'):
                        logger.info(f"  - OCRResult keys: {list(result[0].keys())}")
                        # Log the actual content
                        if 'rec_texts' in result[0]:
                            logger.info(f"  - rec_texts: {result[0]['rec_texts']}")
                        if 'rec_scores' in result[0]:
                            logger.info(f"  - rec_scores: {result[0]['rec_scores']}")
                    else:
                        logger.info(f"  - First element content: {result[0]}")
            else:
                logger.warning(f"OCR returned None or empty result for image: {image_path}")
        
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            traceback.print_exc()
            if "cls" in str(e):
                logger.warning(f"OCR failed with cls parameter, retrying with basic settings: {e}")
                # Create a new simple OCR instance
                from paddleocr import PaddleOCR
                simple_ocr = PaddleOCR(lang='en')
                result = simple_ocr.ocr(image_path)
                logger.info(f"Fallback OCR successful, result type: {type(result)}, length: {len(result) if result else 0}")
            else:
                raise
        
        tokens = []
        
        # Log the complete result structure for debugging
        logger.info(f"Processing OCR result - Type: {type(result)}")
        if result is None:
            logger.warning("OCR returned None result!")
            return []
        elif isinstance(result, (list, dict)):
            logger.info(f"Result length/keys: {len(result) if isinstance(result, list) else list(result.keys())}")
            if isinstance(result, list) and len(result) > 0:
                logger.info(f"First element type: {type(result[0])}")
                if hasattr(result[0], 'keys'):
                    logger.info(f"First element keys: {list(result[0].keys())}")
                logger.info(f"First element preview: {str(result[0])[:200]}...")
        
        # Handle new PaddleOCR v5 result format (list with OCR pipeline results)
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            logger.info("Detected new PaddleOCR v5 pipeline result format")
            
            # Extract OCR results from pipeline format
            ocr_data = result[0]  # First element contains OCR results
            
            # Check for different possible keys in the result
            possible_text_keys = ['rec_texts', 'text_results', 'texts', 'ocr_texts']
            possible_box_keys = ['dt_polys', 'boxes', 'polygons', 'bboxes']
            possible_score_keys = ['rec_scores', 'scores', 'confidences']
            
            rec_texts = []
            dt_polys = []
            rec_scores = []
            
            # Find the actual keys used
            for key in possible_text_keys:
                if key in ocr_data:
                    rec_texts = ocr_data[key]
                    logger.info(f"Found text data in key: {key}")
                    break
                    
            for key in possible_box_keys:
                if key in ocr_data:
                    dt_polys = ocr_data[key]
                    logger.info(f"Found box data in key: {key}")
                    break
                    
            for key in possible_score_keys:
                if key in ocr_data:
                    rec_scores = ocr_data[key]
                    logger.info(f"Found score data in key: {key}")
                    break
            
            logger.info(f"Pipeline format: {len(rec_texts)} texts, {len(rec_scores)} scores, {len(dt_polys)} polygons")
            
            # Process the extracted data (same logic as dict format)
            for i in range(len(rec_texts)):
                if i >= len(rec_scores):
                    break
                    
                text = str(rec_texts[i]).strip() if rec_texts[i] else ""
                confidence = float(rec_scores[i]) if rec_scores[i] is not None else 0.0
                
                # Skip empty text or low confidence
                if not text or confidence < 0.1:
                    continue
                
                # Get polygon and convert to bounding box
                if i < len(dt_polys) and dt_polys[i] is not None and len(dt_polys[i]) > 0:
                    polygon = dt_polys[i]
                    # Convert polygon points to bounding box [x1, y1, x2, y2]
                    x_coords = [point[0] for point in polygon]
                    y_coords = [point[1] for point in polygon]
                    
                    x1 = int(min(x_coords))
                    y1 = int(min(y_coords))
                    x2 = int(max(x_coords))
                    y2 = int(max(y_coords))
                else:
                    # Default bbox if coordinates not available
                    x1, y1, x2, y2 = 0, 0, 100, 20
                    logger.warning(f"No polygon data for text '{text}', using default bbox")
                
                # Create standardized token
                token = {
                    "text": text,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(confidence, 3)
                }
                
                tokens.append(token)
                logger.debug(f"Pipeline Token {i+1}: '{text}' at [{x1},{y1},{x2},{y2}] conf={confidence:.3f}")
        
        # Handle PaddleOCR v5/PaddleX dictionary format
        elif isinstance(result, dict):
            logger.info("Processing OCR result as PaddleOCR v5 dictionary format")
            
            # Extract data from dictionary keys
            dt_polys = result.get('dt_polys', [])
            rec_texts = result.get('rec_texts', [])
            rec_scores = result.get('rec_scores', [])
            
            logger.info(f"OCR dict keys: {list(result.keys())}")
            logger.info(f"Found {len(rec_texts)} text regions, {len(dt_polys)} polygons, {len(rec_scores)} scores")
            
            # Process each detected text
            for i in range(len(rec_texts)):
                if i >= len(rec_scores):
                    break
                    
                text = str(rec_texts[i]).strip() if rec_texts[i] else ""
                confidence = float(rec_scores[i]) if rec_scores[i] is not None else 0.0
                
                # Skip empty text or low confidence
                if not text or confidence < 0.1:
                    continue
                
                # Get polygon and convert to bounding box
                if i < len(dt_polys) and dt_polys[i] is not None and len(dt_polys[i]) > 0:
                    polygon = dt_polys[i]
                    # Convert polygon points to bounding box [x1, y1, x2, y2]
                    x_coords = [point[0] for point in polygon]
                    y_coords = [point[1] for point in polygon]
                    
                    x1 = int(min(x_coords))
                    y1 = int(min(y_coords))
                    x2 = int(max(x_coords))
                    y2 = int(max(y_coords))
                else:
                    # Default bbox if coordinates not available
                    x1, y1, x2, y2 = 0, 0, 100, 20
                    logger.warning(f"No polygon data for text '{text}', using default bbox")
                
                # Create standardized token
                token = {
                    "text": text,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(confidence, 3)
                }
                
                tokens.append(token)
                logger.debug(f"Token {i+1}: '{text}' at [{x1},{y1},{x2},{y2}] conf={confidence:.3f}")
        
        elif isinstance(result, list):
            # Handle PaddleOCR list format - could contain OCRResult objects or traditional format
            logger.info("Processing OCR result as list format")
            logger.info(f"List length: {len(result)}")
            
            if not result:
                logger.warning(f"Empty OCR result list for image: {image_path}")
            elif result[0] is None:
                logger.warning(f"OCR result contains None for image: {image_path}")
            elif hasattr(result[0], 'keys') and hasattr(result[0], 'get'):
                # Modern PaddleX OCRResult object format
                logger.info("Detected PaddleX OCRResult object in list")
                ocr_obj = result[0]
                
                # Extract arrays from OCRResult object (same as dict format)
                dt_polys = ocr_obj.get("dt_polys", [])
                rec_texts = ocr_obj.get("rec_texts", [])
                rec_scores = ocr_obj.get("rec_scores", [])
                
                logger.info(f"OCRResult: {len(rec_texts)} texts, {len(rec_scores)} scores, {len(dt_polys)} polygons")
                
                # Process each detected text (same logic as dict format)
                for i in range(len(rec_texts)):
                    text = str(rec_texts[i]).strip() if i < len(rec_texts) else ""
                    confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                    polygon = dt_polys[i] if i < len(dt_polys) else None
                    
                    # Skip empty text or low confidence
                    if not text or confidence < 0.1:
                        continue
                    
                    # Convert polygon to bounding box [x1, y1, x2, y2] (same logic)
                    if polygon is not None and len(polygon) >= 4:
                        try:
                            x_coords = [point[0] for point in polygon]
                            y_coords = [point[1] for point in polygon]
                            
                            x1 = int(min(x_coords))
                            y1 = int(min(y_coords))
                            x2 = int(max(x_coords))
                            y2 = int(max(y_coords))
                        except (IndexError, ValueError, TypeError) as e:
                            logger.warning(f"Invalid polygon for text '{text}': {e}")
                            x1, y1, x2, y2 = 0, 0, 100, 20  # Default bbox
                    else:
                        x1, y1, x2, y2 = 0, 0, 100, 20
                        logger.warning(f"No polygon data for text '{text}', using default bbox")
                    
                    # Create standardized token
                    token = {
                        "text": text,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(confidence, 3)
                    }
                    
                    tokens.append(token)
                    logger.debug(f"OCRResult Token {i}: '{text}' at [{x1},{y1},{x2},{y2}] conf={confidence:.3f}")
                
                logger.info(f"Extracted {len(tokens)} tokens from OCRResult object")
            
            elif isinstance(result[0], list):
                # Legacy PaddleOCR format: result[0] contains list of [bbox_coords, (text, confidence)]
                logger.info("Detected legacy PaddleOCR list format")
                logger.info(f"Legacy list length: {len(result[0])}")
                
                if not result[0]:
                    logger.warning(f"Empty legacy OCR result for image: {image_path}")
                
                for i, detection in enumerate(result[0]):
                    if not detection or not isinstance(detection, (list, tuple)) or len(detection) < 2:
                        continue
                        
                    bbox_coords = detection[0]  # List of 4 corner points
                    text_info = detection[1]    # (text_string, confidence_score)
                    
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = str(text_info[0]).strip()
                        confidence = float(text_info[1])
                    else:
                        continue
                    
                    if not text or confidence < 0.1:
                        continue
                    
                    # Convert quadrilateral to bounding box
                    x_coords = [point[0] for point in bbox_coords]
                    y_coords = [point[1] for point in bbox_coords]
                    
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    
                    token = {
                        "text": text,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(confidence, 3)
                    }
                    
                    tokens.append(token)
        
        else:
            logger.error(f"Unexpected OCR result type: {type(result)}")
            logger.error(f"Result content (first 500 chars): {str(result)[:500]}")
            if hasattr(result, '__dict__'):
                logger.error(f"Result attributes: {dir(result)}")
        
        logger.info(f"OCR completed: {len(tokens)} tokens extracted from {image_path}")
        return tokens
        
    except Exception as e:
        logger.error(f"OCR processing failed for {image_path}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return empty tokens instead of raising exception to allow pipeline to continue
        logger.warning(f"Returning empty tokens due to OCR error")
        return []


def normalize_bbox(bbox: List[int], img_width: int, img_height: int) -> List[float]:
    """
    Convert pixel bounding box coordinates to normalized coordinates (0-1 range).
    
    Args:
        bbox: Bounding box in pixel coordinates [x1, y1, x2, y2]
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Normalized bounding box [x1_norm, y1_norm, x2_norm, y2_norm]
    """
    if img_width <= 0 or img_height <= 0:
        logger.warning("Invalid image dimensions for normalization")
        return [0.0, 0.0, 0.0, 0.0]
    
    x1, y1, x2, y2 = bbox
    
    # Normalize to 0-1 range
    x1_norm = max(0.0, min(1.0, x1 / img_width))
    y1_norm = max(0.0, min(1.0, y1 / img_height))
    x2_norm = max(0.0, min(1.0, x2 / img_width))
    y2_norm = max(0.0, min(1.0, y2 / img_height))
    
    return [round(x1_norm, 6), round(y1_norm, 6), round(x2_norm, 6), round(y2_norm, 6)]


def get_image_dimensions(image_path: str) -> tuple:
    """
    Get image dimensions efficiently.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (width, height)
    """
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        logger.error(f"Failed to get image dimensions for {image_path}: {e}")
        return (0, 0)


def run_ocr_on_pages(image_paths: List[str], ocr_engine, normalize_coords: bool = False) -> Dict:
    """
    Run OCR on multiple preprocessed images (e.g., PDF pages).
    
    Args:
        image_paths: List of paths to processed image files
        ocr_engine: PaddleOCR instance from get_ocr_engine()
        normalize_coords: Whether to normalize bounding box coordinates
        
    Returns:
        Dictionary containing OCR results for all pages:
        {
           "pages": [
                { 
                    "page_index": 0, 
                    "image_path": "path/to/page_01.jpg",
                    "tokens": [...],
                    "total_tokens": 15,
                    "image_dimensions": [width, height]
                },
                ...
           ],
           "total_pages": 2,
           "total_tokens": 30
        }
    """
    logger.info(f"Running OCR on {len(image_paths)} pages")
    
    pages_data = []
    total_tokens = 0
    
    for page_idx, image_path in enumerate(image_paths):
        logger.info(f"Processing page {page_idx + 1}/{len(image_paths)}: {image_path}")
        
        try:
            # Run OCR on single image
            logger.info(f"CALLING run_ocr_on_image for: {image_path}")
            tokens = run_ocr_on_image(image_path, ocr_engine)
            logger.info(f"RETURNED from run_ocr_on_image: {len(tokens)} tokens from page {page_idx + 1}")
            
            # Get image dimensions for normalization
            img_width, img_height = get_image_dimensions(image_path)
            
            # Normalize bounding boxes if requested
            if normalize_coords and img_width > 0 and img_height > 0:
                for token in tokens:
                    token["bbox_normalized"] = normalize_bbox(token["bbox"], img_width, img_height)
            
            # Create page data
            page_data = {
                "page_index": page_idx,
                "image_path": image_path,
                "tokens": tokens,
                "total_tokens": len(tokens),
                "image_dimensions": [img_width, img_height]
            }
            
            pages_data.append(page_data)
            total_tokens += len(tokens)
            
            logger.info(f"Page {page_idx + 1} processed: {len(tokens)} tokens extracted")
            
        except Exception as e:
            logger.error(f"Failed to process page {page_idx + 1} ({image_path}): {e}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            # Add empty page data for failed pages
            page_data = {
                "page_index": page_idx,
                "image_path": image_path,
                "tokens": [],
                "total_tokens": 0,
                "image_dimensions": [0, 0],
                "error": str(e)
            }
            pages_data.append(page_data)
    
    # Create final result
    result = {
        "pages": pages_data,
        "total_pages": len(image_paths),
        "total_tokens": total_tokens,
        "normalize_coords": normalize_coords
    }
    
    logger.info(f"OCR processing completed: {total_tokens} total tokens from {len(image_paths)} pages")
    return result


def save_ocr_output(job_id: str, ocr_data: dict) -> str:
    """
    Save OCR results to JSON file in the results directory.
    GUARANTEED to save OCR JSON even if empty to prevent FileNotFoundError.
    
    Args:
        job_id: Unique job identifier
        ocr_data: OCR results dictionary from run_ocr_on_pages()
        
    Returns:
        Path to the saved OCR JSON file
        
    Raises:
        Exception: If file saving fails
    """
    logger.info(f"Saving OCR output for job: {job_id}")
    
    try:
        # Construct output path
        script_dir = Path(__file__).parent.parent  # Go up to backend/ directory
        results_dir = script_dir / "tmp" / "results" / job_id
        
        # Ensure OCR data has minimum required structure
        if not ocr_data or not isinstance(ocr_data, dict):
            logger.warning("Empty or invalid OCR data, creating default structure")
            ocr_data = {"pages": [], "total_pages": 0, "total_tokens": 0}
        results_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = results_dir / "ocr.json"
        
        # Add metadata to OCR data
        from datetime import datetime
        ocr_data_with_metadata = {
            "job_id": job_id,
            "timestamp": str(datetime.now()),
            "ocr_engine": "PaddleOCR",
            **ocr_data
        }
        
        # Write JSON file with proper formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_data_with_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"OCR output saved to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to save OCR output for job {job_id}: {e}")
        raise Exception(f"OCR output saving failed: {e}")


def process_document_ocr(preprocessed_image_paths: List[str], job_id: str, lang: str = "en", normalize_coords: bool = True) -> Dict:
    """
    Complete OCR processing pipeline for a document.
    
    Args:
        preprocessed_image_paths: List of preprocessed image paths from Phase 2
        job_id: Unique job identifier
        lang: OCR language code
        normalize_coords: Whether to normalize bounding box coordinates
        
    Returns:
        OCR results dictionary with saved file path
    """
    logger.info(f"Starting complete OCR processing for job: {job_id}")
    
    try:
        # Get OCR engine
        logger.info("Getting OCR engine...")
        try:
            ocr_engine = get_ocr_engine(lang, job_id)
            logger.info(f"OCR engine obtained successfully: {type(ocr_engine)}")
            logger.info(f"OCR engine has predict method: {hasattr(ocr_engine, 'predict')}")
            logger.info(f"OCR engine has ocr method: {hasattr(ocr_engine, 'ocr')}")
        except Exception as engine_error:
            logger.error(f"get_ocr_engine failed: {engine_error}")
            logger.error(f"Engine error traceback: {traceback.format_exc()}")
            raise
        
        # Run OCR on all pages
        logger.info(f"Running OCR on {len(preprocessed_image_paths)} pages...")
        try:
            ocr_results = run_ocr_on_pages(preprocessed_image_paths, ocr_engine, normalize_coords)
            logger.info(f"OCR results obtained: {type(ocr_results)} with {ocr_results.get('total_tokens', 0)} tokens")
        except Exception as pages_error:
            logger.error(f"run_ocr_on_pages failed: {pages_error}")
            logger.error(f"Pages error traceback: {traceback.format_exc()}")
            raise
        
        # Save results
        output_path = save_ocr_output(job_id, ocr_results)
        
        # Add output path to results
        ocr_results["output_file"] = output_path
        
        logger.info(f"Complete OCR processing finished for job: {job_id}")
        return ocr_results
        
    except Exception as e:
        logger.error(f"Complete OCR processing failed for job {job_id}: {e}")
        raise Exception(f"OCR processing pipeline failed: {e}")


if __name__ == "__main__":
    # Simple test
    print("Testing OCR Engine...")
    
    try:
        # Test engine initialization
        ocr = get_ocr_engine("en")
        print("✓ OCR engine initialized successfully")
        
        # Test would require actual image files
        print("✓ OCR engine module ready for use")
        
    except Exception as e:
        print(f"✗ OCR engine test failed: {e}")
        print("Note: Install PaddleOCR with: pip install paddlepaddle paddleocr")