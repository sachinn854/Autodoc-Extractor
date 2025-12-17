"""
Document preprocessing module for OCR and object detection preparation.
Handles PDF conversion, image enhancement, denoising, deskewing, and format standardization.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import List, Union, Optional
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert PDF to list of PIL images.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (default 300)
        
    Returns:
        List of PIL Image objects, one per page
        
    Raises:
        Exception: If PDF conversion fails
    """
    logger.info(f"Converting PDF to images: {pdf_path} at {dpi} DPI")
    
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        logger.info(f"Successfully converted PDF to {len(images)} pages")
        return images
    except Exception as e:
        logger.error(f"Failed to convert PDF {pdf_path}: {e}")
        raise Exception(f"PDF conversion error: {e}")


def load_image(path: str) -> np.ndarray:
    """
    Load image file as BGR numpy array.
    
    Args:
        path: Path to image file
        
    Returns:
        BGR numpy array (OpenCV format)
        
    Raises:
        Exception: If image loading fails
    """
    logger.info(f"Loading image: {path}")
    
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image from {path}")
        
        logger.info(f"Loaded image: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"Failed to load image {path}: {e}")
        raise Exception(f"Image loading error: {e}")


def denoise_image(img: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """
    Remove noise from image while preserving edges.
    
    Args:
        img: BGR input image
        method: "bilateral" or "median"
        
    Returns:
        Denoised BGR image
    """
    logger.info(f"Denoising image using {method} method")
    
    try:
        if method == "bilateral":
            # Parameters: d=9, sigmaColor=75, sigmaSpace=75
            denoised = cv2.bilateralFilter(img, 9, 75, 75)
        elif method == "median":
            # Use 3x3 kernel
            denoised = cv2.medianBlur(img, 3)
        else:
            logger.warning(f"Unknown denoising method {method}, using bilateral")
            denoised = cv2.bilateralFilter(img, 9, 75, 75)
        
        logger.info("Denoising completed")
        return denoised
        
    except Exception as e:
        logger.error(f"Denoising failed: {e}")
        return img  # Return original if denoising fails


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale.
    
    Args:
        img: BGR input image
        
    Returns:
        Single-channel grayscale image (uint8)
    """
    logger.info("Converting to grayscale")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    logger.info(f"Grayscale conversion complete: {gray.shape}")
    return gray


def apply_threshold(gray: np.ndarray, method: str = "adaptive", enable: bool = True) -> np.ndarray:
    """
    Apply thresholding to grayscale image (optional step).
    
    Args:
        gray: Grayscale input image
        method: "adaptive" or "otsu"
        enable: If False, return original image
        
    Returns:
        Thresholded binary image or original if disabled
    """
    if not enable:
        logger.info("Thresholding disabled, returning original")
        return gray
    
    logger.info(f"Applying {method} thresholding")
    
    try:
        if method == "adaptive":
            # Adaptive Gaussian thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif method == "otsu":
            # Global Otsu thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            logger.warning(f"Unknown threshold method {method}, using adaptive")
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        
        logger.info("Thresholding completed")
        return thresh
        
    except Exception as e:
        logger.error(f"Thresholding failed: {e}")
        return gray


def deskew_image(img: np.ndarray, delta_thresh: float = 1.0) -> np.ndarray:
    """
    Detect and correct skew in document images.
    
    Args:
        img: BGR or grayscale input image
        delta_thresh: Minimum angle (degrees) to trigger deskewing
        
    Returns:
        Deskewed BGR image
    """
    logger.info(f"Detecting skew (threshold: {delta_thresh}°)")
    
    try:
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            original_is_color = True
        else:
            gray = img.copy()
            original_is_color = False
        
        # Apply threshold for better contour detection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and get their bounding rectangles
        angles = []
        min_area = 100  # Filter out small contours
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                
                # Normalize angle to [-45, 45] range
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                    
                angles.append(angle)
        
        # Calculate median angle if we have enough samples
        if len(angles) < 3:
            logger.info("Insufficient contours for skew detection, skipping deskew")
            return img if original_is_color else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        median_angle = np.median(angles)
        logger.info(f"Detected skew angle: {median_angle:.2f}°")
        
        # Check if angle exceeds threshold
        if abs(median_angle) < delta_thresh:
            logger.info(f"Skew angle {median_angle:.2f}° below threshold {delta_thresh}°, skipping correction")
            return img if original_is_color else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Rotate image to correct skew
        height, width = gray.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Apply rotation to original image
        if original_is_color:
            rotated = cv2.warpAffine(img, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            rotated = cv2.warpAffine(img, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
        
        logger.info(f"Image deskewed by {median_angle:.2f}°")
        return rotated
        
    except Exception as e:
        logger.error(f"Deskewing failed: {e}")
        return img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def resize_for_ocr(img: np.ndarray, target_height: int = 1200, max_width: int = 3000) -> np.ndarray:
    """
    Resize image for optimal OCR performance while preserving aspect ratio.
    
    Args:
        img: BGR input image
        target_height: Target height in pixels
        max_width: Maximum allowed width
        
    Returns:
        Resized BGR image
    """
    h, w = img.shape[:2]
    logger.info(f"Resizing image from {w}x{h} (target height: {target_height}, max width: {max_width})")
    
    try:
        # Calculate scale factor based on target height
        scale_factor = target_height / h
        new_width = int(w * scale_factor)
        new_height = target_height
        
        # Check if new width exceeds maximum
        if new_width > max_width:
            scale_factor = max_width / w
            new_width = max_width
            new_height = int(h * scale_factor)
        
        # Resize image
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        logger.info(f"Resized to {new_width}x{new_height} (scale: {scale_factor:.3f})")
        return resized
        
    except Exception as e:
        logger.error(f"Resizing failed: {e}")
        return img


def enhance_contrast(img: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    """
    Enhance image contrast using CLAHE or histogram equalization.
    
    Args:
        img: BGR input image
        use_clahe: Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
    Returns:
        Contrast-enhanced BGR image
    """
    logger.info(f"Enhancing contrast (CLAHE: {use_clahe})")
    
    try:
        if use_clahe:
            # Apply CLAHE to each channel separately
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            
            # Split channels
            b, g, r = cv2.split(img)
            
            # Apply CLAHE to each channel
            b_clahe = clahe.apply(b)
            g_clahe = clahe.apply(g)
            r_clahe = clahe.apply(r)
            
            # Merge channels back
            enhanced = cv2.merge([b_clahe, g_clahe, r_clahe])
        else:
            # Convert to YUV and apply histogram equalization to Y channel
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        logger.info("Contrast enhancement completed")
        return enhanced
        
    except Exception as e:
        logger.error(f"Contrast enhancement failed: {e}")
        return img


def sharpen_image(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Sharpen image using unsharp mask technique.
    
    Args:
        img: BGR input image
        amount: Sharpening strength (1.0 = normal)
        
    Returns:
        Sharpened BGR image
    """
    logger.info(f"Sharpening image (amount: {amount})")
    
    try:
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
        
        # Create unsharp mask
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        
        logger.info("Image sharpening completed")
        return sharpened
        
    except Exception as e:
        logger.error(f"Sharpening failed: {e}")
        return img


def save_image(img: np.ndarray, out_path: str, quality: int = 95) -> None:
    """
    Save image to disk with proper encoding and directory creation.
    
    Args:
        img: BGR image to save
        out_path: Output file path
        quality: JPEG quality (1-100)
    """
    logger.info(f"Saving image to {out_path} (quality: {quality})")
    
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Save with specified quality
        pil_img.save(out_path, "JPEG", quality=quality, optimize=True)
        
        logger.info(f"Image saved successfully: {out_path}")
        
    except Exception as e:
        logger.error(f"Failed to save image {out_path}: {e}")
        raise Exception(f"Image saving error: {e}")


def preprocess_document(input_path: str, job_id: str, dpi: int = 300, do_threshold: bool = False) -> List[str]:
    """
    Main preprocessing wrapper function. Processes PDF or image files through
    the complete enhancement pipeline.
    
    Args:
        input_path: Path to input PDF or image file
        job_id: Unique job identifier for organizing temporary files
        dpi: DPI for PDF conversion (default 300)
        do_threshold: Whether to apply thresholding step
        
    Returns:
        List of paths to processed image files
        
    Raises:
        Exception: If preprocessing fails
    """
    logger.info(f"Starting document preprocessing: {input_path} (job_id: {job_id})")
    
    # Create directory structure (use relative path from current script location)
    script_dir = Path(__file__).parent.parent  # Go up to backend/ directory
    base_tmp_dir = script_dir / "tmp"
    uploads_dir = base_tmp_dir / "uploads" / job_id
    preprocessed_dir = base_tmp_dir / "preprocessed" / job_id
    results_dir = base_tmp_dir / "results" / job_id
    
    uploads_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Copy input file to uploads directory (if not already there)
        input_path_obj = Path(input_path)
        raw_input_path = uploads_dir / f"raw_input{input_path_obj.suffix}"
        
        # Check if file already exists at target location (avoid copying to itself)
        if raw_input_path != Path(input_path):
            # Retry logic for Windows file access issues
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.copy2(input_path, raw_input_path)
                    logger.info(f"Copied input file to: {raw_input_path}")
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"File access retry {attempt + 1}/{max_retries}: {e}")
                        time.sleep(0.5)  # Wait 500ms before retry
                    else:
                        raise
        else:
            logger.info(f"File already at target location: {raw_input_path}")
        
        # Determine if input is PDF or image
        if input_path_obj.suffix.lower() == '.pdf':
            logger.info("Processing PDF file")
            # Convert PDF to images
            pil_images = convert_pdf_to_images(str(raw_input_path), dpi=dpi)
            
            # Convert PIL images to numpy arrays
            images = []
            for pil_img in pil_images:
                # Convert PIL to BGR numpy array
                img_array = np.array(pil_img.convert('RGB'))
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                images.append(img_bgr)
        else:
            logger.info("Processing image file")
            # Load single image
            img = load_image(str(raw_input_path))
            images = [img]
        
        processed_paths = []
        
        # Process each page/image through the pipeline
        for page_idx, img in enumerate(images, 1):
            logger.info(f"Processing page {page_idx}/{len(images)}")
            
            try:
                # Apply preprocessing pipeline in order:
                # 1. Denoise
                processed = denoise_image(img, method="bilateral")
                
                # 2. Convert to grayscale
                gray = to_grayscale(processed)
                
                # 3. Apply threshold (optional)
                thresholded = apply_threshold(gray, method="adaptive", enable=do_threshold)
                
                # Convert back to BGR for remaining steps
                if len(thresholded.shape) == 2:
                    processed = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
                else:
                    processed = thresholded
                
                # 4. Deskew
                processed = deskew_image(processed, delta_thresh=1.0)
                
                # 5. Enhance contrast
                processed = enhance_contrast(processed, use_clahe=True)
                
                # 6. Sharpen
                processed = sharpen_image(processed, amount=1.0)
                
                # 7. Resize for OCR
                processed = resize_for_ocr(processed, target_height=1200, max_width=3000)
                
                # Save processed image
                output_filename = f"page_{page_idx:02d}.jpg"
                output_path = preprocessed_dir / output_filename
                save_image(processed, str(output_path), quality=95)
                
                processed_paths.append(str(output_path))
                logger.info(f"Successfully processed page {page_idx}: {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to process page {page_idx}: {e}")
                # Continue with other pages on failure
                continue
        
        if not processed_paths:
            raise Exception("No pages were successfully processed")
        
        logger.info(f"Document preprocessing completed: {len(processed_paths)} pages processed")
        return processed_paths
        
    except Exception as e:
        logger.error(f"Document preprocessing failed: {e}")
        raise Exception(f"Preprocessing error: {e}")


def delete_tmp_job(job_id: str) -> None:
    """
    Clean up temporary files for a specific job.
    
    Args:
        job_id: Job identifier to clean up
    """
    logger.info(f"Cleaning up temporary files for job: {job_id}")
    
    script_dir = Path(__file__).parent.parent  # Go up to backend/ directory
    base_tmp_dir = script_dir / "tmp"
    
    # Remove all job-related directories
    for subdir in ["uploads", "preprocessed", "results"]:
        job_dir = base_tmp_dir / subdir / job_id
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
                logger.info(f"Removed directory: {job_dir}")
            except Exception as e:
                logger.error(f"Failed to remove directory {job_dir}: {e}")
    
    logger.info(f"Cleanup completed for job: {job_id}")


def create_demo_files():
    """Create demo files without cleanup to show folder structure."""
    import uuid
    import cv2
    
    # Create demo image
    demo_img = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    cv2.rectangle(demo_img, (100, 100), (300, 150), (0, 0, 0), -1)
    cv2.rectangle(demo_img, (100, 200), (500, 250), (0, 0, 0), -1)
    
    # Save demo input
    demo_input_path = "tmp/demo_input.jpg"
    os.makedirs(os.path.dirname(demo_input_path), exist_ok=True)
    save_image(demo_img, demo_input_path)
    
    # Process without cleanup
    demo_job_id = "demo_job_123"
    processed_paths = preprocess_document(demo_input_path, demo_job_id)
    
    print(f"\n🎯 Demo files created (job_id: {demo_job_id}):")
    print(f"📁 Raw input: backend/tmp/uploads/{demo_job_id}/raw_input.jpg")
    print(f"📁 Processed: backend/tmp/preprocessed/{demo_job_id}/page_01.jpg") 
    print(f"📁 Results: backend/tmp/results/{demo_job_id}/ (empty, ready for OCR)")
    print(f"\n🔍 Check these folders to see the files!")
    
    return processed_paths


if __name__ == "__main__":
    # Simple test
    import uuid
    
    # Test with a sample image (if available)
    test_job_id = str(uuid.uuid4())
    print(f"Test job ID: {test_job_id}")
    
    # Create demo files without cleanup
    print("\nCreating demo files...")
    create_demo_files()
    
    # This would be called from main.py in actual usage:
    # processed_paths = preprocess_document("sample.pdf", test_job_id)
    # print(f"Processed files: {processed_paths}")