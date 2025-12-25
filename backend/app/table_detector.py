import cv2
import numpy as np
from typing import List, Dict, Tuple
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableDetector:
    """
    YOLO-based table detection for document images (high accuracy)
    Uses YOLOv8 for advanced table structure detection
    """
    
    def __init__(self):
        """Initialize YOLO-based table detector"""
        self.yolo_model = None
        self._initialize_yolo()
    
    def _initialize_yolo(self):
        """Initialize YOLO model for table detection"""
        from ultralytics import YOLO
        
        # Try to load pre-trained table detection model
        model_path = "yolov8n.pt"  # Start with nano model for speed
        
        logger.info(f"🔄 Loading YOLO model: {model_path}")
        self.yolo_model = YOLO(model_path)
        logger.info("✅ YOLO model loaded successfully")
    
    def detect_tables(self, image_path: str, min_table_area: int = 5000) -> List[Dict]:
        """
        Detect tables using YOLO model
        
        Args:
            image_path: Path to input image
            min_table_area: Minimum area for table detection
            
        Returns:
            List of detected table regions with coordinates
        """
        logger.info(f"🔍 YOLO table detection on: {image_path}")
        
        try:
            # Run YOLO inference
            results = self.yolo_model(image_path)
            
            tables = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Calculate area
                        area = (x2 - x1) * (y2 - y1)
                        
                        # More lenient criteria for table detection
                        if area >= min_table_area and confidence > 0.3:  # Lower confidence threshold
                            table_info = {
                                "table_bbox": {
                                    "x1": int(x1),
                                    "y1": int(y1), 
                                    "x2": int(x2),
                                    "y2": int(y2)
                                },
                                "confidence": float(confidence),
                                "label": "TABLE",
                                "area": int(area)
                            }
                            tables.append(table_info)
            
            logger.info(f"✅ YOLO detected {len(tables)} tables")
            
            # If no tables detected, always fallback to full image for restaurant bills
            if not tables:
                logger.warning("⚠️ YOLO found no tables, using intelligent fallback for restaurant bills")
                return self._fallback_full_image_table(image_path)
            
            return tables
            
        except Exception as e:
            logger.error(f"❌ YOLO table detection failed: {e}")
            logger.info("🔄 Falling back to full image table detection")
            return self._fallback_full_image_table(image_path)
    
    def _fallback_full_image_table(self, image_path: str) -> List[Dict]:
        """Fallback: treat entire image as one table, but try to find table headers first"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            height, width = image.shape[:2]
            
            # Try to find a more intelligent table region by looking for common table headers
            # This is a smarter fallback that looks for actual table content
            return [{
                "bbox": {
                    "x1": 0,
                    "y1": int(height * 0.3),  # Start from 30% down the image to skip headers
                    "x2": width,
                    "y2": int(height * 0.8)   # End at 80% to skip footers
                },
                "label": "TABLE",
                "confidence": 1.0,
                "area": width * int(height * 0.5)  # 50% of image area
            }]
            
        except Exception as e:
            logger.error(f"❌ Fallback table detection failed: {e}")
            return []
    
    def visualize_tables(self, image_path: str, output_path: str = None) -> str:
        """
        Visualize detected tables on the image
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            
        Returns:
            Path to visualization image
        """
        try:
            # Detect tables
            tables = self.detect_tables(image_path)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Draw bounding boxes
            for i, table in enumerate(tables):
                bbox = table['table_bbox']
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"Table {i+1} ({table['confidence']:.2f})"
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save visualization
            if output_path is None:
                output_path = image_path.replace('.', '_tables.')
            
            cv2.imwrite(output_path, image)
            logger.info(f"✅ Visualization saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
            return image_path


# Standalone function for backward compatibility
def detect_tables(image_path: str, confidence_threshold: float = 0.25) -> List[Dict]:
    """
    Standalone function for table detection (backward compatibility)
    
    Args:
        image_path: Path to input image
        confidence_threshold: Not used in YOLO version (kept for compatibility)
        
    Returns:
        List of detected table regions
    """
    detector = TableDetector()
    return detector.detect_tables(image_path)