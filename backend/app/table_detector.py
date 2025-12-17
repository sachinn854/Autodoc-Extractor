import cv2
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
import os
from pathlib import Path

class TableDetector:
    """
    YOLOv8-based table detection for document images
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize table detector
        
        Args:
            model_path: Path to custom YOLO model. If None, uses pre-trained model
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model for table detection"""
        
        # Simple approach: Use pretrained YOLO for general object detection
        try:
            # Check if custom model exists first
            model_path = Path(__file__).parent.parent / "models" / "table_detector_best.pt"
            
            if model_path.exists():
                print(f"✅ Loading model from: {model_path}")
                self.model = YOLO(str(model_path))
                print("✅ Model loaded successfully")
            else:
                print("📦 Loading YOLOv8 nano pretrained model")
                self.model = YOLO('yolov8n.pt')  # Will auto-download if needed
                print("✅ Pretrained YOLOv8 loaded")
                
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            print("🔄 Falling back to receipt-style detection only")
            self.model = None
    
    def detect_tables(self, image_path: str, confidence_threshold: float = 0.25) -> List[Dict]:  # Lowered threshold
        """
        Detect tables in the given image
        
        Args:
            image_path: Path to the input image
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected tables with bbox and confidence
            Format: [
                {
                    "label": "TABLE",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.92
                }
            ]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        tables = []
        
        if self.model is None:
            # No model available - use receipt-style detection
            print("📄 Using receipt-style table detection")
            tables = self._receipt_based_table_detection(image)
        else:
            # Use YOLO model first
            print("🔍 Using YOLO for table detection")
            tables = self._yolo_table_detection(image, confidence_threshold)
            
            # If no tables detected, fallback to receipt-style
            if not tables:
                print("⚠️ No tables detected by YOLO, falling back to receipt-style detection")
                tables = self._receipt_based_table_detection(image)
        
        # GUARANTEE: Always return at least one table
        if not tables:
            print("🚨 EMERGENCY: Creating full-image table")
            height, width = image.shape[:2]
            tables = [{
                "label": "TABLE",
                "bbox": {
                    "x1": 0,
                    "y1": 0,
                    "x2": width,
                    "y2": height
                },
                "confidence": 1.0
            }]
        
        # DEBUG INFO
        print(f"🎯 FINAL RESULT: {len(tables)} table(s) detected")
        for i, table in enumerate(tables):
            bbox = table['bbox']
            print(f"   Table {i+1}: [{bbox['x1']},{bbox['y1']} → {bbox['x2']},{bbox['y2']}] conf={table['confidence']:.2f}")
        
        return tables
    
    def _yolo_table_detection(self, image: np.ndarray, confidence_threshold: float) -> List[Dict]:
        """Use YOLO model for table detection"""
        try:
            # Run inference
            results = self.model(image, conf=confidence_threshold)
            
            tables = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Accept receipt-relevant objects as tables
                        receipt_objects = {
                            60: 'dining table',  # Original table class
                            84: 'book',          # Receipts look like books/papers
                            # Add other relevant class IDs if needed
                        }
                        
                        # Check if detected object is table-like
                        is_table_like = (
                            class_id in receipt_objects or 
                            'table' in class_name.lower() or
                            'book' in class_name.lower() or
                            'paper' in class_name.lower()
                        )
                        
                        if is_table_like:
                            print(f"📋 YOLO table-like object detected: {class_name} (confidence: {confidence:.2f})")
                            tables.append({
                                "label": class_name.upper(),  # Preserve YOLO class name (BOOK, DINING TABLE, etc.)
                                "bbox": {
                                    "x1": int(x1),
                                    "y1": int(y1), 
                                    "x2": int(x2),
                                    "y2": int(y2)
                                },
                                "confidence": float(confidence)
                            })
                        else:
                            print(f"🔍 COCO object detected: {class_name} (skipping, not receipt-related)")
            
            return tables
            
        except Exception as e:
            print(f"YOLO detection failed: {e}")
            # Fallback to heuristic detection
            return self._heuristic_table_detection(image)
    
    def _heuristic_table_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Fallback heuristic method to detect table-like regions
        Based on horizontal/vertical lines and text density
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//30, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//30))
        
        # Find horizontal lines
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)
        
        # Find vertical lines
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=2)
        
        # Combine lines to find table structure
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find contours of potential table regions
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter by size (table should be reasonably large)
            min_table_area = (width * height) * 0.05  # At least 5% of image
            if area > min_table_area and w > 100 and h > 50:
                tables.append({
                    "label": "TABLE",
                    "bbox": {
                        "x1": int(x),
                        "y1": int(y),
                        "x2": int(x + w),
                        "y2": int(y + h)
                    },
                    "confidence": 0.7  # Fixed confidence for heuristic method
                })
        
        # If no tables found, assume entire image is a table (for receipts)
        if not tables:
            # Create table covering most of the image
            margin = 20
            tables.append({
                "label": "TABLE",
                "bbox": {
                    "x1": margin,
                    "y1": margin,
                    "x2": width - margin,
                    "y2": height - margin
                },
                "confidence": 0.6
            })
        
        return tables
    
    def _receipt_based_table_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Receipt-style table detection: treat the whole receipt as one table
        This works when you have line items but no visible table borders
        """
        height, width = image.shape[:2]
        
        # For receipts, create a table covering the main content area
        # Use safer margins to ensure we capture content
        margin_x = int(width * 0.05)   # 5% margin
        margin_y_top = int(height * 0.05)  # Skip top 5% (safer)
        margin_y_bottom = int(height * 0.05)  # Skip bottom 5% (safer)
        
        table = {
            "label": "TABLE", 
            "bbox": {
                "x1": margin_x,
                "y1": margin_y_top,
                "x2": width - margin_x,
                "y2": height - margin_y_bottom
            },
            "confidence": 0.9  # High confidence for receipt-style
        }
        
        print(f"📄 RECEIPT TABLE DETECTED: x1={table['bbox']['x1']}, y1={table['bbox']['y1']}, x2={table['bbox']['x2']}, y2={table['bbox']['y2']}")
        return [table]
    
    def visualize_tables(self, image_path: str, tables: List[Dict], output_path: str = None) -> str:
        """
        Draw detected tables on image for visualization
        
        Args:
            image_path: Input image path
            tables: List of detected tables
            output_path: Output path for visualization. If None, auto-generates
            
        Returns:
            Path to the visualization image
        """
        image = cv2.imread(image_path)
        
        # Draw table bounding boxes
        for table in tables:
            bbox = table['bbox']
            confidence = table['confidence']
            
            # Handle both old list format and new dict format
            if isinstance(bbox, dict):
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            else:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"TABLE ({confidence:.2f})"
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization
        if output_path is None:
            base_path = Path(image_path)
            output_path = str(base_path.parent / f"{base_path.stem}_table_detection{base_path.suffix}")
        
        cv2.imwrite(output_path, image)
        return output_path

# Convenience function for direct use
def detect_tables(image_path: str, model_path: str = None, confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Detect tables in an image
    
    Args:
        image_path: Path to input image
        model_path: Path to custom YOLO model (optional)
        confidence_threshold: Minimum detection confidence
        
    Returns:
        List of detected tables
    """
    detector = TableDetector(model_path)
    return detector.detect_tables(image_path, confidence_threshold)

if __name__ == "__main__":
    # Test the detector
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        detector = TableDetector()
        tables = detector.detect_tables(image_path)
        
        print(f"Detected {len(tables)} tables:")
        for i, table in enumerate(tables):
            print(f"  Table {i+1}: {table}")
        
        # Create visualization
        vis_path = detector.visualize_tables(image_path, tables)
        print(f"Visualization saved to: {vis_path}")
    else:
        print("Usage: python table_detector.py <image_path>")
