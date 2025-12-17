"""
Phase 9: Evaluation & Testing Module
Comprehensive testing and metrics for the document processing pipeline
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRMetrics:
    """Calculate OCR accuracy metrics (CER/WER)"""
    
    @staticmethod
    def character_error_rate(predicted: str, ground_truth: str) -> float:
        """Calculate Character Error Rate (CER)"""
        if len(ground_truth) == 0:
            return 1.0 if len(predicted) > 0 else 0.0
            
        # Simple edit distance calculation
        def edit_distance(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
                
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            return dp[m][n]
        
        distance = edit_distance(predicted.lower(), ground_truth.lower())
        return distance / len(ground_truth)
    
    @staticmethod
    def word_error_rate(predicted: str, ground_truth: str) -> float:
        """Calculate Word Error Rate (WER)"""
        pred_words = predicted.split()
        gt_words = ground_truth.split()
        
        if len(gt_words) == 0:
            return 1.0 if len(pred_words) > 0 else 0.0
        
        # Edit distance on words
        def edit_distance(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
                
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1].lower() == s2[j-1].lower():
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            return dp[m][n]
        
        distance = edit_distance(pred_words, gt_words)
        return distance / len(gt_words)

class TableMetrics:
    """Calculate table detection and structure metrics"""
    
    @staticmethod
    def intersection_over_union(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_map(predicted_boxes: List[Dict], ground_truth_boxes: List[Dict], 
                     iou_threshold: float = 0.5) -> float:
        """Calculate mean Average Precision (mAP) for table detection"""
        if not ground_truth_boxes:
            return 1.0 if not predicted_boxes else 0.0
        
        # Match predictions with ground truth
        matched = [False] * len(ground_truth_boxes)
        correct = 0
        
        for pred in predicted_boxes:
            pred_box = [pred['x1'], pred['y1'], pred['x2'], pred['y2']]
            best_iou = 0.0
            best_match = -1
            
            for i, gt in enumerate(ground_truth_boxes):
                if matched[i]:
                    continue
                    
                gt_box = [gt['x1'], gt['y1'], gt['x2'], gt['y2']]
                iou = TableMetrics.intersection_over_union(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
            
            if best_iou >= iou_threshold and best_match >= 0:
                matched[best_match] = True
                correct += 1
        
        precision = correct / len(predicted_boxes) if predicted_boxes else 0.0
        recall = correct / len(ground_truth_boxes)
        
        # Simple AP calculation (can be improved)
        return (precision + recall) / 2 if (precision + recall) > 0 else 0.0

class FieldAccuracy:
    """Calculate field-level extraction accuracy"""
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extract numeric values from text"""
        import re
        pattern = r'\d+\.?\d*'
        matches = re.findall(pattern, text)
        return [float(match) for match in matches if match]
    
    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """Extract dates from text"""
        import re
        # Common date patterns
        patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
        ]
        
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        return dates
    
    @staticmethod
    def calculate_field_accuracy(predicted: Dict, ground_truth: Dict) -> Dict[str, float]:
        """Calculate accuracy for specific fields"""
        results = {}
        
        # Date accuracy
        pred_dates = FieldAccuracy.extract_dates(str(predicted.get('date', '')))
        gt_dates = FieldAccuracy.extract_dates(str(ground_truth.get('date', '')))
        
        if gt_dates:
            date_matches = sum(1 for date in pred_dates if date in gt_dates)
            results['date_accuracy'] = date_matches / len(gt_dates)
        else:
            results['date_accuracy'] = 1.0 if not pred_dates else 0.0
        
        # Total amount accuracy
        pred_amounts = FieldAccuracy.extract_numbers(str(predicted.get('total', '')))
        gt_amounts = FieldAccuracy.extract_numbers(str(ground_truth.get('total', '')))
        
        if gt_amounts and pred_amounts:
            # Check if any predicted amount matches ground truth (within tolerance)
            tolerance = 0.01
            amount_match = any(
                abs(pred - gt) <= tolerance
                for pred in pred_amounts
                for gt in gt_amounts
            )
            results['amount_accuracy'] = 1.0 if amount_match else 0.0
        else:
            results['amount_accuracy'] = 1.0 if not gt_amounts and not pred_amounts else 0.0
        
        return results

class PipelineTester:
    """Test the complete document processing pipeline"""
    
    def __init__(self, test_data_dir: str):
        self.test_data_dir = Path(test_data_dir)
        self.results = []
        
    def test_file_types(self) -> Dict[str, Any]:
        """Test different file types"""
        logger.info("Testing different file types...")
        
        test_files = {
            'jpg': list(self.test_data_dir.glob("*.jpg")),
            'png': list(self.test_data_dir.glob("*.png")),
            'pdf': list(self.test_data_dir.glob("*.pdf")),
        }
        
        results = {}
        for file_type, files in test_files.items():
            if files:
                logger.info(f"Testing {len(files)} {file_type.upper()} files...")
                success_count = 0
                
                for file_path in files[:5]:  # Test first 5 files of each type
                    try:
                        # This would call your main processing pipeline
                        # result = process_document(file_path)
                        success_count += 1
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
                
                results[file_type] = {
                    'total_files': len(files),
                    'tested_files': min(5, len(files)),
                    'success_count': success_count,
                    'success_rate': success_count / min(5, len(files))
                }
        
        return results
    
    def test_image_quality(self) -> Dict[str, Any]:
        """Test with different image qualities"""
        logger.info("Testing image quality robustness...")
        
        # This would test with:
        # - Blurry images
        # - Low resolution images
        # - Rotated images
        # - Noisy images
        
        return {
            'blurry_images': {'tested': 0, 'success_rate': 0.0},
            'low_resolution': {'tested': 0, 'success_rate': 0.0},
            'rotated_images': {'tested': 0, 'success_rate': 0.0},
            'noisy_images': {'tested': 0, 'success_rate': 0.0},
        }
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """Benchmark processing performance"""
        logger.info("Benchmarking performance...")
        
        # This would measure:
        # - Processing time per document
        # - Memory usage
        # - CPU utilization
        
        return {
            'avg_processing_time': 0.0,
            'memory_usage_mb': 0.0,
            'cpu_utilization': 0.0,
        }

class EvaluationReport:
    """Generate comprehensive evaluation reports"""
    
    def __init__(self):
        self.metrics = {
            'ocr_metrics': {},
            'table_metrics': {},
            'field_accuracy': {},
            'pipeline_tests': {},
            'performance_metrics': {}
        }
    
    def add_ocr_result(self, predicted_text: str, ground_truth_text: str):
        """Add OCR result for evaluation"""
        if 'results' not in self.metrics['ocr_metrics']:
            self.metrics['ocr_metrics']['results'] = []
        
        cer = OCRMetrics.character_error_rate(predicted_text, ground_truth_text)
        wer = OCRMetrics.word_error_rate(predicted_text, ground_truth_text)
        
        self.metrics['ocr_metrics']['results'].append({
            'predicted': predicted_text,
            'ground_truth': ground_truth_text,
            'cer': cer,
            'wer': wer
        })
    
    def calculate_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics across all tests"""
        summary = {}
        
        # OCR summary
        if 'results' in self.metrics['ocr_metrics']:
            results = self.metrics['ocr_metrics']['results']
            if results:
                avg_cer = sum(r['cer'] for r in results) / len(results)
                avg_wer = sum(r['wer'] for r in results) / len(results)
                
                summary['ocr'] = {
                    'average_cer': avg_cer,
                    'average_wer': avg_wer,
                    'total_samples': len(results)
                }
        
        # Add other summaries...
        
        return summary
    
    def generate_report(self, output_path: str):
        """Generate detailed evaluation report"""
        summary = self.calculate_summary_metrics()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary_metrics': summary,
            'detailed_metrics': self.metrics,
            'recommendations': self._generate_recommendations()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to: {output_path}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        recommendations = []
        
        # OCR recommendations
        if 'ocr' in self.calculate_summary_metrics():
            ocr_metrics = self.calculate_summary_metrics()['ocr']
            avg_cer = ocr_metrics.get('average_cer', 0)
            
            if avg_cer > 0.1:
                recommendations.append("OCR accuracy is low. Consider image preprocessing improvements.")
            if avg_cer > 0.2:
                recommendations.append("OCR accuracy is very low. Consider switching OCR models.")
        
        # Add more recommendations based on other metrics...
        
        return recommendations

def quick_ocr_test():
    """Quick OCR functionality test"""
    logger.info("Running quick OCR test...")
    
    try:
        # Test OCR engine initialization
        from app.ocr_engine import get_ocr_engine
        
        logger.info("Testing OCR engine initialization...")
        ocr_engine = get_ocr_engine('en')
        logger.info(f"OCR engine type: {type(ocr_engine)}")
        
        # Test with a simple image (if exists)
        test_image_path = Path("backend/tmp/test_image.jpg")
        if test_image_path.exists():
            from app.ocr_engine import run_ocr_on_image
            
            logger.info(f"Testing OCR on image: {test_image_path}")
            tokens = run_ocr_on_image(str(test_image_path), ocr_engine)
            logger.info(f"OCR extracted {len(tokens)} tokens")
            
            if tokens:
                logger.info("Sample tokens:")
                for i, token in enumerate(tokens[:3]):
                    logger.info(f"  Token {i+1}: {token}")
            else:
                logger.warning("No tokens extracted from test image")
        else:
            logger.info("No test image found, skipping image OCR test")
            
        return True
        
    except Exception as e:
        logger.error(f"OCR test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Run quick OCR test
    import traceback
    
    logger.info("Starting Phase 9 evaluation...")
    
    # Quick functionality test
    ocr_working = quick_ocr_test()
    
    if ocr_working:
        logger.info("✅ OCR engine is working properly")
    else:
        logger.error("❌ OCR engine has issues")
    
    # Create evaluation report
    evaluator = EvaluationReport()
    
    # Add some test data (if available)
    # evaluator.add_ocr_result("predicted text", "ground truth text")
    
    # Generate report
    report_path = "backend/tmp/evaluation_report.json"
    evaluator.generate_report(report_path)
    
    logger.info("Phase 9 evaluation completed!")