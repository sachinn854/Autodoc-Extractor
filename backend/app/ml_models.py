import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpenseCategorizer:
    """
    ML-based expense categorization using TF-IDF + Logistic Regression
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize categorizer
        
        Args:
            model_path: Path to saved model directory
        """
        self.model_path = model_path or self._get_default_model_path()
        self.vectorizer = None
        self.classifier = None
        self.categories = [
            'Food', 'Groceries', 'Travel', 'Utilities', 'Stationery', 
            'Medical', 'Entertainment', 'Clothing', 'Electronics', 'Other'
        ]
        self._load_or_train_model()
    
    def _get_default_model_path(self) -> str:
        """Get default model save path"""
        script_dir = Path(__file__).parent.parent
        model_dir = script_dir / "models"
        model_dir.mkdir(exist_ok=True)
        return str(model_dir)
    
    def _load_or_train_model(self):
        """Load existing model or train new one"""
        vectorizer_path = Path(self.model_path) / "category_vectorizer.pkl"
        classifier_path = Path(self.model_path) / "category_classifier.pkl"
        
        if vectorizer_path.exists() and classifier_path.exists():
            try:
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                logger.info("Loaded existing category classification model")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Training new model.")
        
        # Train new model with sample data
        self._train_default_model()
    
    def _train_default_model(self):
        """Train model with sample expense data"""
        logger.info("Training new category classification model")
        
        # Sample training data (would be much larger in production)
        training_data = [
            # Food
            ("bread milk eggs", "Food"),
            ("restaurant dinner lunch", "Food"),
            ("pizza burger sandwich", "Food"),
            ("coffee tea juice", "Food"),
            ("rice dal vegetables", "Food"),
            
            # Groceries  
            ("supermarket grocery store", "Groceries"),
            ("fruits vegetables meat", "Groceries"),
            ("cleaning supplies detergent", "Groceries"),
            ("toilet paper shampoo", "Groceries"),
            
            # Travel
            ("fuel petrol diesel", "Travel"),
            ("bus ticket train", "Travel"),
            ("taxi uber cab", "Travel"),
            ("flight airline", "Travel"),
            ("hotel accommodation", "Travel"),
            
            # Utilities
            ("electricity bill power", "Utilities"),
            ("water bill", "Utilities"),
            ("phone mobile bill", "Utilities"),
            ("internet wifi broadband", "Utilities"),
            ("gas lpg cylinder", "Utilities"),
            
            # Stationery
            ("pen pencil paper", "Stationery"),
            ("notebook diary book", "Stationery"),
            ("marker highlighter", "Stationery"),
            ("stapler clips", "Stationery"),
            
            # Medical
            ("medicine tablet pills", "Medical"),
            ("doctor consultation", "Medical"),
            ("pharmacy medical", "Medical"),
            ("hospital treatment", "Medical"),
            
            # Entertainment
            ("movie cinema ticket", "Entertainment"),
            ("game gaming", "Entertainment"),
            ("music concert", "Entertainment"),
            
            # Electronics
            ("mobile phone laptop", "Electronics"),
            ("charger cable", "Electronics"),
            ("headphones speaker", "Electronics"),
            
            # Clothing
            ("shirt pant dress", "Clothing"),
            ("shoes sandals", "Clothing"),
            ("jacket coat", "Clothing"),
        ]
        
        # Prepare training data
        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]
        
        # Train vectorizer and classifier
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        X = self.vectorizer.fit_transform(texts)
        self.classifier = LogisticRegression(random_state=42)
        self.classifier.fit(X, labels)
        
        # Save trained model
        self._save_model()
        
        logger.info("Category classification model trained successfully")
    
    def _save_model(self):
        """Save trained model to disk"""
        try:
            vectorizer_path = Path(self.model_path) / "category_vectorizer.pkl"
            classifier_path = Path(self.model_path) / "category_classifier.pkl"
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(classifier_path, 'wb') as f:
                pickle.dump(self.classifier, f)
                
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess item description for classification"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def predict_category(self, item_text: str) -> Dict[str, any]:
        """
        Predict category for item description
        
        Args:
            item_text: Item description text
            
        Returns:
            Dictionary with category and confidence
        """
        if not item_text or not self.vectorizer or not self.classifier:
            return {"category": "Other", "confidence": 0.5}
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(item_text)
            
            if not processed_text:
                return {"category": "Other", "confidence": 0.5}
            
            # Vectorize text
            X = self.vectorizer.transform([processed_text])
            
            # Predict category
            prediction = self.classifier.predict(X)[0]
            
            # Get confidence (probability of predicted class)
            probabilities = self.classifier.predict_proba(X)[0]
            max_prob_idx = np.argmax(probabilities)
            confidence = probabilities[max_prob_idx]
            
            return {
                "category": prediction,
                "confidence": float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Category prediction failed: {e}")
            return {"category": "Other", "confidence": 0.5}

class AnomalyDetector:
    """
    Anomaly detection for expense data using Isolation Forest
    """
    
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1,  # 10% of data considered anomalous
            random_state=42
        )
        self.is_trained = False
        self.feature_stats = {}
    
    def _extract_features(self, items: List[Dict], vendor: str = "") -> np.ndarray:
        """
        Extract numerical features for anomaly detection
        
        Args:
            items: List of expense items
            vendor: Vendor name
            
        Returns:
            Feature matrix
        """
        features = []
        
        for item in items:
            item_features = [
                item.get('unit_price', 0.0),
                item.get('qty', 1.0),
                item.get('line_total', 0.0),
                len(vendor) if vendor else 0,  # Vendor name length as proxy
                len(item.get('description', '')),  # Item description length
            ]
            features.append(item_features)
        
        return np.array(features)
    
    def _calculate_feature_stats(self, features: np.ndarray):
        """Calculate basic statistics for features"""
        self.feature_stats = {
            'mean_price': np.mean(features[:, 0]) if len(features) > 0 else 0,
            'std_price': np.std(features[:, 0]) if len(features) > 0 else 1,
            'mean_qty': np.mean(features[:, 1]) if len(features) > 0 else 1,
            'std_qty': np.std(features[:, 1]) if len(features) > 0 else 1,
        }
    
    def train_on_historical_data(self, historical_documents: List[Dict]):
        """Train anomaly detector on historical data"""
        all_features = []
        
        for doc in historical_documents:
            items = doc.get('items', [])
            vendor = doc.get('vendor', '')
            
            if items:
                features = self._extract_features(items, vendor)
                all_features.extend(features)
        
        if len(all_features) > 5:  # Need minimum samples
            all_features = np.array(all_features)
            self._calculate_feature_stats(all_features)
            self.model.fit(all_features)
            self.is_trained = True
            logger.info(f"Anomaly detector trained on {len(all_features)} samples")
        else:
            logger.warning("Insufficient historical data for anomaly training")
    
    def detect_anomalies(self, items: List[Dict], vendor: str = "", date: str = "") -> List[Dict]:
        """
        Detect anomalies in expense items
        
        Args:
            items: List of expense items  
            vendor: Vendor name
            date: Transaction date
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if not items:
            return anomalies
        
        # Rule-based anomaly detection (always works)
        anomalies.extend(self._rule_based_anomalies(items, vendor))
        
        # ML-based detection (if trained)
        if self.is_trained:
            anomalies.extend(self._ml_based_anomalies(items, vendor))
        
        return anomalies
    
    def _rule_based_anomalies(self, items: List[Dict], vendor: str) -> List[Dict]:
        """Simple rule-based anomaly detection"""
        anomalies = []
        
        for item in items:
            unit_price = item.get('unit_price', 0)
            qty = item.get('qty', 1)
            description = item.get('description', '')
            
            # Price-based anomalies
            if unit_price > 10000:  # Very expensive item
                anomalies.append({
                    "type": "PRICE_OUTLIER_HIGH",
                    "item": description,
                    "value": unit_price,
                    "message": f"Unusually high price: {unit_price}"
                })
            
            if unit_price > 0 and unit_price < 0.1:  # Very cheap item
                anomalies.append({
                    "type": "PRICE_OUTLIER_LOW", 
                    "item": description,
                    "value": unit_price,
                    "message": f"Unusually low price: {unit_price}"
                })
            
            # Quantity-based anomalies
            if qty > 100:  # Very large quantity
                anomalies.append({
                    "type": "QUANTITY_OUTLIER",
                    "item": description,
                    "value": qty,
                    "message": f"Unusually high quantity: {qty}"
                })
        
        return anomalies
    
    def _ml_based_anomalies(self, items: List[Dict], vendor: str) -> List[Dict]:
        """ML-based anomaly detection using trained model"""
        anomalies = []
        
        try:
            features = self._extract_features(items, vendor)
            predictions = self.model.predict(features)
            
            for i, (prediction, item) in enumerate(zip(predictions, items)):
                if prediction == -1:  # Anomaly detected
                    anomalies.append({
                        "type": "ML_ANOMALY",
                        "item": item.get('description', 'Unknown'),
                        "value": item.get('unit_price', 0),
                        "message": "Detected as anomalous by ML model"
                    })
        
        except Exception as e:
            logger.error(f"ML anomaly detection failed: {e}")
        
        return anomalies

# Convenience functions for Phase 6
def predict_category(item_text: str) -> Dict[str, any]:
    """Predict category for item description"""
    categorizer = ExpenseCategorizer()
    return categorizer.predict_category(item_text)

def detect_anomalies(items: List[Dict], vendor: str = "", date: str = "") -> List[Dict]:
    """Detect anomalies in expense items"""
    detector = AnomalyDetector()
    return detector.detect_anomalies(items, vendor, date)

if __name__ == "__main__":
    # Test categorization
    categorizer = ExpenseCategorizer()
    
    test_items = [
        "bread and butter",
        "mobile phone", 
        "petrol fuel",
        "doctor consultation",
        "movie ticket"
    ]
    
    print("Category Predictions:")
    for item in test_items:
        result = categorizer.predict_category(item)
        print(f"  '{item}' -> {result['category']} ({result['confidence']:.2f})")
    
    # Test anomaly detection
    detector = AnomalyDetector()
    
    test_expense_items = [
        {"description": "Normal coffee", "unit_price": 5.0, "qty": 1},
        {"description": "Expensive laptop", "unit_price": 50000.0, "qty": 1},
        {"description": "Bulk rice", "unit_price": 50.0, "qty": 200}
    ]
    
    anomalies = detector.detect_anomalies(test_expense_items)
    print(f"\nDetected {len(anomalies)} anomalies:")
    for anomaly in anomalies:
        print(f"  {anomaly['type']}: {anomaly['message']}")
