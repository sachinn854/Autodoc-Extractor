import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import logging
from dateutil import parser as date_parser

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpendingInsights:
    """
    Generate spending insights and analytics from extracted expense data
    """
    
    def __init__(self):
        self.currency_symbols = {
            'MYR': 'RM', 'USD': '$', 'EUR': '€', 'GBP': '£', 'INR': '₹'
        }
    
    def generate_spending_insights(self, extracted_docs: List[Dict]) -> Dict:
        """
        Generate comprehensive spending insights from documents
        
        Args:
            extracted_docs: List of extracted document data
            
        Returns:
            Dictionary with spending insights
        """
        if not extracted_docs:
            return self._empty_insights()
        
        logger.info(f"Generating insights from {len(extracted_docs)} documents")
        
        # Convert to DataFrame for easier analysis
        df = self._documents_to_dataframe(extracted_docs)
        
        if df.empty:
            return self._empty_insights()
        
        insights = {
            "summary": self._generate_summary(df, extracted_docs),
            "time_analysis": self._generate_time_analysis(df),
            "category_analysis": self._generate_category_analysis(df), 
            "vendor_analysis": self._generate_vendor_analysis(df),
            "spending_patterns": self._generate_spending_patterns(df),
            "recommendations": self._generate_recommendations(df, extracted_docs)
        }
        
        logger.info("Spending insights generated successfully")
        return insights
    
    def _documents_to_dataframe(self, docs: List[Dict]) -> pd.DataFrame:
        """Convert extracted documents to pandas DataFrame"""
        rows = []
        
        for doc in docs:
            # Parse document date
            doc_date = self._parse_date(doc.get('date', ''))
            vendor = doc.get('vendor', 'Unknown')
            currency = doc.get('currency', 'MYR')
            
            # Add each item as a row
            items = doc.get('items', [])
            for item in items:
                rows.append({
                    'date': doc_date,
                    'vendor': vendor,
                    'currency': currency,
                    'description': item.get('description', ''),
                    'category': item.get('category', 'Other'),
                    'qty': item.get('qty', 1),
                    'unit_price': item.get('unit_price', 0.0),
                    'line_total': item.get('line_total', 0.0),
                    'document_total': doc.get('total', 0.0)
                })
            
            # If no items, add document-level row
            if not items:
                rows.append({
                    'date': doc_date,
                    'vendor': vendor,
                    'currency': currency,
                    'description': 'Total',
                    'category': 'Other',
                    'qty': 1,
                    'unit_price': doc.get('total', 0.0),
                    'line_total': doc.get('total', 0.0),
                    'document_total': doc.get('total', 0.0)
                })
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        
        try:
            return date_parser.parse(date_str)
        except (ValueError, TypeError):
            return None
    
    def _generate_summary(self, df: pd.DataFrame, docs: List[Dict]) -> Dict:
        """Generate overall spending summary"""
        total_documents = len(docs)
        total_items = len(df)
        
        # Calculate totals by currency
        currency_totals = df.groupby('currency')['line_total'].sum().to_dict()
        
        # Date range
        date_range = {}
        if 'date' in df.columns and not df['date'].isna().all():
            valid_dates = df['date'].dropna()
            if not valid_dates.empty:
                date_range = {
                    'earliest': valid_dates.min().strftime('%Y-%m-%d'),
                    'latest': valid_dates.max().strftime('%Y-%m-%d'),
                    'span_days': (valid_dates.max() - valid_dates.min()).days
                }
        
        return {
            'total_documents': total_documents,
            'total_items': total_items,
            'currency_totals': currency_totals,
            'date_range': date_range,
            'average_document_value': df.groupby('vendor')['document_total'].first().mean() if total_documents > 0 else 0
        }
    
    def _generate_time_analysis(self, df: pd.DataFrame) -> Dict:
        """Generate time-based spending analysis"""
        time_analysis = {
            'daily_spend': {},
            'weekly_spend': {},
            'monthly_spend': {},
            'day_of_week_pattern': {},
            'trends': {}
        }
        
        if 'date' not in df.columns or df['date'].isna().all():
            return time_analysis
        
        df_with_dates = df.dropna(subset=['date']).copy()
        
        if df_with_dates.empty:
            return time_analysis
        
        # Daily spending
        daily = df_with_dates.groupby(df_with_dates['date'].dt.date)['line_total'].sum()
        time_analysis['daily_spend'] = {str(date): float(amount) for date, amount in daily.items()}
        
        # Weekly spending
        df_with_dates['week'] = df_with_dates['date'].dt.to_period('W')
        weekly = df_with_dates.groupby('week')['line_total'].sum()
        time_analysis['weekly_spend'] = {str(week): float(amount) for week, amount in weekly.items()}
        
        # Monthly spending
        df_with_dates['month'] = df_with_dates['date'].dt.to_period('M')
        monthly = df_with_dates.groupby('month')['line_total'].sum()
        time_analysis['monthly_spend'] = {str(month): float(amount) for month, amount in monthly.items()}
        
        # Day of week pattern
        df_with_dates['day_of_week'] = df_with_dates['date'].dt.day_name()
        dow_pattern = df_with_dates.groupby('day_of_week')['line_total'].sum()
        time_analysis['day_of_week_pattern'] = {day: float(amount) for day, amount in dow_pattern.items()}
        
        # Simple trend analysis
        if len(monthly) >= 2:
            recent_months = monthly.tail(3).values
            trend_direction = "increasing" if recent_months[-1] > recent_months[0] else "decreasing"
            trend_magnitude = abs(recent_months[-1] - recent_months[0]) / recent_months[0] * 100 if recent_months[0] > 0 else 0
            
            time_analysis['trends'] = {
                'direction': trend_direction,
                'magnitude_percent': float(trend_magnitude),
                'recent_average': float(np.mean(recent_months))
            }
        
        return time_analysis
    
    def _generate_category_analysis(self, df: pd.DataFrame) -> Dict:
        """Generate category-based spending analysis"""
        category_totals = df.groupby('category')['line_total'].sum().sort_values(ascending=False)
        category_counts = df.groupby('category').size()
        category_avg = df.groupby('category')['line_total'].mean()
        
        top_categories = []
        for category in category_totals.head(10).index:
            top_categories.append({
                'category': category,
                'total_amount': float(category_totals[category]),
                'item_count': int(category_counts[category]),
                'average_amount': float(category_avg[category])
            })
        
        return {
            'top_categories': top_categories,
            'category_distribution': {cat: float(amt) for cat, amt in category_totals.items()},
            'most_frequent_category': category_counts.idxmax() if not category_counts.empty else 'N/A'
        }
    
    def _generate_vendor_analysis(self, df: pd.DataFrame) -> Dict:
        """Generate vendor-based spending analysis"""
        # Group by vendor and sum document totals (not line totals to avoid double counting)
        vendor_docs = df.groupby('vendor').agg({
            'document_total': 'first',  # Take first occurrence of document total
            'date': 'count'  # Count visits
        }).rename(columns={'date': 'visit_count'})
        
        vendor_totals = vendor_docs['document_total'].sort_values(ascending=False)
        vendor_frequency = vendor_docs['visit_count'].sort_values(ascending=False)
        
        top_vendors = []
        for vendor in vendor_totals.head(10).index:
            top_vendors.append({
                'vendor': vendor,
                'total_spent': float(vendor_totals[vendor]),
                'visit_count': int(vendor_frequency[vendor]),
                'average_per_visit': float(vendor_totals[vendor] / vendor_frequency[vendor]) if vendor_frequency[vendor] > 0 else 0
            })
        
        return {
            'top_vendors': top_vendors,
            'vendor_distribution': {vendor: float(total) for vendor, total in vendor_totals.items()},
            'most_frequent_vendor': vendor_frequency.idxmax() if not vendor_frequency.empty else 'N/A',
            'total_unique_vendors': len(vendor_totals)
        }
    
    def _generate_spending_patterns(self, df: pd.DataFrame) -> Dict:
        """Generate spending pattern insights"""
        patterns = {
            'price_ranges': {},
            'quantity_patterns': {},
            'expensive_items': [],
            'bulk_purchases': []
        }
        
        # Price range analysis
        price_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
        price_labels = ['<10', '10-50', '50-100', '100-500', '500-1000', '>1000']
        df['price_range'] = pd.cut(df['unit_price'], bins=price_bins, labels=price_labels, include_lowest=True)
        price_dist = df.groupby('price_range')['line_total'].sum()
        patterns['price_ranges'] = {str(range_): float(amount) for range_, amount in price_dist.items()}
        
        # Quantity patterns
        qty_stats = {
            'average_quantity': float(df['qty'].mean()),
            'max_quantity': float(df['qty'].max()),
            'bulk_threshold': float(df['qty'].quantile(0.9))  # 90th percentile as bulk threshold
        }
        patterns['quantity_patterns'] = qty_stats
        
        # Expensive items (top 95th percentile)
        expensive_threshold = df['unit_price'].quantile(0.95)
        expensive_items = df[df['unit_price'] >= expensive_threshold].nlargest(5, 'unit_price')
        patterns['expensive_items'] = [
            {
                'description': row['description'],
                'price': float(row['unit_price']),
                'vendor': row['vendor'],
                'date': row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'Unknown'
            }
            for _, row in expensive_items.iterrows()
        ]
        
        # Bulk purchases (high quantity)
        bulk_threshold = df['qty'].quantile(0.9)
        bulk_items = df[df['qty'] >= bulk_threshold].nlargest(5, 'qty')
        patterns['bulk_purchases'] = [
            {
                'description': row['description'],
                'quantity': int(row['qty']),
                'total_value': float(row['line_total']),
                'vendor': row['vendor']
            }
            for _, row in bulk_items.iterrows()
        ]
        
        return patterns
    
    def _generate_recommendations(self, df: pd.DataFrame, docs: List[Dict]) -> List[str]:
        """Generate spending recommendations based on patterns"""
        recommendations = []
        
        # Analyze spending patterns for recommendations
        if not df.empty:
            # High spending categories
            top_category = df.groupby('category')['line_total'].sum().idxmax()
            top_category_spend = df.groupby('category')['line_total'].sum().max()
            
            if top_category_spend > 1000:  # Arbitrary threshold
                recommendations.append(
                    f"Your highest spending category is '{top_category}' (${top_category_spend:.2f}). Consider tracking these expenses more closely."
                )
            
            # Frequent vendors
            vendor_counts = df['vendor'].value_counts()
            if len(vendor_counts) > 0 and vendor_counts.iloc[0] > 5:
                top_vendor = vendor_counts.index[0]
                recommendations.append(
                    f"You shop frequently at '{top_vendor}'. Look for loyalty programs or bulk discounts."
                )
            
            # Price variation analysis
            avg_price = df['unit_price'].mean()
            high_price_count = len(df[df['unit_price'] > avg_price * 2])
            
            if high_price_count > len(df) * 0.1:  # More than 10% are high-priced
                recommendations.append(
                    "You have several high-priced items. Consider comparing prices before purchasing."
                )
            
            # Time-based recommendations
            if 'date' in df.columns and not df['date'].isna().all():
                df_with_dates = df.dropna(subset=['date'])
                if not df_with_dates.empty:
                    monthly_spend = df_with_dates.groupby(df_with_dates['date'].dt.to_period('M'))['line_total'].sum()
                    if len(monthly_spend) >= 2:
                        recent_avg = monthly_spend.tail(2).mean()
                        if recent_avg > monthly_spend.mean() * 1.2:
                            recommendations.append(
                                "Your recent spending is above average. Consider reviewing your budget."
                            )
        
        # Add default recommendation if none generated
        if not recommendations:
            recommendations.append("Keep tracking your expenses to identify spending patterns and savings opportunities.")
        
        return recommendations
    
    def _empty_insights(self) -> Dict:
        """Return empty insights structure"""
        return {
            "summary": {
                'total_documents': 0,
                'total_items': 0,
                'currency_totals': {},
                'date_range': {},
                'average_document_value': 0
            },
            "time_analysis": {
                'daily_spend': {},
                'weekly_spend': {},
                'monthly_spend': {},
                'day_of_week_pattern': {},
                'trends': {}
            },
            "category_analysis": {
                'top_categories': [],
                'category_distribution': {},
                'most_frequent_category': 'N/A'
            },
            "vendor_analysis": {
                'top_vendors': [],
                'vendor_distribution': {},
                'most_frequent_vendor': 'N/A',
                'total_unique_vendors': 0
            },
            "spending_patterns": {
                'price_ranges': {},
                'quantity_patterns': {'average_quantity': 0, 'max_quantity': 0, 'bulk_threshold': 0},
                'expensive_items': [],
                'bulk_purchases': []
            },
            "recommendations": ["Start by uploading some expense documents to get insights."]
        }

# Convenience function
def generate_spending_insights(extracted_docs: List[Dict]) -> Dict:
    """Generate spending insights from extracted documents"""
    insights_generator = SpendingInsights()
    return insights_generator.generate_spending_insights(extracted_docs)

# Historical data management
def load_historical_data(data_dir: str = "data/history") -> List[Dict]:
    """Load historical extracted documents for insights"""
    historical_docs = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.info(f"Historical data directory not found: {data_dir}")
        return historical_docs
    
    # Load all extracted JSON files
    for json_file in data_path.glob("extracted_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
                historical_docs.append(doc_data)
        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")
    
    logger.info(f"Loaded {len(historical_docs)} historical documents")
    return historical_docs

def save_to_history(extracted_data: Dict, data_dir: str = "data/history"):
    """Save extracted document to historical data"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"extracted_{timestamp}.json"
    
    filepath = data_path / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved to historical data: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save to history: {e}")

if __name__ == "__main__":
    # Test insights generation
    sample_docs = [
        {
            "vendor": "Grocery Store A",
            "date": "2023-12-01",
            "currency": "MYR",
            "items": [
                {"description": "Bread", "category": "Food", "qty": 2, "unit_price": 3.5, "line_total": 7.0},
                {"description": "Milk", "category": "Food", "qty": 1, "unit_price": 4.5, "line_total": 4.5}
            ],
            "total": 11.5
        },
        {
            "vendor": "Electronics Store", 
            "date": "2023-12-02",
            "currency": "MYR",
            "items": [
                {"description": "Phone Charger", "category": "Electronics", "qty": 1, "unit_price": 25.0, "line_total": 25.0}
            ],
            "total": 25.0
        }
    ]
    
    insights = generate_spending_insights(sample_docs)
    print("Generated Insights:")
    print(json.dumps(insights, indent=2, default=str))
