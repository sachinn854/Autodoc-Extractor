#!/usr/bin/env python3
"""
Debug script to test receipt parsing
"""
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_receipt_parsing():
    """Test receipt parsing with sample data"""
    
    # Sample OCR tokens from your receipt
    sample_tokens = [
        {"text": "Liquor Street", "bbox": [100, 50, 200, 70]},
        {"text": "Invoice Number:", "bbox": [50, 150, 150, 170]},
        {"text": "IN001001259", "bbox": [200, 150, 300, 170]},
        {"text": "Invoice Date:", "bbox": [50, 180, 150, 200]},
        {"text": "20-May-18 22:55", "bbox": [200, 180, 320, 200]},
        {"text": "Item", "bbox": [50, 250, 100, 270]},
        {"text": "Qty.", "bbox": [200, 250, 230, 270]},
        {"text": "Rate", "bbox": [280, 250, 320, 270]},
        {"text": "Total", "bbox": [380, 250, 420, 270]},
        {"text": "Tandoori chicken", "bbox": [50, 300, 180, 320]},
        {"text": "1", "bbox": [200, 300, 210, 320]},
        {"text": "295.00", "bbox": [280, 300, 330, 320]},
        {"text": "309.75", "bbox": [380, 300, 430, 320]},
        {"text": "Lasooni Dal Tadka", "bbox": [50, 330, 180, 350]},
        {"text": "1", "bbox": [200, 330, 210, 350]},
        {"text": "275.00", "bbox": [280, 330, 330, 350]},
        {"text": "288.75", "bbox": [380, 330, 430, 350]},
        {"text": "HYDERABADI MURG BIRYANI", "bbox": [50, 360, 200, 380]},
        {"text": "1", "bbox": [200, 360, 210, 380]},
        {"text": "375.00", "bbox": [280, 360, 330, 380]},
        {"text": "393.75", "bbox": [380, 360, 430, 380]},
        {"text": "Tandoori Roti all food less spicy", "bbox": [50, 390, 220, 410]},
        {"text": "2", "bbox": [200, 390, 210, 410]},
        {"text": "30.00", "bbox": [280, 390, 320, 410]},
        {"text": "63.00", "bbox": [380, 390, 420, 410]},
        {"text": "Tandoori Roti", "bbox": [50, 420, 150, 440]},
        {"text": "1", "bbox": [200, 420, 210, 440]},
        {"text": "30.00", "bbox": [280, 420, 320, 440]},
        {"text": "31.50", "bbox": [380, 420, 420, 440]},
        {"text": "Total Qty:", "bbox": [50, 500, 120, 520]},
        {"text": "6", "bbox": [400, 500, 410, 520]},
        {"text": "Sub Total", "bbox": [50, 530, 120, 550]},
        {"text": "1,035.00", "bbox": [350, 530, 420, 550]},
        {"text": "Total:", "bbox": [50, 600, 100, 620]},
        {"text": "1,139.00", "bbox": [350, 600, 420, 620]}
    ]
    
    # Create mock pages data
    pages_data = [{"tokens": sample_tokens}]
    
    # Test parsing
    from app.parser import BusinessSchemaParser
    parser = BusinessSchemaParser()
    
    print("🧪 Testing receipt parsing...")
    print(f"📊 Input: {len(sample_tokens)} tokens")
    
    # Test header detection
    sorted_tokens = sorted(sample_tokens, key=lambda t: t.get('bbox', [0,0,0,0])[1])
    has_header = parser._has_column_header(sorted_tokens)
    print(f"🔍 Header detected: {has_header}")
    
    # Test item extraction
    items, flags = parser.extract_items_from_ocr(pages_data)
    
    print(f"\n📋 Results:")
    print(f"   Items found: {len(items)}")
    print(f"   Flags: {flags}")
    
    if items:
        print(f"\n📦 Extracted Items:")
        for i, item in enumerate(items, 1):
            desc = item.get('description', 'N/A')
            qty = item.get('qty', 0)
            price = item.get('unit_price', 0)
            total = item.get('line_total', 0)
            print(f"   {i}. {desc}")
            print(f"      Qty: {qty} | Price: ₹{price} | Total: ₹{total}")
    else:
        print("❌ No items extracted!")
        
    return items, flags

if __name__ == "__main__":
    test_receipt_parsing()