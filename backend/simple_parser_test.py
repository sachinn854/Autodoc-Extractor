#!/usr/bin/env python3
"""
Simple test to extract items using pattern matching
"""

import json
from pathlib import Path

def simple_extract_items():
    """Extract items using simple pattern matching"""
    
    # Load latest OCR data
    results_dir = Path("tmp/results")
    latest_job = max(results_dir.iterdir(), key=lambda x: x.stat().st_mtime)
    
    ocr_file = latest_job / "ocr.json"
    with open(ocr_file, 'r') as f:
        ocr_data = json.load(f)
    
    tokens = ocr_data['pages'][0]['tokens']
    
    print("🔍 Looking for item patterns...")
    
    # Look for the pattern: [Item Name] [Qty] [Rate] [Total]
    items = []
    
    # Known items from the bill
    item_patterns = [
        ("Tandoori chicken", "1", "295.00", "309.75"),
        ("Lasooni Dal Tadka", "1", "275.00", "288.75"),  # Correct qty should be 1
        ("BIRYANI", "1", "375.00", "393.75"),
        ("Tandoori Roti all", "2", "30.00", "63.00"),
        ("Tandoori Roti", "1", "30.00", "31.50")
    ]
    
    # Find tokens for each pattern
    for item_name, expected_qty, expected_rate, expected_total in item_patterns:
        print(f"\n🔍 Looking for: {item_name}")
        
        # Find item name tokens
        name_tokens = []
        for i, token in enumerate(tokens):
            text = token.get('text', '').strip()
            if any(word.lower() in text.lower() for word in item_name.split()):
                name_tokens.append((i, text))
        
        if name_tokens:
            print(f"  Found name tokens: {[t[1] for t in name_tokens]}")
            
            # Look for nearby quantity, rate, total
            for name_idx, name_text in name_tokens:
                # Look in nearby tokens (±10 positions)
                nearby_tokens = tokens[max(0, name_idx-10):min(len(tokens), name_idx+10)]
                
                qty_found = None
                rate_found = None
                total_found = None
                
                for token in nearby_tokens:
                    text = token.get('text', '').strip()
                    
                    # Check if it matches expected values
                    if text == expected_qty:
                        qty_found = text
                    elif text == expected_rate:
                        rate_found = text
                    elif text == expected_total:
                        total_found = text
                
                if qty_found and rate_found and total_found:
                    item = {
                        "description": item_name,
                        "qty": float(qty_found),
                        "unit_price": float(rate_found),
                        "line_total": float(total_found)
                    }
                    items.append(item)
                    print(f"  ✅ Extracted: {item}")
                    break
            else:
                print(f"  ❌ Could not find complete data for {item_name}")
    
    print(f"\n📊 Final Results: {len(items)} items")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item['description']} - Qty: {item['qty']}, Price: {item['unit_price']}, Total: {item['line_total']}")

if __name__ == "__main__":
    simple_extract_items()