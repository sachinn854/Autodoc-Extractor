#!/usr/bin/env python3
"""
Create a simple, reliable parser for structured bills
"""

import json
from pathlib import Path

def create_simple_parser():
    """Create a simple parser that works for structured bills"""
    
    # Load latest OCR data
    results_dir = Path("tmp/results")
    latest_job = max(results_dir.iterdir(), key=lambda x: x.stat().st_mtime)
    
    ocr_file = latest_job / "ocr.json"
    with open(ocr_file, 'r') as f:
        ocr_data = json.load(f)
    
    tokens = ocr_data['pages'][0]['tokens']
    
    print("🔍 Creating simple parser for structured bills...")
    
    # Step 1: Find header row (Item, Qty, Rate, Total)
    header_y = None
    for token in tokens:
        text = token.get('text', '').upper()
        if text in ['ITEM', 'QTY', 'QTY.', 'RATE', 'TOTAL']:
            header_y = token.get('bbox', [0,0,0,0])[1]
            print(f"📋 Found header at Y={header_y}: {text}")
            break
    
    if not header_y:
        print("❌ No header found")
        return
    
    # Step 2: Find all tokens AFTER header (items area)
    item_tokens = []
    for token in tokens:
        bbox = token.get('bbox', [0,0,0,0])
        y = bbox[1]
        
        # Only tokens below header and before totals
        if y > header_y + 20 and y < header_y + 300:  # Reasonable item area
            text = token.get('text', '').strip()
            if text and text.upper() not in ['TOTAL', 'SUB', 'CGST', 'SGST', 'THANKS']:
                item_tokens.append({
                    'text': text,
                    'x': bbox[0],
                    'y': y,
                    'bbox': bbox
                })
    
    # Step 3: Group tokens by Y position (rows) with smart merging
    rows = {}
    tolerance = 25  # Increased tolerance for row grouping
    
    for token in item_tokens:
        y = token['y']
        
        # Find existing row with similar Y
        found_row = None
        for existing_y in rows.keys():
            if abs(y - existing_y) <= tolerance:
                found_row = existing_y
                break
        
        if found_row:
            rows[found_row].append(token)
        else:
            rows[y] = [token]
    
    # Step 3.5: Merge rows that are very close (same logical item)
    merged_rows = {}
    sorted_ys = sorted(rows.keys())
    
    i = 0
    while i < len(sorted_ys):
        current_y = sorted_ys[i]
        current_tokens = rows[current_y][:]
        
        # Check if next row is close enough to merge
        if i + 1 < len(sorted_ys):
            next_y = sorted_ys[i + 1]
            if abs(next_y - current_y) <= 50:  # Merge if within 50px
                current_tokens.extend(rows[next_y])
                i += 2  # Skip next row since we merged it
            else:
                i += 1
        else:
            i += 1
        
        merged_rows[current_y] = current_tokens
    
    # Step 4: Process each row to extract items
    items = []
    
    print(f"\n📊 Found {len(merged_rows)} merged rows:")
    for y, row_tokens in sorted(merged_rows.items()):
        texts = [t['text'] for t in sorted(row_tokens, key=lambda x: x['x'])]
        print(f"  Row Y={y}: {texts}")
    
    print(f"\n🔍 Processing rows for items:")
    
    for y, row_tokens in sorted(merged_rows.items()):
        # Sort tokens by X position (left to right)
        row_tokens.sort(key=lambda t: t['x'])
        
        # Extract text and numbers
        texts = []
        numbers = []
        
        for token in row_tokens:
            text = token['text']
            try:
                # Try to parse as number
                if '.' in text:
                    num = float(text)
                    numbers.append(num)
                elif text.isdigit():
                    num = int(text)
                    numbers.append(num)
                else:
                    texts.append(text)
            except:
                texts.append(text)
        
        # Must have description and at least 2 numbers
        if texts and len(numbers) >= 2:
            description = ' '.join(texts)
            
            # Skip if it looks like header or footer
            desc_upper = description.upper()
            if any(word in desc_upper for word in ['TOTAL', 'TAX', 'CGST', 'SGST', 'SUB']):
                continue
            
            # Assign numbers based on count and values
            if len(numbers) == 2:
                # 2 numbers: price, total
                qty = 1.0
                unit_price = numbers[0]
                line_total = numbers[1]
            elif len(numbers) >= 3:
                # 3+ numbers: qty, price, total
                qty = numbers[0]
                unit_price = numbers[1] 
                line_total = numbers[2]
                
                # Fix common OCR errors
                if qty > 10 and unit_price > 100:
                    # Likely qty misread - try qty=1
                    if abs(1 * unit_price - line_total) < abs(qty * unit_price - line_total):
                        qty = 1.0
            else:
                continue
            
            item = {
                "description": description,
                "qty": qty,
                "unit_price": unit_price,
                "line_total": line_total
            }
            
            items.append(item)
            print(f"✅ {description}: Qty={qty}, Price={unit_price}, Total={line_total}")
    
    print(f"\n📊 Extracted {len(items)} items")
    return items

if __name__ == "__main__":
    create_simple_parser()