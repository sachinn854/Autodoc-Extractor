#!/usr/bin/env python3
"""
Debug script to trace exactly what's happening in the parser
"""

import json
from pathlib import Path
from app.parser import BusinessSchemaParser

def debug_latest_job():
    """Debug the latest job to see where parsing fails"""
    
    # Find latest job
    results_dir = Path("tmp/results")
    latest_job = max(results_dir.iterdir(), key=lambda x: x.stat().st_mtime)
    job_id = latest_job.name
    
    print(f"🔍 Debugging job: {job_id}")
    
    # Load OCR data
    ocr_file = latest_job / "ocr.json"
    tables_file = latest_job / "tables.json"
    
    if not ocr_file.exists():
        print("❌ OCR file not found!")
        return
        
    if not tables_file.exists():
        print("❌ Tables file not found!")
        return
    
    with open(ocr_file, 'r') as f:
        ocr_data = json.load(f)
    
    with open(tables_file, 'r') as f:
        tables_data = json.load(f)
    
    print(f"\n📊 OCR Data:")
    print(f"  - Total tokens: {ocr_data.get('total_tokens', 0)}")
    print(f"  - Total pages: {ocr_data.get('total_pages', 0)}")
    
    print(f"\n📋 Tables Data:")
    tables_list = tables_data.get('tables', [])
    print(f"  - Total tables: {len(tables_list)}")
    if tables_list:
        for i, table in enumerate(tables_list):
            print(f"  - Table {i+1}: label='{table.get('label', 'NO_LABEL')}', rows={len(table.get('rows', []))}")
    
    # Test the parser
    parser = BusinessSchemaParser()
    
    print(f"\n🔥 TESTING PARSER LOGIC 🔥")
    
    # Step 1: Check receipt detection
    is_receipt = parser._is_receipt_document(tables_list)
    print(f"📝 Is Receipt: {is_receipt}")
    
    # Step 2: Check structured data (with detailed logging)
    has_structured_data = False
    if tables_list:
        for table in tables_list:
            rows = table.get('rows', [])
            print(f"\n🔍 Checking table with {len(rows)} rows...")
            
            if rows:
                valid_item_rows = 0
                for i, row in enumerate(rows[:10]):
                    cells = row.get('cells', {})
                    if any(cells.values()):
                        row_text = ' '.join(str(v) for v in cells.values() if v).upper()
                        print(f"  Row {i}: {row_text[:80]}")
                        
                        # Check for company/header keywords
                        skip_keywords = ['LIQUOR', 'PRIVATE', 'LIMITED', 'SECTOR', 'FARIDABAD', 
                                        'PHONE', 'GSTIN', 'INVOICE', 'ADDRESS', 'COMPANY',
                                        'PH.', 'NO.', 'DATE:', 'TIME:', 'MAY-', 'JAN-', 'FEB-', 'MAR-', 
                                        'APR-', 'JUN-', 'JUL-', 'AUG-', 'SEP-', 'OCT-', 'NOV-', 'DEC-']
                        should_skip = any(keyword in row_text for keyword in skip_keywords)
                        
                        if should_skip:
                            print(f"    ❌ Skipped (company/header info)")
                            continue
                        
                        # Check for numbers AND letters
                        has_numbers = any(any(c.isdigit() for c in str(v)) or '.' in str(v) for v in cells.values() if v)
                        has_alpha = any(any(c.isalpha() for c in str(v)) for v in cells.values() if v)
                        
                        if has_numbers and has_alpha:
                            valid_item_rows += 1
                            print(f"    ✅ Valid item row (has numbers AND letters)")
                        else:
                            print(f"    ❌ Invalid (numbers: {has_numbers}, letters: {has_alpha})")
                
                print(f"\n  Valid item rows: {valid_item_rows}")
                if valid_item_rows >= 2:
                    has_structured_data = True
                    break
    
    print(f"\n📊 Has Structured Data: {has_structured_data}")
    
    # Step 3: Test OCR parsing
    if not has_structured_data or is_receipt:
        print(f"\n🔍 TESTING OCR PARSING...")
        
        ocr_pages = ocr_data.get('pages', [])
        if ocr_pages:
            page = ocr_pages[0]
            tokens = page.get('tokens', [])
            
            print(f"📝 First page tokens: {len(tokens)}")
            
            # Sort tokens
            sorted_tokens = sorted(tokens, key=lambda t: t.get('bbox', [0,0,0,0])[1])
            
            # Test header detection
            has_header = parser._has_column_header(sorted_tokens)
            print(f"📋 Header Detected: {has_header}")
            
            if has_header:
                print(f"\n✅ SHOULD USE HEADER-BASED PARSING")
                items, flags = parser._extract_items_with_header(sorted_tokens)
            else:
                print(f"\n⚠️ SHOULD USE HEADERLESS PARSING")
                items, flags = parser._extract_items_headerless(sorted_tokens)
            
            print(f"\n📊 PARSING RESULTS:")
            print(f"  - Items found: {len(items)}")
            print(f"  - Confidence flags: {len(flags)}")
            
            for i, item in enumerate(items):
                print(f"  - Item {i+1}: {item.get('description', 'NO_DESC')} | Qty: {item.get('qty', 0)} | Price: {item.get('unit_price', 0)}")
    
    print(f"\n✅ Debug complete!")

if __name__ == "__main__":
    debug_latest_job()