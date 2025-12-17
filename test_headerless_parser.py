"""
Quick test for headerless bill parsing
"""

from backend.app.parser import BusinessSchemaParser

# Mock tokens for headerless receipt
# Format:
# ITEM NAME
# 2  5.50  11.00

mock_tokens_headerless = [
    # Item 1
    {'text': 'KF MODELLING CLAY', 'bbox': [50, 100, 200, 120], 'confidence': 0.95},
    {'text': '1', 'bbox': [300, 100, 320, 120], 'confidence': 0.98},
    {'text': '9.00', 'bbox': [350, 100, 400, 120], 'confidence': 0.97},
    {'text': '9.00', 'bbox': [450, 100, 500, 120], 'confidence': 0.97},
    
    # Item 2
    {'text': 'CRAYOLA CRAYONS PACK', 'bbox': [50, 140, 220, 160], 'confidence': 0.94},
    {'text': '2', 'bbox': [300, 140, 320, 160], 'confidence': 0.98},
    {'text': '5.50', 'bbox': [350, 140, 400, 160], 'confidence': 0.96},
    {'text': '11.00', 'bbox': [450, 140, 500, 160], 'confidence': 0.96},
    
    # Total row (should STOP here)
    {'text': 'SUBTOTAL', 'bbox': [50, 200, 150, 220], 'confidence': 0.93},
    {'text': '20.00', 'bbox': [450, 200, 500, 220], 'confidence': 0.97},
    
    # Footer (should be IGNORED after SUBTOTAL)
    {'text': 'CASH', 'bbox': [50, 240, 100, 260], 'confidence': 0.92},
    {'text': '50.00', 'bbox': [450, 240, 500, 260], 'confidence': 0.96},
    {'text': 'CHANGE', 'bbox': [50, 280, 120, 300], 'confidence': 0.91},
    {'text': '30.00', 'bbox': [450, 280, 500, 300], 'confidence': 0.95},
]

# Mock tokens with header (existing logic)
mock_tokens_with_header = [
    # Header
    {'text': 'ITEM', 'bbox': [50, 50, 100, 70], 'confidence': 0.95},
    {'text': 'QTY', 'bbox': [300, 50, 340, 70], 'confidence': 0.96},
    {'text': 'RATE', 'bbox': [350, 50, 400, 70], 'confidence': 0.96},
    {'text': 'TOTAL', 'bbox': [450, 50, 500, 70], 'confidence': 0.97},
    
    # Item 1
    {'text': 'KF MODELLING CLAY', 'bbox': [50, 100, 200, 120], 'confidence': 0.95},
    {'text': '1', 'bbox': [300, 100, 320, 120], 'confidence': 0.98},
    {'text': '9.00', 'bbox': [350, 100, 400, 120], 'confidence': 0.97},
    {'text': '9.00', 'bbox': [450, 100, 500, 120], 'confidence': 0.97},
]


def test_headerless():
    """Test headerless bill parsing"""
    print("\n" + "="*80)
    print("TEST 1: HEADERLESS RECEIPT")
    print("="*80)
    
    parser = BusinessSchemaParser()
    pages_data = [{'tokens': mock_tokens_headerless}]
    
    items, flags = parser.extract_items_from_ocr(pages_data)
    
    print(f"\nExtracted {len(items)} items:")
    for item in items:
        print(f"  - {item['description']}")
        print(f"    Qty: {item['qty']}, Price: {item['unit_price']:.2f}, Total: {item['line_total']:.2f}")
    
    print(f"\nFlags: {flags}")
    
    assert len(items) == 2, f"Expected 2 items, got {len(items)}"
    assert items[0]['description'] == 'KF MODELLING CLAY'
    assert items[0]['qty'] == 1.0
    assert items[0]['unit_price'] == 9.0
    assert items[0]['line_total'] == 9.0
    
    print("\n✅ Headerless test PASSED!")


def test_with_header():
    """Test header-based parsing (existing logic)"""
    print("\n" + "="*80)
    print("TEST 2: WITH HEADER (EXISTING LOGIC)")
    print("="*80)
    
    parser = BusinessSchemaParser()
    pages_data = [{'tokens': mock_tokens_with_header}]
    
    items, flags = parser.extract_items_from_ocr(pages_data)
    
    print(f"\nExtracted {len(items)} items:")
    for item in items:
        print(f"  - {item['description']}")
        print(f"    Qty: {item['qty']}, Price: {item['unit_price']:.2f}, Total: {item['line_total']:.2f}")
    
    print(f"\nFlags: {flags}")
    
    assert len(items) >= 1, f"Expected at least 1 item, got {len(items)}"
    
    print("\n✅ Header-based test PASSED!")


if __name__ == '__main__':
    print("\n🔥 TESTING HEADERLESS PARSER 🔥\n")
    
    try:
        test_headerless()
        test_with_header()
        print("\n" + "="*80)
        print("✅✅✅ ALL TESTS PASSED ✅✅✅")
        print("="*80 + "\n")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
