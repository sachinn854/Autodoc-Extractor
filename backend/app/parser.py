import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import re
from sklearn.cluster import KMeans
import logging
import csv
import os
from pathlib import Path
from datetime import datetime
import dateutil.parser as date_parser
from decimal import Decimal, InvalidOperation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableParser:
    """
    Parse OCR tokens into structured table format
    Converts raw OCR output + table bboxes → rows/columns/cells
    """
    
    def __init__(self):
        """
        Initialize table parser with default column mappings
        """
        # Default column patterns for receipt/invoice parsing
        self.column_patterns = {
            'ITEM': ['item', 'description', 'desc', 'product', 'name'],
            'QTY': ['qty', 'quantity', 'qu', 'q'],
            'UNIT_PRICE': ['price', 'unit_price', 'rate', 'amount', 'unit'],
            'LINE_TOTAL': ['total', 'line_total', 'sum', 'subtotal']
        }
        
        # Y-threshold for grouping tokens into rows (pixels)
        self.y_threshold = 10
        
        # Minimum tokens required for a valid row
        self.min_tokens_per_row = 1
    
    def parse_table_structure(self, tokens: List[Dict], table_bbox) -> Dict:
        """
        Main function to parse OCR tokens into structured table format
        
        Args:
            tokens: OCR tokens with text, bbox, confidence
            table_bbox: dict with x1,y1,x2,y2 keys OR [x1, y1, x2, y2] list
            
        Returns:
            Structured table with rows and cells
        """
        logger.info(f"Parsing table structure for {len(tokens)} tokens")
        
        # Step 1: Filter tokens inside table bbox
        table_tokens = self.filter_tokens_inside_table(tokens, table_bbox)
        logger.info(f"Found {len(table_tokens)} tokens inside table")
        
        if not table_tokens:
            return self._empty_table_structure(table_bbox)
        
        # Step 2: Group tokens into rows
        rows = self.group_tokens_into_rows(table_tokens)
        logger.info(f"Grouped into {len(rows)} rows")
        
        # Step 3: Detect column structure
        column_zones = self.detect_column_zones(table_tokens)
        logger.info(f"Detected {len(column_zones)} columns")
        
        # Step 4: Assign tokens to columns for each row
        structured_rows = []
        for row_idx, row_tokens in enumerate(rows):
            row_cells = self.assign_tokens_to_columns(row_tokens, column_zones)
            structured_rows.append({
                "row_index": row_idx,
                "cells": row_cells
            })
        
        # Step 5: Merge multi-line items
        structured_rows = self.merge_multiline_items(structured_rows)
        
        # Step 6: Build final structure
        result = {
            "table_bbox": table_bbox,
            "rows": structured_rows
        }
        
        logger.info(f"Final table structure has {len(structured_rows)} rows")
        return result
    
    def filter_tokens_inside_table(self, tokens: List[Dict], table_bbox: List[int]) -> List[Dict]:
        """
        Filter OCR tokens that are inside the table bounding box
        STOPS at TOTAL/SUBTOTAL row to exclude footer content
        
        Args:
            tokens: List of OCR tokens
            table_bbox: dict with x1,y1,x2,y2 OR [x1, y1, x2, y2] list
            
        Returns:
            Filtered tokens inside table (excluding content after TOTAL)
        """
        # Handle both dict and list formats
        if isinstance(table_bbox, dict):
            x1, y1, x2, y2 = table_bbox['x1'], table_bbox['y1'], table_bbox['x2'], table_bbox['y2']
        else:
            x1, y1, x2, y2 = table_bbox
        
        # First, collect all tokens inside bbox
        tokens_inside = []
        for token in tokens:
            bbox = token.get('bbox', [])
            if len(bbox) >= 4:
                token_x1, token_y1, token_x2, token_y2 = bbox[:4]
                center_x = (token_x1 + token_x2) / 2
                center_y = (token_y1 + token_y2) / 2
                
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    tokens_inside.append(token)
        
        # Sort by Y position (top to bottom)
        tokens_inside.sort(key=lambda t: t.get('bbox', [0,0,0,0])[1])
        
        # Now filter, stopping at TOTAL keywords
        filtered_tokens = []
        
        for token in tokens_inside:
            text = token.get('text', '').strip().upper()
            
            # 🛑 STOP at TOTAL/SUBTOTAL keywords - don't include this or anything after
            total_keywords = [
                'TOTAL AMOUNT', 'SUBTOTAL', 'SUB TOTAL', 'GRAND TOTAL',
                'NET TOTAL', 'FINAL AMOUNT', 'AMOUNT PAYABLE', 'BILL TOTAL'
            ]
            
            # Check if this token contains a TOTAL keyword
            is_total_keyword = any(keyword in text for keyword in total_keywords)
            
            # Also check for standalone "TOTAL" with specific patterns
            if not is_total_keyword:
                if text in ['TOTAL', 'TOTAL:', 'AMOUNT', 'AMOUNT:']:
                    is_total_keyword = True
            
            if is_total_keyword:
                logger.info(f"🛑 Table parsing stopped at TOTAL keyword: '{token.get('text', '')}'")
                print(f"🛑 TABLE BUILDING STOPPED AT: '{token.get('text', '')}'")
                break  # Stop processing tokens - exclude TOTAL and everything after
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def group_tokens_into_rows(self, tokens: List[Dict]) -> List[List[Dict]]:
        """
        Group tokens into rows based on vertical proximity (y-coordinates)
        
        Args:
            tokens: List of OCR tokens inside table
            
        Returns:
            List of rows, each row is a list of tokens
        """
        if not tokens:
            return []
        
        # Calculate y-center for each token
        tokens_with_y = []
        for token in tokens:
            bbox = token.get('bbox', [])
            if len(bbox) >= 4:
                y_center = (bbox[1] + bbox[3]) / 2
                tokens_with_y.append((token, y_center))
        
        if not tokens_with_y:
            return []
        
        # Sort by y-coordinate
        tokens_with_y.sort(key=lambda x: x[1])
        
        # Group into rows using y-threshold
        rows = []
        current_row = [tokens_with_y[0][0]]
        current_y = tokens_with_y[0][1]
        
        for token, y_center in tokens_with_y[1:]:
            if abs(y_center - current_y) <= self.y_threshold:
                # Same row
                current_row.append(token)
            else:
                # New row
                if len(current_row) >= self.min_tokens_per_row:
                    rows.append(current_row)
                current_row = [token]
                current_y = y_center
        
        # Add last row
        if len(current_row) >= self.min_tokens_per_row:
            rows.append(current_row)
        
        # Sort tokens within each row by x-coordinate
        for row in rows:
            row.sort(key=lambda token: token.get('bbox', [0])[0])
        
        return rows
    
    def detect_column_zones(self, tokens: List[Dict]) -> List[Dict]:
        """
        Detect column zones based on x-coordinates of tokens
        
        Args:
            tokens: All tokens in the table
            
        Returns:
            List of column zones with x-ranges and labels
        """
        if not tokens:
            return []
        
        # Collect x-centers of all tokens
        x_centers = []
        for token in tokens:
            bbox = token.get('bbox', [])
            if len(bbox) >= 4:
                x_center = (bbox[0] + bbox[2]) / 2
                x_centers.append(x_center)
        
        if not x_centers:
            return []
        
        # Use K-means to cluster x-coordinates into columns
        n_columns = min(4, len(set(x_centers)))  # Max 4 columns for receipts
        if n_columns == 1:
            # Single column case
            min_x = min(x_centers)
            max_x = max(x_centers)
            return [{
                'label': 'ITEM',
                'x_range': [min_x - 50, max_x + 50],
                'x_center': (min_x + max_x) / 2
            }]
        
        # Cluster into multiple columns
        kmeans = KMeans(n_clusters=n_columns, random_state=42)
        x_array = np.array(x_centers).reshape(-1, 1)
        kmeans.fit(x_array)
        
        # Get cluster centers and sort them
        centers = sorted(kmeans.cluster_centers_.flatten())
        
        # Create column zones
        column_zones = []
        column_labels = ['ITEM', 'QTY', 'UNIT_PRICE', 'LINE_TOTAL']
        
        for i, center in enumerate(centers):
            # Calculate x-range for this column
            if i == 0:
                x_min = min(x_centers) - 50
            else:
                x_min = (centers[i-1] + center) / 2
            
            if i == len(centers) - 1:
                x_max = max(x_centers) + 50
            else:
                x_max = (center + centers[i+1]) / 2
            
            column_zones.append({
                'label': column_labels[min(i, len(column_labels)-1)],
                'x_range': [x_min, x_max],
                'x_center': center
            })
        
        return column_zones
    
    def assign_tokens_to_columns(self, row_tokens: List[Dict], column_zones: List[Dict]) -> Dict[str, str]:
        """
        Assign tokens in a row to appropriate columns
        
        Args:
            row_tokens: Tokens in a single row
            column_zones: Detected column zones
            
        Returns:
            Dictionary mapping column labels to text
        """
        cells = {zone['label']: '' for zone in column_zones}
        
        for token in row_tokens:
            bbox = token.get('bbox', [])
            text = token.get('text', '').strip()
            
            if len(bbox) >= 4 and text:
                token_x_center = (bbox[0] + bbox[2]) / 2
                
                # Find which column this token belongs to
                best_column = None
                min_distance = float('inf')
                
                for zone in column_zones:
                    x_min, x_max = zone['x_range']
                    if x_min <= token_x_center <= x_max:
                        # Token is within column range
                        distance = abs(token_x_center - zone['x_center'])
                        if distance < min_distance:
                            min_distance = distance
                            best_column = zone['label']
                
                # Assign token to best column
                if best_column:
                    if cells[best_column]:
                        cells[best_column] += ' ' + text
                    else:
                        cells[best_column] = text
        
        # Clean up cells
        for column, text in cells.items():
            cells[column] = ' '.join(text.split())  # Remove extra whitespace
        
        return cells
    
    def merge_multiline_items(self, rows: List[Dict]) -> List[Dict]:
        """
        Merge multi-line item descriptions with their corresponding data rows
        
        Args:
            rows: List of structured rows
            
        Returns:
            Rows with merged multi-line items
        """
        if len(rows) <= 1:
            return rows
        
        merged_rows = []
        i = 0
        
        while i < len(rows):
            current_row = rows[i].copy()
            
            # Check if next row might be a continuation
            if i + 1 < len(rows):
                next_row = rows[i + 1]
                
                # Check if next row has only ITEM text and empty QTY/PRICE
                next_cells = next_row.get('cells', {})
                item_text = next_cells.get('ITEM', '').strip()
                qty_text = next_cells.get('QTY', '').strip()
                price_text = next_cells.get('UNIT_PRICE', '').strip()
                total_text = next_cells.get('LINE_TOTAL', '').strip()
                
                # If next row has item text but no qty/price, it's likely a continuation
                if (item_text and 
                    not qty_text and 
                    not price_text and 
                    not total_text and
                    not self._is_numeric_like(item_text)):
                    
                    # Merge item text into current row
                    current_item = current_row.get('cells', {}).get('ITEM', '')
                    if current_item:
                        current_row['cells']['ITEM'] = current_item + ' ' + item_text
                    else:
                        current_row['cells']['ITEM'] = item_text
                    
                    i += 2  # Skip the merged row
                else:
                    i += 1
            else:
                i += 1
            
            merged_rows.append(current_row)
        
        return merged_rows
    
    def _is_numeric_like(self, text: str) -> bool:
        """
        Check if text looks like a number (price, quantity, etc.)
        """
        # Remove common currency symbols and spaces
        cleaned = re.sub(r'[₹$€£,\s]', '', text)
        
        # Check if it's a number (int or float)
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    
    def _empty_table_structure(self, table_bbox) -> Dict:
        """
        Return empty table structure
        """
        return {
            "table_bbox": table_bbox,
            "rows": []
        }

# Convenience functions
def parse_tokens_to_table(tokens: List[Dict], table_bbox: List[int]) -> Dict:
    """
    Parse OCR tokens into structured table format
    
    Args:
        tokens: OCR tokens from Phase 3
        table_bbox: Table bounding box from table detector
        
    Returns:
        Structured table data
    """
    parser = TableParser()
    return parser.parse_table_structure(tokens, table_bbox)

def process_multiple_tables(tokens: List[Dict], tables: List[Dict]) -> List[Dict]:
    """
    Process multiple tables in the same document
    
    Args:
        tokens: All OCR tokens from the document
        tables: List of detected table bboxes
        
    Returns:
        List of structured tables
    """
    parser = TableParser()
    structured_tables = []
    
    for table in tables:
        table_bbox = table.get('bbox')
        if table_bbox:
            # Debug info
            if isinstance(table_bbox, dict):
                print(f"📊 Processing table: [{table_bbox['x1']},{table_bbox['y1']} → {table_bbox['x2']},{table_bbox['y2']}]")
            structured_table = parser.parse_table_structure(tokens, table_bbox)
            structured_tables.append(structured_table)
    
    return structured_tables

class BusinessSchemaParser:
    """
    Phase 5: Convert table structure to clean business schema
    Takes Phase 4 output and creates standardized receipt/invoice data
    """
    
    def __init__(self):
        """
        Initialize business schema parser with validation rules
        """
        # Common OCR text corrections
        self.ocr_corrections = {
            # Numbers
            'O': '0', 'o': '0', 'D': '0',
            'I': '1', 'l': '1', '|': '1',
            'S': '5', 's': '5',
            'Z': '2', 'z': '2',
            'G': '6', 'g': '6',
            'T': '7', 't': '7',
            'B': '8', 'b': '8',
            'g': '9', 'q': '9',
        }
        
        # Currency mappings (comprehensive)
        self.currency_symbols = {
            'RM': 'MYR', 'MYR': 'MYR',
            'Rs': 'INR', 'Rs.': 'INR', '₹': 'INR', 'INR': 'INR', 'RUPEES': 'INR',
            '$': 'USD', 'USD': 'USD', 'US$': 'USD',
            '€': 'EUR', 'EUR': 'EUR',
            '£': 'GBP', 'GBP': 'GBP',
            '¥': 'JPY', 'JPY': 'JPY',
            'SGD': 'SGD', 'S$': 'SGD',
            'AED': 'AED'
        }
        
        # Common vendor name patterns (for fuzzy matching)
        self.vendor_indicators = [
            'SDN BHD', 'PTE LTD', 'LTD', 'INC', 'CORP', 'CO',
            'STORE', 'SHOP', 'MART', 'SUPERMARKET', 'PHARMACY',
            'PRIVATE LIMITED', 'PVT LTD', 'LLP', 'LLC'
        ]
        
        # Total/tax keywords (comprehensive)
        self.total_keywords = [
            'total', 'grand total', 'net total', 'amount', 'balance',
            'amount payable', 'payable', 'balance due', 'final amount',
            'net amount', 'total amount', 'gross total'
        ]
        
        self.tax_keywords = [
            'tax', 'gst', 'vat', 'service tax', 'sst', 'sales tax',
            'cgst', 'sgst', 'igst', 'vat tax', 'service charge',
            'govt tax', 'state tax'
        ]
        
        self.subtotal_keywords = [
            'subtotal', 'sub total', 'sub-total', 'net', 'net amount',
            'before tax', 'pre-tax'
        ]
        
        self.discount_keywords = [
            'discount', 'disc', 'off', 'less', 'deduction', 'rebate',
            'promo', 'voucher', 'coupon'
        ]
        
        # Confidence threshold for suspicious OCR
        self.confidence_threshold = 0.5
    
    def normalize_bbox(self, bbox) -> List[int]:
        """
        Normalize bbox from PaddleOCR polygon format to rectangle [x1,y1,x2,y2]
        Handles both [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] and [x1,y1,x2,y2] formats
        """
        if isinstance(bbox, list):
            # Already rectangle format
            if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                return bbox
            # Polygon format [[x1,y1],[x2,y2],...]
            elif len(bbox) > 0 and isinstance(bbox[0], (list, tuple)):
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        return bbox
        
        # Currency mappings
        self.currency_symbols = {
            'RM': 'MYR', 'MYR': 'MYR',
            '$': 'USD', 'USD': 'USD',
            '€': 'EUR', 'EUR': 'EUR',
            '£': 'GBP', 'GBP': 'GBP',
            '₹': 'INR', 'INR': 'INR',
            '¥': 'JPY', 'JPY': 'JPY',
        }
        
        # Common vendor name patterns (for fuzzy matching)
        self.vendor_indicators = [
            'SDN BHD', 'PTE LTD', 'LTD', 'INC', 'CORP', 'CO',
            'STORE', 'SHOP', 'MART', 'SUPERMARKET', 'PHARMACY'
        ]
        
        # Total/tax keywords
        self.total_keywords = ['total', 'grand total', 'net total', 'amount', 'balance']
        self.tax_keywords = ['tax', 'gst', 'vat', 'service tax', 'sst']
        
        # Confidence threshold for suspicious OCR
        self.confidence_threshold = 0.5
    
    def process_document_to_schema(self, job_id: str, tables_data: dict, ocr_data: dict) -> dict:
        """
        Main function to convert Phase 4 tables to business schema
        
        Args:
            job_id: Job identifier
            tables_data: Output from Phase 4 table detection
            ocr_data: Output from Phase 3 OCR
            
        Returns:
            Business schema dictionary
        """
        print("\n" + "="*80)
        print("🔥 ENTERED ITEM PARSING LOGIC (process_document_to_schema) 🔥")
        print("="*80 + "\n")
        logger.info(f"Processing document to business schema for job {job_id}")
        
        # Step 1: Normalize OCR bbox format (PaddleOCR polygon -> rectangle)
        ocr_pages = ocr_data.get('pages', [])
        for page in ocr_pages:
            if 'tokens' in page:
                for token in page['tokens']:
                    if 'bbox' in token:
                        token['bbox'] = self.normalize_bbox(token['bbox'])
        
        # Step 2: Extract header information from OCR tokens
        header_info = self.extract_header_fields(ocr_pages)
        
        # Step 3: Check if this is a RECEIPT (not a table-based invoice)
        tables_list = tables_data.get('tables', [])
        print(f"\n📋 Tables found: {len(tables_list)}")
        if tables_list:
            print(f"   Table labels: {[t.get('label', 'NO_LABEL') for t in tables_list]}")
        
        is_receipt = self._is_receipt_document(tables_list)
        print(f"\n🧾 Is Receipt? {is_receipt}")
        
        if is_receipt:
            print("🧾 RECEIPT DETECTED: Bypassing table parsing, using OCR-based extraction\n")
            logger.info("🧾 RECEIPT DETECTED: Bypassing table parsing, using OCR-based extraction")
        
        # Step 4: Process items - PREFER STRUCTURED TABLES if available
        items = []
        confidence_flags = []
        
        # Check if tables have actual structured data (rows with cells)
        has_structured_data = False
        if tables_list:
            for table in tables_list:
                rows = table.get('rows', [])
                if rows:
                    # Check if rows have proper cell data
                    for row in rows[:5]:  # Check first 5 rows
                        cells = row.get('cells', {})
                        if any(cells.values()):  # Has any non-empty cell
                            has_structured_data = True
                            break
                if has_structured_data:
                    break
        
        print(f"📊 Has structured table data: {has_structured_data}")
        
        if has_structured_data:
            # PREFER structured tables - data is already parsed!
            logger.info("📊 Using STRUCTURED TABLE DATA (tables.json has rows)")
            for table in tables_list:
                table_items, table_flags = self.parse_table_to_items(table)
                items.extend(table_items)
                confidence_flags.extend(table_flags)
        elif is_receipt or not tables_list:
            # Receipt format OR no tables - extract directly from OCR
            logger.info("📝 Using receipt-style OCR parsing (no table structure)")
            items, confidence_flags = self.extract_items_from_ocr(ocr_pages)
        else:
            # Tables exist but no structured data - fallback to OCR
            logger.warning("⚠️ Tables exist but have no structured data - falling back to OCR extraction")
            items, confidence_flags = self.extract_items_from_ocr(ocr_pages)
        
        # Step 3: Extract totals and amounts
        amounts = self.extract_amounts(ocr_pages, tables_list)
        
        # Step 4: Validate and correct calculations
        validation_flags = self.validate_totals(items, amounts)
        confidence_flags.extend(validation_flags)
        
        # Step 5: Build final schema
        business_schema = self.build_final_output(
            header_info, items, amounts, confidence_flags
        )
        
        # Step 6: Save outputs
        json_path, csv_path = self.save_business_schema(job_id, business_schema)
        logger.info(f"Business schema saved to {json_path} and {csv_path}")
        
        return business_schema
    
    def normalize_text(self, text: str, confidence: float = 1.0) -> str:
        """
        Fix common OCR mistakes and normalize text
        
        Args:
            text: Raw OCR text
            confidence: OCR confidence score
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Basic cleanup
        normalized = text.strip()
        
        # Split into words and process carefully
        words = normalized.split()
        corrected_words = []
        
        for word in words:
            # Only apply aggressive OCR corrections to numeric-looking content
            if self._is_numeric_context(word):
                # Apply OCR corrections only for numeric contexts
                corrected = word
                
                # O -> 0 in numbers
                corrected = re.sub(r'O(?=\d)', '0', corrected)  # O before digit
                corrected = re.sub(r'(?<=\d)O(?=\d|\.|\b)', '0', corrected)  # O after digit
                corrected = re.sub(r'^O(?=\.)', '0', corrected)  # O at start before decimal
                
                # I/l -> 1 in numbers
                corrected = re.sub(r'[Il](?=\d)', '1', corrected)  # I/l before digit
                corrected = re.sub(r'(?<=\d)[Il](?=\d|\b)', '1', corrected)  # I/l after digit
                
                # Common decimal fixes
                corrected = re.sub(r'(\d),(\d{2})\b', r'\1.\2', corrected)  # xx,xx -> xx.xx
                corrected = re.sub(r'(\d)O(\d)', r'\g<1>0\2', corrected)  # xOx -> x0x
                
                corrected_words.append(corrected)
            else:
                # For text content, minimal corrections only
                corrected_words.append(word)
        
        # Join back
        normalized = ' '.join(corrected_words)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def normalize_bbox(self, bbox) -> List[int]:
        """
        Normalize bbox from PaddleOCR polygon format to rectangle [x1,y1,x2,y2]
        Handles both [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] and [x1,y1,x2,y2] formats
        """
        if isinstance(bbox, list):
            # Already rectangle format
            if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                return bbox
            # Polygon format [[x1,y1],[x2,y2],...]
            elif len(bbox) > 0 and isinstance(bbox[0], (list, tuple)):
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        return bbox
    
    def _is_receipt_document(self, tables_list: List[dict]) -> bool:
        """
        Detect if document is a receipt (not a structured table invoice)
        Receipts are text streams, not grids - should skip table parsing
        """
        if not tables_list:
            return True  # No tables = likely receipt
        
        # Check table labels for receipt indicators
        receipt_indicators = ['RECEIPT', 'BOOK', 'GOODS', 'FALLBACK', 'PAPER', 'DINING']
        for table in tables_list:
            label = str(table.get('label', 'NO_LABEL')).upper()
            
            # NO_LABEL or missing label = treat as receipt
            if label in ['NO_LABEL', 'NONE', '']:
                logger.info(f"🧾 No label found - treating as receipt")
                return True
            
            # Check for receipt indicators in label
            if any(indicator in label for indicator in receipt_indicators):
                logger.info(f"🧾 Receipt indicator found in label: '{label}'")
                return True
            
            # Check if table has rows - if not, it's likely a receipt bbox fallback
            if not table.get('rows'):
                logger.info("🧾 Table has no rows - treating as receipt")
                return True
        
        return False
    
    def extract_header_fields(self, pages_data: dict) -> dict:
        """
        Extract vendor, date, currency from OCR tokens
        
        Args:
            pages_data: OCR pages data
            
        Returns:
            Dictionary with header fields
        """
        header = {
            'vendor': '',
            'date': None,
            'currency': 'MYR'  # Default currency
        }
        
        # Collect all tokens from all pages - handle both dict and list formats
        all_tokens = []
        
        if isinstance(pages_data, dict):
            # Handle dictionary format: {'page_1': {...}, 'page_2': {...}}
            for page_key, page_data in pages_data.items():
                tokens = page_data.get('tokens', [])
                all_tokens.extend(tokens)
        elif isinstance(pages_data, list):
            # Handle list format: [{'page_index': 0, 'tokens': [...]}, ...]
            for page_data in pages_data:
                if isinstance(page_data, dict):
                    tokens = page_data.get('tokens', [])
                    all_tokens.extend(tokens)
        else:
            logger.warning(f"Unexpected pages_data format: {type(pages_data)}, treating as empty")
            return header
        
        if not all_tokens:
            return header
        
        # Sort tokens by position (top-left first)
        sorted_tokens = sorted(all_tokens, key=lambda t: (t.get('bbox', [0, 0])[1], t.get('bbox', [0, 0])[0]))
        
        # Extract vendor (top-most, longest text)
        vendor = self._extract_vendor(sorted_tokens[:10])  # Check top 10 tokens
        if vendor:
            header['vendor'] = vendor
        
        # Extract date
        date = self._extract_date(all_tokens)
        if date:
            header['date'] = date
        
        # Extract currency
        currency = self._extract_currency(all_tokens)
        if currency:
            header['currency'] = currency
        
        return header
    
    def parse_table_to_items(self, table: dict) -> Tuple[List[dict], List[str]]:
        """
        Convert table rows to standardized item format
        
        Args:
            table: Table structure from Phase 4
            
        Returns:
            Tuple of (items list, confidence flags)
        """
        items = []
        flags = []
        
        rows = table.get('rows', [])
        
        for row in rows:
            cells = row.get('cells', {})
            
            # Skip header rows (all caps, common keywords)
            if self._is_header_row(cells):
                continue
            
            # Parse item from row
            item = self._parse_item_row(cells)
            
            if item and item.get('description'):
                items.append(item)
                
                # Check for suspicious data
                if item.get('qty', 0) <= 0:
                    flags.append(f"SUSPICIOUS_QTY: {item.get('description', 'Unknown')}")
                
                if item.get('unit_price', 0) <= 0 and item.get('line_total', 0) > 0:
                    flags.append(f"MISSING_UNIT_PRICE: {item.get('description', 'Unknown')}")
        
        return items, flags
    
    def extract_items_from_ocr(self, pages_data: List[dict]) -> Tuple[List[dict], List[str]]:
        """
        Extract items from raw OCR data when no tables are detected (receipt format)
        Routes to header-based OR headerless parsing based on detection.
        
        Args:
            pages_data: OCR pages data (array format)
            
        Returns:
            Tuple of (items list, confidence flags)
        """
        print("\n" + "="*80)
        print("🔥 ENTERED extract_items_from_ocr 🔥")
        print("="*80 + "\n")
        
        items = []
        flags = []
        
        # Collect all tokens from all pages
        all_tokens = []
        for page_data in pages_data:
            if isinstance(page_data, dict):
                tokens = page_data.get('tokens', [])
                all_tokens.extend(tokens)
        
        if not all_tokens:
            return items, flags
        
        print("\n=== RAW OCR TOKENS (First 25) ===")
        for i, t in enumerate(all_tokens[:25]):
            print(f"{i}: {t.get('text', '')}")
        print("="*40 + "\n")
        
        logger.info(f"Attempting receipt-style parsing from {len(all_tokens)} OCR tokens")
        
        # Sort tokens by Y position (top to bottom)
        sorted_tokens = sorted(all_tokens, key=lambda t: t.get('bbox', [0,0,0,0])[1])
        
        # 🔍 DETECT: Does this receipt have column headers?
        has_header = self._has_column_header(sorted_tokens)
        
        if has_header:
            print("\n✅ HEADER DETECTED → Using header-based parsing\n")
            logger.info("Header detected: using column-based parsing")
            return self._extract_items_with_header(sorted_tokens)
        else:
            print("\n⚠️ NO HEADER → Using headerless parsing\n")
            logger.info("No header detected: using proximity-based parsing")
            return self._extract_items_headerless(sorted_tokens)
        
        # 🔍 DETECT: Does this receipt have column headers?
        has_header = self._has_column_header(sorted_tokens)
        
        if has_header:
            print("\n✅ HEADER DETECTED → Using header-based parsing\n")
            logger.info("Header detected: using column-based parsing")
            return self._extract_items_with_header(sorted_tokens)
        else:
            print("\n⚠️ NO HEADER → Using headerless parsing\n")
            logger.info("No header detected: using proximity-based parsing")
            return self._extract_items_headerless(sorted_tokens)
    
    def _has_column_header(self, sorted_tokens: List[dict]) -> bool:
        """
        Detect if receipt has column header row (QTY, RATE, TOTAL keywords)
        
        Args:
            sorted_tokens: Tokens sorted by Y position
            
        Returns:
            True if header found, False otherwise
        """
        # Search for header keywords in first 50 tokens
        header_keywords = ['QTY', 'QUANTITY', 'RATE', 'PRICE', 'UNIT PRICE', 'TOTAL', 'AMOUNT']
        
        for i, token in enumerate(sorted_tokens[:50]):
            text = token.get('text', '').strip().upper()
            text_clean = text.replace('.', '').replace(':', '').strip()
            
            # Check if this is a header keyword
            if any(keyword == text_clean for keyword in header_keywords):
                # Verify nearby tokens have more header keywords
                nearby_tokens = sorted_tokens[max(0, i-2):min(len(sorted_tokens), i+5)]
                nearby_texts = [t.get('text', '').upper().replace('.', '').replace(':', '').strip() 
                               for t in nearby_tokens]
                header_count = sum(1 for nt in nearby_texts if nt in header_keywords)
                
                if header_count >= 2:  # At least 2 header keywords nearby
                    logger.info(f"Header detected at token #{i}: '{text}'")
                    return True
        
        return False
    
    def _extract_items_with_header(self, sorted_tokens: List[dict]) -> Tuple[List[dict], List[str]]:
        """
        EXISTING LOGIC: Extract items using column header positions
        
        Args:
            sorted_tokens: Tokens sorted by Y position
            
        Returns:
            Tuple of (items list, confidence flags)
        """
        items = []
        flags = []
        
        # 🔍 STEP 1: Find column header row and detect column positions
        column_header_index = -1
        column_positions = {'qty': None, 'rate': None, 'total': None}
        
        # STRICT column keywords - must be standalone words
        qty_keywords = ['QTY', 'QUANTITY', 'QTY.', 'QUANTITY.']
        rate_keywords = ['RATE', 'PRICE', 'UNIT PRICE', 'RATE.', 'PRICE.']
        total_keywords = ['TOTAL', 'AMOUNT', 'TOTAL.', 'AMOUNT.']
        
        # Track if we found the main header row
        found_header_row = False
        
        print("\n🔍 SEARCHING FOR COLUMN HEADER...")
        for i, token in enumerate(sorted_tokens):
            text = token.get('text', '').strip()
            text_upper = text.upper()
            bbox = token.get('bbox', [0,0,0,0])
            x_pos = bbox[0]
            
            # Skip if we already found header row and this is much later (footer area)
            if found_header_row and i > column_header_index + 50:
                break
            
            # STRICT MATCHING: Must be exact keyword or keyword with punctuation only
            text_clean = text_upper.replace('.', '').replace(':', '').strip()
            
            # Detect QTY column (exact match only)
            if text_clean in qty_keywords or text_upper in qty_keywords:
                if not found_header_row:  # Only accept first occurrence
                    column_positions['qty'] = x_pos
                    print(f"  📊 QTY column detected at x={x_pos} (matched: {text})")
            
            # Detect RATE/PRICE column (exact match only)
            if text_clean in rate_keywords or text_upper in rate_keywords:
                if not found_header_row:
                    column_positions['rate'] = x_pos
                    print(f"  💰 RATE/PRICE column detected at x={x_pos} (matched: {text})")
            
            # Detect TOTAL column (exact match only, not "Sub Total" or "Total Qty")
            if text_clean == 'TOTAL' or text_upper == 'TOTAL' or text_upper == 'TOTAL.':
                # Make sure it's NOT part of a compound phrase
                if 'SUB' not in text_upper and 'QTY' not in text_upper and 'GRAND' not in text_upper:
                    if not found_header_row:
                        column_positions['total'] = x_pos
                        print(f"  💵 TOTAL column detected at x={x_pos} (matched: {text})")
            
            # Mark header row ONLY if we found at least 2 columns on this line
            header_keywords = ['ITEM', 'DESCRIPTION', 'DESC', 'QTY', 'QUANTITY', 'RATE', 'PRICE', 'TOTAL', 'AMOUNT']
            if any(kw == text_clean for kw in header_keywords):
                if not found_header_row:
                    # Check if this looks like a real header row by checking nearby tokens
                    nearby_tokens = sorted_tokens[max(0, i-2):min(len(sorted_tokens), i+5)]
                    nearby_texts = [t.get('text', '').upper().replace('.', '').replace(':', '').strip() for t in nearby_tokens]
                    header_count = sum(1 for nt in nearby_texts if nt in header_keywords)
                    
                    if header_count >= 2:  # At least 2 header keywords nearby
                        column_header_index = i
                        found_header_row = True
                        print(f"\n✅✅✅ COLUMN HEADER ROW at #{i} ✅✅✅\n")
                        logger.info(f"🎯 Found column header row at token #{i}")
        
        # Print detected columns
        print(f"\n📊 DETECTED COLUMNS: {column_positions}\n")
        
        # Start scanning AFTER column header
        start_index = column_header_index + 1 if column_header_index >= 0 else 0
        logger.info(f"📍 Starting item scan from token #{start_index}")
        
        # 🔍 STEP 2: Define strict item validation
        def is_valid_item_row(text: str, index: int) -> bool:
            """Returns True if text is a valid product description"""
            if not text or len(text.strip()) < 3:
                return False
            
            text_clean = text.strip()
            text_upper = text_clean.upper()
            import re
            
            # ❌ REJECT: Barcode (10-15 digits)
            if re.match(r'^\d{10,15}$', text_clean):
                logger.debug(f"  ❌ Barcode rejected: {text_clean}")
                return False
            
            # ❌ REJECT: Contains colon (field labels like "Date:", "Cashier:")
            if ':' in text_clean:
                logger.debug(f"  ❌ Field label rejected: {text_clean}")
                return False
            
            # ❌ REJECT: Header/footer keywords (comprehensive)
            reject_keywords = [
                # Transaction fields
                'DATE', 'TIME', 'CASHIER', 'MEMBER', 'INVOICE', 'RECEIPT', 'DOCUMENT',
                'TERMINAL', 'BANK CARD', 'CARD NUMBER', 'APPROVAL', 'TRANSACTION',
                
                # Totals (all variations)
                'TOTAL', 'SUBTOTAL', 'SUB TOTAL', 'SUB-TOTAL', 'GRAND TOTAL',
                'NET TOTAL', 'GROSS TOTAL', 'FINAL AMOUNT', 'AMOUNT PAYABLE',
                'BALANCE DUE', 'PAYABLE',
                
                # Tax (all variations)
                'TAX', 'GST', 'SST', 'VAT', 'CGST', 'SGST', 'IGST',
                'SERVICE TAX', 'SALES TAX', 'SERVICE CHARGE',
                
                # Payment
                'ROUNDING', 'CHANGE', 'CASH', 'PAYMENT', 'BALANCE', 'TENDER', 'PAID',
                
                # Discounts
                'DISCOUNT', 'DISC', 'OFF', 'LESS', 'DEDUCTION', 'REBATE',
                'PROMO', 'VOUCHER', 'COUPON',
                
                # Footer
                'THANK', 'THANKS', 'WELCOME', 'VISIT', 'GOODS SOLD',
                'NOT BE RETURNED', 'EXCHANGEABLE', 
                
                # Company
                'SDN', 'BHD', 'REG NO', 'PRIVATE LIMITED', 'PVT LTD', 'LLP', 'LLC'
            ]
            for keyword in reject_keywords:
                if keyword in text_upper:
                    logger.debug(f"  ❌ Metadata keyword rejected: {text_clean}")
                    return False
            
            # ❌ REJECT: Pure numbers or codes
            if text_clean.replace('-', '').replace('/', '').replace(' ', '').replace('.', '').replace('*', '').isdigit():
                logger.debug(f"  ❌ Pure number rejected: {text_clean}")
                return False
            
            # ❌ REJECT: More than 60% digits
            digit_count = sum(c.isdigit() for c in text_clean)
            if len(text_clean) > 0 and (digit_count / len(text_clean)) > 0.6:
                logger.debug(f"  ❌ Too many digits rejected: {text_clean}")
                return False
            
            # ❌ REJECT: Must contain alphabetic characters
            if not any(c.isalpha() for c in text_clean):
                logger.debug(f"  ❌ No letters rejected: {text_clean}")
                return False
            
            # ❌ REJECT: Too short without unit indicators
            if len(text_clean) < 5:
                if not any(unit in text_upper for unit in ['PC', 'KG', 'PCS', 'UNIT']):
                    logger.debug(f"  ❌ Too short rejected: {text_clean}")
                    return False
            
            # ✅ ACCEPT: Passed all filters
            return True
        
        # 🔍 STEP 3: Extract items starting after column header
        logger.info("=" * 80)
        logger.info("SCANNING TOKENS FOR VALID ITEMS")
        logger.info("=" * 80)
        
        for i in range(start_index, len(sorted_tokens)):
            token = sorted_tokens[i]
            text = token.get('text', '').strip()
            
            # 🛑 STOP: Check if this is Total/Grand Total keyword (end of items)
            text_upper = text.upper()
            total_keywords = ['SUBTOTAL', 'SUB TOTAL', 'GRAND TOTAL', 'NET TOTAL', 
                            'TOTAL AMOUNT', 'FINAL AMOUNT', 'AMOUNT PAYABLE']
            if any(keyword in text_upper for keyword in total_keywords):
                logger.info(f"🛑 TOTAL keyword detected: '{text}' - stopping item extraction")
                print(f"\n🛑 TOTAL DETECTED: '{text}' - Stopping here\n")
                break  # Stop processing further tokens
            
            is_valid = is_valid_item_row(text, i)
            status = "✅ VALID" if is_valid else "❌ REJECT"
            logger.info(f"Token #{i}: {status} | '{text}'")
            
            if not is_valid:
                continue
            
            # ✅ Valid item candidate - extract full details
            logger.info(f"  ✅ VALID ITEM CANDIDATE")
            item = self._extract_item_from_token_area(token, sorted_tokens, i, column_positions)
            
            # 🔍 STEP 4: Final validation - must have price > 0 AND description > 5
            if (item and 
                item.get('description') and 
                len(item.get('description', '')) > 5 and
                item.get('line_total', 0) > 0):
                
                print(f"\n🔥 ADDING ITEM TO LIST: '{item.get('description')}'")
                print(f"   Qty: {item.get('qty')} | Unit: RM{item.get('unit_price', 0):.2f} | Total: RM{item.get('line_total', 0):.2f}\n")
                items.append(item)
                logger.info(f"  ✅✅✅ ITEM EXTRACTED: '{item.get('description')}'")
                logger.info(f"      Qty: {item.get('qty')} | Unit: RM{item.get('unit_price', 0):.2f} | Total: RM{item.get('line_total', 0):.2f}")
                logger.info("-" * 80)
        
        logger.info("=" * 80)
        logger.info(f"FINAL RESULT: {len(items)} items extracted")
        logger.info("=" * 80)
        
        if not items:
            flags.append("NO_ITEMS_EXTRACTED: Could not identify valid item patterns in OCR data")
            logger.warning("No items extracted from receipt-style parsing")
        
        return items, flags
    
    def _extract_items_headerless(self, sorted_tokens: List[dict]) -> Tuple[List[dict], List[str]]:
        """
        NEW LOGIC: Extract items WITHOUT column headers (proximity + semantic inference)
        
        Args:
            sorted_tokens: Tokens sorted by Y position
            
        Returns:
            Tuple of (items list, confidence flags)
        """
        items = []
        flags = []
        
        print("\n" + "="*80)
        print("🔥 HEADERLESS PARSING MODE 🔥")
        print("="*80 + "\n")
        
        logger.info("=" * 80)
        logger.info("HEADERLESS PARSING: Using proximity + semantic inference")
        logger.info("=" * 80)
        
        # Group tokens into rows by Y position
        rows = self._group_tokens_into_rows_headerless(sorted_tokens)
        logger.info(f"Grouped {len(sorted_tokens)} tokens into {len(rows)} rows")
        
        print(f"\n📊 GROUPED INTO {len(rows)} ROWS\n")
        
        # Process each row
        for row_idx, row_tokens in enumerate(rows):
            # 🛑 STOP: Check if this is Total/Grand Total row (end of items)
            if self._is_total_row(row_tokens):
                logger.info(f"🛑 TOTAL ROW detected at row {row_idx} - stopping item extraction")
                print(f"\n🛑 TOTAL ROW DETECTED - Stopping here\n")
                break  # Stop processing further rows
            
            # Check if this is a valid item row
            if not self._is_valid_item_row_headerless(row_tokens):
                logger.debug(f"Row {row_idx}: Skipped (not item row)")
                continue
            
            # Extract item from row
            item = self._extract_item_from_row_headerless(row_tokens)
            
            if item and item.get('description') and len(item.get('description', '')) > 3:
                # Validate math: qty × price ≈ total
                qty = item.get('qty', 1.0)
                price = item.get('unit_price', 0.0)
                total = item.get('line_total', 0.0)
                
                if total > 0:
                    expected_total = qty * price
                    error = abs(total - expected_total)
                    tolerance = max(0.05, total * 0.02)  # 2% tolerance or 5 cents
                    
                    if error <= tolerance:
                        print(f"\n✅ ITEM: {item['description']}")
                        print(f"   Qty: {qty} | Price: {price:.2f} | Total: {total:.2f}")
                        items.append(item)
                        logger.info(f"✅ Item extracted: {item['description']} (Qty:{qty}, Price:{price:.2f}, Total:{total:.2f})")
                    else:
                        logger.debug(f"❌ Math mismatch: {item['description']} ({qty}×{price:.2f}={expected_total:.2f} ≠ {total:.2f})")
                        flags.append(f"MATH_MISMATCH: {item['description']}")
        
        logger.info("=" * 80)
        logger.info(f"HEADERLESS RESULT: {len(items)} items extracted")
        logger.info("=" * 80)
        
        if not items:
            flags.append("NO_ITEMS_EXTRACTED_HEADERLESS: Could not parse items without column headers")
            logger.warning("No items extracted from headerless parsing")
        
        print(f"\n📦 FINAL: {len(items)} items extracted\n")
        
        return items, flags
    
    def _group_tokens_into_rows_headerless(self, sorted_tokens: List[dict]) -> List[List[dict]]:
        """
        Group tokens into rows by Y coordinate proximity
        
        Args:
            sorted_tokens: Tokens sorted by Y position
            
        Returns:
            List of rows (each row is list of tokens)
        """
        if not sorted_tokens:
            return []
        
        rows = []
        current_row = [sorted_tokens[0]]
        current_y = sorted_tokens[0].get('bbox', [0,0,0,0])[1]
        
        y_threshold = 15  # Y tolerance for same row
        
        for token in sorted_tokens[1:]:
            bbox = token.get('bbox', [0,0,0,0])
            token_y = bbox[1]
            
            if abs(token_y - current_y) <= y_threshold:
                # Same row
                current_row.append(token)
            else:
                # New row
                if current_row:
                    # Sort current row by X position (left to right)
                    current_row.sort(key=lambda t: t.get('bbox', [0,0,0,0])[0])
                    rows.append(current_row)
                current_row = [token]
                current_y = token_y
        
        # Add last row
        if current_row:
            current_row.sort(key=lambda t: t.get('bbox', [0,0,0,0])[0])
            rows.append(current_row)
        
        return rows
    
    def _is_total_row(self, row_tokens: List[dict]) -> bool:
        """
        Check if row is a Total/Grand Total/Sub Total row (end of items section)
        
        Args:
            row_tokens: Tokens in the row
            
        Returns:
            True if this is a total row (should stop parsing)
        """
        if not row_tokens:
            return False
        
        # Combine all text in row
        row_text = ' '.join(t.get('text', '').strip() for t in row_tokens if t.get('text', '').strip())
        row_text_upper = row_text.upper()
        
        # Total keywords that indicate end of items
        total_keywords = [
            'SUBTOTAL', 'SUB TOTAL', 'SUB-TOTAL',
            'TOTAL AMOUNT', 'AMOUNT TOTAL', 'TOTAL PRICE', 'TOTAL QTY',
            'GRAND TOTAL', 'NET TOTAL', 'GROSS TOTAL',
            'FINAL AMOUNT', 'AMOUNT PAYABLE', 'PAYABLE AMOUNT',
            'BILL TOTAL', 'INVOICE TOTAL',
            'TOTAL:', 'AMOUNT:'
        ]
        
        # Check for exact or close matches (case-insensitive, whitespace-flexible)
        for keyword in total_keywords:
            if keyword in row_text_upper:
                logger.info(f"🛑 Total row detected: '{row_text}' (matched: {keyword})")
                print(f"🛑 STOPPING AT: '{row_text}' (matched: {keyword})")
                return True
        
        # Also check for single word "TOTAL" or "AMOUNT" if it's the main text
        # But NOT if it's part of "LINE TOTAL" column header
        if 'TOTAL' in row_text_upper and 'LINE' not in row_text_upper:
            # Must have a numeric value after it (indicating summary row)
            has_number = any(self._parse_amount(t.get('text', '')) > 0 for t in row_tokens)
            if has_number:
                logger.info(f"🛑 Total row detected: '{row_text}' (TOTAL keyword with number)")
                print(f"🛑 STOPPING AT: '{row_text}' (TOTAL with number)")
                return True
        
        # Check for standalone "AMOUNT" with a large number (likely total)
        if row_text_upper.startswith('AMOUNT') or 'AMOUNT' == row_text_upper.strip():
            has_number = any(self._parse_amount(t.get('text', '')) > 0 for t in row_tokens)
            if has_number:
                logger.info(f"🛑 Total row detected: '{row_text}' (AMOUNT keyword)")
                print(f"🛑 STOPPING AT: '{row_text}' (AMOUNT keyword)")
                return True
        
        return False
    
    def _is_valid_item_row_headerless(self, row_tokens: List[dict]) -> bool:
        """
        Check if row is a valid item row (not header/footer/summary)
        
        Args:
            row_tokens: Tokens in the row
            
        Returns:
            True if valid item row
        """
        if not row_tokens:
            return False
        
        # Combine all text in row
        row_text = ' '.join(t.get('text', '').strip() for t in row_tokens if t.get('text', '').strip())
        row_text_upper = row_text.upper()
        
        # Must have some alphabetic content
        if not any(c.isalpha() for c in row_text):
            return False
        
        # Reject header/footer keywords
        reject_keywords = [
            'DATE', 'TIME', 'CASHIER', 'MEMBER', 'INVOICE', 'RECEIPT',
            'SUBTOTAL', 'SUB TOTAL', 'GRAND TOTAL', 'NET TOTAL',
            'TAX', 'GST', 'SST', 'VAT', 'CGST', 'SGST',
            'ROUNDING', 'CHANGE', 'CASH', 'PAYMENT', 'BALANCE', 'TENDER',
            'DISCOUNT', 'DISC', 'OFF', 'LESS',
            'THANK', 'THANKS', 'WELCOME', 'VISIT',
            'SDN', 'BHD', 'REG NO', 'PRIVATE LIMITED',
            # Payment/Terminal info
            'TERMINAL', 'BANK CARD', 'CARD NUMBER', 'APPROVAL',
            'TRANSACTION', 'REFERENCE', 'REF NO',
            # Header keywords
            'QTY', 'QUANTITY', 'RATE', 'PRICE', 'AMOUNT'
        ]
        
        for keyword in reject_keywords:
            if keyword in row_text_upper:
                return False
        
        # Must contain at least one numeric value
        has_number = any(self._parse_amount(t.get('text', '')) > 0 for t in row_tokens)
        if not has_number:
            return False
        
        return True
    
    def _extract_item_from_row_headerless(self, row_tokens: List[dict]) -> Optional[dict]:
        """
        Extract item from row using proximity + semantic inference
        
        Strategy:
        - Find description anchor (longest alphabetic text, leftmost)
        - Find numeric tokens (right of description)
        - Assign roles by position and magnitude:
          * Small integer (1-20) = Qty
          * First decimal = Unit Price
          * Last/largest = Line Total
        
        Args:
            row_tokens: Tokens in row (sorted left to right)
            
        Returns:
            Item dict or None
        """
        if not row_tokens:
            return None
        
        # Separate text tokens and numeric tokens
        text_tokens = []
        numeric_values = []
        
        for token in row_tokens:
            text = token.get('text', '').strip()
            num_val = self._parse_amount(text)
            
            if num_val > 0:
                numeric_values.append(num_val)
            elif any(c.isalpha() for c in text):
                text_tokens.append(text)
        
        # Build description from text tokens
        description = ' '.join(text_tokens).strip()
        
        if not description or len(description) < 3:
            return None
        
        if not numeric_values:
            return None
        
        # Validate numeric values - reject abnormally large numbers (card/approval codes)
        # Typical max price: $10,000 per item
        MAX_REASONABLE_PRICE = 10000.0
        numeric_values = [v for v in numeric_values if v <= MAX_REASONABLE_PRICE]
        
        if not numeric_values:
            return None
        
        # Assign numeric roles by position and magnitude
        qty = 1.0
        unit_price = 0.0
        line_total = 0.0
        
        if len(numeric_values) == 1:
            # Single number = price/total
            unit_price = numeric_values[0]
            line_total = numeric_values[0]
        
        elif len(numeric_values) == 2:
            # Two numbers: determine which is qty/price/total
            val1, val2 = numeric_values[0], numeric_values[1]
            
            # If first is small integer (≤20), treat as qty
            if val1 <= 20 and val1 == int(val1):
                qty = val1
                line_total = val2
                unit_price = line_total / qty if qty > 0 else line_total
            else:
                # Both are prices: first=unit_price, second=line_total
                unit_price = val1
                line_total = val2
                qty = line_total / unit_price if unit_price > 0 else 1.0
        
        elif len(numeric_values) >= 3:
            # Three+ numbers: standard format [qty, price, total]
            qty = numeric_values[0]
            unit_price = numeric_values[1]
            line_total = numeric_values[2]
            
            # Validate: if math doesn't work, try alternative
            expected = qty * unit_price
            if abs(line_total - expected) > (line_total * 0.2):
                # Try: [price, qty, total]
                alt_price = numeric_values[0]
                alt_qty = numeric_values[1]
                alt_total = numeric_values[2]
                alt_expected = alt_qty * alt_price
                
                if abs(alt_total - alt_expected) < abs(line_total - expected):
                    qty = alt_qty
                    unit_price = alt_price
                    line_total = alt_total
        
        # Final validation
        if unit_price <= 0 or line_total <= 0:
            return None
        
        return {
            'description': description,
            'qty': qty,
            'unit_price': unit_price,
            'line_total': line_total
        }
    
    def _extract_item_from_token_area(self, desc_token: dict, all_tokens: List[dict], desc_index: int, column_positions: dict = None) -> dict:
        """
        SIMPLE SEQUENTIAL EXTRACTION - tokens are already in correct order!
        """
        desc_bbox = desc_token.get('bbox', [0,0,0,0])
        desc_y = desc_bbox[1]
        desc_text = desc_token.get('text', '').strip()
        
        # Collect all tokens on SAME LINE, sorted by X position
        same_line_tokens = []
        
        for token in all_tokens:
            token_text = token.get('text', '').strip()
            token_bbox = token.get('bbox', [0,0,0,0])
            token_y = token_bbox[1]
            token_x = token_bbox[0]
            
            if not token_text:
                continue
            
            # Same line check (±15px Y tolerance)
            if abs(token_y - desc_y) > 15:
                continue
            
            same_line_tokens.append({
                'text': token_text,
                'x': token_x,
                'y': token_y,
                'bbox': token_bbox
            })
        
        # Sort by X position (left to right)
        same_line_tokens.sort(key=lambda t: t['x'])
        
        # Extract description (leftmost text tokens)
        description_parts = []
        numeric_tokens = []
        
        for token in same_line_tokens:
            text = token['text']
            num_val = self._parse_amount(text)
            
            if num_val > 0:
                # It's a number
                numeric_tokens.append(num_val)
            elif any(c.isalpha() for c in text):
                # It's text - part of description
                description_parts.append(text)
        
        description = ' '.join(description_parts).strip()
        
        if not description or len(description) < 3:
            return None
        
        # ASSIGN NUMBERS BY POSITION (left to right)
        qty = 1.0
        unit_price = 0.0
        line_total = 0.0
        
        if len(numeric_tokens) == 1:
            # Single number = price
            unit_price = numeric_tokens[0]
            line_total = unit_price
        
        elif len(numeric_tokens) == 2:
            # Two numbers = qty, total OR price, total
            if numeric_tokens[0] < 20:  # Small number = qty
                qty = numeric_tokens[0]
                line_total = numeric_tokens[1]
                unit_price = line_total / qty if qty > 0 else line_total
            else:  # Both prices
                unit_price = numeric_tokens[0]
                line_total = numeric_tokens[1]
        
        elif len(numeric_tokens) >= 3:
            # Three+ numbers = qty, price, total (standard receipt format)
            qty = numeric_tokens[0]
            unit_price = numeric_tokens[1]
            line_total = numeric_tokens[2]
        
        # Validation: Check if math works (qty × price ≈ total)
        if qty > 0 and unit_price > 0 and line_total > 0:
            expected_total = qty * unit_price
            error_ratio = abs(line_total - expected_total) / max(expected_total, line_total)
            
            # If math is way off (>20% error), try alternative interpretation
            if error_ratio > 0.20:
                # Maybe qty and price are swapped?
                if len(numeric_tokens) >= 3:
                    # Try: price, qty, total
                    alt_price = numeric_tokens[0]
                    alt_qty = numeric_tokens[1]
                    alt_total = numeric_tokens[2]
                    alt_expected = alt_qty * alt_price
                    alt_error = abs(alt_total - alt_expected) / max(alt_expected, alt_total)
                    
                    if alt_error < error_ratio:  # Better match
                        qty = alt_qty
                        unit_price = alt_price
                        line_total = alt_total
        
        # Final validation
        if unit_price == 0 and line_total == 0:
            return None
        
        return {
            'description': description,
            'qty': qty,
            'unit_price': unit_price,
            'line_total': line_total
        }
    
    def _score_as_quantity(self, value: float, text: str) -> float:
        """Score how likely this value is a quantity (0-1)"""
        score = 0.0
        
        # Typical quantities: 1-100
        if 1 <= value <= 100:
            score += 0.5
        elif value < 1 or value > 1000:
            return 0.0  # Not a valid quantity
        
        # Integer quantities are more common
        if value == int(value):
            score += 0.3
        
        # Has unit indicators
        if any(unit in text.upper() for unit in ['PC', 'PCS', 'KG', 'G', 'L', 'ML']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_as_unit_price(self, value: float, text: str) -> float:
        """Score how likely this value is a unit price (0-1)"""
        score = 0.0
        
        # Typical unit prices: 0.1 - 10000
        if 0.1 <= value <= 10000:
            score += 0.5
        else:
            return 0.0
        
        # Has decimal places (prices often do)
        if value != int(value):
            score += 0.2
        
        # Has currency symbols
        if any(sym in text for sym in ['RM', 'RS', '₹', '$', '€']):
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_as_line_total(self, value: float, text: str) -> float:
        """Score how likely this value is a line total (0-1)"""
        score = 0.0
        
        # Line totals typically larger than unit prices
        if value >= 1:
            score += 0.5
        
        # Has decimal places
        if value != int(value):
            score += 0.2
        
        # Has currency symbols
        if any(sym in text for sym in ['RM', 'RS', '₹', '$', '€']):
            score += 0.3
        
        return min(score, 1.0)
    
    def _find_best_assignment(self, numeric_tokens: List[dict], column_positions: dict) -> dict:
        """
        Find the best qty/price/total assignment using spatial + semantic + mathematical validation
        """
        if len(numeric_tokens) == 0:
            return None
        
        # Generate all possible assignments
        candidates = []
        
        if len(numeric_tokens) == 1:
            # Single number = treat as price
            candidates.append({
                'qty': 1.0,
                'unit_price': numeric_tokens[0]['value'],
                'line_total': numeric_tokens[0]['value']
            })
        
        elif len(numeric_tokens) == 2:
            # Two numbers: try both interpretations
            val1, val2 = numeric_tokens[0]['value'], numeric_tokens[1]['value']
            
            # Option 1: [qty, price]
            candidates.append({
                'qty': val1,
                'unit_price': val2,
                'line_total': val1 * val2
            })
            
            # Option 2: [price, total]
            candidates.append({
                'qty': 1.0,
                'unit_price': val1,
                'line_total': val2
            })
        
        elif len(numeric_tokens) >= 3:
            # Three+ numbers: try [qty, price, total]
            # Sort by X position (left to right)
            sorted_tokens = sorted(numeric_tokens, key=lambda t: t['x'])
            
            # Standard layout: [qty, price, total]
            candidates.append({
                'qty': sorted_tokens[0]['value'],
                'unit_price': sorted_tokens[1]['value'],
                'line_total': sorted_tokens[2]['value']
            })
            
            # Alternative: [price, qty, total] (some receipts)
            candidates.append({
                'qty': sorted_tokens[1]['value'],
                'unit_price': sorted_tokens[0]['value'],
                'line_total': sorted_tokens[2]['value']
            })
        
        # Score each candidate
        best_candidate = None
        best_score = -1
        
        for candidate in candidates:
            score = self._score_assignment(candidate, numeric_tokens, column_positions)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate if best_score > 0.3 else None
    
    def _score_assignment(self, assignment: dict, numeric_tokens: List[dict], column_positions: dict) -> float:
        """
        Score an assignment based on semantic validity + spatial matching + mathematical consistency
        """
        score = 0.0
        
        qty = assignment['qty']
        unit_price = assignment['unit_price']
        line_total = assignment['line_total']
        
        # SEMANTIC VALIDATION (40 points)
        # Quantity should be reasonable (1-100 typical)
        if 1 <= qty <= 100:
            score += 15
        elif qty > 100:
            score -= 10  # Penalty for unreasonable qty
        
        # Unit price should be reasonable (0.1-10000)
        if 0.1 <= unit_price <= 10000:
            score += 15
        
        # Total should be >= unit price
        if line_total >= unit_price:
            score += 10
        else:
            score -= 20  # Big penalty
        
        # MATHEMATICAL VALIDATION (40 points)
        expected_total = qty * unit_price
        if expected_total > 0:
            error_ratio = abs(line_total - expected_total) / expected_total
            if error_ratio < 0.02:  # Within 2%
                score += 40
            elif error_ratio < 0.05:  # Within 5%
                score += 30
            elif error_ratio < 0.10:  # Within 10%
                score += 20
            else:
                score -= 10  # Math doesn't work
        
        # SPATIAL VALIDATION (20 points)
        if column_positions and column_positions.get('qty') and column_positions.get('rate'):
            # Find which token matches which value
            for token in numeric_tokens:
                val = token['value']
                x = token['x']
                
                # Check if qty is in qty column
                if val == qty and column_positions.get('qty'):
                    if abs(x - column_positions['qty']) < 80:
                        score += 7
                
                # Check if price is in rate column
                if val == unit_price and column_positions.get('rate'):
                    if abs(x - column_positions['rate']) < 80:
                        score += 7
                
                # Check if total is in total column
                if val == line_total and column_positions.get('total'):
                    if abs(x - column_positions['total']) < 80:
                        score += 6
        
        return score
    
    def extract_amounts(self, pages_data: List[dict], tables: List[dict]) -> dict:
        """
        Extract total amounts, tax, subtotal from document
        
        Args:
            pages_data: OCR pages data
            tables: Table data
            
        Returns:
            Dictionary with amounts
        """
        amounts = {
            'subtotal': 0.0,
            'tax': 0.0,
            'total': 0.0
        }
        
        # Collect all text tokens - handle both dict and list formats
        all_tokens = []
        
        if isinstance(pages_data, dict):
            # Handle dictionary format
            for page_key, page_data in pages_data.items():
                tokens = page_data.get('tokens', [])
                all_tokens.extend(tokens)
        elif isinstance(pages_data, list):
            # Handle list format
            for page_data in pages_data:
                if isinstance(page_data, dict):
                    tokens = page_data.get('tokens', [])
                    all_tokens.extend(tokens)
        
        # Look for total amounts from OCR
        ocr_total = self._find_amount_by_keywords(all_tokens, self.total_keywords)
        amounts['tax'] = self._find_amount_by_keywords(all_tokens, self.tax_keywords)
        
        # Calculate subtotal from items (MOST ACCURATE)
        if amounts['subtotal'] == 0.0:
            item_total = 0.0
            for table in tables:
                for row in table.get('rows', []):
                    cells = row.get('cells', {})
                    line_total = self._parse_amount(cells.get('LINE_TOTAL', '0'))
                    item_total += line_total
            amounts['subtotal'] = item_total
        
        # Calculate total from items sum + tax (MORE ACCURATE than OCR)
        calculated_total = amounts['subtotal'] + amounts['tax']
        
        # Use calculated total if available, otherwise use OCR total
        if calculated_total > 0:
            amounts['total'] = calculated_total
            logger.info(f"Using calculated total: ${calculated_total:.2f} (items sum + tax)")
        elif ocr_total > 0:
            amounts['total'] = ocr_total
            logger.info(f"Using OCR-extracted total: ${ocr_total:.2f}")
        else:
            amounts['total'] = 0.0
        
        return amounts
    
    def validate_totals(self, items: List[dict], amounts: dict) -> List[str]:
        """
        Validate mathematical consistency of amounts
        
        Args:
            items: Parsed items list
            amounts: Extracted amounts
            
        Returns:
            List of validation flags
        """
        flags = []
        
        # Calculate sum of line totals
        calculated_subtotal = sum(item.get('line_total', 0) for item in items)
        
        # Check subtotal consistency
        if abs(calculated_subtotal - amounts['subtotal']) > 0.01:
            flags.append(f"SUBTOTAL_MISMATCH: calculated={calculated_subtotal:.2f}, found={amounts['subtotal']:.2f}")
        
        # Check total consistency
        expected_total = amounts['subtotal'] + amounts['tax']
        if abs(expected_total - amounts['total']) > 0.01:
            flags.append(f"TOTAL_MISMATCH: expected={expected_total:.2f}, found={amounts['total']:.2f}")
        
        # Validate individual items
        for item in items:
            qty = item.get('qty', 0)
            unit_price = item.get('unit_price', 0)
            line_total = item.get('line_total', 0)
            
            expected_line_total = qty * unit_price
            if abs(expected_line_total - line_total) > 0.01 and qty > 0 and unit_price > 0:
                flags.append(f"LINE_TOTAL_MISMATCH: {item.get('description', 'Unknown')}")
        
        return flags
    
    def build_final_output(self, header: dict, items: List[dict], amounts: dict, flags: List[str]) -> dict:
        """
        Build the final business schema output
        
        Args:
            header: Header information
            items: Parsed items
            amounts: Extracted amounts
            flags: Validation flags
            
        Returns:
            Complete business schema
        """
        return {
            'vendor': header.get('vendor', ''),
            'date': header.get('date'),
            'currency': header.get('currency', 'MYR'),
            'items': items,
            'tax': amounts.get('tax', 0.0),
            'subtotal': amounts.get('subtotal', 0.0),
            'total': amounts.get('total', 0.0),
            'confidence_flags': flags,
            'processing_timestamp': datetime.now().isoformat(),
            'item_count': len(items)
        }
    
    def save_business_schema(self, job_id: str, schema: dict) -> Tuple[str, str]:
        """
        Save business schema to JSON and CSV files
        
        Args:
            job_id: Job identifier
            schema: Business schema dictionary
            
        Returns:
            Tuple of (json_path, csv_path)
        """
        # Setup paths
        results_dir = Path(__file__).parent.parent / "tmp" / "results" / job_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = results_dir / "extracted.json"
        csv_path = results_dir / "extracted.csv"
        
        # Save JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False, default=str)
        
        # Save CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header info
            writer.writerow(['Vendor', schema.get('vendor', '')])
            writer.writerow(['Date', schema.get('date', '')])
            writer.writerow(['Currency', schema.get('currency', '')])
            writer.writerow(['Total', schema.get('total', 0)])
            writer.writerow(['Tax', schema.get('tax', 0)])
            writer.writerow([''])  # Empty row
            
            # Write items header
            writer.writerow(['Description', 'Qty', 'Unit Price', 'Line Total'])
            
            # Write items
            for item in schema.get('items', []):
                writer.writerow([
                    item.get('description', ''),
                    item.get('qty', 0),
                    item.get('unit_price', 0),
                    item.get('line_total', 0)
                ])
        
        logger.info(f"Business schema saved to {json_path} and {csv_path}")
        return str(json_path), str(csv_path)
    
    # Helper methods
    def _is_numeric_context(self, text: str) -> bool:
        """Check if text appears to be numeric (price, quantity)"""
        # Strip whitespace
        text = text.strip()
        
        # Check if text is primarily numeric
        # Remove common currency symbols and see what's left
        cleaned = re.sub(r'[\s$€£¥₹RM]', '', text)
        
        # If more than 50% of remaining characters are digits/decimal points, it's numeric
        if not cleaned:
            return False
        
        numeric_chars = len(re.findall(r'[\d.,]', cleaned))
        return numeric_chars / len(cleaned) > 0.5
    
    def _extract_vendor(self, top_tokens: List[dict]) -> str:
        """Extract vendor name from top tokens"""
        candidates = []
        
        for token in top_tokens:
            text = token.get('text', '').strip()
            confidence = token.get('confidence', 1.0)
            
            # Skip very short text or low confidence
            if len(text) < 3 or confidence < 0.3:
                continue
                
            # Skip if text looks like pure numbers, dates, or addresses
            if (self._is_numeric_context(text) or 
                re.search(r'^\d+[/-]\d+', text) or  # date-like
                re.search(r'^No\.\s*\d+', text, re.IGNORECASE)):  # address-like
                continue
            
            # Check if it contains vendor indicators
            text_upper = text.upper()
            for indicator in self.vendor_indicators:
                if indicator in text_upper:
                    return text
            
            # Add as candidate if it's substantial text
            if len(text) > 5:
                candidates.append(text)
        
        # Return longest candidate if no clear vendor found
        return max(candidates, key=len) if candidates else ''
    
    def _extract_date(self, tokens: List[dict]) -> Optional[str]:
        """Extract date from tokens"""
        for token in tokens:
            text = token.get('text', '').strip()
            
            # Try to parse as date
            try:
                # Common date patterns
                if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text):
                    parsed_date = date_parser.parse(text, fuzzy=True)
                    return parsed_date.strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                continue
        
        return None
    
    def _extract_currency(self, tokens: List[dict]) -> str:
        """Extract currency from tokens"""
        for token in tokens:
            text = token.get('text', '').strip().upper()
            
            for symbol, code in self.currency_symbols.items():
                if symbol in text:
                    return code
        
        return 'MYR'  # Default
    
    def _is_header_row(self, cells: dict) -> bool:
        """Check if row is a header row"""
        text_values = [v.upper() for v in cells.values() if v]
        
        header_keywords = ['ITEM', 'QTY', 'QUANTITY', 'PRICE', 'TOTAL', 'AMOUNT', 'DESCRIPTION']
        return any(keyword in ' '.join(text_values) for keyword in header_keywords)
    
    def _parse_item_row(self, cells: dict) -> Optional[dict]:
        """Parse a single table row into an item - with strict validation"""
        description = ''
        qty = 1
        unit_price = 0.0
        line_total = 0.0
        
        # Extract description
        for key in ['ITEM', 'DESCRIPTION', 'DESC', 'PRODUCT']:
            if key in cells and cells[key]:
                description = cells[key].strip()
                break
        
        # VALIDATION 1: Must have description
        if not description or len(description) < 3:
            return None
        
        # VALIDATION 2: Skip metadata rows (these are NOT items)
        metadata_keywords = [
            'INVOICE', 'DATE', 'TOTAL QTY', 'SUB TOTAL', 'SUBTOTAL',
            'CGST', 'SGST', 'GST', 'TAX', 'DISCOUNT', 'GRAND TOTAL',
            'PAYMENT', 'CASH', 'CHANGE', 'BALANCE', 'TENDER',
            'THANK', 'WELCOME', 'VISIT AGAIN'
        ]
        desc_upper = description.upper()
        if any(kw in desc_upper for kw in metadata_keywords):
            return None
        
        # Skip rows with colons (field labels like "Invoice Number:")
        if ':' in description:
            return None
        
        # Extract quantity
        qty_str = ''
        for key in ['QTY', 'QUANTITY', 'Q']:
            if key in cells and cells[key]:
                qty_str = cells[key].strip()
                break
        
        # Extract unit price
        price_str = ''
        for key in ['UNIT_PRICE', 'PRICE', 'RATE', 'UNIT']:
            if key in cells and cells[key]:
                price_str = cells[key].strip()
                break
        
        # Extract line total
        total_str = ''
        for key in ['LINE_TOTAL', 'TOTAL', 'AMOUNT', 'SUM']:
            if key in cells and cells[key]:
                total_str = cells[key].strip()
                break
        
        # VALIDATION 3: Must have EITHER unit_price OR line_total with actual numeric value
        if not price_str and not total_str:
            return None
        
        # Parse values
        if qty_str:
            qty = self._parse_quantity(qty_str)
        if price_str:
            unit_price = self._parse_amount(price_str)
        if total_str:
            line_total = self._parse_amount(total_str)
        
        # VALIDATION 4: At least one price field must be > 0
        if unit_price == 0.0 and line_total == 0.0:
            return None
        
        # Infer missing values
        if unit_price == 0.0 and line_total > 0.0 and qty > 0:
            unit_price = line_total / qty
        
        if line_total == 0.0 and unit_price > 0.0 and qty > 0:
            line_total = unit_price * qty
        
        return {
            'description': description,
            'qty': qty,
            'unit_price': round(unit_price, 2),
            'line_total': round(line_total, 2)
        }
    
    def _parse_quantity(self, text: str) -> int:
        """Parse quantity from text"""
        if not text:
            return 1
        
        # Extract first number
        match = re.search(r'\d+', text)
        if match:
            return max(1, int(match.group()))
        
        return 1
    
    def _parse_amount(self, text: str) -> float:
        """Parse monetary amount from text - SIMPLE for structured data"""
        if not text:
            return 0.0
        
        text_clean = text.strip()
        
        # Remove currency symbols and extra spaces
        text_clean = re.sub(r'[RM$₹€£¥]', '', text_clean).strip()
        
        # Handle comma as thousands separator: 1,234.56 → 1234.56
        if ',' in text_clean and '.' in text_clean:
            text_clean = text_clean.replace(',', '')
        # Handle comma as decimal separator (European style): 123,45 → 123.45
        elif ',' in text_clean:
            parts = text_clean.split(',')
            if len(parts) == 2 and len(parts[1]) <= 2:
                text_clean = text_clean.replace(',', '.')
            else:
                text_clean = text_clean.replace(',', '')
        
        # Extract just the number
        match = re.search(r'\d+\.?\d*', text_clean)
        if match:
            try:
                return float(match.group())
            except:
                return 0.0
        
        return 0.0
    
    def _find_amount_by_keywords(self, tokens: List[dict], keywords: List[str]) -> float:
        """Find amount associated with specific keywords"""
        for i, token in enumerate(tokens):
            text = token.get('text', '').lower()
            
            # Check if token contains keyword
            if any(keyword in text for keyword in keywords):
                # Look for number in same token or nearby tokens
                for j in range(max(0, i-2), min(len(tokens), i+3)):
                    nearby_text = tokens[j].get('text', '')
                    amount = self._parse_amount(nearby_text)
                    if amount > 0:
                        return amount
        
        return 0.0

# Convenience function for Phase 5
def process_document_to_business_schema(job_id: str) -> dict:
    """
    Process a completed job to business schema (Phase 5)
    
    Args:
        job_id: Job identifier
        
    Returns:
        Business schema dictionary
    """
    # Load Phase 4 table data
    results_dir = Path(__file__).parent.parent / "tmp" / "results" / job_id
    
    tables_file = results_dir / "tables.json"
    ocr_file = results_dir / "ocr.json"  # Fixed: Match ocr_engine.py filename
    
    if not tables_file.exists():
        raise FileNotFoundError(f"Tables data not found for job {job_id}")
    
    # Load OCR data with tolerance for missing files
    if not ocr_file.exists():
        logger.warning(f"OCR data not found for job {job_id}, using empty OCR data")
        ocr_data = {"pages": [], "total_pages": 0, "total_tokens": 0}
    else:
        try:
            with open(ocr_file, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
                # Validate OCR data structure
                if not isinstance(ocr_data, dict) or "pages" not in ocr_data:
                    logger.warning(f"Invalid OCR data structure for job {job_id}, using empty OCR data")
                    ocr_data = {"pages": [], "total_pages": 0, "total_tokens": 0}
        except Exception as e:
            logger.error(f"Failed to load OCR data for job {job_id}: {e}, using empty OCR data")
            ocr_data = {"pages": [], "total_pages": 0, "total_tokens": 0}
    
    # Load tables data
    with open(tables_file, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    
    # Process to business schema
    parser = BusinessSchemaParser()
    return parser.process_document_to_schema(job_id, tables_data, ocr_data)

if __name__ == "__main__":
    # Test the parser with sample data
    sample_tokens = [
        {"text": "ITEM", "bbox": [50, 100, 150, 120], "confidence": 0.9},
        {"text": "QTY", "bbox": [200, 100, 250, 120], "confidence": 0.9},
        {"text": "PRICE", "bbox": [300, 100, 380, 120], "confidence": 0.9},
        {"text": "KF MODELLING CLAY", "bbox": [50, 150, 180, 170], "confidence": 0.8},
        {"text": "1", "bbox": [210, 150, 230, 170], "confidence": 0.9},
        {"text": "9.00", "bbox": [320, 150, 360, 170], "confidence": 0.9},
    ]
    
    table_bbox = {"x1": 40, "y1": 90, "x2": 400, "y2": 200}
    
    parser = TableParser()
    result = parser.parse_table_structure(sample_tokens, table_bbox)
    
    print("Parsed table structure:")
    print(json.dumps(result, indent=2))
    
    # Test business schema parser
    print("\n" + "="*50)
    print("Testing Business Schema Parser (Phase 5)")
    
    business_parser = BusinessSchemaParser()
    
    # Test text normalization
    test_text = "9.OO"  # Common OCR error
    normalized = business_parser.normalize_text(test_text)
    print(f"Text normalization: '{test_text}' -> '{normalized}'")
    
    # Test amount parsing
    test_amounts = ["RM 9.50", "$12,34", "15.00", "Total: 25.75"]
    for amount_text in test_amounts:
        parsed = business_parser._parse_amount(amount_text)
        print(f"Amount parsing: '{amount_text}' -> {parsed}")
    
    print("Phase 5 parser ready!")
