#!/usr/bin/env python3
"""
Validate Dockerfile syntax and structure
"""

import re
import sys
from pathlib import Path

def validate_dockerfile():
    """Validate Dockerfile for common issues"""
    
    dockerfile_path = Path("Dockerfile")
    
    if not dockerfile_path.exists():
        print("❌ Dockerfile not found!")
        return False
    
    print("🔍 Validating Dockerfile...")
    
    with open(dockerfile_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    warnings = []
    
    # Check for basic structure
    if not re.search(r'^FROM\s+', content, re.MULTILINE):
        issues.append("No FROM instruction found")
    
    # Check for multi-stage build
    from_count = len(re.findall(r'^FROM\s+', content, re.MULTILINE))
    if from_count >= 2:
        print(f"✅ Multi-stage build detected ({from_count} stages)")
    
    # Check for WORKDIR
    if not re.search(r'^WORKDIR\s+', content, re.MULTILINE):
        warnings.append("No WORKDIR instruction found")
    
    # Check for COPY/ADD instructions
    copy_count = len(re.findall(r'^(COPY|ADD)\s+', content, re.MULTILINE))
    if copy_count > 0:
        print(f"✅ Found {copy_count} COPY/ADD instructions")
    
    # Check for RUN instructions
    run_count = len(re.findall(r'^RUN\s+', content, re.MULTILINE))
    if run_count > 0:
        print(f"✅ Found {run_count} RUN instructions")
    
    # Check for CMD or ENTRYPOINT
    if not re.search(r'^(CMD|ENTRYPOINT)\s+', content, re.MULTILINE):
        issues.append("No CMD or ENTRYPOINT instruction found")
    else:
        print("✅ CMD/ENTRYPOINT found")
    
    # Check for common syntax issues
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Check for unescaped quotes in RUN commands
        if line.startswith('RUN') and '"' in line:
            if line.count('"') % 2 != 0:
                issues.append(f"Line {i}: Unmatched quotes in RUN command")
    
    # Check for PaddleOCR model download
    if 'paddleocr' in content.lower() or 'download_models.py' in content:
        print("✅ PaddleOCR model download found")
    else:
        warnings.append("PaddleOCR model download not found")
    
    # Check for environment variables
    env_count = len(re.findall(r'^ENV\s+', content, re.MULTILINE))
    if env_count > 0:
        print(f"✅ Found {env_count} environment variables")
    
    # Report results
    print("\n📊 Validation Results:")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✅ Dockerfile validation passed!")
    return True

if __name__ == '__main__':
    success = validate_dockerfile()
    sys.exit(0 if success else 1)