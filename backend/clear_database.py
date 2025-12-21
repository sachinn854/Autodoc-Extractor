#!/usr/bin/env python3
"""
Simple script to clear database and cache
"""
import os
import sys
import shutil
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.database import SessionLocal, User, Document

def clear_everything():
    """Clear database, cache, and temp files"""
    print("🧹 Starting complete cleanup...")
    
    # 1. Clear database
    print("1️⃣ Clearing database...")
    db = SessionLocal()
    try:
        # Count before deletion
        users_count = db.query(User).count()
        docs_count = db.query(Document).count()
        
        # Delete all
        db.query(Document).delete()
        db.query(User).delete()
        db.commit()
        
        print(f"   ✅ Deleted {users_count} users and {docs_count} documents")
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        db.rollback()
    finally:
        db.close()
    
    # 2. Clear job status file
    print("2️⃣ Clearing job status...")
    job_files = [
        Path("tmp/job_status.json"),
        Path("backend/tmp/job_status.json")
    ]
    for job_file in job_files:
        if job_file.exists():
            job_file.unlink()
            print(f"   ✅ Deleted {job_file}")
    
    # 3. Clear temp directories
    print("3️⃣ Clearing temp files...")
    temp_dirs = [
        Path("tmp"),
        Path("backend/tmp"),
        Path("backend/data/history")
    ]
    
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                print(f"   ✅ Cleared {temp_dir}")
            except Exception as e:
                print(f"   ⚠️ Could not clear {temp_dir}: {e}")
    
    # 4. Recreate essential directories
    print("4️⃣ Recreating directories...")
    essential_dirs = [
        Path("tmp/uploads"),
        Path("tmp/preprocessed"),
        Path("tmp/results"),
        Path("backend/tmp/uploads"),
        Path("backend/tmp/preprocessed"),
        Path("backend/tmp/results"),
        Path("backend/data")
    ]
    
    for dir_path in essential_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created {dir_path}")
    
    print("\n🎉 Complete cleanup finished!")
    print("✅ Database cleared")
    print("✅ Cache cleared") 
    print("✅ Temp files cleared")
    print("✅ Directories recreated")
    print("\n👍 Ready for fresh testing!")

if __name__ == "__main__":
    clear_everything()