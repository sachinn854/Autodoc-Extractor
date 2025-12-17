"""
Simple database migration to add email verification fields
Run this once to update existing database
"""
import sqlite3
import os

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'autodoc.db')

print(f"Migrating database: {DB_PATH}")

try:
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if columns already exist
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Add is_verified column if not exists
    if 'is_verified' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN is_verified BOOLEAN DEFAULT 0")
        print("✅ Added is_verified column")
    else:
        print("⏭️  is_verified column already exists")
    
    # Add verification_token column if not exists
    if 'verification_token' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN verification_token TEXT")
        print("✅ Added verification_token column")
    else:
        print("⏭️  verification_token column already exists")
    
    # Commit changes
    conn.commit()
    print("\n✅ Database migration completed successfully!")
    
    # Optional: Mark existing users as verified
    cursor.execute("UPDATE users SET is_verified = 1 WHERE is_verified IS NULL OR is_verified = 0")
    affected = cursor.rowcount
    conn.commit()
    
    if affected > 0:
        print(f"✅ Marked {affected} existing user(s) as verified")
    
except Exception as e:
    print(f"❌ Migration failed: {e}")
finally:
    if conn:
        conn.close()

print("\n🎉 Migration complete! Restart your backend server.")
