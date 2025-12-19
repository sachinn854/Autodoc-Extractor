"""
Authentication utilities: password hashing, JWT tokens, user verification, email verification
"""
from datetime import datetime, timedelta
from typing import Optional
import jwt
import bcrypt
import secrets
import smtplib
import logging
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.database import get_db, User

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = "your-secret-key-change-this-in-production-12345"  # TODO: Move to env variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# HTTP Bearer token scheme
security = HTTPBearer()


def hash_password(password: str) -> str:
    """Hash a password for storing using bcrypt"""
    # Convert password to bytes and hash it
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash using bcrypt"""
    password_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> dict:
    """Decode and verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )



def generate_otp() -> str:
    """Generate a 6-digit OTP code"""
    return str(random.randint(100000, 999999))


def is_otp_expired(expires_at: datetime) -> bool:
    """Check if OTP has expired"""
    return datetime.utcnow() > expires_at


def get_otp_expiry_time() -> datetime:
    """Get OTP expiry time (10 minutes from now)"""
    return datetime.utcnow() + timedelta(minutes=10)




def send_otp_email(email: str, otp_code: str) -> bool:
    """
    Send OTP verification email to user
    
    Args:
        email: User's email address
        otp_code: 6-digit OTP code
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Email configuration from environment variables
        SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
        SMTP_EMAIL = os.getenv("SMTP_EMAIL")
        SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
        
        # Check if SMTP is configured
        if not SMTP_EMAIL or not SMTP_PASSWORD or SMTP_EMAIL == "your-email@gmail.com":
            print(f"⚠️ SMTP not configured for {email}")
            print(f"📧 OTP Code for manual verification: {otp_code}")
            print(f"📧 To enable email OTP:")
            print(f"   1. Set SMTP_EMAIL environment variable to your Gmail")
            print(f"   2. Set SMTP_PASSWORD to your Gmail app password")
            print(f"   3. Restart the application")
            return False
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Your OTP Code - Autodoc Extractor"
        msg['From'] = SMTP_EMAIL
        msg['To'] = email
        
        # HTML email body
        html = f"""
        <html>
          <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="text-align: center; margin-bottom: 30px;">
              <h1 style="color: #3b82f6;">📄 Autodoc Extractor</h1>
              <h2 style="color: #333;">Email Verification</h2>
            </div>
            
            <div style="background: #f8fafc; padding: 30px; border-radius: 10px; text-align: center;">
              <p style="font-size: 18px; margin-bottom: 20px;">Your verification code is:</p>
              
              <div style="background: #3b82f6; color: white; font-size: 32px; font-weight: bold; 
                          padding: 20px; border-radius: 8px; letter-spacing: 8px; margin: 20px 0;">
                {otp_code}
              </div>
              
              <p style="color: #666; font-size: 14px; margin-top: 20px;">
                This code will expire in <strong>10 minutes</strong>
              </p>
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background: #fff3cd; border-radius: 8px;">
              <p style="margin: 0; color: #856404; font-size: 14px;">
                <strong>Security Note:</strong> Never share this code with anyone. 
                Our team will never ask for your OTP code.
              </p>
            </div>
            
            <div style="text-align: center; margin-top: 30px; color: #666; font-size: 12px;">
              <p>If you didn't request this code, please ignore this email.</p>
              <p>© 2024 Autodoc Extractor - AI Document Processing</p>
            </div>
          </body>
        </html>
        """
        
        part = MIMEText(html, 'html')
        msg.attach(part)
        
        # Send email with debug logging
        print(f"🔄 Attempting to send email to {email}")
        print(f"📧 SMTP Server: {SMTP_SERVER}:{SMTP_PORT}")
        print(f"📧 From: {SMTP_EMAIL}")
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            print("🔄 Connecting to SMTP server...")
            server.starttls()
            print("🔄 Starting TLS...")
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            print("🔄 Logged in successfully...")
            server.send_message(msg)
            print("🔄 Message sent...")
        
        print(f"✅ OTP email sent to {email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send OTP email: {e}")
        print(f"📧 Manual OTP for {email}: {otp_code}")
        return False



def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get current authenticated user from JWT token
    Usage: current_user: User = Depends(get_current_user)
    """
    token = credentials.credentials
    payload = decode_token(token)
    
    user_id: int = payload.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # For now, allow unverified users to upload (email verification optional)
    # TODO: Enable strict verification when SMTP is properly configured
    if not user.is_verified:
        logger.warning(f"⚠️ Unverified user accessing system: {user.email}")
    
    return user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password"""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user
