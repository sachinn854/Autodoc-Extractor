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


def generate_verification_token() -> str:
    """Generate a random verification token"""
    return secrets.token_urlsafe(32)


def send_verification_email(email: str, token: str, frontend_url: str = None):
    """
    Send verification email to user
    
    Args:
        email: User's email address
        token: Verification token
        frontend_url: Frontend base URL (auto-detected if None)
    """
    try:
        # Auto-detect frontend URL
        if frontend_url is None:
            if os.getenv("RENDER"):
                # On Render, use the service URL
                frontend_url = f"https://{os.getenv('RENDER_SERVICE_NAME', 'autodoc-extractor')}.onrender.com"
            else:
                frontend_url = "http://localhost:3000"
        
        # Email configuration from environment variables
        SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
        SMTP_EMAIL = os.getenv("SMTP_EMAIL")
        SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
        
        # Check if SMTP is configured
        if not SMTP_EMAIL or not SMTP_PASSWORD or SMTP_EMAIL == "your-email@gmail.com":
            print(f"⚠️ SMTP not configured for {email}")
            print(f"📧 To enable email verification:")
            print(f"   1. Set SMTP_EMAIL environment variable to your Gmail")
            print(f"   2. Set SMTP_PASSWORD to your Gmail app password")
            print(f"   3. Restart the application")
            print(f"📧 For now, users can signup and use the app without verification")
            return False
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Verify Your Email - Autodoc Extractor"
        msg['From'] = SMTP_EMAIL
        msg['To'] = email
        
        # Verification link
        verify_link = f"{frontend_url}/verify-email?token={token}"
        
        # HTML email body
        html = f"""
        <html>
          <body style="font-family: Arial, sans-serif;">
            <h2>Welcome to Autodoc Extractor!</h2>
            <p>Thank you for signing up. Please verify your email address by clicking the button below:</p>
            <p style="margin: 30px 0;">
              <a href="{verify_link}" 
                 style="background-color: #4CAF50; color: white; padding: 12px 24px; 
                        text-decoration: none; border-radius: 4px; display: inline-block;">
                Verify Email Address
              </a>
            </p>
            <p>Or copy and paste this link into your browser:</p>
            <p><a href="{verify_link}">{verify_link}</a></p>
            <p style="color: #666; font-size: 12px; margin-top: 40px;">
              If you didn't create an account, please ignore this email.
            </p>
          </body>
        </html>
        """
        
        part = MIMEText(html, 'html')
        msg.attach(part)
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        
        print(f"✅ Verification email sent to {email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send verification email: {e}")
        print(f"📧 Manual verification link: {frontend_url}/verify-email?token={token}")
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
