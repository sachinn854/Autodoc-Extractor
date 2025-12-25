"""
Configuration Management
Centralized settings loaded from environment variables
Compatible with Pydantic 1.x
"""

import os
from functools import lru_cache
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseModel):
    """Application settings loaded from environment variables"""
    
    # Application
    app_name: str = os.getenv("APP_NAME", "AutoDoc Extractor API")
    version: str = os.getenv("VERSION", "2.0.0")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    environment: str = os.getenv("ENVIRONMENT", "production")
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/autodoc.db")
    
    # CORS
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    
    # File Upload
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    allowed_file_extensions: List[str] = [".pdf", ".jpg", ".jpeg", ".png"]
    
    # OCR
    ocr_language: str = os.getenv("OCR_LANGUAGE", "eng")
    ocr_dpi: int = int(os.getenv("OCR_DPI", "300"))
    
    # Rate Limiting
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    upload_rate_limit: str = os.getenv("UPLOAD_RATE_LIMIT", "10/minute")
    login_rate_limit: str = os.getenv("LOGIN_RATE_LIMIT", "5/minute")
    general_rate_limit: str = os.getenv("GENERAL_RATE_LIMIT", "100/minute")
    
    # SMTP (Optional)
    smtp_server: Optional[str] = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_email: Optional[str] = os.getenv("SMTP_EMAIL", "")
    smtp_password: Optional[str] = os.getenv("SMTP_PASSWORD", "")
    
    # Admin
    admin_verification_key: str = os.getenv("ADMIN_VERIFICATION_KEY", "admin123")
    
    class Config:
        # Pydantic 1.x configuration
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

