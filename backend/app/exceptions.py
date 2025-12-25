"""
Standardized exception handlers for consistent error responses
"""
from fastapi import HTTPException, status


class DocumentNotFoundException(HTTPException):
    """Raised when a document/job is not found"""
    def __init__(self, job_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with job_id '{job_id}' not found"
        )


class OCRNotFoundException(HTTPException):
    """Raised when OCR results are not found"""
    def __init__(self, job_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"OCR results not found for job '{job_id}'. Please run OCR processing first."
        )


class ProcessingFailedException(HTTPException):
    """Raised when document processing fails"""
    def __init__(self, job_id: str, reason: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Processing failed for job '{job_id}': {reason}"
        )


class InvalidFileTypeException(HTTPException):
    """Raised when uploaded file type is not supported"""
    def __init__(self, file_type: str, allowed_types: list):
        super().__init__(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file_type}'. Allowed types: {', '.join(allowed_types)}"
        )


class FileTooLargeException(HTTPException):
    """Raised when uploaded file exceeds size limit"""
    def __init__(self, size_mb: float, max_size_mb: int):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
        )


class AuthenticationException(HTTPException):
    """Raised for authentication failures"""
    def __init__(self, detail: str = "Invalid credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationException(HTTPException):
    """Raised for authorization failures"""
    def __init__(self, detail: str = "Not authorized to access this resource"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )


class RateLimitExceededException(HTTPException):
    """Raised when rate limit is exceeded"""
    def __init__(self, detail: str = "Too many requests. Please try again later."):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": "60"}
        )
