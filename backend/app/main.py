"""
Phase 7: Complete FastAPI Backend & Orchestration
Full document processing pipeline with proper API structure
"""

import os
import uuid
import json
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import logging

# Import our modules
from app.preprocessing import preprocess_document, delete_tmp_job
from app.ocr_engine import process_document_ocr
from app.table_detector import detect_tables
from app.parser import process_multiple_tables, process_document_to_business_schema
from app.ml_models import ExpenseCategorizer, predict_category, detect_anomalies
from app.insights import generate_spending_insights, load_historical_data, save_to_history

# Import database and auth modules
from app.database import Base, engine, get_db, init_db, User, Document
from app.auth import (
    hash_password, authenticate_user, create_access_token, 
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES,
    generate_verification_token, send_verification_email
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models for API Responses
class UploadResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    process_url: str
    message: str

class ProcessResponse(BaseModel):
    job_id: str
    status: str
    result_url: Optional[str] = None
    error: Optional[str] = None
    message: str

class ResultResponse(BaseModel):
    job_id: str
    status: str
    extracted: Optional[Dict[str, Any]] = None
    insights: Optional[Dict[str, Any]] = None
    csv_url: Optional[str] = None
    files: Optional[Dict[str, str]] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None

class StatusResponse(BaseModel):
    job_id: str
    status: str  # "uploaded", "processing", "completed", "failed"
    progress: Optional[str] = None
    error: Optional[str] = None
    estimated_time_remaining: Optional[int] = None

# FastAPI app initialization
app = FastAPI(
    title="Autodoc Extractor API",
    description="Complete document processing pipeline with ML insights",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Job status tracking (in production, use Redis/database)
JOB_STATUS = {}

def load_job_status():
    """Load job status from file"""
    global JOB_STATUS
    status_file = Path("tmp/job_status.json")
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                JOB_STATUS = json.load(f)
        except:
            JOB_STATUS = {}

def save_job_status():
    """Save job status to file"""
    status_file = Path("tmp/job_status.json")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, 'w') as f:
        json.dump(JOB_STATUS, f, indent=2)

def update_job_status(job_id: str, status: str, progress: str = None, error: str = None):
    """Update job status for tracking"""
    JOB_STATUS[job_id] = {
        "status": status,
        "progress": progress,
        "error": error,
        "updated_at": datetime.now().isoformat()
    }
    save_job_status()

# Load existing job status on startup
load_job_status()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    init_db()
    logger.info("Database initialized")


# ==================== AUTH ENDPOINTS ====================

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    created_at: datetime
    is_active: bool


@app.post("/auth/signup", response_model=TokenResponse)
async def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """
    Create a new user account
    
    Args:
        request: Signup details (email, password, full_name)
        
    Returns:
        JWT token and user info
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Generate verification token
    verification_token = generate_verification_token()
    
    # Create new user
    hashed_password = hash_password(request.password)
    new_user = User(
        email=request.email,
        password_hash=hashed_password,
        full_name=request.full_name,
        verification_token=verification_token,
        is_verified=False  # Email not verified yet
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Send verification email
    try:
        send_verification_email(new_user.email, verification_token)
    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
    
    # Create access token
    access_token = create_access_token(
        data={"user_id": new_user.id, "email": new_user.email}
    )
    
    logger.info(f"New user registered: {new_user.email} (unverified)")
    
    return TokenResponse(
        access_token=access_token,
        user={
            "id": new_user.id,
            "email": new_user.email,
            "full_name": new_user.full_name,
            "created_at": new_user.created_at.isoformat(),
            "is_active": new_user.is_active
        }
    )


@app.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Login with email and password
    
    Args:
        request: Login credentials (email, password)
        
    Returns:
        JWT token and user info
    """
    # Authenticate user
    user = authenticate_user(db, request.email, request.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    
    # Check if email is verified
    if not user.is_verified:
        raise HTTPException(
            status_code=403,
            detail="Email not verified. Please check your email for verification link."
        )
    
    # Create access token
    access_token = create_access_token(
        data={"user_id": user.id, "email": user.email}
    )
    
    logger.info(f"User logged in: {user.email}")
    
    return TokenResponse(
        access_token=access_token,
        user={
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "created_at": user.created_at.isoformat(),
            "is_active": user.is_active
        }
    )


@app.get("/auth/verify-email")
async def verify_email(token: str, db: Session = Depends(get_db)):
    """
    Verify user's email address using verification token
    
    Args:
        token: Email verification token
        
    Returns:
        Success message
    """
    # Find user by verification token
    user = db.query(User).filter(User.verification_token == token).first()
    
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired verification token"
        )
    
    # Mark user as verified
    user.is_verified = True
    user.verification_token = None  # Clear token after use
    db.commit()
    
    logger.info(f"Email verified for user: {user.email}")
    
    return {
        "message": "Email verified successfully! You can now login.",
        "email": user.email
    }


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user info
    
    Returns:
        Current user details
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        created_at=current_user.created_at,
        is_active=current_user.is_active
    )


# ==================== DOCUMENT ENDPOINTS ====================


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> UploadResponse:
    """
    Upload a document and create a processing job.
    Requires authentication.
    
    Args:
        file: Uploaded file (PDF, JPG, PNG)
        current_user: Authenticated user (from JWT token)
        
    Returns:
        UploadResponse with job ID and processing URL
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    logger.info(f"Creating upload job: {job_id} for user: {current_user.email}")
    
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png'}
        file_ext = Path(file.filename or "unknown").suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
            )
        
        # Create upload directory
        script_dir = Path(__file__).parent.parent
        upload_dir = script_dir / "tmp" / "uploads" / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        input_file_path = upload_dir / f"raw_input{file_ext}"
        
        with open(input_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            buffer.flush()  # Ensure all data is written
        
        # Ensure file is fully written and handle is closed
        import os
        os.sync() if hasattr(os, 'sync') else None
        
        # Create document record in database
        file_size = os.path.getsize(input_file_path)
        new_document = Document(
            user_id=current_user.id,
            job_id=job_id,
            filename=file.filename or "unknown",
            status="processing"
        )
        db.add(new_document)
        db.commit()
        db.refresh(new_document)
        
        # Initialize job status
        update_job_status(job_id, "uploaded", "File uploaded successfully")
        
        logger.info(f"File uploaded successfully: {input_file_path}, Document ID: {new_document.id}")
        
        # Start automatic processing in background
        # Create a new database session for background task
        from app.database import SessionLocal
        background_db = SessionLocal()
        background_tasks.add_task(run_complete_pipeline, job_id, False, 300, "en", True, background_db)
        
        update_job_status(job_id, "processing", "Starting document processing...")
        
        return UploadResponse(
            job_id=job_id,
            status="processing",
            filename=file.filename or "unknown",
            process_url=f"/process/{job_id}",
            message="File uploaded and processing started automatically."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed for job {job_id}: {e}")
        
        # Clean up on failure
        try:
            delete_tmp_job(job_id)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/process/{job_id}", response_model=ProcessResponse)
async def process_document_pipeline(
    job_id: str,
    background_tasks: BackgroundTasks,
    do_threshold: bool = False,
    dpi: int = 300,
    lang: str = "en",
    normalize_coords: bool = True
) -> ProcessResponse:
    """
    Start complete document processing pipeline (Phases 2-6).
    
    Args:
        job_id: Job identifier from upload
        background_tasks: FastAPI background tasks
        do_threshold: Whether to apply thresholding
        dpi: DPI for PDF conversion
        lang: OCR language code
        normalize_coords: Whether to normalize bounding boxes
        
    Returns:
        ProcessResponse with status and result URL
    """
    logger.info(f"Starting pipeline processing for job: {job_id}")
    
    try:
        # Validate job exists
        script_dir = Path(__file__).parent.parent
        upload_dir = script_dir / "tmp" / "uploads" / job_id
        
        if not upload_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Check if already processing
        if job_id in JOB_STATUS and JOB_STATUS[job_id]["status"] == "processing":
            return ProcessResponse(
                job_id=job_id,
                status="processing",
                message="Job is already being processed"
            )
        
        # Start background processing
        update_job_status(job_id, "processing", "Pipeline started")
        background_tasks.add_task(
            run_complete_pipeline,
            job_id, do_threshold, dpi, lang, normalize_coords
        )
        
        logger.info(f"Background processing started for job: {job_id}")
        
        return ProcessResponse(
            job_id=job_id,
            status="processing",
            result_url=f"/result/{job_id}",
            message="Processing started. Use result_url to check status and get results."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start processing for job {job_id}: {e}")
        update_job_status(job_id, "failed", error=str(e))
        
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")
        
        # Step 1: Preprocess the document
        processed_paths = preprocess_document(
            input_path=str(input_file_path),
            job_id=job_id,
            dpi=dpi,
            do_threshold=do_threshold
        )
        
        logger.info(f"Preprocessing completed: {len(processed_paths)} pages")
        
        # Step 2: Run OCR on preprocessed images
        ocr_results = process_document_ocr(
            preprocessed_image_paths=processed_paths,
            job_id=job_id,
            lang=lang,
            normalize_coords=normalize_coords
        )
        
        logger.info(f"OCR completed: {ocr_results['total_tokens']} tokens extracted")
        
        # Step 3: Table detection and parsing (Phase 4)
        table_results = process_tables_for_job(job_id, processed_paths, ocr_results)
        logger.info(f"Table processing completed: {len(table_results.get('tables', []))} tables found")
        
        # Step 4: Business schema extraction (Phase 5)
        try:
            business_schema = process_document_to_business_schema(job_id)
            logger.info(f"Business schema extraction completed: {business_schema.get('item_count', 0)} items extracted")
        except Exception as e:
            logger.warning(f"Business schema extraction failed: {e}")
            business_schema = {"error": str(e), "status": "failed"}
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "job_id": job_id,
                "preprocessing": {
                    "processed_files": processed_paths,
                    "total_pages": len(processed_paths)
                },
                "ocr": {
                    "total_tokens": ocr_results['total_tokens'],
                    "total_pages": ocr_results['total_pages'],
                    "output_file": ocr_results['output_file'],
                    "language": lang,
                    "normalized_coordinates": normalize_coords
                },
                "tables": table_results,
                "business_schema": business_schema,
                "message": f"Successfully processed {len(processed_paths)} pages with {ocr_results['total_tokens']} tokens, {len(table_results.get('tables', []))} tables, and {business_schema.get('item_count', 0)} extracted items"
            }
        )
        
    except Exception as e:
        logger.error(f"Complete document processing failed for job {job_id}: {e}")
        
        # Clean up on failure
        try:
            delete_tmp_job(job_id)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


async def run_complete_pipeline(
    job_id: str, 
    do_threshold: bool = False,
    dpi: int = 300,
    lang: str = "en", 
    normalize_coords: bool = True,
    db: Session = None  # Optional database session for updating document
):
    """
    Background task: Run complete processing pipeline (Phases 2-6)
    
    Args:
        job_id: Job identifier
        do_threshold: Whether to apply thresholding
        dpi: DPI for PDF conversion
        lang: OCR language code
        normalize_coords: Whether to normalize bounding boxes
        db: Optional database session for updating document record
    """
    start_time = datetime.now()
    logger.info(f"Starting complete pipeline for job {job_id}")
    
    try:
        # Get file paths
        script_dir = Path(__file__).parent.parent
        upload_dir = script_dir / "tmp" / "uploads" / job_id
        results_dir = script_dir / "tmp" / "results" / job_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Find uploaded file
        input_files = list(upload_dir.glob("raw_input.*"))
        if not input_files:
            raise Exception("No input file found")
        
        input_file_path = input_files[0]
        
        # Phase 2: Preprocessing
        update_job_status(job_id, "processing", "Phase 2: Preprocessing document")
        processed_paths = preprocess_document(
            input_path=str(input_file_path),
            job_id=job_id,
            dpi=dpi,
            do_threshold=do_threshold
        )
        logger.info(f"Phase 2 completed: {len(processed_paths)} pages preprocessed")
        
        # Phase 3: OCR
        logger.info(f"Starting Phase 3: OCR processing with {len(processed_paths)} images")
        logger.info(f"Preprocessed image paths: {processed_paths}")
        update_job_status(job_id, "processing", "Phase 3: Running OCR")
        
        # Test OCR import
        logger.info("Testing OCR import...")
        try:
            from app.ocr_engine import process_document_ocr
            logger.info("OCR import successful")
        except Exception as import_err:
            logger.error(f"OCR import failed: {import_err}")
            raise
            
        try:
            logger.info("Calling process_document_ocr...")
            logger.info(f"Paths to process: {processed_paths}")
            
            ocr_results = process_document_ocr(
                preprocessed_image_paths=processed_paths,
                job_id=job_id,
                lang=lang,
                normalize_coords=normalize_coords
            )
            
            logger.info(f"process_document_ocr returned successfully!")
            logger.info(f"Phase 3 completed: {ocr_results.get('total_tokens', 0)} tokens extracted")
            
        except Exception as e:
            logger.error(f"Phase 3 OCR processing failed: {e}")
            logger.error(f"OCR error traceback: {traceback.format_exc()}")
            
            # GUARANTEE: Save empty OCR results to prevent FileNotFoundError downstream
            from app.ocr_engine import save_ocr_output
            empty_ocr_results = {"pages": [], "total_pages": 0, "total_tokens": 0}
            save_ocr_output(job_id, empty_ocr_results)
            logger.info("Saved empty OCR results to prevent downstream errors")
            
            # Continue with empty OCR data instead of failing
            ocr_results = empty_ocr_results
            logger.info("Continuing pipeline with empty OCR data")
        
        # Phase 4: Table Detection & Parsing
        update_job_status(job_id, "processing", "Phase 4: Detecting tables and structure")
        table_results = process_tables_for_job(job_id, processed_paths, ocr_results)
        logger.info(f"Phase 4 completed: {len(table_results.get('tables', []))} tables found")
        
        # Phase 5: Business Schema Extraction
        update_job_status(job_id, "processing", "Phase 5: Extracting business schema")
        business_results = process_document_to_business_schema(job_id)
        logger.info(f"Phase 5 completed: Business schema extracted")
        
        # Phase 6: ML Insights
        update_job_status(job_id, "processing", "Phase 6: Generating ML insights")
        insights_results = process_insights_for_job(job_id, business_results, include_historical=True)
        logger.info(f"Phase 6 completed: ML insights generated")
        
        # Save processing summary
        processing_time = (datetime.now() - start_time).total_seconds()
        summary = {
            "job_id": job_id,
            "status": "completed",
            "processing_time_seconds": processing_time,
            "phases_completed": ["preprocessing", "ocr", "tables", "business_schema", "insights"],
            "files_created": {
                "preprocessed_pages": len(processed_paths),
                "ocr_tokens": ocr_results.get('total_tokens', 0),
                "tables_detected": len(table_results.get('tables', [])),
                "items_extracted": len(business_results.get('items', [])),
                "categories_assigned": len(insights_results.get('categories', []))
            },
            "completed_at": datetime.now().isoformat()
        }
        
        summary_file = results_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Update database if session provided
        if db:
            try:
                doc = db.query(Document).filter(Document.job_id == job_id).first()
                if doc:
                    doc.status = "completed"
                    # Save extracted data as JSON string
                    doc.extracted_data = json.dumps(business_results, ensure_ascii=False)
                    # Save summary fields
                    doc.vendor = business_results.get('vendor', '')
                    doc.date = business_results.get('date', '')
                    doc.total_amount = str(business_results.get('total', ''))
                    doc.item_count = len(business_results.get('items', []))
                    doc.updated_at = datetime.utcnow()
                    db.commit()
                    logger.info(f"Document record updated in database for job {job_id}")
            except Exception as db_error:
                logger.error(f"Failed to update database for job {job_id}: {db_error}")
        
        # Update final status
        update_job_status(job_id, "completed", f"Pipeline completed in {processing_time:.1f}s")
        
        logger.info(f"Complete pipeline finished for job {job_id} in {processing_time:.1f}s")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Pipeline failed for job {job_id}: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Update database if session provided
        if db:
            try:
                doc = db.query(Document).filter(Document.job_id == job_id).first()
                if doc:
                    doc.status = "failed"
                    doc.error_message = error_msg
                    doc.updated_at = datetime.utcnow()
                    db.commit()
                    logger.info(f"Document record marked as failed in database for job {job_id}")
            except Exception as db_error:
                logger.error(f"Failed to update database error for job {job_id}: {db_error}")
        
        # Save error details
        try:
            script_dir = Path(__file__).parent.parent
            results_dir = script_dir / "tmp" / "results" / job_id
            results_dir.mkdir(parents=True, exist_ok=True)
            
            error_file = results_dir / "error.log"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Pipeline Error for Job {job_id}\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"Error: {error_msg}\n\n")
                f.write("Full Traceback:\n")
                f.write(traceback.format_exc())
        except:
            pass
        
        update_job_status(job_id, "failed", error=error_msg)


@app.post("/process-tables/{job_id}")
async def process_tables_endpoint(job_id: str) -> JSONResponse:
    """
    Process tables for an existing job (Phase 4).
    Requires that preprocessing and OCR have already been completed.
    
    Args:
        job_id: Existing job ID with completed OCR
        
    Returns:
        JSON response with detected tables and structured data
    """
    logger.info(f"Starting table processing for job: {job_id}")
    
    try:
        # Check if job exists and get paths
        script_dir = Path(__file__).parent.parent
        preprocessed_dir = script_dir / "tmp" / "preprocessed" / job_id
        results_dir = script_dir / "tmp" / "results" / job_id
        
        if not preprocessed_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or preprocessing not completed")
        
        if not results_dir.exists():
            raise HTTPException(status_code=404, detail=f"OCR not completed for job {job_id}")
        
        # Load OCR results
        ocr_file = results_dir / "ocr_results.json"
        if not ocr_file.exists():
            raise HTTPException(status_code=404, detail=f"OCR results not found for job {job_id}")
        
        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_results = json.load(f)
        
        # Get preprocessed image paths
        processed_paths = []
        for page_file in sorted(preprocessed_dir.glob("page_*.jpg")):
            processed_paths.append(str(page_file))
        
        if not processed_paths:
            raise HTTPException(status_code=404, detail=f"No preprocessed images found for job {job_id}")
        
        # Process tables
        table_results = process_tables_for_job(job_id, processed_paths, ocr_results)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "job_id": job_id,
                "tables": table_results,
                "message": f"Table processing completed: {len(table_results.get('tables', []))} tables found"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Table processing failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Table processing failed: {str(e)}")


@app.post("/extract-business-schema/{job_id}")
async def extract_business_schema_endpoint(job_id: str) -> JSONResponse:
    """
    Extract business schema for an existing job (Phase 5).
    Requires that table processing (Phase 4) has already been completed.
    
    Args:
        job_id: Existing job ID with completed table processing
        
    Returns:
        JSON response with business schema (vendor, items, totals)
    """
    logger.info(f"Starting business schema extraction for job: {job_id}")
    
    try:
        # Process to business schema
        business_schema = process_document_to_business_schema(job_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "job_id": job_id,
                "business_schema": business_schema,
                "message": f"Business schema extraction completed: {business_schema.get('item_count', 0)} items extracted"
            }
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Business schema extraction failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Business schema extraction failed: {str(e)}")


@app.post("/process-insights/{job_id}")
async def process_insights_endpoint(job_id: str, include_historical: bool = True) -> JSONResponse:
    """
    Generate ML insights for an existing job (Phase 6).
    
    Args:
        job_id: Existing job ID with completed business schema
        include_historical: Whether to include historical data for better insights
        
    Returns:
        JSON response with ML insights and analytics
    """
    logger.info(f"Starting insights generation for job: {job_id}")
    
    try:
        script_dir = Path(__file__).parent.parent
        results_dir = script_dir / "tmp" / "results" / job_id
        
        if not results_dir.exists():
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Load business schema results
        schema_file = results_dir / "extracted.json"
        if not schema_file.exists():
            raise HTTPException(status_code=404, detail=f"Business schema not found for job {job_id}. Run Phase 5 first.")
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        # Process insights
        insights_results = process_insights_for_job(job_id, extracted_data, include_historical)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "job_id": job_id,
                "insights": insights_results,
                "message": "ML insights generated successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Insights generation failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")


def process_insights_for_job(job_id: str, extracted_data: dict, include_historical: bool = True) -> dict:
    """Generate ML insights for business schema data"""
    logger.info(f"Processing ML insights for job {job_id}")
    
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "tmp" / "results" / job_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Enhance items with category predictions
        categorizer = ExpenseCategorizer()
        enhanced_items = []
        
        for item in extracted_data.get('items', []):
            description = item.get('description', '')
            existing_category = item.get('category', 'Other')
            
            if description:
                category_result = categorizer.predict_category(description)
                
                # Use ML category if confidence is high or no existing category
                if category_result['confidence'] > 0.7 or existing_category == 'Other':
                    item['category'] = category_result['category']
                    item['category_confidence'] = category_result['confidence']
                else:
                    item['category_confidence'] = 0.5
            
            enhanced_items.append(item)
        
        # Step 2: Detect anomalies
        anomalies = detect_anomalies(
            enhanced_items, 
            extracted_data.get('vendor', ''), 
            extracted_data.get('date', '')
        )
        
        # Step 3: Generate spending insights
        enhanced_data = extracted_data.copy()
        enhanced_data['items'] = enhanced_items
        
        documents_for_insights = [enhanced_data]
        
        # Load historical data if requested
        if include_historical:
            try:
                historical_docs = load_historical_data()
                documents_for_insights.extend(historical_docs)
                logger.info(f"Included {len(historical_docs)} historical documents")
            except Exception as e:
                logger.warning(f"Failed to load historical data: {e}")
        
        spending_insights = generate_spending_insights(documents_for_insights)
        
        # Step 4: Compile final insights
        final_insights = {
            "job_id": job_id,
            "categories": [
                {
                    "item": item['description'],
                    "category": item['category'],
                    "confidence": item.get('category_confidence', 0.5)
                }
                for item in enhanced_items if item.get('description')
            ],
            "anomalies": anomalies,
            "spending_insights": spending_insights,
            "processing_stats": {
                "items_categorized": len(enhanced_items),
                "anomalies_detected": len(anomalies),
                "historical_docs_included": len(documents_for_insights) - 1
            }
        }
        
        # Step 5: Save insights
        output_file = results_dir / "insights.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_insights, f, indent=2, ensure_ascii=False, default=str)
        
        # Save enhanced data to history
        try:
            save_to_history(enhanced_data)
        except Exception as e:
            logger.warning(f"Failed to save to history: {e}")
        
        final_insights['output_file'] = str(output_file)
        return final_insights
        
    except Exception as e:
        logger.error(f"Insights processing failed: {e}")
        return {
            "job_id": job_id,
            "categories": [],
            "anomalies": [],
            "spending_insights": {},
            "error": str(e)
        }


def process_tables_for_job(job_id: str, processed_paths: List[str], ocr_results: dict) -> dict:
    """
    Process tables for a job with preprocessed images and OCR results
    
    Args:
        job_id: Job identifier
        processed_paths: List of preprocessed image paths
        ocr_results: OCR results dictionary
        
    Returns:
        Dictionary with table detection and parsing results
    """
    logger.info(f"Processing tables for job {job_id}")
    
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "tmp" / "results" / job_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_tables = []
    page_table_count = 0
    
    try:
        # Process each page
        for page_idx, image_path in enumerate(processed_paths):
            page_num = page_idx + 1
            logger.info(f"Processing tables for page {page_num}: {image_path}")
            
            # Step 1: Detect tables in the image
            logger.info(f"Page {page_num}: Starting table detection on {image_path}")
            detected_tables = detect_tables(image_path, confidence_threshold=0.25)  # Lowered threshold
            logger.info(f"Page {page_num}: Found {len(detected_tables)} table candidates")
            
            # Log each detected table
            for i, table in enumerate(detected_tables):
                logger.info(f"  Table {i+1}: bbox={table['bbox']}, confidence={table['confidence']}, label={table['label']}")
            
            # DEBUG: Save table detection visualization
            if detected_tables:
                try:
                    from app.table_detector import TableDetector
                    debug_detector = TableDetector()
                    debug_dir = results_dir / "debug_tables"
                    debug_dir.mkdir(exist_ok=True)
                    viz_path = debug_detector.visualize_tables(
                        image_path, 
                        detected_tables, 
                        str(debug_dir / f"page_{page_num:02d}_tables.jpg")
                    )
                    logger.info(f"Table debug visualization saved: {viz_path}")
                except Exception as viz_e:
                    logger.warning(f"Failed to save table visualization: {viz_e}")
            
            if not detected_tables:
                logger.warning(f"Page {page_num}: No tables detected, checking OCR fallback")
                continue
            
            # Step 2: Get OCR tokens for this page
            page_tokens = []
            
            # OCR results.pages is an array, find the page by index
            ocr_pages = ocr_results.get('pages', [])
            logger.info(f"Page {page_num}: Looking for page in {len(ocr_pages)} OCR pages")
            
            if isinstance(ocr_pages, list):
                # Array format: [{page_index: 0, tokens: [...]}, ...]
                for page_data in ocr_pages:
                    if page_data.get('page_index') == page_idx:  # page_idx = page_num - 1
                        page_tokens = page_data.get('tokens', [])
                        logger.info(f"Page {page_num}: Found {len(page_tokens)} OCR tokens (array format)")
                        break
            elif isinstance(ocr_pages, dict):
                # Dict format: {page_01: {tokens: [...]}, ...}  
                page_key = f"page_{page_num:02d}"
                if page_key in ocr_pages:
                    page_tokens = ocr_pages[page_key].get('tokens', [])
                    logger.info(f"Page {page_num}: Found {len(page_tokens)} OCR tokens (dict format)")
            
            if not page_tokens:
                logger.warning(f"Page {page_num}: No OCR tokens found, skipping table parsing")
                continue
            
            # Step 3: Parse table structure for each detected table
            structured_tables = process_multiple_tables(page_tokens, detected_tables)
            
            # Add page information to each table
            for table_idx, structured_table in enumerate(structured_tables):
                structured_table['page_number'] = page_num
                structured_table['table_index'] = table_idx
                structured_table['image_path'] = image_path
                all_tables.append(structured_table)
                page_table_count += 1
            
            logger.info(f"Page {page_num}: Parsed {len(structured_tables)} tables")
        
        # Save table results
        table_output = {
            "job_id": job_id,
            "total_tables": len(all_tables),
            "total_pages_processed": len(processed_paths),
            "tables": all_tables
        }
        
        # Save to file
        output_file = results_dir / "tables.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(table_output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Table results saved to: {output_file}")
        
        # Add output file path to results
        table_output['output_file'] = str(output_file)
        
        return table_output
        
    except Exception as e:
        logger.error(f"Table processing failed: {e}")
        return {
            "job_id": job_id,
            "total_tables": 0,
            "total_pages_processed": 0,
            "tables": [],
            "error": str(e)
        }


@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str) -> JSONResponse:
    """
    Clean up temporary files for a completed job.
    
    Args:
        job_id: Job identifier to clean up
        
    Returns:
        JSON response confirming cleanup
    """
    try:
        delete_tmp_job(job_id)
        
        # Remove from job status tracking
        if job_id in JOB_STATUS:
            del JOB_STATUS[job_id]
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Cleaned up job {job_id}"
            }
        )
    except Exception as e:
        logger.error(f"Cleanup failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "autodoc-extractor",
        "version": "2.0.0",
        "active_jobs": len(JOB_STATUS),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/jobs")
async def list_jobs():
    """List all active jobs."""
    return {
        "active_jobs": len(JOB_STATUS),
        "jobs": {
            job_id: {
                "status": info["status"],
                "progress": info.get("progress"),
                "updated_at": info.get("updated_at")
            }
            for job_id, info in JOB_STATUS.items()
        }
    }


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str) -> StatusResponse:
    """
    Get the current status of a processing job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        StatusResponse with current job status
    """
    if job_id not in JOB_STATUS:
        raise HTTPException(
            status_code=404, 
            detail=f"Job not found: {job_id}"
        )
    
    job_info = JOB_STATUS[job_id]
    return StatusResponse(
        job_id=job_id,
        status=job_info["status"],
        progress=job_info.get("progress"),
        error=job_info.get("error")
    )


# ==================== DOCUMENT HISTORY ENDPOINTS ====================

class DocumentListItem(BaseModel):
    id: int
    job_id: str
    filename: str
    status: str
    created_at: datetime
    updated_at: datetime
    items_count: Optional[int] = None
    vendor: Optional[str] = None
    total_amount: Optional[str] = None

class DocumentsListResponse(BaseModel):
    documents: List[DocumentListItem]
    total: int


@app.get("/my-documents", response_model=DocumentsListResponse)
async def get_my_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 50
):
    """
    Get authenticated user's document history
    
    Args:
        skip: Number of documents to skip (pagination)
        limit: Maximum number of documents to return
        
    Returns:
        List of user's documents with metadata
    """
    # Query user's documents
    documents_query = db.query(Document).filter(
        Document.user_id == current_user.id
    ).order_by(Document.created_at.desc())
    
    total = documents_query.count()
    documents = documents_query.offset(skip).limit(limit).all()
    
    # Format response
    document_list = []
    for doc in documents:
        items_count = None
        if doc.extracted_data:
            try:
                import json
                data = json.loads(doc.extracted_data) if isinstance(doc.extracted_data, str) else doc.extracted_data
                items = data.get('items', [])
                items_count = len(items) if isinstance(items, list) else None
            except:
                pass
        
        document_list.append(DocumentListItem(
            id=doc.id,
            job_id=doc.job_id,
            filename=doc.filename,
            status=doc.status,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            items_count=items_count,
            vendor=doc.vendor,
            total_amount=doc.total_amount
        ))
    
    return DocumentsListResponse(
        documents=document_list,
        total=total
    )


class SaveExtractedDataRequest(BaseModel):
    items: List[Dict[str, Any]]

class SaveExtractedDataResponse(BaseModel):
    success: bool
    message: str
    items_saved: int


@app.patch("/result/{job_id}/save", response_model=SaveExtractedDataResponse)
async def save_extracted_data(
    job_id: str,
    request: SaveExtractedDataRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Save edited extracted data for a document
    
    Args:
        job_id: Job identifier
        request: Updated items list
        
    Returns:
        Success status and number of items saved
    """
    # Find document
    doc = db.query(Document).filter(
        Document.job_id == job_id,
        Document.user_id == current_user.id  # Ensure user owns document
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Document not found or you don't have permission to edit it"
        )
    
    # Update extracted data
    if not doc.extracted_data:
        doc.extracted_data = {}
    
    doc.extracted_data['items'] = request.items
    doc.updated_at = datetime.utcnow()
    
    db.commit()
    
    logger.info(f"Saved edited data for job {job_id}: {len(request.items)} items")
    
    return SaveExtractedDataResponse(
        success=True,
        message="Extracted data saved successfully",
        items_saved=len(request.items)
    )


@app.get("/result/{job_id}")
async def get_job_result(job_id: str):
    """
    Get the final results of a completed processing job.
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        JSON response with extracted data and insights
    """
    if job_id not in JOB_STATUS:
        raise HTTPException(
            status_code=404, 
            detail=f"Job not found: {job_id}"
        )
    
    job_info = JOB_STATUS[job_id]
    
    # Check if job is completed
    if job_info["status"] != "completed":
        return {
            "job_id": job_id,
            "status": job_info["status"],
            "progress": job_info.get("progress", "In progress..."),
            "message": "Job is still processing. Please check status endpoint."
        }
    
    # Load results from saved files
    try:
        script_dir = Path(__file__).parent.parent
        results_dir = script_dir / "tmp" / "results" / job_id
        
        # Load different result files
        result_data = {
            "job_id": job_id,
            "status": "completed",
            "extracted_data": {},
            "insights": {},
            "tables": [],
            "ocr_results": {}
        }
        
        # Load insights.json if exists
        insights_file = results_dir / "insights.json"
        if insights_file.exists():
            with open(insights_file, 'r', encoding='utf-8') as f:
                result_data["insights"] = json.load(f)
        
        # Load tables.json if exists
        tables_file = results_dir / "tables.json"
        if tables_file.exists():
            with open(tables_file, 'r', encoding='utf-8') as f:
                result_data["tables"] = json.load(f)
        
        # Load OCR results if exists
        ocr_file = results_dir / "ocr_results.json"
        if ocr_file.exists():
            with open(ocr_file, 'r', encoding='utf-8') as f:
                result_data["ocr_results"] = json.load(f)
        
        # Load extracted data if exists  
        extracted_file = results_dir / "extracted.json"  # Fixed: correct filename
        if extracted_file.exists():
            with open(extracted_file, 'r', encoding='utf-8') as f:
                result_data["extracted_data"] = json.load(f)
        
        return result_data
        
    except Exception as e:
        logger.error(f"Failed to load results for job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load results: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Ensure tmp directory exists
    script_dir = Path(__file__).parent.parent  # Go up to backend/ directory
    (script_dir / "tmp").mkdir(parents=True, exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
