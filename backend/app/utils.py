"""
Utility functions for Phase 7: Backend orchestration
Helper functions for job management, file operations, and error handling
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directories():
    """
    Ensure all required directories exist
    """
    script_dir = Path(__file__).parent.parent
    
    directories = [
        script_dir / "tmp",
        script_dir / "tmp" / "uploads",
        script_dir / "tmp" / "preprocessed", 
        script_dir / "tmp" / "results",
        script_dir / "models",
        script_dir / "data" / "history"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    logger.info("All required directories ensured")

def get_job_directories(job_id: str) -> Dict[str, Path]:
    """
    Get all directory paths for a job
    
    Args:
        job_id: Job identifier
        
    Returns:
        Dictionary with directory paths
    """
    script_dir = Path(__file__).parent.parent
    
    return {
        "upload": script_dir / "tmp" / "uploads" / job_id,
        "preprocessed": script_dir / "tmp" / "preprocessed" / job_id,
        "results": script_dir / "tmp" / "results" / job_id
    }

def validate_job_exists(job_id: str) -> bool:
    """
    Check if a job exists (has upload directory)
    
    Args:
        job_id: Job identifier
        
    Returns:
        True if job exists, False otherwise
    """
    dirs = get_job_directories(job_id)
    return dirs["upload"].exists()

def get_job_files(job_id: str) -> Dict[str, List[Path]]:
    """
    Get all files associated with a job
    
    Args:
        job_id: Job identifier
        
    Returns:
        Dictionary with file lists by category
    """
    dirs = get_job_directories(job_id)
    
    files = {
        "uploads": [],
        "preprocessed": [],
        "results": []
    }
    
    # Upload files
    if dirs["upload"].exists():
        files["uploads"] = list(dirs["upload"].iterdir())
    
    # Preprocessed files
    if dirs["preprocessed"].exists():
        files["preprocessed"] = list(dirs["preprocessed"].iterdir())
    
    # Result files
    if dirs["results"].exists():
        files["results"] = list(dirs["results"].iterdir())
    
    return files

def calculate_job_size(job_id: str) -> Dict[str, int]:
    """
    Calculate total size of files for a job
    
    Args:
        job_id: Job identifier
        
    Returns:
        Dictionary with sizes in bytes
    """
    files = get_job_files(job_id)
    
    sizes = {
        "uploads": 0,
        "preprocessed": 0,
        "results": 0,
        "total": 0
    }
    
    for category, file_list in files.items():
        category_size = 0
        for file_path in file_list:
            if file_path.is_file():
                try:
                    category_size += file_path.stat().st_size
                except OSError:
                    continue
        
        sizes[category] = category_size
        sizes["total"] += category_size
    
    return sizes

def cleanup_old_jobs(max_age_hours: int = 24, max_jobs: int = 100) -> Dict[str, int]:
    """
    Clean up old job directories
    
    Args:
        max_age_hours: Maximum age in hours
        max_jobs: Maximum number of jobs to keep
        
    Returns:
        Cleanup statistics
    """
    script_dir = Path(__file__).parent.parent
    tmp_dir = script_dir / "tmp"
    
    if not tmp_dir.exists():
        return {"cleaned": 0, "kept": 0, "errors": 0}
    
    stats = {"cleaned": 0, "kept": 0, "errors": 0}
    
    # Get all job directories
    job_dirs = []
    for subdir in ["uploads", "preprocessed", "results"]:
        subdir_path = tmp_dir / subdir
        if subdir_path.exists():
            for job_dir in subdir_path.iterdir():
                if job_dir.is_dir():
                    job_dirs.append((job_dir, job_dir.stat().st_mtime))
    
    # Sort by modification time (newest first)
    job_dirs.sort(key=lambda x: x[1], reverse=True)
    
    current_time = datetime.now().timestamp()
    max_age_seconds = max_age_hours * 3600
    
    for i, (job_dir, mod_time) in enumerate(job_dirs):
        age_seconds = current_time - mod_time
        
        # Clean if too old or beyond max jobs limit
        if age_seconds > max_age_seconds or i >= max_jobs:
            try:
                shutil.rmtree(job_dir)
                stats["cleaned"] += 1
                logger.info(f"Cleaned up old job directory: {job_dir}")
            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Failed to clean up {job_dir}: {e}")
        else:
            stats["kept"] += 1
    
    logger.info(f"Cleanup completed: {stats}")
    return stats

def save_job_metadata(job_id: str, metadata: Dict) -> bool:
    """
    Save metadata for a job
    
    Args:
        job_id: Job identifier
        metadata: Metadata dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        dirs = get_job_directories(job_id)
        dirs["results"].mkdir(parents=True, exist_ok=True)
        
        metadata_file = dirs["results"] / "metadata.json"
        
        # Add timestamp
        metadata["saved_at"] = datetime.now().isoformat()
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save metadata for job {job_id}: {e}")
        return False

def load_job_metadata(job_id: str) -> Optional[Dict]:
    """
    Load metadata for a job
    
    Args:
        job_id: Job identifier
        
    Returns:
        Metadata dictionary or None if not found
    """
    try:
        dirs = get_job_directories(job_id)
        metadata_file = dirs["results"] / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata for job {job_id}: {e}")
        return None

def get_system_stats() -> Dict[str, any]:
    """
    Get system statistics
    
    Returns:
        Dictionary with system stats
    """
    script_dir = Path(__file__).parent.parent
    tmp_dir = script_dir / "tmp"
    
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_jobs": 0,
        "total_size_mb": 0,
        "directories": {
            "uploads": 0,
            "preprocessed": 0, 
            "results": 0
        }
    }
    
    if not tmp_dir.exists():
        return stats
    
    total_size = 0
    
    for subdir_name in ["uploads", "preprocessed", "results"]:
        subdir = tmp_dir / subdir_name
        if subdir.exists():
            job_count = len([d for d in subdir.iterdir() if d.is_dir()])
            stats["directories"][subdir_name] = job_count
            
            # Calculate size
            try:
                for item in subdir.rglob("*"):
                    if item.is_file():
                        total_size += item.stat().st_size
            except OSError:
                pass
    
    stats["total_jobs"] = max(stats["directories"].values())
    stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
    
    return stats

def create_csv_from_json(json_file: Path, csv_file: Path) -> bool:
    """
    Create CSV file from extracted JSON data
    
    Args:
        json_file: Path to JSON file
        csv_file: Path to output CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import pandas as pd
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract items for CSV
        items = data.get('items', [])
        
        if not items:
            # Create empty CSV with headers
            df = pd.DataFrame(columns=[
                'description', 'category', 'qty', 'unit_price', 'line_total'
            ])
        else:
            df = pd.DataFrame(items)
        
        # Add document-level information
        if items:
            df['vendor'] = data.get('vendor', '')
            df['date'] = data.get('date', '')
            df['currency'] = data.get('currency', '')
            df['document_total'] = data.get('total', 0.0)
        
        # Save CSV
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"CSV created: {csv_file} ({len(items)} items)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create CSV: {e}")
        return False

def validate_file_type(filename: str) -> bool:
    """
    Validate if file type is supported
    
    Args:
        filename: Name of the file
        
    Returns:
        True if supported, False otherwise
    """
    allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png'}
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_extensions

def get_file_info(file_path: Path) -> Dict[str, any]:
    """
    Get file information
    
    Args:
        file_path: Path to file
        
    Returns:
        File information dictionary
    """
    try:
        stat = file_path.stat()
        
        return {
            "name": file_path.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": file_path.suffix.lower()
        }
    except Exception as e:
        logger.error(f"Failed to get file info for {file_path}: {e}")
        return {"error": str(e)}

# Initialize directories on import
ensure_directories()