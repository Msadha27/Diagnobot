"""
File upload routes – images, PDFs, and plain text
"""

import logging
import uuid
import aiofiles
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/tiff"}
ALLOWED_DOC_TYPES = {"application/pdf", "text/plain", "text/csv"}
MAX_IMAGE_MB = 50
MAX_DOC_MB = 100


def _check_size(content: bytes, max_mb: int, label: str) -> None:
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(
            status_code=413,
            detail=f"{label} too large. Maximum size: {max_mb} MB (uploaded: {size_mb:.1f} MB)",
        )


@router.post("/upload/image", tags=["upload"])
async def upload_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload an X-ray or dermatology image for analysis."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type '{file.content_type}'. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    content = await file.read()
    _check_size(content, MAX_IMAGE_MB, "Image file")

    # Save with unique name
    suffix = Path(file.filename).suffix or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    save_path = UPLOAD_DIR / unique_name

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(content)

    logger.info(f"Image uploaded: {file.filename} → {save_path}")

    return {
        "status": "success",
        "message": "Image uploaded successfully",
        "original_filename": file.filename,
        "saved_as": str(save_path),
        "file_id": unique_name,
        "size_bytes": len(content),
        "content_type": file.content_type,
    }


@router.post("/upload/pdf", tags=["upload"])
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a medical PDF report for text extraction."""
    if file.content_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document type. Allowed: {', '.join(ALLOWED_DOC_TYPES)}",
        )

    content = await file.read()
    _check_size(content, MAX_DOC_MB, "Document file")

    suffix = Path(file.filename).suffix or ".pdf"
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    save_path = UPLOAD_DIR / unique_name

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(content)

    logger.info(f"Document uploaded: {file.filename} → {save_path}")

    return {
        "status": "success",
        "message": "Document uploaded successfully",
        "original_filename": file.filename,
        "saved_as": str(save_path),
        "file_id": unique_name,
        "size_bytes": len(content),
    }


@router.post("/upload/text", tags=["upload"])
async def upload_text(text: str = Form(...), patient_id: Optional[str] = Form(None)) -> Dict[str, Any]:
    """Submit patient symptoms or notes as plain text."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    text_id = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"{text_id}.txt"

    async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
        await f.write(text)

    logger.info(f"Text input saved: {save_path}")

    return {
        "status": "success",
        "message": "Text submitted successfully",
        "text_id": text_id,
        "saved_as": str(save_path),
        "patient_id": patient_id,
        "word_count": len(text.split()),
    }
