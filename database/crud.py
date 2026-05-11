"""
CRUD operations for DiagnoBot database models
"""

import logging
import uuid
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from models.database import AnalysisRecord, UploadedFile, MedicalReport

logger = logging.getLogger(__name__)


# ==================== ANALYSIS RECORDS ====================

async def create_analysis_record(
    db: AsyncSession,
    analysis_type: str,
    patient_id: Optional[str] = None,
    input_file: Optional[str] = None,
) -> Optional[AnalysisRecord]:
    record = AnalysisRecord(
        analysis_type=analysis_type,
        patient_id=patient_id,
        input_file=input_file,
        status="pending",
    )
    db.add(record)
    try:
        await db.flush()
    except OperationalError as exc:
        await db.rollback()
        logger.warning("Analysis history unavailable; continuing without DB record: %s", exc)
        return None
    return record


async def update_analysis_result(
    db: AsyncSession,
    record_id: int,
    result: Dict[str, Any],
    status: str = "success",
    model_used: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Optional[AnalysisRecord]:
    if record_id is None:
        return None
    record = await db.get(AnalysisRecord, record_id)
    if record:
        record.result = result
        record.status = status
        record.model_used = model_used
        record.confidence = confidence
    return record


# ==================== UPLOADED FILES ====================

async def save_upload_record(
    db: AsyncSession,
    file_id: str,
    original_filename: str,
    saved_path: str,
    file_type: str,
    size_bytes: int,
    content_type: str,
) -> Optional[UploadedFile]:
    record = UploadedFile(
        file_id=file_id,
        original_filename=original_filename,
        saved_path=saved_path,
        file_type=file_type,
        size_bytes=size_bytes,
        content_type=content_type,
    )
    db.add(record)
    try:
        await db.flush()
    except OperationalError as exc:
        await db.rollback()
        logger.warning("Upload history unavailable; continuing without DB record: %s", exc)
        return None
    return record


# ==================== MEDICAL REPORTS ====================

async def save_medical_report(
    db: AsyncSession,
    full_report: str,
    report_type: str,
    generation_model: str,
    summary: Optional[str] = None,
    patient_id: Optional[str] = None,
    clinical_findings: Optional[Dict[str, Any]] = None,
) -> Optional[MedicalReport]:
    report = MedicalReport(
        report_id=uuid.uuid4().hex,
        patient_id=patient_id,
        report_type=report_type,
        full_report=full_report,
        summary=summary,
        generation_model=generation_model,
        clinical_findings=clinical_findings,
    )
    db.add(report)
    try:
        await db.flush()
    except OperationalError as exc:
        await db.rollback()
        logger.warning("Report history unavailable; continuing without DB record: %s", exc)
        return None
    return report
