"""
Analytics and History routes for DiagnoBot
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, select, desc

from database.connection import get_db
from models.database import AnalysisRecord, MedicalReport, UploadedFile

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/admin/reset-data", response_model=Dict[str, Any])
async def reset_analysis_database(
    db: AsyncSession = Depends(get_db),
):
    """
    Clear saved analysis state while keeping the database schema ready for new runs.
    """
    try:
        deleted_reports = await db.execute(delete(MedicalReport))
        deleted_uploads = await db.execute(delete(UploadedFile))
        deleted_records = await db.execute(delete(AnalysisRecord))
        await db.commit()

        return {
            "status": "success",
            "message": "Analysis database cleared. New analyses will start from a clean state.",
            "deleted": {
                "analysis_records": deleted_records.rowcount or 0,
                "uploaded_files": deleted_uploads.rowcount or 0,
                "medical_reports": deleted_reports.rowcount or 0,
            },
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to reset analysis database: {e}")
        raise HTTPException(status_code=500, detail="Could not reset analysis database")


@router.get("/history", response_model=Dict[str, Any])
async def get_analysis_history(
    limit: int = 20,
    offset: int = 0,
    analysis_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve the history of past analyses.
    """
    try:
        query = select(AnalysisRecord).order_by(desc(AnalysisRecord.created_at))
        
        if analysis_type:
            query = query.where(AnalysisRecord.analysis_type == analysis_type)
            
        # Count total
        # result_count = await db.execute(select(func.count()).select_from(query.subquery()))
        # total = result_count.scalar()
        
        # Get records
        result = await db.execute(query.limit(limit).offset(offset))
        records = result.scalars().all()
        
        return {
            "status": "success",
            "count": len(records),
            "offset": offset,
            "limit": limit,
            "records": [
                {
                    "id": r.id,
                    "type": r.analysis_type,
                    "patient_id": r.patient_id,
                    "input_file": r.input_file,
                    "status": r.status,
                    "confidence": r.confidence,
                    "created_at": r.created_at,
                    "result_summary": r.result.get("clinical_summary") if r.result else None
                } for r in records
            ]
        }
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve history")


@router.get("/reports", response_model=Dict[str, Any])
async def get_report_history(
    limit: int = 20,
    offset: int = 0,
    patient_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve the history of generated medical reports.
    """
    try:
        query = select(MedicalReport).order_by(desc(MedicalReport.created_at))
        
        if patient_id:
            query = query.where(MedicalReport.patient_id == patient_id)
            
        result = await db.execute(query.limit(limit).offset(offset))
        reports = result.scalars().all()
        
        return {
            "status": "success",
            "count": len(reports),
            "reports": [
                {
                    "report_id": r.report_id,
                    "patient_id": r.patient_id,
                    "type": r.report_type,
                    "created_at": r.created_at,
                    "summary": r.summary
                } for r in reports
            ]
        }
    except Exception as e:
        logger.error(f"Failed to fetch reports: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve reports")


@router.get("/record/{record_id}", response_model=Dict[str, Any])
async def get_analysis_detail(
    record_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get full details of a specific analysis record.
    """
    record = await db.get(AnalysisRecord, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
        
    return {
        "status": "success",
        "record": {
            "id": record.id,
            "type": record.analysis_type,
            "patient_id": record.patient_id,
            "input_file": record.input_file,
            "result": record.result,
            "status": record.status,
            "model_used": record.model_used,
            "confidence": record.confidence,
            "created_at": record.created_at
        }
    }
