from uuid import uuid4
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from models.database import Patient, AnalysisRecord, MedicalReport

router = APIRouter()


class PatientCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)


@router.post("/patients")
async def register_patient(
    payload: PatientCreate,
    db: AsyncSession = Depends(get_db)
):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Patient name is required")

    patient_id = f"PT-{datetime.now().strftime('%Y%m%d')}-{uuid4().hex[:6].upper()}"
    patient = Patient(patient_id=patient_id, name=name)

    db.add(patient)
    await db.flush()

    return {
        "message": "Patient registered successfully",
        "patient_id": patient.patient_id,
        "name": patient.name,
        "created_at": patient.created_at
    }


@router.get("/patients")
async def list_patients(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Patient).order_by(desc(Patient.created_at))
    )
    patients = result.scalars().all()

    return {
        "count": len(patients),
        "patients": [
            {
                "patient_id": p.patient_id,
                "name": p.name,
                "created_at": p.created_at
            }
            for p in patients
        ]
    }


@router.get("/patients/{patient_id}")
async def get_patient(
    patient_id: str,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Patient).where(Patient.patient_id == patient_id)
    )
    patient = result.scalar_one_or_none()

    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    analyses_result = await db.execute(
        select(AnalysisRecord)
        .where(AnalysisRecord.patient_id == patient_id)
        .order_by(desc(AnalysisRecord.created_at))
    )
    analyses = analyses_result.scalars().all()

    reports_result = await db.execute(
        select(MedicalReport)
        .where(MedicalReport.patient_id == patient_id)
        .order_by(desc(MedicalReport.created_at))
    )
    reports = reports_result.scalars().all()

    return {
        "patient": {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "created_at": patient.created_at
        },
        "analysis_count": len(analyses),
        "analyses": [
            {
                "id": a.id,
                "analysis_type": a.analysis_type,
                "status": a.status,
                "confidence": a.confidence,
                "created_at": a.created_at,
                "result": a.result
            }
            for a in analyses
        ],
        "reports": [
            {
                "report_id": r.report_id,
                "report_type": r.report_type,
                "summary": r.summary,
                "created_at": r.created_at
            }
            for r in reports
        ]
    }
