from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import Patient
from database.connection import get_db
from database.crud import create_analysis_record, update_analysis_result

router = APIRouter()


class VitalsInput(BaseModel):
    patient_id: str
    raw_red: Optional[int] = None
    raw_ir: Optional[int] = None
    pulse_bpm: Optional[float] = Field(default=None, gt=0, le=250)
    spo2_percent: Optional[float] = Field(default=None, gt=0, le=100)
    finger_detected: bool
    signal_quality: str
    measurement_status: str = "experimental"


@router.post("/vitals")
async def save_vitals(
    data: VitalsInput,
    db: AsyncSession = Depends(get_db)
):
    patient = await db.get(Patient, data.patient_id)

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    quality = data.signal_quality.lower()

    # BPM can be saved independently of SpO2.
    pulse_valid = (
        data.finger_detected
        and data.pulse_bpm is not None
    )

    record = await create_analysis_record(
        db,
        analysis_type="vitals",
        patient_id=data.patient_id
    )

    if record is None:
        raise HTTPException(
            status_code=503,
            detail="Could not save vitals"
        )

    result = {
        "raw_red": data.raw_red,
        "raw_ir": data.raw_ir,
        "pulse_bpm": data.pulse_bpm if pulse_valid else None,
        "pulse_status": "experimental" if pulse_valid else "unavailable",
        "spo2_percent": None,
        "finger_detected": data.finger_detected,
        "signal_quality": quality,
        "measurement_status": data.measurement_status,
        "note": "Pulse estimate is experimental. SpO2 is not calibrated."
    }

    await update_analysis_result(
        db,
        record.id,
        result,
        status="success" if pulse_valid else "incomplete",
        model_used="MAX30102"
    )

    return {
        "message": "Sensor data saved",
        "record_id": record.id,
        "patient_id": data.patient_id,
        "status": "success" if pulse_valid else "incomplete",
        "vitals": result
    }
