"""
Dermatology analysis routes
Skin condition detection using Derm CNN (HAM10000) + webcam support
"""

import logging
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from database.connection import get_db
from database import crud
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)
router = APIRouter()


class DermDependencies:
    """Dependencies for dermatology routes"""

    def __init__(self):
        self.analyzer = None
        self.qwen_analyzer = None

    async def initialize(self, model_manager) -> None:
        from ml_pipeline.vision.derm_cnn import create_dermatology_analyzer
        from ml_pipeline.vision.qwen_vl_analyzer import create_qwen_vl_analyzer
        self.analyzer = await create_dermatology_analyzer(model_manager)
        self.qwen_analyzer = await create_qwen_vl_analyzer(model_manager)

    async def get_analyzers(self):
        if not self.analyzer or not self.qwen_analyzer:
            from main import model_manager
            if model_manager is None:
                raise HTTPException(status_code=503, detail="Backend models are still initializing.")
            await self.initialize(model_manager)
        return self.analyzer, self.qwen_analyzer


derm_deps = DermDependencies()


async def get_derm_analyzers():
    return await derm_deps.get_analyzers()


@router.post("/dermatology/detect", tags=["dermatology"])
async def detect_skin_condition(
    file: UploadFile = File(...),
    return_detailed: bool = True,
    analyzers=Depends(get_derm_analyzers),
    db: AsyncSession = Depends(get_db),
    patient_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect skin conditions from an uploaded image.

    **Parameters:**
    - `file`: Skin image (JPEG / PNG)
    - `return_detailed`: Include all top-5 predictions

    **Returns:** Classification, confidence, severity, and clinical advice.
    """
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(allowed)}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        analyzer, qwen_analyzer = analyzers

        # Create pending database record
        db_record = await crud.create_analysis_record(
            db,
            analysis_type="dermatology",
            patient_id=patient_id,
            input_file=file.filename
        )
        
        # 1. CNN Analysis
        result = await analyzer.analyze_skin_image(tmp_path, return_detailed=return_detailed)

        # 2. VLM Description
        qwen_result = await qwen_analyzer.analyze_skin(image_path=tmp_path)
        if qwen_result.get("status") == "success":
            result["description"] = qwen_result.get("description")
            result["vlm_model"] = qwen_result.get("model")

        # Update database with results
        confidence = result.get("classification", {}).get("confidence", 0.0)
        await crud.update_analysis_result(
            db,
            record_id=db_record.id,
            result=result,
            status="success",
            model_used=f"DermCNN + {result.get('vlm_model', 'Qwen2-VL')}",
            confidence=float(confidence)
        )

        return result

    except Exception as e:
        logger.error(f"Dermatology detection failed: {e}")
        # Update record with error status
        if 'db_record' in locals():
            await crud.update_analysis_result(
                db, 
                record_id=db_record.id, 
                result={"error": str(e)}, 
                status="error"
            )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if "tmp_path" in locals() and Path(tmp_path).exists():
            Path(tmp_path).unlink()


@router.get("/dermatology/camera", tags=["dermatology"])
async def start_camera(
    device_id: int = 0,
    analyzers=Depends(get_derm_analyzers),
) -> Dict[str, Any]:
    """Start the webcam stream for real-time skin analysis."""
    analyzer, _ = analyzers
    return analyzer.start_webcam(device_id=device_id)


@router.post("/dermatology/capture", tags=["dermatology"])
async def capture_and_analyze(
    analyzers=Depends(get_derm_analyzers),
    db: AsyncSession = Depends(get_db),
    patient_id: Optional[str] = None
) -> Dict[str, Any]:
    """Capture one frame from the active webcam and analyze it."""
    analyzer, _ = analyzers
    result = await analyzer.capture_and_analyze()

    if result.get("status") == "success":
        # Create database record for the live capture
        db_record = await crud.create_analysis_record(
            db,
            analysis_type="dermatology_live",
            patient_id=patient_id,
            input_file="webcam_capture"
        )

        # Update with result
        confidence = result.get("classification", {}).get("confidence", 0.0)
        await crud.update_analysis_result(
            db,
            record_id=db_record.id,
            result=result,
            status="success",
            model_used="DermCNN (Live)",
            confidence=float(confidence)
        )

    return result


@router.delete("/dermatology/camera", tags=["dermatology"])
async def stop_camera(
    analyzers=Depends(get_derm_analyzers),
) -> Dict[str, Any]:
    """Stop the active webcam stream."""
    analyzer, _ = analyzers
    return analyzer.stop_webcam()
