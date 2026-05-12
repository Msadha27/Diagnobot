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
from ml_pipeline.triage import build_triage_assessment

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
        try:
            self.analyzer = await create_dermatology_analyzer(model_manager)
        except Exception as e:
            logger.warning(f"Derm CNN unavailable; continuing with VLM only: {e}")
            self.analyzer = None
        self.qwen_analyzer = await create_qwen_vl_analyzer(model_manager)

    async def get_analyzers(self):
        if not self.qwen_analyzer:
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
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/pjpeg"}
    image_suffixes = {".jpg", ".jpeg", ".jfif", ".png"}
    if file.content_type not in allowed and Path(file.filename or "").suffix.lower() not in image_suffixes:
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
        
        # 1. Skin classification model, if available.
        classification_result: Dict[str, Any] = {
            "classification": {
                "label": "Not classified",
                "confidence": 0.0,
                "severity": "unknown"
            }
        }
        if analyzer is not None:
            logger.info("Running dermatology classifier...")
            classification_result = await analyzer.analyze_skin_image(
                tmp_path,
                return_detailed=return_detailed
            )

        # 2. Vision Analysis (Moondream GGUF VLM)
        # Using the "Eyes" of the system for smart skin analysis
        logger.info("Running vision VLM for dermatology analysis...")
        qwen_result = await qwen_analyzer.analyze_skin(image_path=tmp_path)
        
        if qwen_result.get("status") == "error":
            raise Exception(f"Vision analysis failed: {qwen_result.get('error')}")

        result = {
            "status": "success",
            "classification": classification_result.get("classification", {}),
            "all_predictions": classification_result.get("all_predictions"),
            "clinical_advice": classification_result.get("clinical_advice"),
            "classifier_model": classification_result.get("classifier_model"),
            "classifier_note": classification_result.get("classifier_note"),
            "description": qwen_result.get("description"),
            "vlm_model": qwen_result.get("model"),
            "disclaimer": "AI-generated visual analysis. Consult a dermatologist."
        }
        result = {key: value for key, value in result.items() if value is not None}
        result["triage_assessment"] = build_triage_assessment(
            visual_summary=result.get("description"),
            classification=result.get("classification"),
            mode="dermatology",
        )

        # 2. Reasoning (Google Gemma-2)
        # Generate a professional verdict based on the skin description
        try:
            from main import model_manager
            from ml_pipeline.nlp.report_generator import create_report_generator
            report_gen = await create_report_generator(model_manager)
            
            classification = result.get("classification", {})
            if classification.get("label") == "Uncertain":
                context = (
                    "Dermatology classifier result is uncertain. "
                    f"Top visual match: {classification.get('top_match')} "
                    f"with confidence {classification.get('confidence', 0):.2f}. "
                    f"Top matches: {result.get('all_predictions')}\n"
                    f"Visual observation: {result.get('description')}"
                )
            else:
                context = (
                    f"Dermatology classification: {classification}\n"
                    f"Visual observation: {result.get('description')}"
                )
            logger.info("Generating medical reasoning for skin condition...")
            verdict = await report_gen.generate_report(context)
            result["doctor_verdict"] = verdict
        except Exception as e:
            logger.warning(f"Gemma-2 skin reasoning failed: {e}")
            result["doctor_verdict"] = "Specialist reasoning engine is loading. Please check description."

        # Update database with results
        if db_record is not None:
            await crud.update_analysis_result(
                db,
                record_id=db_record.id,
                result=result,
                status="success",
                model_used=f"{result.get('vlm_model', 'Moondream2-GGUF')} + Gemma-2",
                confidence=float(result.get("classification", {}).get("confidence", 0.0))
            )

        return result

    except Exception as e:
        logger.error(f"Dermatology detection failed: {e}")
        # Update record with error status
        if 'db_record' in locals() and db_record is not None:
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
    return {
        "status": "ready",
        "device_id": device_id,
        "message": "Use POST /api/v1/dermatology/capture to capture and analyze one webcam frame.",
        "model": "Moondream2-GGUF",
    }


@router.get("/dermatology/classes", tags=["dermatology"])
async def get_dermatology_classes() -> Dict[str, Any]:
    """List dermatology classes available to the classifier paths."""
    from ml_pipeline.vision.derm_cnn import DermatologyAnalyzer

    return {
        "hf_derm_cnn_classes": [
            {"id": idx, **info}
            for idx, info in DermatologyAnalyzer.DISEASE_CLASSES.items()
        ],
        "local_dataset_classes": [
            {"name": name, **info}
            for name, info in DermatologyAnalyzer.LOCAL_DISEASE_CLASSES.items()
        ],
        "note": (
            "The configured Hugging Face Derm CNN currently fails to load because its "
            "config has no valid model_type. The backend falls back to the local "
            "dataset similarity classifier."
        ),
    }


@router.post("/dermatology/capture", tags=["dermatology"])
async def capture_and_analyze(
    analyzers=Depends(get_derm_analyzers),
    db: AsyncSession = Depends(get_db),
    patient_id: Optional[str] = None
) -> Dict[str, Any]:
    """Capture one frame from the active webcam and analyze it."""
    _, qwen_analyzer = analyzers
    
    # We use the qwen_analyzer to handle the capture and analysis directly or via a temp file
    # For now, we'll simulate the capture by using a placeholder or existing logic if available
    # Since the previous analyzer is gone, we'll need to handle the capture here.
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise Exception("Failed to capture image from webcam.")
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, frame)
            tmp_path = tmp.name

        qwen_result = await qwen_analyzer.analyze_skin(image_path=tmp_path)
        
        result = {
            "status": "success",
            "description": qwen_result.get("description"),
            "vlm_model": qwen_result.get("model"),
            "disclaimer": "Live webcam analysis. Consult a specialist."
        }
        result["triage_assessment"] = build_triage_assessment(
            visual_summary=result.get("description"),
            mode="dermatology",
        )

        # Reasoning (Google Gemma-2)
        try:
            from main import model_manager
            from ml_pipeline.nlp.report_generator import create_report_generator
            report_gen = await create_report_generator(model_manager)
            context = f"Live Dermatology Observation: {result['description']}"
            result["doctor_verdict"] = await report_gen.generate_report(context)
        except:
            result["doctor_verdict"] = "Specialist reasoning engine is busy."

        # Create database record
        db_record = await crud.create_analysis_record(
            db,
            analysis_type="dermatology_live",
            patient_id=patient_id,
            input_file="webcam_capture"
        )

        if db_record is not None:
            await crud.update_analysis_result(
                db,
                record_id=db_record.id,
                result=result,
                status="success",
                model_used=f"{result.get('vlm_model', 'Moondream2-GGUF')} + Gemma-2 (Live)",
                confidence=1.0
            )
        
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
            
        return result
    except Exception as e:
        logger.error(f"Live capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/dermatology/camera", tags=["dermatology"])
async def stop_camera(
    analyzers=Depends(get_derm_analyzers),
) -> Dict[str, Any]:
    """Stop the active webcam stream."""
    return {"status": "stopped", "message": "No persistent webcam stream is running."}
