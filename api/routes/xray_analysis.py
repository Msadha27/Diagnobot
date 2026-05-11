"""
X-Ray Analysis Routes
Endpoints for chest X-ray analysis using TorchXRayVision (DenseNet121) + vision VLM
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import Dict, Any, List, Optional
import logging
import tempfile
from pathlib import Path

from database.connection import get_db
from database import crud
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)
router = APIRouter()


class XRayDependencies:
    """Holds lazy-initialized XRayAnalyzer and vision VLM analyzer instances."""

    def __init__(self):
        self.analyzer = None
        self.qwen_analyzer = None

    async def initialize(self, model_manager) -> None:
        from ml_pipeline.vision.xray_analyzer import create_xray_analyzer
        from ml_pipeline.vision.qwen_vl_analyzer import create_qwen_vl_analyzer

        # Both use lazy-loading internally
        self.analyzer = await create_xray_analyzer(model_manager)
        self.qwen_analyzer = await create_qwen_vl_analyzer(model_manager)

    async def get_analyzers(self):
        if not self.analyzer or not self.qwen_analyzer:
            from main import model_manager
            if model_manager is None:
                raise HTTPException(
                    status_code=503,
                    detail="Backend models are still initializing.",
                )
            await self.initialize(model_manager)
        return self.analyzer, self.qwen_analyzer


# Module-level singleton – wired to the app lifespan
xray_deps = XRayDependencies()


async def get_xray_analyzers():
    """FastAPI dependency that provides both X-ray analyzers."""
    return await xray_deps.get_analyzers()


# ==================== ENDPOINTS ====================

@router.post("/xray/analyze", tags=["xray"])
async def analyze_xray(
    file: UploadFile = File(...),
    return_bbox: bool = True,
    return_confidence: bool = True,
    analyzers=Depends(get_xray_analyzers),
    db: AsyncSession = Depends(get_db),
    patient_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a chest X-ray image.

    **Parameters:**
    - `file`: X-ray image (JPEG, PNG, TIFF)
    - `return_bbox`: Include detected bounding boxes
    - `return_confidence`: Include confidence scores

    **Returns:** Clinical findings, anomaly list, and recommendations.
    """
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/tiff"}

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: {', '.join(allowed_types)}",
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Processing X-ray: {file.filename}")

        analyzer, qwen_analyzer = analyzers

        # Create pending database record
        db_record = await crud.create_analysis_record(
            db,
            analysis_type="xray",
            patient_id=patient_id,
            input_file=file.filename
        )

        # 1. CNN Analysis (Pathology labels)
        result = await analyzer.analyze_xray(
            image_path=tmp_path,
            return_bbox=return_bbox,
            return_confidence=return_confidence,
        )

        # 2. Vision VLM Analysis (Natural language description)
        # Note: This is computationally expensive on CPU.
        qwen_result = await qwen_analyzer.analyze_xray(image_path=tmp_path)

        if qwen_result.get("status") == "success":
            result["description"] = qwen_result.get("description")
            result["vlm_model"] = qwen_result.get("model")
            if "disclaimer" in qwen_result:
                result["disclaimer"] = qwen_result["disclaimer"]

        # 3. Clinical Reasoning (The "Brain" - Google Gemma-2)
        # Generate a final professional verdict combining all AI findings
        try:
            from main import model_manager
            from ml_pipeline.nlp.report_generator import create_report_generator
            report_gen = await create_report_generator(model_manager)
            
            # Combine findings into a context string for Gemma
            findings_text = ", ".join([f["name"] for f in result.get("findings", [])])
            context = f"CNN Detections: {findings_text or 'None'}. Visual Description: {result.get('description', 'N/A')}"
            
            logger.info("Generating final Doctor's Verdict with Gemma-2...")
            verdict = await report_gen.generate_report(context)
            result["doctor_verdict"] = verdict
        except Exception as e:
            logger.warning(f"Gemma-2 reasoning failed: {e}")
            result["doctor_verdict"] = "Medical reasoning engine is currently unavailable. Please review findings manually."

        # Update database with results
        confidence = result.get("findings", [{}])[0].get("confidence", 0.0) if result.get("findings") else 0.0
        await crud.update_analysis_result(
            db,
            record_id=db_record.id,
            result=result,
            status="success",
            model_used=f"TorchXRayVision + {result.get('vlm_model', 'Moondream2-GGUF')}",
            confidence=float(confidence)
        )

        return result
    except Exception as e:
        logger.error(f"X-ray analysis failed: {e}")
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


@router.get("/xray/models", tags=["xray"])
async def get_xray_models() -> Dict[str, Any]:
    """List available X-ray analysis models and their capabilities."""
    return {
        "available_models": [
            {
                "name": "TorchXRayVision",
                "type": "Pre-trained CNN",
                "capability": "Multi-label Pathology Classification & Feature extraction",
                "parameters": "DenseNet121",
                "training_data": "MIMIC, CheXpert, ImageNet-CXR",
            },
            {
                "name": "Vision VLM",
                "type": "Vision-language model",
                "capability": "Natural-language description of visible X-ray findings",
                "parameters": "Moondream2 GGUF or PaliGemma depending on settings",
                "training_data": "General vision-language pretraining",
            },
        ],
        "accuracy_notes": {
            "torchxrayvision": "~88% on multi-label classification",
        },
    }


@router.post("/xray/batch-analyze", tags=["xray"])
async def batch_analyze_xrays(
    files: List[UploadFile] = File(...),
    analyzers=Depends(get_xray_analyzers),
) -> Dict[str, Any]:
    """
    Analyze up to 10 X-ray images in a single request.

    **Returns:** Per-file results + batch summary.
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch.")

    results = []
    errors = []

    for file in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            analyzer, qwen_analyzer = analyzers
            result = await analyzer.analyze_xray(tmp_path)
            
            # Optionally add descriptions in batch (might be slow!)
            qwen_result = await qwen_analyzer.analyze_xray(tmp_path)
            if qwen_result.get("status") == "success":
                result["description"] = qwen_result.get("description")
            results.append({"filename": file.filename, "result": result})
            Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    return {
        "status": "success" if not errors else "partial",
        "total_files": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors or None,
    }
