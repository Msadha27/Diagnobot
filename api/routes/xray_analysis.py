"""
X-Ray Analysis Routes
Endpoints for chest X-ray analysis using TorchXRayVision (DenseNet121) + Qwen2-VL
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import Dict, Any, List
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()


class XRayDependencies:
    """Holds the lazy-initialized XRayAnalyzer instance."""

    def __init__(self):
        self.analyzer = None

    async def initialize(self, model_manager) -> None:
        from ml_pipeline.vision.xray_analyzer import create_xray_analyzer
        self.analyzer = await create_xray_analyzer(model_manager)

    async def get_analyzer(self):
        if not self.analyzer:
            from main import model_manager
            if model_manager is None:
                raise HTTPException(
                    status_code=503,
                    detail="Backend models are still initializing.",
                )
            await self.initialize(model_manager)
        return self.analyzer


# Module-level singleton – wired to the app lifespan
xray_deps = XRayDependencies()


async def get_xray_analyzer():
    """FastAPI dependency that provides the XRayAnalyzer."""
    return await xray_deps.get_analyzer()


# ==================== ENDPOINTS ====================

@router.post("/xray/analyze", tags=["xray"])
async def analyze_xray(
    file: UploadFile = File(...),
    return_bbox: bool = True,
    return_confidence: bool = True,
    analyzer=Depends(get_xray_analyzer),
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

        result = await analyzer.analyze_xray(
            image_path=tmp_path,
            return_bbox=return_bbox,
            return_confidence=return_confidence,
        )

        Path(tmp_path).unlink(missing_ok=True)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"X-ray analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


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
        ],
        "accuracy_notes": {
            "torchxrayvision": "~88% on multi-label classification",
        },
    }


@router.post("/xray/batch-analyze", tags=["xray"])
async def batch_analyze_xrays(
    files: List[UploadFile] = File(...),
    analyzer=Depends(get_xray_analyzer),
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

            result = await analyzer.analyze_xray(tmp_path)
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