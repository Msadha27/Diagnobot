"""
Dermatology analysis routes
Skin condition detection using Derm CNN (HAM10000) + webcam support
"""

import logging
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter()


class DermDependencies:
    """Dependencies for dermatology routes"""

    def __init__(self):
        self.analyzer = None

    async def initialize(self, model_manager) -> None:
        from ml_pipeline.vision.derm_cnn import create_dermatology_analyzer
        self.analyzer = await create_dermatology_analyzer(model_manager)

    async def get_analyzer(self):
        if not self.analyzer:
            raise HTTPException(status_code=503, detail="Dermatology analyzer not initialized")
        return self.analyzer


derm_deps = DermDependencies()


async def get_derm_analyzer():
    return await derm_deps.get_analyzer()


@router.post("/dermatology/detect", tags=["dermatology"])
async def detect_skin_condition(
    file: UploadFile = File(...),
    return_detailed: bool = True,
    analyzer=Depends(get_derm_analyzer),
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

        result = await analyzer.analyze_skin_image(tmp_path, return_detailed=return_detailed)

        Path(tmp_path).unlink(missing_ok=True)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dermatology detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/dermatology/camera", tags=["dermatology"])
async def start_camera(
    device_id: int = 0,
    analyzer=Depends(get_derm_analyzer),
) -> Dict[str, Any]:
    """Start the webcam stream for real-time skin analysis."""
    return analyzer.start_webcam(device_id=device_id)


@router.post("/dermatology/capture", tags=["dermatology"])
async def capture_and_analyze(
    analyzer=Depends(get_derm_analyzer),
) -> Dict[str, Any]:
    """Capture one frame from the active webcam and analyze it."""
    return await analyzer.capture_and_analyze()


@router.delete("/dermatology/camera", tags=["dermatology"])
async def stop_camera(
    analyzer=Depends(get_derm_analyzer),
) -> Dict[str, Any]:
    """Stop the active webcam stream."""
    return analyzer.stop_webcam()
