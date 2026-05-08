"""
Medical report generation routes
BioGPT + BioBart + ClinicalT5
"""

import logging
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)
router = APIRouter()


class ReportDependencies:
    def __init__(self):
        self.generator = None

    async def initialize(self, model_manager) -> None:
        from ml_pipeline.nlp.report_generator import create_report_generator
        self.generator = await create_report_generator(model_manager)

    async def get_generator(self):
        if not self.generator:
            from main import model_manager
            if model_manager is None:
                raise HTTPException(status_code=503, detail="Backend models are still initializing.")
            await self.initialize(model_manager)
        return self.generator


report_deps = ReportDependencies()


async def get_report_generator():
    return await report_deps.get_generator()


@router.post("/report/generate", tags=["reports"])
async def generate_report(
    clinical_findings: Dict[str, Any] = Body(...),
    patient_info: Optional[Dict[str, str]] = Body(None),
    max_length: int = Body(512),
) -> Dict[str, Any]:
    """
    Generate a structured medical report from clinical findings.

    **Body:**
    - `clinical_findings`: Dictionary of findings (from X-ray or dermatology analysis)
    - `patient_info`: Optional – age, gender, etc.
    - `max_length`: Maximum token length of the generated report
    """
    try:
        generator = await get_report_generator()
        return await generator.generate_report_from_context(
            clinical_findings=clinical_findings,
            patient_info=patient_info,
            max_length=max_length,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/summarize", tags=["reports"])
async def summarize_report(
    report_text: str = Body(..., embed=True),
    max_length: int = Body(256, embed=True),
) -> Dict[str, Any]:
    """Summarize a long medical report using ClinicalT5."""
    try:
        generator = await get_report_generator()
        return await generator.summarize_report(report_text=report_text, max_length=max_length)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/from-input", tags=["reports"])
async def report_from_patient_input(
    patient_input: str = Body(..., embed=True),
    symptoms: Optional[List[str]] = Body(None, embed=True),
    medical_history: Optional[str] = Body(None, embed=True),
) -> Dict[str, Any]:
    """Convert raw patient description into a formal clinical report using BioBart."""
    try:
        generator = await get_report_generator()
        return await generator.convert_patient_input_to_report(
            patient_input=patient_input,
            symptoms=symptoms,
            medical_history=medical_history,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Patient input conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
