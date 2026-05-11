"""
NLP analysis routes for clinical text, PDFs, and patient notes.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from ml_pipeline.nlp.clinical_extractor import extract_clinical_data

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/nlp/analyze-text", tags=["nlp"])
async def analyze_clinical_text(
    text: str = Body(..., embed=True),
) -> Dict[str, Any]:
    """Extract symptoms, lab values, risk flags, and a short triage summary."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    extraction = extract_clinical_data(text)
    logger.info("NLP text analysis requested (%s chars)", len(text))
    return {
        "status": "success",
        "input_text": text[:200] + ("..." if len(text) > 200 else ""),
        "word_count": len(text.split()),
        "model": "Rule-based clinical extractor + report generator",
        "extraction": extraction,
    }


@router.post("/nlp/understand", tags=["nlp"])
async def understand_medical_context(
    text: str = Body(..., embed=True),
    extract_symptoms: bool = Body(True, embed=True),
) -> Dict[str, Any]:
    """Extract medical context and possible triage areas from free-form text."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    extraction = extract_clinical_data(text)
    logger.info("Medical context understanding requested")
    return {
        "status": "success",
        "input_text": text[:200],
        "model": "Rule-based clinical extractor",
        "extract_symptoms": extract_symptoms,
        "symptoms": extraction["symptoms"] if extract_symptoms else [],
        "condition_hints": extraction["condition_hints"],
        "risk_flags": extraction["risk_flags"],
        "summary": extraction["summary"],
    }


@router.post("/nlp/extract-entities", tags=["nlp"])
async def extract_medical_entities(
    text: str = Body(..., embed=True),
) -> Dict[str, Any]:
    """Extract symptom and lab-value entities from clinical text."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    extraction = extract_clinical_data(text)
    entities = [
        {"type": "symptom", "text": symptom}
        for symptom in extraction["symptoms"]
    ] + [
        {"type": "lab_value", "text": name, "value": value}
        for name, value in extraction["lab_values"].items()
    ]

    logger.info("Medical entity extraction requested")
    return {
        "status": "success",
        "input_text": text[:200],
        "model": "Rule-based clinical extractor",
        "entities": entities,
    }
