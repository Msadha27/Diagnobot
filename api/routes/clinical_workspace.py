"""
Emergency triage and doctor workspace routes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException

from ml_pipeline.nlp.clinical_extractor import extract_clinical_data

router = APIRouter()


SPECIALIST_RULES = [
    {
        "specialist": "Emergency Medicine",
        "urgency": "emergency",
        "terms": ["chest pain", "difficulty breathing", "unconscious", "heavy bleeding", "stroke", "seizure"],
        "reason": "Potential life-threatening symptom reported.",
    },
    {
        "specialist": "Dermatologist",
        "urgency": "soon",
        "terms": ["rash", "acne", "itching", "skin", "wound", "ulcer", "burn", "discharge"],
        "reason": "Skin, wound, burn, or rash symptoms reported.",
    },
    {
        "specialist": "Ophthalmologist",
        "urgency": "soon",
        "terms": ["red eyes", "yellow eyes", "eye pain", "blurred vision", "vision", "sclera"],
        "reason": "Eye color or vision-related symptom reported.",
    },
    {
        "specialist": "Orthopedic Doctor",
        "urgency": "soon",
        "terms": ["fracture", "bone", "joint", "swelling", "sprain", "xray", "x-ray"],
        "reason": "Bone, joint, injury, or X-ray concern reported.",
    },
    {
        "specialist": "Gynecologist",
        "urgency": "routine",
        "terms": ["pregnancy", "period", "vaginal", "pelvic", "gyno", "gynec", "menstrual"],
        "reason": "Gynecological or pregnancy-related symptom reported.",
    },
    {
        "specialist": "General Physician",
        "urgency": "routine",
        "terms": ["fever", "cough", "fatigue", "weakness", "headache", "vomiting"],
        "reason": "General medical symptom reported.",
    },
]


@router.post("/triage/emergency", tags=["triage"])
async def emergency_triage(
    symptoms: str = Body("", embed=True),
    visual_summary: Optional[str] = Body(None, embed=True),
    temperature: Optional[float] = Body(None, embed=True),
) -> Dict[str, Any]:
    """Create a patient-facing triage summary and specialist recommendation."""
    text = " ".join(item for item in [symptoms, visual_summary or "", f"temperature {temperature}" if temperature else ""] if item)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Symptoms or visual summary are required.")

    extraction = extract_clinical_data(text)
    recommendation = _recommend_specialist(text, extraction)
    urgency = _highest_urgency([recommendation["urgency"], *_urgency_from_extraction(extraction, temperature)])

    return {
        "status": "success",
        "mode": "emergency",
        "urgency": urgency,
        "recommended_specialist": recommendation["specialist"],
        "reason": recommendation["reason"],
        "extracted_symptoms": extraction.get("symptoms", []),
        "risk_flags": extraction.get("risk_flags", []),
        "patient_summary": _patient_summary(urgency, recommendation, extraction),
        "next_steps": _patient_next_steps(urgency, recommendation["specialist"]),
        "disclaimer": "Triage support only. Seek qualified medical care for diagnosis and treatment.",
    }


@router.post("/doctor/second-opinion", tags=["doctor"])
async def doctor_second_opinion(
    patient_id: Optional[str] = Body(None, embed=True),
    clinical_findings: str = Body("", embed=True),
    doctor_plan: str = Body("", embed=True),
    patient_history: Optional[str] = Body(None, embed=True),
) -> Dict[str, Any]:
    """Review a clinician-entered plan and return cautious decision support."""
    combined = "\n".join(item for item in [patient_history or "", clinical_findings, doctor_plan] if item)
    if not combined.strip():
        raise HTTPException(status_code=400, detail="Clinical findings or doctor plan are required.")

    extraction = extract_clinical_data(combined)
    cautions = _doctor_cautions(combined, extraction)

    return {
        "status": "success",
        "mode": "doctor",
        "patient_id": patient_id,
        "detected_context": {
            "symptoms": extraction.get("symptoms", []),
            "important_findings": extraction.get("abnormal_labs", [])[:6],
            "risk_flags": extraction.get("risk_flags", []),
        },
        "second_opinion": {
            "summary": _doctor_summary(extraction, doctor_plan),
            "cautions": cautions,
            "suggested_checks": _suggested_checks(combined, extraction),
        },
        "disclaimer": "Clinical decision support only. Final decisions remain with the treating clinician.",
    }


def _recommend_specialist(text: str, extraction: Dict[str, Any]) -> Dict[str, str]:
    lowered = text.lower()
    for rule in SPECIALIST_RULES:
        if any(term in lowered for term in rule["terms"]):
            return {
                "specialist": rule["specialist"],
                "urgency": rule["urgency"],
                "reason": rule["reason"],
            }
    if extraction.get("abnormal_labs"):
        return {
            "specialist": "General Physician",
            "urgency": "routine",
            "reason": "Abnormal report values need clinical correlation.",
        }
    return {
        "specialist": "General Physician",
        "urgency": "routine",
        "reason": "No specific specialist pattern was detected.",
    }


def _urgency_from_extraction(extraction: Dict[str, Any], temperature: Optional[float]) -> List[str]:
    urgencies = []
    if extraction.get("risk_flags"):
        urgencies.append("soon")
    if temperature is not None and temperature >= 103:
        urgencies.append("emergency")
    return urgencies


def _highest_urgency(values: List[str]) -> str:
    order = {"routine": 1, "soon": 2, "emergency": 3}
    return max(values or ["routine"], key=lambda value: order.get(value, 1))


def _patient_summary(urgency: str, recommendation: Dict[str, str], extraction: Dict[str, Any]) -> str:
    symptoms = extraction.get("symptoms", [])
    symptom_text = ", ".join(symptoms) if symptoms else "reported symptoms"
    return (
        f"Triage level: {urgency}. Based on {symptom_text}, DiagnoBot suggests "
        f"{recommendation['specialist']} review. Reason: {recommendation['reason']}"
    )


def _patient_next_steps(urgency: str, specialist: str) -> List[str]:
    if urgency == "emergency":
        return [
            "Seek emergency medical care now.",
            "Do not wait for app results if symptoms are severe or worsening.",
            f"Share this triage summary with the {specialist} team.",
        ]
    if urgency == "soon":
        return [
            f"Book {specialist} review as soon as practical.",
            "Monitor symptoms and seek urgent care if pain, breathing difficulty, bleeding, or fever worsens.",
            "Carry uploaded reports/images to the consultation.",
        ]
    return [
        f"Schedule routine review with a {specialist}.",
        "Keep reports and symptom notes ready.",
        "Repeat or follow up tests only if advised by a clinician.",
    ]


def _doctor_cautions(text: str, extraction: Dict[str, Any]) -> List[str]:
    lowered = text.lower()
    cautions = []
    if extraction.get("risk_flags"):
        cautions.append("Risk flags are present; confirm clinical stability before routine management.")
    if "antibiotic" in lowered and not any(term in lowered for term in ["culture", "pus", "infection", "wbc"]):
        cautions.append("Antibiotic plan mentioned; verify infection evidence and local guidelines.")
    if "steroid" in lowered and any(term in lowered for term in ["infection", "wound", "fever"]):
        cautions.append("Steroid plan with possible infection symptoms needs careful review.")
    if not cautions:
        cautions.append("No major rule-based caution detected; correlate with examination and guidelines.")
    return cautions


def _suggested_checks(text: str, extraction: Dict[str, Any]) -> List[str]:
    lowered = text.lower()
    checks = []
    abnormal_keys = {lab.get("key") for lab in extraction.get("abnormal_labs", [])}
    if {"wbc_count", "neutrophils", "lymphocytes"}.intersection(abnormal_keys) or "fever" in lowered:
        checks.append("Review vitals, fever history, infection focus, and repeat CBC if needed.")
    if {"ldl", "triglyceride", "cholesterol"}.intersection(abnormal_keys):
        checks.append("Review cardiovascular risk factors and fasting lipid follow-up.")
    if "pain" in lowered or "fracture" in lowered:
        checks.append("Confirm pain severity, neurovascular status, and imaging findings.")
    if not checks:
        checks.append("Confirm diagnosis, red flags, allergies, medications, and follow-up plan.")
    return checks


def _doctor_summary(extraction: Dict[str, Any], doctor_plan: str) -> str:
    findings = extraction.get("summary", "No structured finding extracted.")
    plan_text = doctor_plan[:300] if doctor_plan else "No treatment plan entered."
    return f"Second opinion summary: {findings} Entered plan reviewed: {plan_text}"
