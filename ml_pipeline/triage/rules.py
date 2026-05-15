"""
Explainable triage rules for combining visual, symptom, and report signals.

The goal is not diagnosis. These rules turn extracted warning signs into a
transparent urgency recommendation that can be shown and challenged.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


EMERGENCY_TERMS = {
    "difficulty breathing": "difficulty breathing reported",
    "shortness of breath": "shortness of breath reported",
    "chest pain": "chest pain reported",
    "unconscious": "loss of consciousness reported",
    "heavy bleeding": "heavy bleeding reported",
    "necrosis": "possible dead/dark tissue described",
    "rapidly spreading": "rapidly spreading symptoms described",
    "sepsis": "sepsis-related term reported",
}

SOON_TERMS = {
    "rash": "rash described",
    "lesion": "lesion described",
    "pus": "possible pus/discharge described",
    "discharge": "possible discharge described",
    "swelling": "swelling described",
    "redness": "redness described",
    "pain": "pain described",
    "wound": "wound described",
    "ulcer": "ulcer described",
    "bleeding": "bleeding described",
    "fever": "fever reported",
    "yellow eyes": "yellow eyes reported",
    "blurred vision": "blurred vision reported",
}

SPECIALIST_BY_MODE = {
    "skin": "Dermatologist",
    "dermatology": "Dermatologist",
    "wound": "Dermatologist / General Surgeon",
    "eye": "Ophthalmologist",
    "fever": "General Physician",
    "xray": "Emergency Medicine / Pulmonologist",
}


def build_triage_assessment(
    *,
    symptoms: Optional[str] = None,
    visual_summary: Optional[str] = None,
    temperature: Optional[float] = None,
    classification: Optional[Dict[str, Any]] = None,
    extraction: Optional[Dict[str, Any]] = None,
    mode: str = "general",
) -> Dict[str, Any]:
    """Build an explainable triage recommendation from available signals."""

    text = _combined_text(symptoms, visual_summary, extraction)
    lowered = text.lower()
    reasons: List[str] = []
    red_flags: List[str] = []
    urgency_votes: List[str] = []
    has_clinical_signal = False

    for term, reason in EMERGENCY_TERMS.items():
        if _has_positive_term(lowered, term):
            urgency_votes.append("emergency")
            red_flags.append(reason)
            has_clinical_signal = True

    for term, reason in SOON_TERMS.items():
        if _has_positive_term(lowered, term):
            urgency_votes.append("soon")
            reasons.append(reason)
            has_clinical_signal = True

    if temperature is not None:
        if temperature >= 103:
            urgency_votes.append("emergency")
            red_flags.append(f"temperature {temperature:g} F is very high")
            has_clinical_signal = True
        elif temperature >= 100.4:
            urgency_votes.append("soon")
            reasons.append(f"temperature {temperature:g} F suggests fever")
            has_clinical_signal = True

    has_clinical_signal = _add_extraction_votes(extraction, urgency_votes, reasons, red_flags) or has_clinical_signal
    _add_classification_votes(classification, urgency_votes, reasons, red_flags, has_clinical_signal)

    urgency = _highest_urgency(urgency_votes)
    specialist = _recommended_specialist(mode, lowered, extraction)

    return {
        "urgency": urgency,
        "recommended_specialist": specialist,
        "reasons": _unique(reasons)[:8],
        "red_flags": _unique(red_flags)[:8],
        "next_steps": _next_steps(urgency, specialist),
        "confidence_policy": _confidence_policy(classification),
        "rule_basis": [
            "visible red flags",
            "reported symptoms",
            "temperature threshold",
            "extracted lab/report risk flags",
            "classifier severity and uncertainty",
        ],
        "safety_note": "Triage support only; this is not a diagnosis or treatment plan.",
    }


def _combined_text(
    symptoms: Optional[str],
    visual_summary: Optional[str],
    extraction: Optional[Dict[str, Any]],
) -> str:
    parts = [symptoms or "", visual_summary or ""]
    if extraction:
        parts.extend(extraction.get("symptoms", []))
        parts.extend(extraction.get("risk_flags", []))
        parts.extend(extraction.get("condition_hints", []))
        parts.append(extraction.get("summary", ""))
    return " ".join(str(part) for part in parts if part)


def _add_extraction_votes(
    extraction: Optional[Dict[str, Any]],
    urgency_votes: List[str],
    reasons: List[str],
    red_flags: List[str],
) -> bool:
    if not extraction:
        return False

    found_signal = False
    for flag in extraction.get("risk_flags", []):
        lowered = str(flag).lower()
        if "urgent" in lowered or "very high temperature" in lowered:
            urgency_votes.append("emergency")
            red_flags.append(str(flag))
        else:
            urgency_votes.append("soon")
            reasons.append(str(flag))
        found_signal = True

    for lab in extraction.get("abnormal_labs", [])[:6]:
        severity = lab.get("severity")
        status = lab.get("status")
        name = lab.get("name", "abnormal lab")
        if severity == "high":
            urgency_votes.append("soon")
            reasons.append(f"{name} has high-severity abnormality")
            found_signal = True
        elif status in {"positive", "high", "low", "borderline_positive"}:
            urgency_votes.append("soon")
            reasons.append(f"{name} is {str(status).replace('_', ' ')}")
            found_signal = True

    return found_signal


def _add_classification_votes(
    classification: Optional[Dict[str, Any]],
    urgency_votes: List[str],
    reasons: List[str],
    red_flags: List[str],
    has_clinical_signal: bool,
) -> None:
    if not classification:
        return

    label = classification.get("disease") or classification.get("label") or classification.get("top_match")
    severity = str(classification.get("severity", "")).lower()
    confidence = classification.get("confidence")

    if severity == "urgent":
        urgency_votes.append("emergency")
        red_flags.append(f"classifier flagged urgent severity for {label}")
    elif severity == "high":
        urgency_votes.append("soon")
        reasons.append(f"classifier flagged high severity for {label}")
    elif severity == "medium":
        urgency_votes.append("soon")
        reasons.append(f"classifier flagged medium severity for {label}")

    if has_clinical_signal and isinstance(confidence, (int, float)) and confidence < 0.45:
        urgency_votes.append("soon")
        reasons.append("classifier confidence is low, so clinician review is preferred")


def _recommended_specialist(
    mode: str,
    lowered_text: str,
    extraction: Optional[Dict[str, Any]],
) -> str:
    if any(term in lowered_text for term in ["difficulty breathing", "chest pain", "unconscious", "heavy bleeding"]):
        return "Emergency Medicine"
    if any(term in lowered_text for term in ["eye", "sclera", "vision", "yellow eyes", "red eyes"]):
        return "Ophthalmologist"
    if any(term in lowered_text for term in ["wound", "ulcer", "rash", "skin", "itching", "discharge"]):
        return "Dermatologist"
    if extraction and extraction.get("abnormal_labs"):
        return "General Physician"
    return SPECIALIST_BY_MODE.get(mode, "General Physician")


def _has_positive_term(text: str, term: str) -> bool:
    """Return True when a term appears outside a nearby negation phrase."""
    pattern = re.compile(rf"\b{re.escape(term)}\b")
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 90):match.start()]
        if not _is_negated_context(prefix):
            return True
    return False


def _is_negated_context(prefix: str) -> bool:
    negation_patterns = [
        r"\bno\b.{0,80}$",
        r"\bnot\b.{0,80}$",
        r"\bwithout\b.{0,80}$",
        r"\bnone\b.{0,80}$",
        r"\babsent\b.{0,80}$",
        r"\bdenies\b.{0,80}$",
        r"\bnegative for\b.{0,80}$",
        r"\bno visible\b.{0,80}$",
        r"\bno signs? of\b.{0,80}$",
        r"\bnot accompanied by\b.{0,80}$",
    ]
    return any(re.search(pattern, prefix) for pattern in negation_patterns)


def _highest_urgency(votes: Iterable[str]) -> str:
    order = {"routine": 1, "soon": 2, "emergency": 3}
    values = list(votes) or ["routine"]
    return max(values, key=lambda value: order.get(value, 1))


def _next_steps(urgency: str, specialist: str) -> List[str]:
    if urgency == "emergency":
        return [
            "Seek emergency medical care now.",
            "Do not wait for app output if symptoms are severe or worsening.",
            f"Share image/report findings with the {specialist} team.",
        ]
    if urgency == "soon":
        return [
            f"Arrange {specialist} review as soon as practical.",
            "Monitor fever, pain, spreading redness, bleeding, discharge, or breathing symptoms.",
            "Carry uploaded images, reports, and symptom notes to the consultation.",
        ]
    return [
        f"Schedule routine review with a {specialist}.",
        "Keep symptom notes and reports ready for the visit.",
        "Escalate sooner if new red flags appear.",
    ]


def _confidence_policy(classification: Optional[Dict[str, Any]]) -> str:
    if not classification:
        return "No classifier confidence was available; recommendation is based on visible/text red flags."
    confidence = classification.get("confidence")
    if isinstance(confidence, (int, float)) and confidence < 0.45:
        return "Low classifier confidence: treat class label as a visual hint and prioritize clinician review."
    return "Classifier output is used as a triage signal, not as a diagnosis."


def _unique(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(item for item in items if item))
