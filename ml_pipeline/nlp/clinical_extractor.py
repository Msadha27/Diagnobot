"""
Lightweight clinical text extraction for uploaded reports and patient notes.

This module is intentionally deterministic and fast. It gives the dashboard and
API a dependable extraction layer even when the local reasoning LLM is not
loaded yet.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


SYMPTOM_TERMS = [
    "fever",
    "high fever",
    "cough",
    "headache",
    "vomiting",
    "nausea",
    "fatigue",
    "weakness",
    "pain",
    "chest pain",
    "shortness of breath",
    "rash",
    "itching",
    "redness",
    "swelling",
    "discharge",
    "bleeding",
    "wound",
    "ulcer",
    "blurred vision",
    "yellow eyes",
    "red eyes",
    "pale eyes",
]

CONDITION_HINTS = {
    "infection/inflammation": ["fever", "high fever", "swelling", "redness", "discharge", "wound"],
    "respiratory concern": ["cough", "shortness of breath", "chest pain"],
    "dermatology concern": ["rash", "itching", "redness", "acne", "eczema", "wound", "ulcer"],
    "eye/systemic concern": ["yellow eyes", "red eyes", "pale eyes", "blurred vision"],
}

LAB_PATTERNS = {
    "hemoglobin": r"\b(?:hb|hemoglobin)\s*[:=-]?\s*(\d+(?:\.\d+)?)",
    "wbc": r"\b(?:wbc|white blood cells?)\s*[:=-]?\s*(\d+(?:\.\d+)?)",
    "platelets": r"\b(?:platelets?|plt)\s*[:=-]?\s*(\d+(?:\.\d+)?)",
    "glucose": r"\b(?:glucose|blood sugar|rbs|fbs)\s*[:=-]?\s*(\d+(?:\.\d+)?)",
    "bilirubin": r"\b(?:bilirubin)\s*[:=-]?\s*(\d+(?:\.\d+)?)",
    "temperature": r"\b(?:temperature|temp|fever)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(?:f|c|°f|°c)?",
}


def extract_clinical_data(text: str) -> Dict[str, Any]:
    cleaned = _clean_text(text)
    lowered = cleaned.lower()

    symptoms = _find_terms(lowered, SYMPTOM_TERMS)
    labs = _extract_labs(cleaned)
    condition_hints = _condition_hints(symptoms)
    risk_flags = _risk_flags(lowered, symptoms, labs)
    summary = _summary(cleaned, symptoms, condition_hints, risk_flags)

    return {
        "status": "success",
        "word_count": len(cleaned.split()),
        "character_count": len(cleaned),
        "symptoms": symptoms,
        "lab_values": labs,
        "condition_hints": condition_hints,
        "risk_flags": risk_flags,
        "summary": summary,
        "extracted_text_preview": cleaned[:800],
        "disclaimer": "Rule-based extraction for decision support only; clinician review is required.",
    }


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def _find_terms(lowered: str, terms: List[str]) -> List[str]:
    found = []
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            found.append(term)
    return found


def _extract_labs(text: str) -> Dict[str, float]:
    labs: Dict[str, float] = {}
    for name, pattern in LAB_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                labs[name] = float(match.group(1))
            except ValueError:
                continue
    return labs


def _condition_hints(symptoms: List[str]) -> List[str]:
    symptom_set = set(symptoms)
    hints = []
    for hint, terms in CONDITION_HINTS.items():
        if symptom_set.intersection(terms):
            hints.append(hint)
    return hints


def _risk_flags(lowered: str, symptoms: List[str], labs: Dict[str, float]) -> List[str]:
    flags = []
    urgent_terms = ["severe", "unconscious", "difficulty breathing", "heavy bleeding", "chest pain"]
    if any(term in lowered for term in urgent_terms):
        flags.append("urgent clinical review")
    if "high fever" in symptoms:
        flags.append("fever red flag")
    if labs.get("temperature", 0) >= 103:
        flags.append("very high temperature")
    if labs.get("bilirubin", 0) >= 2:
        flags.append("raised bilirubin/jaundice concern")
    if labs.get("wbc", 0) >= 11000:
        flags.append("possible elevated WBC")
    return flags


def _summary(
    text: str,
    symptoms: List[str],
    condition_hints: List[str],
    risk_flags: List[str],
) -> str:
    parts = []
    if symptoms:
        parts.append(f"Extracted symptoms: {', '.join(symptoms)}.")
    if condition_hints:
        parts.append(f"Possible clinical areas to review: {', '.join(condition_hints)}.")
    if risk_flags:
        parts.append(f"Risk flags: {', '.join(risk_flags)}.")
    if not parts:
        first_sentence = text.split(".")[0][:240]
        parts.append(first_sentence or "No obvious symptoms or lab signals were extracted.")
    parts.append("Use this as a triage summary, not a diagnosis.")
    return " ".join(parts)
