"""
Lightweight clinical text extraction for uploaded reports and patient notes.

This module is intentionally deterministic and fast. It gives the dashboard and
API a dependable extraction layer even when the local reasoning LLM is not
loaded yet.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


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

LAB_DEFINITIONS = [
    {"key": "hemoglobin", "label": "Hemoglobin", "aliases": ["Hemoglobin", "Hb"], "low": 13.0, "high": 16.5, "unit": "g/dL"},
    {"key": "rbc_count", "label": "RBC Count", "aliases": ["RBC Count"], "low": 4.5, "high": 5.5, "unit": "million/cmm"},
    {"key": "hematocrit", "label": "Hematocrit", "aliases": ["Hematocrit", "HCT"], "low": 40.0, "high": 49.0, "unit": "%"},
    {"key": "mcv", "label": "MCV", "aliases": ["MCV"], "low": 83.0, "high": 101.0, "unit": "fL"},
    {"key": "mch", "label": "MCH", "aliases": ["MCH"], "low": 27.1, "high": 32.5, "unit": "pg"},
    {"key": "mchc", "label": "MCHC", "aliases": ["MCHC"], "low": 32.5, "high": 36.7, "unit": "g/dL"},
    {"key": "rdw_cv", "label": "RDW CV", "aliases": ["RDW CV", "RDW"], "low": 11.6, "high": 14.0, "unit": "%"},
    {"key": "wbc_count", "label": "WBC Count", "aliases": ["WBC Count", "Total WBC"], "low": 4000.0, "high": 10000.0, "unit": "/cmm"},
    {"key": "neutrophils", "label": "Neutrophils", "aliases": ["Neutrophils"], "low": 40.0, "high": 80.0, "unit": "%"},
    {"key": "lymphocytes", "label": "Lymphocytes", "aliases": ["Lymphocytes"], "low": 20.0, "high": 40.0, "unit": "%"},
    {"key": "eosinophils", "label": "Eosinophils", "aliases": ["Eosinophils"], "low": 1.0, "high": 6.0, "unit": "%"},
    {"key": "monocytes", "label": "Monocytes", "aliases": ["Monocytes"], "low": 2.0, "high": 10.0, "unit": "%"},
    {"key": "platelet_count", "label": "Platelet Count", "aliases": ["Platelet Count", "Platelets"], "low": 150000.0, "high": 410000.0, "unit": "/cmm"},
    {"key": "mpv", "label": "MPV", "aliases": ["MPV"], "low": 7.5, "high": 10.3, "unit": "fL"},
    {"key": "esr", "label": "ESR", "aliases": ["ESR", "Erythrocyte Sedimentation Rate"], "low": 0.0, "high": 14.0, "unit": "mm/hr"},
    {"key": "cholesterol", "label": "Total Cholesterol", "aliases": ["Cholesterol"], "low": None, "high": 200.0, "unit": "mg/dL"},
    {"key": "triglyceride", "label": "Triglyceride", "aliases": ["Triglyceride", "Triglycerides"], "low": None, "high": 150.0, "unit": "mg/dL"},
    {"key": "hdl", "label": "HDL Cholesterol", "aliases": ["HDL Cholesterol", "HDL"], "low": 40.0, "high": None, "unit": "mg/dL", "higher_is_better": True},
    {"key": "ldl", "label": "Direct LDL", "aliases": ["Direct LDL", "LDL"], "low": None, "high": 100.0, "unit": "mg/dL"},
    {"key": "vldl", "label": "VLDL", "aliases": ["VLDL"], "low": 15.0, "high": 35.0, "unit": "mg/dL"},
    {"key": "chol_hdl_ratio", "label": "CHOL/HDL Ratio", "aliases": ["CHOL/HDL Ratio"], "low": None, "high": 5.0, "unit": "ratio"},
    {"key": "ldl_hdl_ratio", "label": "LDL/HDL Ratio", "aliases": ["LDL/HDL Ratio"], "low": None, "high": 3.5, "unit": "ratio"},
    {"key": "urine_ph", "label": "Urine pH", "aliases": ["Urine pH", "pH"], "low": 4.5, "high": 8.0, "unit": "pH"},
    {"key": "specific_gravity", "label": "Specific Gravity", "aliases": ["Specific Gravity", "Sp. Gravity"], "low": 1.005, "high": 1.030, "unit": ""},
    {"key": "urine_pus_cells", "label": "Urine Pus Cells", "aliases": ["Pus Cells", "WBC / HPF", "WBC/HPF"], "low": None, "high": 5.0, "unit": "/HPF"},
    {"key": "urine_rbc", "label": "Urine RBC", "aliases": ["RBC / HPF", "RBC /HPF", "RBC/HPF", "Red Blood Cells"], "low": None, "high": 2.0, "unit": "/HPF"},
    {"key": "epithelial_cells", "label": "Epithelial Cells", "aliases": ["Epithelial Cells"], "low": None, "high": 5.0, "unit": "/HPF"},
    {"key": "urine_protein", "label": "Urine Protein", "aliases": ["Protein", "Albumin"], "qualitative": True, "normal_values": ["negative", "nil", "absent", "trace"], "unit": ""},
    {"key": "urine_glucose", "label": "Urine Glucose", "aliases": ["Urine Glucose", "Sugar", "Glucose"], "qualitative": True, "normal_values": ["negative", "nil", "absent"], "unit": ""},
    {"key": "urine_ketones", "label": "Urine Ketones", "aliases": ["Ketones", "Ketone"], "qualitative": True, "normal_values": ["negative", "nil", "absent"], "unit": ""},
    {"key": "urine_nitrite", "label": "Urine Nitrite", "aliases": ["Nitrite"], "qualitative": True, "normal_values": ["negative", "nil", "absent"], "unit": ""},
    {"key": "urine_leukocyte_esterase", "label": "Leukocyte Esterase", "aliases": ["Leukocyte Esterase", "Leucocyte Esterase"], "qualitative": True, "normal_values": ["negative", "nil", "absent"], "unit": ""},
    {"key": "pregnancy_test", "label": "Urine Pregnancy Test", "aliases": ["Urine Pregnancy Test", "Pregnancy Test", "UPT"], "qualitative": True, "normal_values": ["negative"], "unit": "", "positive_is_expected": True},
    {"key": "beta_hcg", "label": "Beta hCG", "aliases": ["Beta hCG", "Beta-HCG", "bHCG", "hCG"], "low": None, "high": 5.0, "unit": "mIU/mL", "positive_threshold": 25.0},
]


def extract_clinical_data(text: str) -> Dict[str, Any]:
    cleaned = _clean_text(text)
    lowered = cleaned.lower()

    lab_results = _extract_lab_results(cleaned)
    abnormal_statuses = {"high", "low", "borderline_high", "borderline_low", "positive", "borderline_positive"}
    abnormal_labs = _sort_abnormal_labs(
        [lab for lab in lab_results if lab["status"] in abnormal_statuses]
    )
    normal_labs = [lab for lab in lab_results if lab["status"] == "normal"]
    labs = {lab["key"]: lab["value"] for lab in lab_results}

    is_urine_report = _is_urine_report(cleaned, lab_results)
    is_lab_report = len(lab_results) >= 3
    symptoms = [] if is_lab_report else _find_terms(lowered, SYMPTOM_TERMS)
    condition_hints = _condition_hints(symptoms, abnormal_labs)
    risk_flags = _risk_flags(lowered, symptoms, labs, abnormal_labs, is_lab_report)
    summary = _summary(cleaned, symptoms, condition_hints, risk_flags, abnormal_labs, normal_labs)

    return {
        "status": "success",
        "word_count": len(cleaned.split()),
        "character_count": len(cleaned),
        "symptoms": symptoms,
        "lab_values": labs,
        "lab_results": lab_results,
        "abnormal_labs": abnormal_labs,
        "normal_labs": normal_labs,
        "report_kind": "urine_pregnancy_report" if is_urine_report else "lab_report" if is_lab_report else "clinical_text",
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


def _extract_lab_results(text: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    matches = _find_lab_matches(text)

    for index, match_info in enumerate(matches):
        next_start = matches[index + 1]["start"] if index + 1 < len(matches) else match_info["start"] + 190
        parsed = _extract_one_lab(text, match_info, next_start)
        if parsed:
            results.append(parsed)

    return results


def _find_lab_matches(text: str) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    used_keys = set()

    for definition in LAB_DEFINITIONS:
        best_match = None
        for alias in definition["aliases"]:
            match = re.search(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE)
            if match and (best_match is None or match.start() < best_match.start()):
                best_match = match
        if best_match and definition["key"] not in used_keys:
            used_keys.add(definition["key"])
            matches.append({"definition": definition, "start": best_match.start(), "end": best_match.end()})

    return sorted(matches, key=lambda item: item["start"])


def _extract_one_lab(
    text: str,
    match_info: Dict[str, Any],
    next_start: int,
) -> Optional[Dict[str, Any]]:
    definition = match_info["definition"]
    window = _normalize_lab_window(text[match_info["start"]: min(next_start, match_info["start"] + 220)])
    if definition.get("qualitative"):
        return _extract_qualitative_lab(window, definition)

    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", window)]
    if not numbers:
        return None

    value = _pick_lab_value(numbers, definition)
    if value is None:
        return None

    status = _lab_status(value, definition)
    interpretation = _lab_interpretation(definition["key"], status, value)
    return {
        "key": definition["key"],
        "name": definition["label"],
        "value": value,
        "unit": definition["unit"],
        "reference_low": definition.get("low"),
        "reference_high": definition.get("high"),
        "status": status,
        "severity": _lab_severity(status, value, definition),
        "target": _lab_target(status, definition),
        "interpretation": interpretation,
        "suggestions": _lab_suggestions(definition["key"], status),
    }
    return None


def _extract_qualitative_lab(window: str, definition: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    normalized = window.lower()
    value = _qualitative_value(normalized)
    if value is None:
        return None

    normal_values = set(definition.get("normal_values", []))
    status = "normal" if value in normal_values else "positive"
    if definition.get("positive_is_expected") and value in {"positive", "detected", "present"}:
        status = "positive"

    interpretation = _lab_interpretation(definition["key"], status, value)
    return {
        "key": definition["key"],
        "name": definition["label"],
        "value": value,
        "unit": definition.get("unit", ""),
        "reference_low": definition.get("low"),
        "reference_high": definition.get("high"),
        "status": status,
        "severity": _lab_severity(status, 0, definition),
        "target": _lab_target(status, definition),
        "interpretation": interpretation,
        "suggestions": _lab_suggestions(definition["key"], status),
    }


def _qualitative_value(text: str) -> Optional[str]:
    for token in ["positive", "detected", "present", "negative", "absent", "nil", "trace", "+++", "++", "+"]:
        if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text):
            return token
    return None


def _normalize_lab_window(window: str) -> str:
    window = re.sub(r"(Colorimetric|impedance|Derived|Calculated|Microscopic|photometry|oxidase|method|measured)(\d)", r"\1 \2", window, flags=re.IGNORECASE)
    window = re.sub(r"(\d)(Calculated|Direct|VLDL|HDL|LDL|Triglyceride|Cholesterol|Platelet|MPV|ESR)", r"\1 \2", window, flags=re.IGNORECASE)
    window = re.sub(r"MgCl2?(\d)", r"MgCl2 \1", window, flags=re.IGNORECASE)
    return window


def _pick_lab_value(numbers: List[float], definition: Dict[str, Any]) -> Optional[float]:
    low = definition.get("low")
    high = definition.get("high")

    if definition["key"] == "hdl":
        plausible = [number for number in numbers if 10 <= number <= 120]
        return plausible[-1] if plausible else numbers[-1]
    if definition["key"] == "ldl":
        plausible = [number for number in numbers if 20 <= number <= 250]
        return plausible[-1] if plausible else numbers[-1]
    if definition["key"] == "vldl":
        plausible = [number for number in numbers if 1 <= number <= 100]
        return plausible[-1] if plausible else numbers[-1]
    if definition["key"] == "specific_gravity":
        plausible = [number for number in numbers if 1.0 <= number <= 1.05]
        return plausible[-1] if plausible else numbers[-1]
    if definition["key"] in {"urine_pus_cells", "urine_rbc", "epithelial_cells"}:
        plausible = [number for number in numbers if 0 <= number <= 200]
        return plausible[0] if plausible else numbers[0]
    if definition["key"] == "beta_hcg":
        plausible = [number for number in numbers if 0 <= number <= 500000]
        return plausible[-1] if plausible else numbers[-1]

    if len(numbers) >= 3:
        return numbers[-1]
    if len(numbers) == 1:
        return numbers[0]
    if low is not None and high is not None:
        candidates = [number for number in numbers if number != low and number != high]
        if candidates:
            return candidates[-1]
    return numbers[-1]


def _lab_status(value: float, definition: Dict[str, Any]) -> str:
    low = definition.get("low")
    high = definition.get("high")
    higher_is_better = definition.get("higher_is_better", False)

    if definition["key"] == "beta_hcg":
        if value >= definition.get("positive_threshold", 25.0):
            return "positive"
        if value > definition.get("high", 5.0):
            return "borderline_positive"
        return "normal"

    if low is not None and value < low:
        return "low" if not higher_is_better else "borderline_low"
    if high is not None and value > high:
        return "high" if not higher_is_better else "normal"
    return "normal"


def _lab_severity(status: str, value: float, definition: Dict[str, Any]) -> str:
    if status == "normal":
        return "none"
    if status in {"positive", "borderline_positive"}:
        if definition["key"] in {"urine_nitrite", "urine_leukocyte_esterase", "urine_protein", "urine_glucose", "urine_ketones"}:
            return "medium"
        if definition["key"] in {"pregnancy_test", "beta_hcg"}:
            return "info"
        return "medium"
    high = definition.get("high")
    low = definition.get("low")
    if high and value > high * 1.25:
        return "high"
    if low and value < low * 0.75:
        return "high"
    if status.startswith("borderline"):
        return "low"
    return "medium"


def _sort_abnormal_labs(labs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    priority_keys = {
        "pregnancy_test": 0,
        "beta_hcg": 1,
        "urine_nitrite": 2,
        "urine_leukocyte_esterase": 3,
        "urine_pus_cells": 4,
        "urine_rbc": 5,
        "urine_protein": 6,
        "urine_glucose": 7,
        "urine_ketones": 8,
        "wbc_count": 9,
        "mpv": 10,
        "platelet_count": 11,
        "triglyceride": 12,
        "ldl": 13,
    }
    severity_rank = {"high": 0, "medium": 1, "info": 2, "low": 3, "none": 4}
    status_rank = {"positive": 0, "borderline_positive": 1, "high": 2, "low": 2}

    return sorted(
        labs,
        key=lambda lab: (
            priority_keys.get(lab.get("key"), 50),
            severity_rank.get(lab.get("severity"), 5),
            status_rank.get(lab.get("status"), 5),
        ),
    )


def _lab_target(status: str, definition: Dict[str, Any]) -> str:
    low = definition.get("low")
    high = definition.get("high")
    if definition.get("qualitative"):
        if status == "normal":
            return "maintain negative/normal result"
        if definition["key"] == "pregnancy_test":
            return "confirm pregnancy status with clinician/serial beta hCG if needed"
        return "bring back to negative/normal if clinically abnormal"
    if definition["key"] == "beta_hcg" and status in {"positive", "borderline_positive"}:
        return "interpret against pregnancy timing; repeat/confirm if clinically needed"
    if status in {"high", "borderline_high"} and high is not None:
        return f"bring toward <= {high:g} {definition['unit']}"
    if status in {"low", "borderline_low"} and low is not None:
        return f"increase toward >= {low:g} {definition['unit']}"
    return "maintain within reference range"


def _lab_interpretation(key: str, status: str, value: float) -> str:
    if status == "normal":
        return "Within the configured local reference range."

    interpretations = {
        "wbc_count": {
            "high": "May indicate infection, inflammation, stress response, or other immune activation.",
            "low": "May indicate reduced immune cell count; clinical correlation is needed.",
        },
        "lymphocytes": {
            "low": "Can be seen with acute stress, some infections, steroid use, or immune suppression.",
            "high": "Can be seen with some viral infections or chronic immune stimulation.",
        },
        "mpv": {
            "high": "Suggests larger platelets on average; interpret together with platelet count and inflammation markers.",
            "low": "Suggests smaller platelets on average; interpret with platelet count and clinical context.",
        },
        "triglyceride": {
            "high": "Associated with increased cardiometabolic risk and may reflect diet, insulin resistance, alcohol intake, or genetics.",
        },
        "ldl": {
            "high": "LDL above the configured optimal threshold may increase cardiovascular risk over time.",
        },
        "hdl": {
            "low": "HDL below the configured threshold may reduce protective lipid profile.",
        },
        "hemoglobin": {
            "low": "May suggest anemia or blood loss depending on age, sex, and clinical context.",
            "high": "May suggest hemoconcentration or other causes; clinical correlation is needed.",
        },
        "platelet_count": {
            "low": "May increase bleeding risk depending on severity; clinician review is needed.",
            "high": "May reflect inflammation, iron deficiency, or other causes; clinician review is needed.",
        },
        "esr": {
            "high": "Non-specific marker that can rise with inflammation or infection.",
        },
        "urine_pus_cells": {
            "high": "Elevated pus cells can suggest urinary tract inflammation or infection.",
        },
        "urine_rbc": {
            "high": "Blood cells in urine may occur with infection, stones, contamination, trauma, or kidney/urinary disease.",
        },
        "epithelial_cells": {
            "high": "High epithelial cells can reflect sample contamination or urinary tract irritation.",
        },
        "specific_gravity": {
            "high": "Concentrated urine may suggest dehydration or reduced fluid intake.",
            "low": "Dilute urine may reflect high fluid intake or impaired concentrating ability; clinical context matters.",
        },
        "urine_ph": {
            "high": "Alkaline urine can be diet-related or seen with some infections.",
            "low": "Acidic urine can be diet-related, dehydration-related, or metabolic; clinical context matters.",
        },
        "urine_protein": {
            "positive": "Protein in urine can suggest kidney stress, infection, hypertension-related changes, or pregnancy-related concern.",
        },
        "urine_glucose": {
            "positive": "Glucose in urine may suggest elevated blood glucose or diabetes risk, especially if persistent.",
        },
        "urine_ketones": {
            "positive": "Ketones in urine can occur with fasting, dehydration, vomiting, diabetes, or pregnancy-related nausea.",
        },
        "urine_nitrite": {
            "positive": "Positive nitrite can support possible bacterial urinary tract infection.",
        },
        "urine_leukocyte_esterase": {
            "positive": "Positive leukocyte esterase can support urinary tract inflammation or infection.",
        },
        "pregnancy_test": {
            "positive": "Positive urine pregnancy test suggests pregnancy and should be confirmed/dated clinically.",
        },
        "beta_hcg": {
            "positive": "Beta hCG above the positive threshold supports pregnancy or hCG-producing conditions; interpret with dates and clinical context.",
            "borderline_positive": "Borderline beta hCG may need repeat testing in 48 hours or clinician correlation.",
        },
    }
    return interpretations.get(key, {}).get(
        status,
        f"Value is {status.replace('_', ' ')} compared with the configured reference threshold.",
    )


def _lab_suggestions(key: str, status: str) -> List[str]:
    if status == "normal":
        return ["Maintain current healthy habits and routine follow-up as advised."]

    suggestions = {
        "wbc_count": {
            "high": [
                "Correlate with fever, cough, wound, urinary symptoms, or other infection signs.",
                "Repeat CBC or consult a clinician if symptoms are present or value stays elevated.",
                "Avoid self-medicating with antibiotics without medical advice.",
            ],
            "low": [
                "Review recent viral illness, medications, and immune history with a clinician.",
                "Repeat CBC if advised to confirm the trend.",
            ],
        },
        "lymphocytes": {
            "low": [
                "Correlate with recent illness, stress, medication use, and immune history.",
                "Repeat CBC if clinically indicated.",
            ],
            "high": [
                "Review for viral symptoms or persistent lymph node swelling with a clinician.",
            ],
        },
        "mpv": {
            "high": [
                "Interpret with platelet count rather than alone.",
                "Review inflammation, cardiovascular risk, and recent illness with a clinician.",
            ],
        },
        "triglyceride": {
            "high": [
                "Reduce sugary drinks, refined carbohydrates, fried foods, and excess alcohol.",
                "Increase regular aerobic activity if medically safe.",
                "Discuss fasting repeat lipid profile and cardiometabolic risk with a clinician.",
            ],
        },
        "ldl": {
            "high": [
                "Reduce saturated fat and trans fat; prefer fiber-rich foods, nuts, legumes, and vegetables.",
                "Increase regular exercise if medically safe.",
                "Discuss cardiovascular risk and whether lifestyle-only management is enough.",
            ],
        },
        "hdl": {
            "low": [
                "Increase regular physical activity if medically safe.",
                "Avoid smoking and improve dietary quality with unsaturated fats from nuts, seeds, and fish.",
            ],
        },
        "hemoglobin": {
            "low": [
                "Review iron, B12, folate intake, bleeding history, and fatigue symptoms with a clinician.",
                "Do not start iron unless deficiency is confirmed or advised.",
            ],
            "high": [
                "Ensure hydration and review smoking, altitude exposure, or breathing issues with a clinician.",
            ],
        },
        "platelet_count": {
            "low": [
                "Seek clinician review, especially if bruising, bleeding, or very low values are present.",
            ],
            "high": [
                "Review inflammation, iron deficiency, recent infection, and repeat testing with a clinician.",
            ],
        },
        "esr": {
            "high": [
                "Correlate with pain, fever, infection symptoms, autoimmune symptoms, and CRP if available.",
            ],
        },
        "urine_pus_cells": {
            "high": [
                "Correlate with burning urination, fever, urgency, flank pain, and urine culture if symptomatic.",
                "Hydrate unless medically restricted and consult a clinician if symptoms are present.",
            ],
        },
        "urine_rbc": {
            "high": [
                "Repeat clean-catch urine test if contamination is possible.",
                "Review stones, infection symptoms, menstruation contamination, trauma, and kidney history with a clinician.",
            ],
        },
        "epithelial_cells": {
            "high": [
                "Consider repeat clean-catch sample because contamination is common.",
            ],
        },
        "specific_gravity": {
            "high": [
                "Increase water intake if medically safe and review dehydration symptoms.",
            ],
            "low": [
                "Review fluid intake, medication use, and kidney concentrating issues if persistent.",
            ],
        },
        "urine_ph": {
            "high": [
                "Correlate with diet and UTI markers such as nitrite, leukocyte esterase, and pus cells.",
            ],
            "low": [
                "Review hydration, diet, vomiting/diarrhea, diabetes risk, and ketones if present.",
            ],
        },
        "urine_protein": {
            "positive": [
                "Repeat urine test and review blood pressure, kidney history, swelling, and pregnancy status.",
                "In pregnancy, proteinuria should be reviewed promptly with blood pressure and obstetric assessment.",
            ],
        },
        "urine_glucose": {
            "positive": [
                "Check blood glucose/HbA1c or gestational diabetes screening if pregnant.",
                "Review diabetes symptoms such as thirst, frequent urination, and weight changes.",
            ],
        },
        "urine_ketones": {
            "positive": [
                "Review fasting, vomiting, dehydration, diabetes risk, or pregnancy nausea.",
                "Hydrate and seek care if vomiting, weakness, or high blood sugar is present.",
            ],
        },
        "urine_nitrite": {
            "positive": [
                "Consider UTI evaluation and urine culture, especially with urinary symptoms or pregnancy.",
            ],
        },
        "urine_leukocyte_esterase": {
            "positive": [
                "Correlate with pus cells, nitrite, symptoms, and urine culture if needed.",
            ],
        },
        "pregnancy_test": {
            "positive": [
                "Confirm pregnancy status and estimate gestational age with clinician guidance.",
                "Begin/continue prenatal care and avoid medications not reviewed for pregnancy safety.",
            ],
        },
        "beta_hcg": {
            "positive": [
                "Interpret beta hCG with last menstrual period, ultrasound timing, and repeat levels if advised.",
                "Seek urgent care for severe pelvic pain, dizziness, or bleeding.",
            ],
            "borderline_positive": [
                "Repeat beta hCG after 48 hours or follow clinician advice.",
            ],
        },
    }
    return suggestions.get(key, {}).get(
        status,
        ["Review this abnormal value with a qualified clinician and repeat testing if advised."],
    )


def _condition_hints(symptoms: List[str], abnormal_labs: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    symptom_set = set(symptoms)
    hints = []
    for hint, terms in CONDITION_HINTS.items():
        if symptom_set.intersection(terms):
            hints.append(hint)
    abnormal_keys = {lab["key"] for lab in abnormal_labs or []}
    if {"wbc_count", "neutrophils"}.intersection(abnormal_keys):
        hints.append("possible infection/inflammation marker")
    if {"triglyceride", "ldl", "cholesterol"}.intersection(abnormal_keys):
        hints.append("cardiometabolic/lipid risk review")
    if {"platelet_count", "mpv"}.intersection(abnormal_keys):
        hints.append("platelet index review")
    if {"urine_pus_cells", "urine_nitrite", "urine_leukocyte_esterase"}.intersection(abnormal_keys):
        hints.append("urinary tract infection/inflammation review")
    if {"urine_protein", "urine_glucose", "urine_ketones"}.intersection(abnormal_keys):
        hints.append("metabolic/kidney/pregnancy urine marker review")
    if {"pregnancy_test", "beta_hcg"}.intersection(abnormal_keys):
        hints.append("pregnancy confirmation/obstetric review")
    return hints


def _risk_flags(
    lowered: str,
    symptoms: List[str],
    labs: Dict[str, float],
    abnormal_labs: Optional[List[Dict[str, Any]]] = None,
    is_lab_report: bool = False,
) -> List[str]:
    flags = []
    urgent_terms = ["severe", "unconscious", "difficulty breathing", "heavy bleeding", "chest pain"]
    if not is_lab_report and any(term in lowered for term in urgent_terms):
        flags.append("urgent clinical review")
    if "high fever" in symptoms:
        flags.append("fever red flag")
    if labs.get("temperature", 0) >= 103:
        flags.append("very high temperature")
    if labs.get("bilirubin", 0) >= 2:
        flags.append("raised bilirubin/jaundice concern")
    if labs.get("wbc", 0) >= 11000:
        flags.append("possible elevated WBC")
    for lab in abnormal_labs or []:
        if lab["key"] == "wbc_count" and lab["status"] in {"high", "borderline_high"}:
            flags.append("WBC above reference range")
        if lab["key"] == "mpv" and lab["status"] == "high":
            flags.append("MPV above reference range")
        if lab["key"] == "triglyceride" and lab["status"] in {"high", "borderline_high"}:
            flags.append("triglyceride above desirable range")
        if lab["key"] == "ldl" and lab["status"] in {"high", "borderline_high"}:
            flags.append("LDL above optimal range")
        if lab["key"] in {"urine_nitrite", "urine_leukocyte_esterase", "urine_pus_cells"} and lab["status"] in {"high", "positive"}:
            flags.append("possible UTI/inflammation marker")
        if lab["key"] == "urine_protein" and lab["status"] == "positive":
            flags.append("urine protein positive")
        if lab["key"] == "urine_glucose" and lab["status"] == "positive":
            flags.append("urine glucose positive")
        if lab["key"] == "urine_ketones" and lab["status"] == "positive":
            flags.append("urine ketones positive")
        if lab["key"] in {"pregnancy_test", "beta_hcg"} and lab["status"] in {"positive", "borderline_positive"}:
            flags.append("pregnancy marker positive/borderline")
    return list(dict.fromkeys(flags))


def _is_urine_report(text: str, lab_results: List[Dict[str, Any]]) -> bool:
    lowered = text.lower()
    urine_terms = ["urine", "urinalysis", "routine urine", "pregnancy test", "upt", "pus cells", "specific gravity"]
    urine_keys = {
        "urine_ph",
        "specific_gravity",
        "urine_pus_cells",
        "urine_rbc",
        "epithelial_cells",
        "urine_protein",
        "urine_glucose",
        "urine_ketones",
        "urine_nitrite",
        "urine_leukocyte_esterase",
        "pregnancy_test",
        "beta_hcg",
    }
    return any(term in lowered for term in urine_terms) or any(lab["key"] in urine_keys for lab in lab_results)


def _summary(
    text: str,
    symptoms: List[str],
    condition_hints: List[str],
    risk_flags: List[str],
    abnormal_labs: Optional[List[Dict[str, Any]]] = None,
    normal_labs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    parts = []
    if abnormal_labs:
        formatted = ", ".join(
            f"{lab['name']} {_format_lab_value(lab['value'])} {lab['unit']} ({lab['status'].replace('_', ' ')})"
            for lab in abnormal_labs[:6]
        )
        parts.append(f"Abnormal/borderline lab findings: {formatted}.")
    elif normal_labs:
        parts.append(f"Extracted {len(normal_labs)} lab values with no major abnormality detected by local ranges.")
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


def _format_lab_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)
