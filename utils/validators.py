"""
Input validators for DiagnoBot API
"""

import re
from pathlib import Path
from typing import Optional


def validate_patient_id(patient_id: Optional[str]) -> bool:
    """Validate patient ID format (alphanumeric + hyphens)."""
    if patient_id is None:
        return True
    return bool(re.match(r"^[a-zA-Z0-9\-_]{1,50}$", patient_id))


def validate_image_path(path: str) -> bool:
    """Check that path points to a supported image file that exists."""
    p = Path(path)
    if not p.exists():
        return False
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".dcm"}


def validate_text_input(text: str, min_length: int = 5, max_length: int = 10_000) -> bool:
    """Validate free-form text input length."""
    cleaned = text.strip()
    return min_length <= len(cleaned) <= max_length


def sanitize_text(text: str) -> str:
    """Basic text sanitization – strip leading/trailing whitespace and null bytes."""
    return text.replace("\x00", "").strip()
