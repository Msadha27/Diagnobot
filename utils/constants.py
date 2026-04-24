"""
Application-wide constants and enumerations
"""

from enum import Enum


# ==================== MODEL NAMES ====================

class ModelName:
    MOONDREAM2 = "moondream2"
    DERM_CNN = "derm_cnn"
    XRAY_VISION = "xray_vision"
    CLINICAL_BERT = "clinical_bert"
    BIOGPT = "biogpt"
    BIOBART = "biobart"
    CLINICAL_T5 = "clinical_t5"


# ==================== ANALYSIS TYPES ====================

class AnalysisType(str, Enum):
    XRAY = "xray"
    DERMATOLOGY = "dermatology"
    NLP = "nlp"
    REPORT = "report"


# ==================== STATUS CODES ====================

class Status(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    PARTIAL = "partial"


# ==================== SEVERITY ====================

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    UNKNOWN = "unknown"


# ==================== SKIN DISEASES (HAM10000) ====================

SKIN_DISEASE_CLASSES = {
    0: {"name": "Melanoma", "severity": Severity.URGENT, "code": "MEL"},
    1: {"name": "Melanocytic nevus", "severity": Severity.LOW, "code": "NV"},
    2: {"name": "Basal cell carcinoma", "severity": Severity.HIGH, "code": "BCC"},
    3: {"name": "Actinic keratosis", "severity": Severity.MEDIUM, "code": "AK"},
    4: {"name": "Benign keratosis", "severity": Severity.LOW, "code": "BKL"},
    5: {"name": "Dermatofibroma", "severity": Severity.LOW, "code": "DF"},
    6: {"name": "Vascular lesion", "severity": Severity.MEDIUM, "code": "VASC"},
}

# ==================== X-RAY ANOMALIES ====================

XRAY_ANOMALY_CLASSES = [
    "pneumonia", "tuberculosis", "nodule", "mass",
    "consolidation", "infiltrate", "pneumothorax",
    "pleural effusion", "atelectasis", "fibrosis",
    "emphysema", "cardiomegaly",
]

# ==================== FILE LIMITS ====================

MAX_IMAGE_MB = 50
MAX_PDF_MB = 100
MAX_BATCH_FILES = 10

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/tiff"}
ALLOWED_DOC_TYPES = {"application/pdf", "text/plain", "text/csv"}
