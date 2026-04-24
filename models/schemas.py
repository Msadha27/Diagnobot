"""
Pydantic schemas for DiagnoBot API request/response models
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ==================== ENUMS ====================

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    UNKNOWN = "unknown"


class AnalysisType(str, Enum):
    XRAY = "xray"
    DERMATOLOGY = "dermatology"
    NLP = "nlp"
    REPORT = "report"


# ==================== PATIENT ====================

class PatientInfo(BaseModel):
    patient_id: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=130)
    gender: Optional[str] = None
    medical_history: Optional[str] = None


# ==================== X-RAY ====================

class XRayFinding(BaseModel):
    type: str
    name: Optional[str] = None
    description: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    detected: bool = True
    source: str = "moondream2"


class XRayAnalysisResponse(BaseModel):
    status: str
    image_path: str
    image_size: tuple
    findings: List[Dict[str, Any]]
    clinical_summary: str
    anomaly_count: int
    recommendations: List[str]


# ==================== DERMATOLOGY ====================

class SkinClassification(BaseModel):
    disease: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: SeverityLevel
    code: str


class DermatologyResponse(BaseModel):
    status: str
    image_path: str
    classification: SkinClassification
    clinical_advice: Dict[str, Any]
    all_predictions: Optional[List[Dict[str, Any]]] = None


# ==================== REPORT ====================

class ReportFromContextRequest(BaseModel):
    clinical_findings: Dict[str, Any]
    patient_info: Optional[PatientInfo] = None
    max_length: int = Field(512, ge=64, le=2048)


class PatientInputReportRequest(BaseModel):
    patient_input: str = Field(..., min_length=10)
    symptoms: Optional[List[str]] = None
    medical_history: Optional[str] = None


class SummarizeReportRequest(BaseModel):
    report_text: str = Field(..., min_length=50)
    max_length: int = Field(256, ge=64, le=1024)


class ReportResponse(BaseModel):
    status: str
    full_report: Optional[str] = None
    summary: Optional[str] = None
    formal_report: Optional[str] = None
    generation_model: str


# ==================== NLP ====================

class NLPAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=5)
    extract_symptoms: bool = True


class NLPAnalysisResponse(BaseModel):
    status: str
    input_text: str
    word_count: int
    model: str
    entities: Optional[List[Dict[str, Any]]] = None


# ==================== UPLOAD ====================

class UploadResponse(BaseModel):
    status: str
    message: str
    original_filename: Optional[str] = None
    saved_as: str
    file_id: str
    size_bytes: int
