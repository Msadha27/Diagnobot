"""
SQLAlchemy ORM models for DiagnoBot database
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class AnalysisRecord(Base):
    """Stores each analysis request and its result."""

    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(String(50), nullable=False)  # xray, dermatology, nlp, report
    patient_id = Column(String(100), nullable=True)
    input_file = Column(String(500), nullable=True)
    result = Column(JSON, nullable=True)
    status = Column(String(20), default="pending")  # pending, success, error
    model_used = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class UploadedFile(Base):
    """Tracks uploaded files."""

    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String(100), unique=True, index=True)
    original_filename = Column(String(500))
    saved_path = Column(String(500))
    file_type = Column(String(50))  # image, pdf, text
    size_bytes = Column(Integer)
    content_type = Column(String(100))
    is_processed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class MedicalReport(Base):
    """Stores generated medical reports."""

    __tablename__ = "medical_reports"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(100), unique=True, index=True)
    patient_id = Column(String(100), nullable=True)
    report_type = Column(String(50))  # clinical_analysis, patient_input_conversion
    full_report = Column(Text)
    summary = Column(Text, nullable=True)
    generation_model = Column(String(100))
    clinical_findings = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
