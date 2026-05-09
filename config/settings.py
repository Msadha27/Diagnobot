"""
Settings and configuration for DiagnoBot backend
Load from .env file or environment variables
"""

from pydantic_settings import BaseSettings
from typing import List, Literal
import os

# torch is optional – not needed when running in dev/mock mode
try:
    import torch as _torch
    _CUDA_AVAILABLE: bool = _torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False


class Settings(BaseSettings):
    """Application settings"""
    
    # ==================== APP CONFIG ====================
    APP_NAME: str = "DiagnoBot"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # ==================== CORS ====================
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
    ]
    
    # ==================== DATABASE ====================
    # For development, use SQLite:
    DATABASE_URL: str = "sqlite+aiosqlite:///./diagnobot.db"
    # PostgreSQL (Production)
    # DATABASE_URL: str = "postgresql://user:password@localhost:5432/diagnobot"
    
    # ==================== ML MODELS ====================
    USE_GPU: bool = _CUDA_AVAILABLE
    DEVICE: str = "cuda" if _CUDA_AVAILABLE else "cpu"
    MODEL_CACHE_DIR: str = "./models_cache"
    
    # Model configurations
    MOONDREAM2_REVISION: str = "2025-06-21"
    MOONDREAM2_DEVICE_MAP: str = "auto"
    
    CLINICAL_BERT_MODEL: str = "emilyalsentzer/Bio_ClinicalBERT"
    BIOGPT_MODEL: str = "microsoft/biogpt"
    BIOBART_MODEL: str = "GanjinZero/biobart-base"
    CLINICAL_T5_MODEL: str = "luqh/ClinicalT5-large"
    DERM_CNN_MODEL: str = "iamhmh/derm-cnn-ham10000"
    
    # ==================== FILE UPLOAD ====================
    MAX_IMAGE_SIZE_MB: int = 50
    MAX_FILE_SIZE_MB: int = 100
    UPLOAD_DIR: str = "./uploads"
    
    SUPPORTED_IMAGE_FORMATS: List[str] = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/tiff",
        "image/dicom"
    ]
    
    SUPPORTED_DOC_FORMATS: List[str] = [
        "application/pdf",
        "text/plain",
        "text/csv"
    ]
    
    # ==================== CAMERA/WEBCAM ====================
    WEBCAM_ENABLED: bool = True
    WEBCAM_DEVICE_ID: int = 0
    WEBCAM_RESOLUTION: tuple = (640, 480)
    WEBCAM_FPS: int = 30
    
    # ==================== MODEL INFERENCE ====================
    # Timeouts (seconds)
    XRAY_INFERENCE_TIMEOUT: int = 30
    DERMATOLOGY_INFERENCE_TIMEOUT: int = 20
    NLP_INFERENCE_TIMEOUT: int = 25
    
    # Batch processing
    BATCH_SIZE: int = 4
    ENABLE_BATCH_PROCESSING: bool = False
    
    # ==================== API KEYS & AUTH ====================
    API_KEY_ENABLED: bool = False
    API_KEYS: List[str] = []  # Load from .env
    
    # ==================== LOGGING ====================
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "./logs/diagnobot.log"
    
    # ==================== REDIS CACHE ====================
    REDIS_URL: str = "redis://localhost:6379/0"
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Load settings
settings = Settings()

# Create required runtime directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.LOG_FILE) or "logs", exist_ok=True)
