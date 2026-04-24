"""
DiagnoBot Backend - FastAPI Main Application
Integrates medical vision models (Moondream2, Derm CNN, TorchXRayVision)
and NLP models (BioGPT, Bio_ClinicalBERT, ClinicalT5, BioBart)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

# Local imports
from config.settings import settings
from config.logging_config import setup_logging
from ml_pipeline.model_manager import ModelManager
from api.routes import health, upload, xray_analysis, dermatology, report_generation, nlp_analysis
from api.middleware.error_handlers import setup_error_handlers

# Setup logging
setup_logging(log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)
logger = logging.getLogger(__name__)

# Global model manager (singleton)
model_manager: ModelManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load all ML models
    Shutdown: Clean up resources
    """
    global model_manager

    logger.info("🚀 Starting DiagnoBot Backend...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"GPU Available: {settings.USE_GPU}")

    try:
        # Initialize model manager with lazy loading
        model_manager = ModelManager(
            use_gpu=settings.USE_GPU,
            model_cache_dir=settings.MODEL_CACHE_DIR,
            device=settings.DEVICE
        )
        logger.info("✅ Model Manager initialized")

        # Pre-load critical models on startup
        await model_manager.preload_models(
            models=[
                'moondream2',
                'clinical_bert',
                'biogpt'
            ]
        )
        logger.info("✅ Critical models preloaded")

    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {str(e)}")
        raise

    yield  # Application runs here

    # Cleanup on shutdown
    logger.info("🛑 Shutting down DiagnoBot Backend...")
    if model_manager:
        await model_manager.cleanup()
    logger.info("✅ Cleanup complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="DiagnoBot Medical Analysis Backend",
    description="AI-powered medical diagnosis system with vision + NLP models",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup global error handlers
setup_error_handlers(app)


# ==================== ROUTES ====================

# Health check
app.include_router(health.router, prefix="/api/v1", tags=["health"])

# File upload
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])

# X-Ray Analysis (Moondream2 + TorchXRayVision)
app.include_router(xray_analysis.router, prefix="/api/v1", tags=["xray"])

# Dermatology (Derm CNN + Webcam)
app.include_router(dermatology.router, prefix="/api/v1", tags=["dermatology"])

# Medical Report Generation (BioGPT + BioBart)
app.include_router(report_generation.router, prefix="/api/v1", tags=["reports"])

# NLP Analysis (Bio_ClinicalBERT + ClinicalT5)
app.include_router(nlp_analysis.router, prefix="/api/v1", tags=["nlp"])


# ==================== ROOT & UTILITY ENDPOINTS ====================

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint – API info"""
    return {
        "name": "DiagnoBot Medical Analysis Backend",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "models_loaded": await get_model_status()
    }


async def get_model_status() -> Dict[str, str]:
    """Get status of all loaded models"""
    if model_manager is None:
        return {"status": "initializing"}
    return await model_manager.get_status()


@app.get("/api/v1/system/info")
async def system_info() -> Dict[str, Any]:
    """Get system and model information"""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    return {
        "environment": settings.ENVIRONMENT,
        "gpu_available": settings.USE_GPU,
        "device": settings.DEVICE,
        "model_cache": settings.MODEL_CACHE_DIR,
        "models": await model_manager.get_status(),
        "max_image_size_mb": settings.MAX_IMAGE_SIZE_MB,
        "supported_formats": settings.SUPPORTED_IMAGE_FORMATS
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower()
    )
