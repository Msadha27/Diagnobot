"""
DiagnoBot Backend - FastAPI Main Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any
from pathlib import Path

from config.settings import settings
from config.logging_config import setup_logging

from database.connection import init_db, create_tables
from ml_pipeline.model_manager import ModelManager

# Analyzers are initialized by the route dependencies during lifespan

from api.routes import health, upload, xray_analysis, dermatology, report_generation, nlp_analysis, analytics, clinical_workspace
from api.middleware.error_handlers import setup_error_handlers

setup_logging(log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)
logger = logging.getLogger(__name__)

# ================= GLOBAL INSTANCES =================

model_manager: ModelManager = None


# ================= LIFESPAN =================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager

    logger.info("[START] Starting DiagnoBot Backend...")

    try:
        # 0️⃣ INIT DATABASE
        init_db(settings.DATABASE_URL)
        await create_tables()

        # 1️⃣ INIT MODEL MANAGER
        model_manager = ModelManager(
            use_gpu=settings.USE_GPU,
            model_cache_dir=settings.MODEL_CACHE_DIR,
            device=settings.DEVICE
        )

        # 2️⃣ PRELOAD MODELS
        # We comment this out to rely entirely on LAZY LOADING. 
        # This prevents 100% RAM exhaustion and freezes during server setup on CPU.
        # await model_manager.preload_models([
        #     "qwen_vl",           # Qwen2-VL-2B-Instruct (replaces Moondream2)
        #     "derm_cnn",
        #     "biogpt",
        #     "biobart",
        #     "clinical_t5"
        # ])

        logger.info("[SUCCESS] Models loaded")

        # 3️⃣ INITIALIZE ROUTES  
        # All route analyzers will now initialize LAZILY when their endpoint is hit.

        logger.info("[SUCCESS] All analyzers initialized")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield

    # ================= SHUTDOWN =================
    logger.info("🛑 Shutting down...")
    if model_manager:
        await model_manager.cleanup()
    logger.info("✅ Cleanup complete")


# ================= APP =================

app = FastAPI(
    title="DiagnoBot Medical Analysis Backend",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_error_handlers(app)

FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="dashboard")


# ================= ROUTES =================

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(xray_analysis.router, prefix="/api/v1", tags=["xray"])
app.include_router(dermatology.router, prefix="/api/v1", tags=["dermatology"])
app.include_router(report_generation.router, prefix="/api/v1", tags=["reports"])
app.include_router(nlp_analysis.router, prefix="/api/v1", tags=["nlp"])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
app.include_router(clinical_workspace.router, prefix="/api/v1", tags=["workspace"])


# ================= ROOT =================

@app.get("/")
async def root():
    return {
        "status": "running",
        "dashboard": "/dashboard",
        "models": await get_model_status()
    }


@app.get("/app", include_in_schema=False)
async def dashboard():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard files not found")
    return FileResponse(index_path)


async def get_model_status():
    if model_manager is None:
        return {"status": "initializing"}
    return await model_manager.get_status()


@app.get("/api/v1/system/info")
async def system_info():
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    return {
        "device": settings.DEVICE,
        "models": await model_manager.get_status()
    }


# ================= RUN =================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        reload_excludes=["*.log", "logs/*", "models_cache/*"]
    )
