"""
DiagnoBot – DEV SERVER (no ML models loaded)
Use this to test all API routes without needing GPU/PyTorch installed.
Run: python dev_server.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# ── Lightweight logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DiagnoBot Medical Analysis Backend  [DEV MODE]",
    description=(
        "AI-powered medical diagnosis backend.\n\n"
        "⚠️ **DEV MODE** – ML models are NOT loaded. "
        "All analysis endpoints return mock responses so you can test routing, "
        "auth, uploads and error handling without a GPU."
    ),
    version="1.0.0-dev",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Error handlers ────────────────────────────────────────────────────────────
from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError

@app.exception_handler(HTTPException)
async def http_exc(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": True, "status_code": exc.status_code,
                                 "detail": exc.detail, "path": str(request.url.path)})

@app.exception_handler(RequestValidationError)
async def val_exc(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422,
                        content={"error": True, "status_code": 422,
                                 "detail": "Validation error", "errors": exc.errors()})

# ── Routes ────────────────────────────────────────────────────────────────────
from api.routes import health, upload, nlp_analysis
from api.routes.xray_analysis import router as xray_router
from api.routes.dermatology import router as derm_router
from api.routes.report_generation import router as report_router

# Real routes (no model needed)
app.include_router(health.router,       prefix="/api/v1")
app.include_router(upload.router,       prefix="/api/v1")
app.include_router(nlp_analysis.router, prefix="/api/v1")

# Routes that need an analyzer – override them with mock stubs below
# (We include the routers for their GET /models endpoints, etc.)
app.include_router(xray_router,    prefix="/api/v1")
app.include_router(derm_router,    prefix="/api/v1")
app.include_router(report_router,  prefix="/api/v1")


# ── Mock overrides for model-dependent POST endpoints ─────────────────────────
from fastapi import File, UploadFile, Body
from typing import Dict, Any, Optional, List

@app.post("/api/v1/xray/analyze", tags=["xray [mock]"])
async def mock_xray_analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    """MOCK – returns a fake X-ray analysis result."""
    return {
        "status": "success [MOCK – no model loaded]",
        "image_filename": file.filename,
        "findings": [
            {"type": "general", "description": "Mock caption: chest X-ray image", "confidence": 0.95, "source": "mock"},
            {"type": "anomaly", "name": "pneumonia", "detected": True, "confidence": 0.82, "source": "mock"},
        ],
        "clinical_summary": "Mock finding: possible consolidation in right lower lobe.",
        "anomaly_count": 1,
        "recommendations": ["Confirm with clinical assessment.", "Consider antibiotic therapy."],
    }


@app.post("/api/v1/dermatology/detect", tags=["dermatology [mock]"])
async def mock_derm_detect(file: UploadFile = File(...)) -> Dict[str, Any]:
    """MOCK – returns a fake dermatology classification."""
    return {
        "status": "success [MOCK – no model loaded]",
        "image_filename": file.filename,
        "classification": {
            "disease": "Melanocytic nevus",
            "confidence": 0.91,
            "severity": "low",
            "code": "NV",
        },
        "clinical_advice": {
            "urgency": "LOW",
            "recommendation": "Likely benign. Routine monitoring recommended.",
            "next_steps": ["Document baseline characteristics", "Monitor for changes (ABCDE rule)"],
            "confidence_level": "High confidence",
        },
    }


@app.post("/api/v1/report/generate", tags=["reports [mock]"])
async def mock_report_generate(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """MOCK – returns a fake generated medical report."""
    return {
        "status": "success [MOCK – no model loaded]",
        "full_report": (
            "MEDICAL REPORT (MOCK)\n\n"
            "Clinical Findings: Based on the provided data, the patient presents with "
            "findings consistent with mock analysis.\n\n"
            "Assessment: Further clinical correlation is recommended.\n"
        ),
        "summary": "Mock summary: no significant abnormalities in mock mode.",
        "generation_model": "BioGPT [MOCK]",
    }


@app.post("/api/v1/report/summarize", tags=["reports [mock]"])
async def mock_report_summarize(report_text: str = Body(..., embed=True)) -> Dict[str, Any]:
    words = report_text.split()
    return {
        "status": "success [MOCK – no model loaded]",
        "summary": f"Mock summary of {len(words)}-word report.",
        "original_length": len(words),
        "summary_length": 8,
        "compression_ratio": round(8 / len(words), 3) if words else 0,
        "model": "ClinicalT5 [MOCK]",
    }


@app.post("/api/v1/report/from-input", tags=["reports [mock]"])
async def mock_report_from_input(
    patient_input: str = Body(..., embed=True),
    symptoms: Optional[List[str]] = Body(None, embed=True),
) -> Dict[str, Any]:
    return {
        "status": "success [MOCK – no model loaded]",
        "formal_report": f"MOCK FORMAL REPORT\n\nChief Complaint: {patient_input[:100]}\n"
                          f"Symptoms: {', '.join(symptoms or ['not provided'])}",
        "original_input": patient_input,
        "generation_model": "BioBart [MOCK]",
    }


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "DiagnoBot Medical Analysis Backend",
        "mode": "DEV (no ML models)",
        "version": "1.0.0-dev",
        "docs": "/docs",
        "redoc": "/redoc",
        "note": "All analysis endpoints return mock responses in dev mode.",
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting DiagnoBot DEV server (mock mode – no ML models)...")
    logger.info("📖 Swagger UI → http://127.0.0.1:8000/docs")
    logger.info("📖 ReDoc      → http://127.0.0.1:8000/redoc")

    uvicorn.run(
        "dev_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
