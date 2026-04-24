"""
Health check routes for DiagnoBot API
"""

import logging
from fastapi import APIRouter
from typing import Dict, Any
import platform
import os

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["health"])
async def health_check() -> Dict[str, Any]:
    """Basic health check – always returns 200 if the server is up."""
    return {
        "status": "healthy",
        "service": "DiagnoBot Medical Analysis Backend",
        "version": "1.0.0",
    }


@router.get("/status", tags=["health"])
async def system_status() -> Dict[str, Any]:
    """Detailed system status including CPU, memory, and GPU info."""
    gpu_info: Dict[str, Any] = {"available": _CUDA_AVAILABLE}
    if _CUDA_AVAILABLE:
        gpu_info.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
            }
        )

    system_info = {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    
    if _PSUTIL_AVAILABLE:
        system_info.update({
            "memory_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
            "memory_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
            "memory_percent": psutil.virtual_memory().percent,
        })
        
    return {
        "status": "running",
        "system": system_info,
        "gpu": gpu_info,
    }
