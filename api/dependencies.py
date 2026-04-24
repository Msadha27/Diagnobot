"""
FastAPI shared dependencies
Provides model manager and database session to routes via Depends()
"""

from fastapi import Depends, HTTPException
from typing import Optional


def get_model_manager():
    """
    Return the global ModelManager instance.
    Import is deferred to avoid circular imports at startup.
    """
    from main import model_manager

    if model_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Model manager not initialized. Server may still be starting up.",
        )
    return model_manager
