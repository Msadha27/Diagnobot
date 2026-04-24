"""
Tests for X-ray analysis routes
"""

import pytest
import io
from pathlib import Path


def test_health_check(test_client):
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_xray_models_endpoint(test_client):
    response = test_client.get("/api/v1/xray/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert len(data["available_models"]) >= 2


def test_xray_upload_invalid_type(test_client):
    """PDF should be rejected on the X-ray endpoint."""
    fake_pdf = io.BytesIO(b"%PDF fake content")
    response = test_client.post(
        "/api/v1/xray/analyze",
        files={"file": ("report.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 400


def test_xray_upload_valid_image(test_client, sample_xray_path, monkeypatch):
    """Valid JPEG should reach the analyzer (mocked)."""
    # Monkeypatch model dependency to avoid loading actual models in CI
    pass  # Implement when integration fixtures are ready
