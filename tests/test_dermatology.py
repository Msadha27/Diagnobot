"""
Tests for dermatology analysis routes
"""

import pytest
import io


def test_derm_detect_invalid_type(test_client):
    """PDF should be rejected by the dermatology endpoint."""
    fake_pdf = io.BytesIO(b"%PDF fake content")
    response = test_client.post(
        "/api/v1/dermatology/detect",
        files={"file": ("report.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 400


def test_derm_detect_valid_jpeg(test_client, sample_skin_image_path, monkeypatch):
    """Valid JPEG should reach the analyzer (mocked)."""
    pass  # Implement when integration fixtures are ready
