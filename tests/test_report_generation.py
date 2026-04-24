"""
Tests for medical report generation routes
"""

import pytest


def test_report_from_input_empty(test_client):
    """Empty patient input should fail validation."""
    response = test_client.post(
        "/api/v1/report/from-input",
        json={"patient_input": ""},
    )
    assert response.status_code in (400, 422)


def test_summarize_empty_text(test_client):
    """Empty report text should fail validation."""
    response = test_client.post(
        "/api/v1/report/summarize",
        json={"report_text": ""},
    )
    assert response.status_code in (400, 422)
