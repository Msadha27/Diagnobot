from ml_pipeline.triage import build_triage_assessment
import pytest


def test_visual_wound_red_flags_escalate_to_soon():
    result = build_triage_assessment(
        symptoms="pain and fever",
        visual_summary="Wound with redness, swelling and pus discharge",
        temperature=101.2,
        mode="wound",
    )

    assert result["urgency"] == "soon"
    assert "Dermatologist" in result["recommended_specialist"]
    assert result["reasons"]


def test_emergency_terms_override_routine_context():
    result = build_triage_assessment(
        symptoms="rash with difficulty breathing",
        visual_summary="mild redness",
        mode="skin",
    )

    assert result["urgency"] == "emergency"
    assert result["recommended_specialist"] == "Emergency Medicine"
    assert result["red_flags"]


def test_low_classifier_confidence_prefers_review():
    result = build_triage_assessment(
        visual_summary="skin rash",
        classification={"label": "Uncertain", "confidence": 0.21, "severity": "unknown"},
        mode="skin",
    )

    assert result["urgency"] == "soon"
    assert "Low classifier confidence" in result["confidence_policy"]


@pytest.mark.asyncio
async def test_emergency_triage_returns_explainable_assessment(client):
    response = await client.post(
        "/api/v1/triage/emergency",
        json={
            "symptoms": "skin rash with difficulty breathing",
            "visual_summary": "red swollen rash",
            "temperature": 99.0,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == "emergency"
    assert data["recommended_specialist"] == "Emergency Medicine"
    assert data["triage_assessment"]["red_flags"]
