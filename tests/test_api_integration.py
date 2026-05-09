import pytest
import io
from unittest.mock import AsyncMock, MagicMock
from api.routes.xray_analysis import get_xray_analyzers
from api.routes.dermatology import get_derm_analyzers

@pytest.mark.asyncio
async def test_xray_multimodal_flow(client, monkeypatch):
    """Test the full X-ray analysis flow with mocked models."""
    
    # 1. Mock the analyzers
    mock_cnn = AsyncMock()
    mock_cnn.analyze_xray.return_value = {
        "status": "success",
        "findings": [{"name": "pneumonia", "confidence": 0.85}],
        "clinical_summary": "Possible pneumonia detected."
    }
    
    mock_vlm = AsyncMock()
    mock_vlm.analyze_xray.return_value = {
        "status": "success",
        "description": "Visual evidence of consolidation in the right lung.",
        "model": "Qwen2-VL"
    }
    
    # Override the analyzer dependency
    async def mock_get_analyzers():
        return mock_cnn, mock_vlm
    
    from main import app
    app.dependency_overrides[get_xray_analyzers] = mock_get_analyzers
    
    # 2. Upload and analyze
    fake_image = io.BytesIO(b"fake image data")
    response = await client.post(
        "/api/v1/xray/analyze",
        files={"file": ("test.jpg", fake_image, "image/jpeg")},
        params={"patient_id": "P123"}
    )
    
    # 3. Verify API Response
    assert response.status_code == 200
    data = response.json()
    assert data["description"] == "Visual evidence of consolidation in the right lung."
    assert "pneumonia" in str(data["findings"])
    
    # 4. Verify History Recording
    history_resp = await client.get("/api/v1/history")
    assert history_resp.status_code == 200
    history_data = history_resp.json()
    assert history_data["count"] >= 1
    assert history_data["records"][0]["patient_id"] == "P123"
    assert history_data["records"][0]["type"] == "xray"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_dermatology_multimodal_flow(client, monkeypatch):
    """Test the full Dermatology detection flow with mocked models."""
    
    # 1. Mock the analyzers
    mock_cnn = AsyncMock()
    mock_cnn.analyze_skin_image.return_value = {
        "status": "success",
        "classification": {"disease": "Melanoma", "confidence": 0.92, "severity": "urgent"},
        "clinical_advice": {"urgency": "URGENT"}
    }
    
    mock_vlm = AsyncMock()
    mock_vlm.analyze_skin.return_value = {
        "status": "success",
        "description": "Dark asymmetrical lesion with irregular borders.",
        "model": "Qwen2-VL"
    }
    
    async def mock_get_analyzers():
        return mock_cnn, mock_vlm
    
    from main import app
    app.dependency_overrides[get_derm_analyzers] = mock_get_analyzers
    
    # 2. Upload and detect
    fake_image = io.BytesIO(b"fake image data")
    response = await client.post(
        "/api/v1/dermatology/detect",
        files={"file": ("skin.jpg", fake_image, "image/jpeg")},
        params={"patient_id": "P456"}
    )
    
    # 3. Verify Response
    assert response.status_code == 200
    data = response.json()
    assert data["description"] == "Dark asymmetrical lesion with irregular borders."
    assert data["classification"]["disease"] == "Melanoma"
    
    # 4. Verify History
    history_resp = await client.get("/api/v1/history?analysis_type=dermatology")
    history_data = history_resp.json()
    assert history_data["count"] >= 1
    assert history_data["records"][0]["patient_id"] == "P456"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_analysis_error_logging(client, monkeypatch):
    """Ensure failed analyses are still recorded in the database with 'error' status."""
    
    mock_cnn = AsyncMock()
    mock_cnn.analyze_xray.side_effect = Exception("Model Crash!")
    
    async def mock_get_analyzers():
        return mock_cnn, AsyncMock()
    
    from main import app
    app.dependency_overrides[get_xray_analyzers] = mock_get_analyzers
    
    # 1. Trigger failing analysis
    fake_image = io.BytesIO(b"fake image data")
    await client.post(
        "/api/v1/xray/analyze",
        files={"file": ("crash.jpg", fake_image, "image/jpeg")}
    )
    
    # 2. Verify Database Record Status
    history_resp = await client.get("/api/v1/history")
    history_data = history_resp.json()
    assert history_data["records"][0]["status"] == "error"

    app.dependency_overrides.clear()
