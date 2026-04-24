"""
Pytest configuration and shared fixtures for DiagnoBot tests
"""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_client():
    """Synchronous test client for the FastAPI app."""
    from main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_xray_path(tmp_path):
    """Create a dummy X-ray image for testing."""
    from PIL import Image
    import numpy as np

    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    path = tmp_path / "test_xray.jpg"
    img.save(str(path))
    return str(path)


@pytest.fixture
def sample_skin_image_path(tmp_path):
    """Create a dummy skin image for testing."""
    from PIL import Image
    import numpy as np

    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    path = tmp_path / "test_skin.jpg"
    img.save(str(path))
    return str(path)
