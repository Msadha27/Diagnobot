"""
Pytest configuration and shared fixtures for DiagnoBot tests
"""

import pytest
import asyncio
from typing import Generator, AsyncGenerator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from database.connection import get_db
from models.database import Base

# Setup in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_db_engine():
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

@pytest.fixture
async def db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    connection = await test_db_engine.connect()
    transaction = await connection.begin()
    
    session_factory = sessionmaker(
        connection, class_=AsyncSession, expire_on_commit=False
    )
    session = session_factory()

    yield session

    await session.close()
    await transaction.rollback()
    await connection.close()

@pytest.fixture
async def client(db_session) -> AsyncGenerator[AsyncClient, None]:
    # Override get_db dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


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
