"""
Database connection and session management
"""

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from models.database import Base

logger = logging.getLogger(__name__)

# Loaded later to avoid circular imports
_engine = None
_async_session_factory = None


def init_db(database_url: str) -> None:
    """Initialize the database engine and create all tables."""
    global _engine, _async_session_factory

    # Convert sync URL to async (e.g., postgresql → postgresql+asyncpg)
    async_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

    connect_args = {}
    if async_url.startswith("sqlite"):
        connect_args = {"timeout": 30}

    _engine = create_async_engine(
        async_url,
        echo=False,
        future=True,
        connect_args=connect_args,
    )
    _async_session_factory = sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )

    logger.info(f"Database engine created: {async_url[:30]}...")


async def get_db() -> AsyncSession:
    """FastAPI dependency – yields a DB session."""
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables() -> None:
    """Create all database tables."""
    if _engine is None:
        raise RuntimeError("Database not initialized.")

    async with _engine.begin() as conn:
        if str(_engine.url).startswith("sqlite"):
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA busy_timeout=30000"))
        await conn.run_sync(Base.metadata.create_all)

    logger.info("✅ Database tables created")
