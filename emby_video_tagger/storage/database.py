"""Database connection and session management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from emby_video_tagger.config.settings import DatabaseConfig
from emby_video_tagger.storage.models import Base


class Database:
    """Database connection and session management."""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize database with configuration."""
        self.config = config
        self._engine: Optional[AsyncEngine] = None
        self._sessionmaker: Optional[async_sessionmaker] = None
    
    async def initialize(self):
        """Initialize database engine and create tables."""
        # Convert sqlite URL to async format if needed
        db_url = self.config.url
        if db_url.startswith("sqlite://"):
            db_url = db_url.replace("sqlite://", "sqlite+aiosqlite://")
        
        # Create async engine
        self._engine = create_async_engine(
            db_url,
            echo=self.config.echo,
            poolclass=NullPool,  # Disable connection pooling for SQLite
        )
        
        # Create session factory
        self._sessionmaker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create tables if they don't exist
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        if not self._sessionmaker:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()