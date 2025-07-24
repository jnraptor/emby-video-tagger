"""Pytest configuration and shared fixtures."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from emby_video_tagger.config.settings import (
    AppConfig,
    EmbyConfig,
    ProcessingConfig,
    AIProviderConfig,
    AIProvider,
    LMStudioConfig,
    OllamaConfig,
    DatabaseConfig,
    SchedulerConfig,
    LoggingConfig,
)
from emby_video_tagger.core.models import (
    Video,
    Frame,
    ProcessingResult,
    TaskStatus,
    VideoTask,
    FrameAnalysis,
)
from emby_video_tagger.storage.models import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config(temp_dir: Path) -> AppConfig:
    """Create a mock configuration for testing."""
    return AppConfig(
        emby=EmbyConfig(
            server_url="http://test.emby.local:8096",
            api_key="test-api-key",
            user_id="test-user-id",
        ),
        ai=AIProviderConfig(
            provider=AIProvider.LMSTUDIO,
            lmstudio=LMStudioConfig(
                model_name="test-model",
            ),
            ollama=OllamaConfig(
                model_name="llava",
                base_url="http://localhost:11434",
            ),
        ),
        processing=ProcessingConfig(
            days_back=7,
            max_frames_per_video=5,
            scene_threshold=30.0,
            max_concurrent_videos=2,
            max_concurrent_frames=3,
        ),
        database=DatabaseConfig(
            url=f"sqlite:///{temp_dir}/test.db",
            echo=False,
        ),
        scheduler=SchedulerConfig(
            enabled=True,
            hour=2,
            minute=0,
        ),
        logging=LoggingConfig(
            level="DEBUG",
            format="json",
            file=temp_dir / "test.log",
        ),
        path_mappings={
            "/remote/path": str(temp_dir),
        },
    )


@pytest_asyncio.fixture
async def async_db_session(mock_config: AppConfig) -> AsyncGenerator[AsyncSession, None]:
    """Create an async database session for testing."""
    # Create engine
    engine = create_async_engine(
        mock_config.database.url.replace("sqlite://", "sqlite+aiosqlite://"),
        echo=False,
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session factory
    async_session_factory = async_sessionmaker(
        engine, expire_on_commit=False
    )
    
    # Yield session
    async with async_session_factory() as session:
        yield session
    
    # Cleanup
    await engine.dispose()


@pytest.fixture
def sample_video() -> Video:
    """Create a sample Video for testing."""
    return Video(
        id="test-video-123",
        name="Test Movie",
        path="/remote/path/movies/test_movie.mp4",
        date_created=datetime.now(),
        existing_tags=["existing", "tags"],
        metadata={"type": "Movie"},
    )


@pytest.fixture
def sample_frame(temp_dir: Path) -> Frame:
    """Create a sample Frame for testing."""
    frame_path = temp_dir / "frame_001.jpg"
    frame_path.write_bytes(b"fake image data")
    
    return Frame(
        path=str(frame_path),
        frame_number=1,
        timestamp=10.5,
        video_id="test-video-123",
    )


@pytest.fixture
def sample_processing_result() -> ProcessingResult:
    """Create a sample ProcessingResult for testing."""
    return ProcessingResult(
        video_id="test-video-123",
        status=TaskStatus.COMPLETED,
        tags=["action", "adventure", "sci-fi"],
        frames_processed=5,
        processing_time=12.5,
        error=None,
    )


@pytest.fixture
def sample_video_task() -> VideoTask:
    """Create a sample VideoTask for testing."""
    return VideoTask(
        id=1,
        emby_id="test-video-123",
        file_path="/remote/path/movies/test_movie.mp4",
        status=TaskStatus.PENDING,
        created_at=datetime.now(),
    )


@pytest.fixture
def sample_frame_analysis(sample_frame: Frame) -> FrameAnalysis:
    """Create a sample FrameAnalysis for testing."""
    return FrameAnalysis(
        frame=sample_frame,
        subjects=["person", "car"],
        activities=["driving", "talking"],
        settings=["city", "street"],
        styles=["action", "dramatic"],
        moods=["tense", "exciting"],
    )


@pytest.fixture
def mock_emby_service() -> AsyncMock:
    """Create a mock Emby service."""
    mock = AsyncMock()
    mock.get_recent_videos.return_value = []
    mock.get_video_by_id.return_value = None
    mock.update_video_tags.return_value = None
    return mock


@pytest.fixture
def mock_frame_extractor() -> AsyncMock:
    """Create a mock frame extractor."""
    mock = AsyncMock()
    mock.extract_frames.return_value = []
    return mock


@pytest.fixture
def mock_vision_processor() -> AsyncMock:
    """Create a mock vision processor."""
    mock = AsyncMock()
    mock.analyze_frame.return_value = []
    mock.analyze_frames.return_value = []
    return mock


@pytest.fixture
def mock_task_repository() -> AsyncMock:
    """Create a mock task repository."""
    mock = AsyncMock()
    mock.create_task.return_value = None
    mock.update_task.return_value = None
    mock.get_task_by_video_id.return_value = None
    mock.get_failed_tasks.return_value = []
    mock.get_statistics.return_value = {
        "total_tasks": 0,
        "successful_tasks": 0,
        "failed_tasks": 0,
        "total_tags_generated": 0,
        "average_processing_time": 0.0,
    }
    return mock


@pytest.fixture
def mock_http_response():
    """Create a mock HTTP response."""
    mock = MagicMock()
    mock.status = 200
    mock.json = AsyncMock(return_value={})
    mock.text = AsyncMock(return_value="")
    mock.raise_for_status = MagicMock()
    return mock


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp session."""
    session = AsyncMock()
    return session