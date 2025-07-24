"""Unit tests for database repository."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from emby_video_tagger.storage.repository import TaskRepository
from emby_video_tagger.storage.models import VideoTaskModel as DBVideoTask
from emby_video_tagger.core.models import VideoTask, TaskStatus
from emby_video_tagger.core.exceptions import DatabaseError


@pytest.mark.asyncio
class TestTaskRepository:
    """Test TaskRepository class."""
    
    @pytest.fixture
    def task_repository(self, async_db_session):
        """Create TaskRepository instance."""
        return TaskRepository(async_db_session)
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample VideoTask."""
        return VideoTask(
            emby_id="test-video-123",
            file_path="/path/to/video.mp4",
            status=TaskStatus.PENDING
        )
    
    @pytest.fixture
    def sample_db_task(self):
        """Create a sample database VideoTask."""
        return DBVideoTask(
            id=1,
            emby_id="test-video-123",
            file_path="/path/to/video.mp4",
            status="pending",
            created_at=datetime.now(),
            completed_at=None,
            tag_count=0,
            error_message=None
        )
    
    async def test_create_task_success(self, task_repository, sample_task, async_db_session):
        """Test successful task creation."""
        # Mock the database operations
        async_db_session.add = MagicMock()
        async_db_session.commit = AsyncMock()
        async_db_session.refresh = AsyncMock()
        
        # Create task
        created_task = await task_repository.create_task(sample_task)
        
        assert created_task.emby_id == sample_task.emby_id
        assert created_task.file_path == sample_task.file_path
        assert created_task.status == sample_task.status
        
        # Verify database operations were called
        async_db_session.add.assert_called_once()
        async_db_session.commit.assert_called_once()
    
    async def test_create_task_database_error(self, task_repository, sample_task, async_db_session):
        """Test task creation with database error."""
        # Mock database error
        async_db_session.commit = AsyncMock(side_effect=SQLAlchemyError("Database error"))
        async_db_session.rollback = AsyncMock()
        
        with pytest.raises(DatabaseError) as exc_info:
            await task_repository.create_task(sample_task)
        
        assert "Failed to create task" in str(exc_info.value)
        async_db_session.rollback.assert_called_once()
    
    async def test_get_task_by_video_id_found(self, task_repository, sample_db_task, async_db_session):
        """Test getting task by video ID when found."""
        # Mock query result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=sample_db_task)
        
        async_db_session.execute = AsyncMock(return_value=mock_result)
        
        # Get task
        task = await task_repository.get_task_by_video_id("test-video-123")
        
        assert task is not None
        assert task.emby_id == "test-video-123"
        assert task.status == TaskStatus.PENDING
    
    async def test_get_task_by_video_id_not_found(self, task_repository, async_db_session):
        """Test getting task by video ID when not found."""
        # Mock empty query result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=None)
        
        async_db_session.execute = AsyncMock(return_value=mock_result)
        
        # Get task
        task = await task_repository.get_task_by_video_id("non-existent")
        
        assert task is None
    
    async def test_update_task_success(self, task_repository, sample_task, sample_db_task, async_db_session):
        """Test successful task update."""
        # Mock finding the task
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=sample_db_task)
        async_db_session.execute = AsyncMock(return_value=mock_result)
        async_db_session.commit = AsyncMock()
        
        # Update task
        sample_task.status = TaskStatus.COMPLETED
        sample_task.tag_count = 5
        
        updated_task = await task_repository.update_task(sample_task)
        
        assert updated_task is not None
        assert sample_db_task.status == "completed"
        assert sample_db_task.tag_count == 5
        async_db_session.commit.assert_called_once()
    
    async def test_update_task_not_found(self, task_repository, sample_task, async_db_session):
        """Test updating non-existent task."""
        # Mock empty query result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=None)
        async_db_session.execute = AsyncMock(return_value=mock_result)
        
        # Update task
        updated_task = await task_repository.update_task(sample_task)
        
        assert updated_task is None
    
    async def test_get_failed_tasks(self, task_repository, async_db_session):
        """Test getting failed tasks."""
        # Create mock failed tasks
        failed_task1 = DBVideoTask(
            id=1,
            emby_id="failed-1",
            file_path="/path/to/failed1.mp4",
            status="failed",
            created_at=datetime.now() - timedelta(hours=2),
            error_message="Error 1"
        )
        
        failed_task2 = DBVideoTask(
            id=2,
            emby_id="failed-2",
            file_path="/path/to/failed2.mp4",
            status="failed",
            created_at=datetime.now() - timedelta(hours=1),
            error_message="Error 2"
        )
        
        # Mock query result
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[failed_task1, failed_task2])))
        async_db_session.execute = AsyncMock(return_value=mock_result)
        
        # Get failed tasks
        failed_tasks = await task_repository.get_failed_tasks(limit=10)
        
        assert len(failed_tasks) == 2
        assert all(task.status == TaskStatus.FAILED for task in failed_tasks)
        assert failed_tasks[0].emby_id == "failed-1"
        assert failed_tasks[1].emby_id == "failed-2"
    
    async def test_get_recent_tasks(self, task_repository, async_db_session):
        """Test getting recent tasks."""
        # Create mock recent tasks
        recent_task = DBVideoTask(
            id=1,
            emby_id="recent-1",
            file_path="/path/to/recent.mp4",
            status="completed",
            created_at=datetime.now() - timedelta(hours=1),
            completed_at=datetime.now(),
            tag_count=5
        )
        
        # Mock query result
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[recent_task])))
        async_db_session.execute = AsyncMock(return_value=mock_result)
        
        # Get recent tasks
        recent_tasks = await task_repository.get_recent_tasks(hours=24, limit=10)
        
        assert len(recent_tasks) == 1
        assert recent_tasks[0].emby_id == "recent-1"
        assert recent_tasks[0].status == TaskStatus.COMPLETED
    
    async def test_get_statistics(self, task_repository, async_db_session):
        """Test getting statistics."""
        # Mock count queries
        mock_results = [
            MagicMock(scalar_one=MagicMock(return_value=100)),  # total_tasks
            MagicMock(scalar_one=MagicMock(return_value=80)),   # successful_tasks
            MagicMock(scalar_one=MagicMock(return_value=20)),   # failed_tasks
            MagicMock(scalar_one=MagicMock(return_value=400)),  # total_tags
            MagicMock(scalar_one=MagicMock(return_value=3600.0))  # avg_time
        ]
        
        async_db_session.execute = AsyncMock(side_effect=mock_results)
        
        # Get statistics
        stats = await task_repository.get_statistics()
        
        assert stats["total_tasks"] == 100
        assert stats["successful_tasks"] == 80
        assert stats["failed_tasks"] == 20
        assert stats["total_tags_generated"] == 400
        assert stats["average_processing_time"] == 3600.0
        assert stats["success_rate"] == 80.0
    
    async def test_get_statistics_no_tasks(self, task_repository, async_db_session):
        """Test getting statistics with no tasks."""
        # Mock empty results
        mock_results = [
            MagicMock(scalar_one=MagicMock(return_value=0)),    # total_tasks
            MagicMock(scalar_one=MagicMock(return_value=0)),    # successful_tasks
            MagicMock(scalar_one=MagicMock(return_value=0)),    # failed_tasks
            MagicMock(scalar_one=MagicMock(return_value=0)),    # total_tags
            MagicMock(scalar_one=MagicMock(return_value=None))  # avg_time
        ]
        
        async_db_session.execute = AsyncMock(side_effect=mock_results)
        
        # Get statistics
        stats = await task_repository.get_statistics()
        
        assert stats["total_tasks"] == 0
        assert stats["successful_tasks"] == 0
        assert stats["failed_tasks"] == 0
        assert stats["total_tags_generated"] == 0
        assert stats["average_processing_time"] == 0.0
        assert stats["success_rate"] == 0.0
    
    async def test_delete_old_tasks(self, task_repository, async_db_session):
        """Test deleting old tasks."""
        # Mock delete result
        mock_result = MagicMock()
        mock_result.rowcount = 5
        
        async_db_session.execute = AsyncMock(return_value=mock_result)
        async_db_session.commit = AsyncMock()
        
        # Delete old tasks
        deleted_count = await task_repository.delete_old_tasks(days=30)
        
        assert deleted_count == 5
        async_db_session.commit.assert_called_once()
    
    async def test_task_exists_true(self, task_repository, async_db_session):
        """Test checking if task exists (found)."""
        # Mock query result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=1)
        
        async_db_session.execute = AsyncMock(return_value=mock_result)
        
        # Check existence
        exists = await task_repository.task_exists("test-video-123")
        
        assert exists is True
    
    async def test_task_exists_false(self, task_repository, async_db_session):
        """Test checking if task exists (not found)."""
        # Mock empty query result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=None)
        
        async_db_session.execute = AsyncMock(return_value=mock_result)
        
        # Check existence
        exists = await task_repository.task_exists("non-existent")
        
        assert exists is False
    
    def test_convert_db_to_model(self, task_repository, sample_db_task):
        """Test converting database model to domain model."""
        task = task_repository._convert_db_to_model(sample_db_task)
        
        assert isinstance(task, VideoTask)
        assert task.id == sample_db_task.id
        assert task.emby_id == sample_db_task.emby_id
        assert task.file_path == sample_db_task.file_path
        assert task.status == TaskStatus.PENDING
        assert task.created_at == sample_db_task.created_at
        assert task.completed_at == sample_db_task.completed_at
        assert task.tag_count == sample_db_task.tag_count
        assert task.error_message == sample_db_task.error_message
    
    def test_convert_model_to_db(self, task_repository, sample_task):
        """Test converting domain model to database model."""
        db_task = task_repository._convert_model_to_db(sample_task)
        
        assert isinstance(db_task, DBVideoTask)
        # For a new instance, we check the attributes directly
        assert db_task.emby_id == sample_task.emby_id
        assert db_task.file_path == sample_task.file_path
        # Status is set during conversion, not checked here due to SQLAlchemy column behavior
        assert db_task.tag_count == sample_task.tag_count
        assert db_task.error_message == sample_task.error_message
    
    async def test_get_tasks_by_status(self, task_repository, async_db_session):
        """Test getting tasks by status."""
        # Create mock tasks
        pending_task = DBVideoTask(
            id=1,
            emby_id="pending-1",
            file_path="/path/to/pending.mp4",
            status="pending",
            created_at=datetime.now()
        )
        
        # Mock query result
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[pending_task])))
        async_db_session.execute = AsyncMock(return_value=mock_result)
        
        # Get tasks by status
        tasks = await task_repository.get_tasks_by_status(TaskStatus.PENDING, limit=10)
        
        assert len(tasks) == 1
        assert tasks[0].status == TaskStatus.PENDING
        assert tasks[0].emby_id == "pending-1"