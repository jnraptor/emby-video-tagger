"""Unit tests for core models."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from emby_video_tagger.core.models import (
    TaskStatus,
    Video,
    Frame,
    ProcessingResult,
    VideoTask,
    FrameAnalysis,
)


class TestVideo:
    """Test Video model."""
    
    def test_video_creation(self):
        """Test creating Video instance."""
        video = Video(
            id="video-123",
            name="Test Movie",
            path="/path/to/movie.mp4",
            date_created=datetime.now(),
            existing_tags=["action", "adventure"],
            metadata={"type": "Movie", "year": 2024}
        )
        
        assert video.id == "video-123"
        assert video.name == "Test Movie"
        assert video.path == "/path/to/movie.mp4"
        assert len(video.existing_tags) == 2
        assert video.metadata["type"] == "Movie"
    
    def test_needs_processing_no_ai_tags(self):
        """Test needs_processing when no AI tags present."""
        video = Video(
            id="video-123",
            name="Test Movie",
            path="/path/to/movie.mp4",
            date_created=datetime.now(),
            existing_tags=["action", "adventure"]
        )
        
        assert video.needs_processing is True
    
    def test_needs_processing_with_ai_tags(self):
        """Test needs_processing when AI tags present."""
        video = Video(
            id="video-123",
            name="Test Movie",
            path="/path/to/movie.mp4",
            date_created=datetime.now(),
            existing_tags=["action", "ai-generated", "adventure"]
        )
        
        assert video.needs_processing is False
    
    def test_needs_processing_with_auto_tagged(self):
        """Test needs_processing with auto-tagged indicator."""
        video = Video(
            id="video-123",
            name="Test Movie",
            path="/path/to/movie.mp4",
            date_created=datetime.now(),
            existing_tags=["action", "Auto-Tagged", "adventure"]
        )
        
        assert video.needs_processing is False
    
    @patch("pathlib.Path.stat")
    def test_file_size_mb(self, mock_stat):
        """Test file_size_mb property."""
        mock_stat.return_value = MagicMock(st_size=1024 * 1024 * 100)  # 100 MB
        
        video = Video(
            id="video-123",
            name="Test Movie",
            path="/path/to/movie.mp4",
            date_created=datetime.now()
        )
        
        assert video.file_size_mb == 100.0
    
    @patch("pathlib.Path.stat", side_effect=Exception("File not found"))
    def test_file_size_mb_error(self, mock_stat):
        """Test file_size_mb when file doesn't exist."""
        video = Video(
            id="video-123",
            name="Test Movie",
            path="/path/to/movie.mp4",
            date_created=datetime.now()
        )
        
        assert video.file_size_mb == 0.0
    
    def test_video_str(self):
        """Test string representation of Video."""
        video = Video(
            id="video-123",
            name="Test Movie",
            path="/path/to/movie.mp4",
            date_created=datetime.now()
        )
        
        assert str(video) == "Video(id=video-123, name=Test Movie)"


class TestFrame:
    """Test Frame model."""
    
    def test_frame_creation(self):
        """Test creating Frame instance."""
        frame = Frame(
            path="/tmp/frame_001.jpg",
            frame_number=1,
            timestamp=10.5,
            video_id="video-123"
        )
        
        assert frame.path == "/tmp/frame_001.jpg"
        assert frame.frame_number == 1
        assert frame.timestamp == 10.5
        assert frame.video_id == "video-123"
    
    @patch("pathlib.Path.unlink")
    @patch("pathlib.Path.exists", return_value=True)
    def test_cleanup_success(self, mock_exists, mock_unlink):
        """Test successful frame cleanup."""
        frame = Frame(
            path="/tmp/frame_001.jpg",
            frame_number=1,
            timestamp=10.5,
            video_id="video-123"
        )
        
        frame.cleanup()
        
        mock_exists.assert_called_once()
        mock_unlink.assert_called_once()
    
    @patch("pathlib.Path.unlink", side_effect=Exception("Permission denied"))
    @patch("pathlib.Path.exists", return_value=True)
    def test_cleanup_error(self, mock_exists, mock_unlink):
        """Test frame cleanup with error (should not raise)."""
        frame = Frame(
            path="/tmp/frame_001.jpg",
            frame_number=1,
            timestamp=10.5,
            video_id="video-123"
        )
        
        # Should not raise exception
        frame.cleanup()
        
        mock_exists.assert_called_once()
        mock_unlink.assert_called_once()
    
    def test_frame_str(self):
        """Test string representation of Frame."""
        frame = Frame(
            path="/tmp/frame_001.jpg",
            frame_number=1,
            timestamp=10.5,
            video_id="video-123"
        )
        
        assert str(frame) == "Frame(video_id=video-123, frame=1, time=10.50s)"


class TestProcessingResult:
    """Test ProcessingResult model."""
    
    def test_processing_result_success(self):
        """Test successful ProcessingResult."""
        result = ProcessingResult(
            video_id="video-123",
            status=TaskStatus.COMPLETED,
            tags=["action", "adventure", "sci-fi"],
            frames_processed=5,
            processing_time=12.5
        )
        
        assert result.video_id == "video-123"
        assert result.is_successful is True
        assert result.tag_count == 3
        assert result.error is None
    
    def test_processing_result_failure(self):
        """Test failed ProcessingResult."""
        result = ProcessingResult(
            video_id="video-123",
            status=TaskStatus.FAILED,
            error="Failed to extract frames"
        )
        
        assert result.is_successful is False
        assert result.tag_count == 0
        assert result.error == "Failed to extract frames"
    
    def test_processing_result_str_success(self):
        """Test string representation of successful result."""
        result = ProcessingResult(
            video_id="video-123",
            status=TaskStatus.COMPLETED,
            tags=["action", "adventure"]
        )
        
        assert str(result) == "ProcessingResult(video_id=video-123, status=completed, tags=2)"
    
    def test_processing_result_str_failure(self):
        """Test string representation of failed result."""
        result = ProcessingResult(
            video_id="video-123",
            status=TaskStatus.FAILED,
            error="Test error"
        )
        
        assert str(result) == "ProcessingResult(video_id=video-123, status=failed, error=Test error)"


class TestVideoTask:
    """Test VideoTask model."""
    
    def test_video_task_creation(self):
        """Test creating VideoTask instance."""
        task = VideoTask(
            emby_id="video-123",
            file_path="/path/to/video.mp4"
        )
        
        assert task.emby_id == "video-123"
        assert task.file_path == "/path/to/video.mp4"
        assert task.status == TaskStatus.PENDING
        assert task.created_at is not None
        assert task.completed_at is None
    
    def test_video_task_with_id(self):
        """Test VideoTask with explicit ID."""
        created_time = datetime.now()
        task = VideoTask(
            id=42,
            emby_id="video-123",
            file_path="/path/to/video.mp4",
            created_at=created_time
        )
        
        assert task.id == 42
        assert task.created_at == created_time
    
    def test_processing_duration_completed(self):
        """Test processing_duration for completed task."""
        created_time = datetime.now()
        completed_time = created_time + timedelta(seconds=30)
        
        task = VideoTask(
            emby_id="video-123",
            file_path="/path/to/video.mp4",
            created_at=created_time,
            completed_at=completed_time
        )
        
        assert task.processing_duration == 30.0
    
    def test_processing_duration_not_completed(self):
        """Test processing_duration for incomplete task."""
        task = VideoTask(
            emby_id="video-123",
            file_path="/path/to/video.mp4"
        )
        
        assert task.processing_duration is None
    
    def test_mark_completed(self):
        """Test marking task as completed."""
        task = VideoTask(
            emby_id="video-123",
            file_path="/path/to/video.mp4"
        )
        
        task.mark_completed(tag_count=5)
        
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.tag_count == 5
        assert task.error_message is None
    
    def test_mark_failed(self):
        """Test marking task as failed."""
        task = VideoTask(
            emby_id="video-123",
            file_path="/path/to/video.mp4"
        )
        
        task.mark_failed("Test error message")
        
        assert task.status == TaskStatus.FAILED
        assert task.completed_at is not None
        assert task.error_message == "Test error message"
    
    def test_video_task_str(self):
        """Test string representation of VideoTask."""
        task = VideoTask(
            id=42,
            emby_id="video-123",
            file_path="/path/to/video.mp4"
        )
        
        assert str(task) == "VideoTask(id=42, emby_id=video-123, status=pending)"


class TestFrameAnalysis:
    """Test FrameAnalysis model."""
    
    def test_frame_analysis_creation(self, sample_frame):
        """Test creating FrameAnalysis instance."""
        analysis = FrameAnalysis(
            frame=sample_frame,
            subjects=["person", "car"],
            activities=["driving"],
            settings=["city", "street"],
            styles=["action"],
            moods=["tense"]
        )
        
        assert analysis.frame == sample_frame
        assert len(analysis.subjects) == 2
        assert "driving" in analysis.activities
        assert "city" in analysis.settings
    
    def test_all_tags(self, sample_frame):
        """Test all_tags property."""
        analysis = FrameAnalysis(
            frame=sample_frame,
            subjects=["person", "car"],
            activities=["driving", "talking"],
            settings=["city"],
            styles=["action", "dramatic"],
            moods=["tense", "exciting"]
        )
        
        all_tags = analysis.all_tags
        
        assert len(all_tags) == 8  # All unique tags
        assert "person" in all_tags
        assert "driving" in all_tags
        assert "city" in all_tags
        assert "action" in all_tags
        assert "tense" in all_tags
    
    def test_all_tags_with_duplicates(self, sample_frame):
        """Test all_tags removes duplicates."""
        analysis = FrameAnalysis(
            frame=sample_frame,
            subjects=["person", "car"],
            activities=["driving", "person"],  # Duplicate
            settings=["city", "car"],  # Duplicate
        )
        
        all_tags = analysis.all_tags
        
        assert len(all_tags) == 4  # Only unique tags
        assert all_tags.count("person") == 1
        assert all_tags.count("car") == 1
    
    def test_merge_with(self, sample_frame):
        """Test merging two FrameAnalysis instances."""
        analysis1 = FrameAnalysis(
            frame=sample_frame,
            subjects=["person"],
            activities=["walking"],
            settings=["park"]
        )
        
        analysis2 = FrameAnalysis(
            frame=sample_frame,
            subjects=["dog", "person"],  # Duplicate person
            activities=["running"],
            settings=["park", "trail"],  # Duplicate park
            styles=["nature"],
            moods=["peaceful"]
        )
        
        analysis1.merge_with(analysis2)
        
        assert len(analysis1.subjects) == 2  # person, dog (no duplicates)
        assert "dog" in analysis1.subjects
        assert len(analysis1.activities) == 2  # walking, running
        assert len(analysis1.settings) == 2  # park, trail (no duplicates)
        assert len(analysis1.styles) == 1  # nature
        assert len(analysis1.moods) == 1  # peaceful


class TestTaskStatus:
    """Test TaskStatus enum."""
    
    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.PROCESSING.value == "processing"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
    
    def test_task_status_comparison(self):
        """Test TaskStatus enum comparison."""
        assert TaskStatus.PENDING != TaskStatus.COMPLETED
        assert TaskStatus.FAILED != TaskStatus.PROCESSING
        assert TaskStatus.COMPLETED == TaskStatus.COMPLETED