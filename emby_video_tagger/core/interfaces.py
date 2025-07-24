"""Abstract base classes and interfaces for Emby Video Tagger."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from .models import Video, Frame, ProcessingResult, TaskStatus, FrameAnalysis


class IEmbyService(ABC):
    """Interface for Emby API operations."""
    
    @abstractmethod
    async def get_recent_videos(
        self, 
        days_back: int = 7,
        limit: int = 100
    ) -> List[Video]:
        """Retrieve recently added videos from Emby."""
        pass
    
    @abstractmethod
    async def get_video_by_id(self, video_id: str) -> Optional[Video]:
        """Get specific video by ID."""
        pass
    
    @abstractmethod
    async def update_video_tags(
        self, 
        video_id: str, 
        tags: List[str]
    ) -> bool:
        """Update video tags in Emby."""
        pass
    
    @abstractmethod
    async def batch_update_tags(
        self, 
        updates: List[Tuple[str, List[str]]]
    ) -> Dict[str, bool]:
        """Batch update multiple videos' tags."""
        pass


class IFrameExtractor(ABC):
    """Interface for video frame extraction."""
    
    @abstractmethod
    async def extract_frames(
        self,
        video_path: str,
        max_frames: int = 10,
        strategy: str = "scene_detection"
    ) -> List[Frame]:
        """Extract representative frames from video."""
        pass
    
    @abstractmethod
    def validate_video(self, video_path: str) -> bool:
        """Check if video file is valid and accessible."""
        pass
    
    @abstractmethod
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata (duration, fps, resolution, etc.)."""
        pass


class IVisionProcessor(ABC):
    """Interface for AI vision processing."""
    
    @abstractmethod
    async def analyze_frame(self, frame: Frame) -> FrameAnalysis:
        """Analyze single frame and return analysis."""
        pass
    
    @abstractmethod
    async def analyze_frames_batch(
        self, 
        frames: List[Frame]
    ) -> List[FrameAnalysis]:
        """Analyze multiple frames in batch."""
        pass
    
    @abstractmethod
    def merge_tags(
        self, 
        analyses: List[FrameAnalysis]
    ) -> List[str]:
        """Merge and deduplicate tags from multiple frame analyses."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the AI model being used."""
        pass


class ITaskRepository(ABC):
    """Interface for task persistence."""
    
    @abstractmethod
    async def create_task(
        self,
        video_id: str,
        file_path: str
    ) -> int:
        """Create new processing task."""
        pass
    
    @abstractmethod
    async def update_task_status(
        self,
        video_id: str,
        status: TaskStatus,
        tag_count: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update task status."""
        pass
    
    @abstractmethod
    async def get_task_by_video_id(
        self, 
        video_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get task by video ID."""
        pass
    
    @abstractmethod
    async def get_pending_tasks(
        self, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all pending tasks."""
        pass
    
    @abstractmethod
    async def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get processing statistics."""
        pass
    
    @abstractmethod
    async def get_failed_tasks(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get failed tasks for retry."""
        pass


class IPathMapper(ABC):
    """Interface for path mapping between systems."""
    
    @abstractmethod
    def map_path(self, source_path: str) -> str:
        """Map source path to destination path."""
        pass
    
    @abstractmethod
    def validate_mapping(self, source_path: str) -> bool:
        """Check if path mapping exists and is valid."""
        pass
    
    @abstractmethod
    def add_mapping(
        self, 
        source_prefix: str, 
        dest_prefix: str
    ) -> None:
        """Add new path mapping."""
        pass


class IVideoTaggingOrchestrator(ABC):
    """Main orchestration interface."""
    
    @abstractmethod
    async def process_recent_videos(
        self,
        days_back: int = 5,
        max_concurrent: int = 3
    ) -> List[ProcessingResult]:
        """Process recently added videos."""
        pass
    
    @abstractmethod
    async def process_single_video(
        self,
        video_id: str
    ) -> ProcessingResult:
        """Process a single video by ID."""
        pass
    
    @abstractmethod
    async def process_videos_batch(
        self,
        video_ids: List[str],
        max_concurrent: int = 3
    ) -> List[ProcessingResult]:
        """Process multiple videos concurrently."""
        pass
    
    @abstractmethod
    async def reprocess_failed_videos(
        self,
        max_retries: int = 3
    ) -> List[ProcessingResult]:
        """Retry processing for failed videos."""
        pass


class ILogger(ABC):
    """Interface for logging operations."""
    
    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        pass
    
    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message."""
        pass
    
    @abstractmethod
    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log critical message."""
        pass