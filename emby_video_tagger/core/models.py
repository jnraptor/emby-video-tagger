"""Core data models for Emby Video Tagger."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path


class TaskStatus(Enum):
    """Status of a video processing task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Video:
    """Represents a video from Emby."""
    id: str
    name: str
    path: str
    date_created: datetime
    existing_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def needs_processing(self) -> bool:
        """Check if video needs AI tag processing."""
        ai_indicators = ["ai-generated", "auto-tagged", "vision-analyzed"]
        return not any(
            indicator in tag.lower() 
            for tag in self.existing_tags 
            for indicator in ai_indicators
        )
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        try:
            return Path(self.path).stat().st_size / (1024 * 1024)
        except:
            return 0.0
    
    def __str__(self) -> str:
        """String representation of video."""
        return f"Video(id={self.id}, name={self.name})"


@dataclass
class Frame:
    """Represents an extracted video frame."""
    path: str
    frame_number: int
    timestamp: float
    video_id: str
    
    def cleanup(self) -> None:
        """Remove temporary frame file."""
        try:
            frame_path = Path(self.path)
            if frame_path.exists():
                frame_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors
    
    def __str__(self) -> str:
        """String representation of frame."""
        return f"Frame(video_id={self.video_id}, frame={self.frame_number}, time={self.timestamp:.2f}s)"


@dataclass
class ProcessingResult:
    """Result of video processing."""
    video_id: str
    status: TaskStatus
    tags: List[str] = field(default_factory=list)
    frames_processed: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if processing was successful."""
        return self.status == TaskStatus.COMPLETED
    
    @property
    def tag_count(self) -> int:
        """Get number of tags generated."""
        return len(self.tags)
    
    def __str__(self) -> str:
        """String representation of result."""
        if self.is_successful:
            return f"ProcessingResult(video_id={self.video_id}, status={self.status.value}, tags={self.tag_count})"
        else:
            return f"ProcessingResult(video_id={self.video_id}, status={self.status.value}, error={self.error})"


@dataclass
class VideoTask:
    """Represents a video processing task in the database."""
    id: Optional[int] = None
    emby_id: str = ""
    file_path: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tag_count: int = 0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamps."""
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def processing_duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.created_at and self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None
    
    def mark_completed(self, tag_count: int) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.tag_count = tag_count
        self.error_message = None
    
    def mark_failed(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error
    
    def __str__(self) -> str:
        """String representation of task."""
        return f"VideoTask(id={self.id}, emby_id={self.emby_id}, status={self.status.value})"


@dataclass
class FrameAnalysis:
    """Result of AI frame analysis."""
    frame: Frame
    subjects: List[str] = field(default_factory=list)
    activities: List[str] = field(default_factory=list)
    settings: List[str] = field(default_factory=list)
    styles: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)
    
    @property
    def all_tags(self) -> List[str]:
        """Get all tags from all categories."""
        tags = []
        tags.extend(self.subjects)
        tags.extend(self.activities)
        tags.extend(self.settings)
        tags.extend(self.styles)
        tags.extend(self.moods)
        return list(set(tags))  # Remove duplicates
    
    def merge_with(self, other: "FrameAnalysis") -> None:
        """Merge tags from another analysis."""
        self.subjects.extend(other.subjects)
        self.activities.extend(other.activities)
        self.settings.extend(other.settings)
        self.styles.extend(other.styles)
        self.moods.extend(other.moods)
        
        # Remove duplicates
        self.subjects = list(set(self.subjects))
        self.activities = list(set(self.activities))
        self.settings = list(set(self.settings))
        self.styles = list(set(self.styles))
        self.moods = list(set(self.moods))