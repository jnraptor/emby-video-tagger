"""SQLAlchemy database models for Emby Video Tagger."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Enum, Text
from sqlalchemy.ext.declarative import declarative_base

from emby_video_tagger.core.models import TaskStatus

Base = declarative_base()


class VideoTaskModel(Base):
    """SQLAlchemy model for video processing tasks."""
    
    __tablename__ = "video_tasks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    emby_id = Column(String(255), unique=True, nullable=False, index=True)
    file_path = Column(Text, nullable=False)
    status = Column(
        Enum(TaskStatus),
        nullable=False,
        default=TaskStatus.PENDING,
        index=True
    )
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    tag_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    
    def __repr__(self):
        """String representation of the model."""
        return (
            f"<VideoTask(id={self.id}, emby_id={self.emby_id}, "
            f"status={self.status.value})>"
        )
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "emby_id": self.emby_id,
            "file_path": self.file_path,
            "status": self.status.value if self.status is not None else None,
            "created_at": self.created_at.isoformat() if self.created_at is not None else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at is not None else None,
            "tag_count": self.tag_count,
            "error_message": self.error_message
        }