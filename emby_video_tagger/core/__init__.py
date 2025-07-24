"""Core domain models and interfaces for Emby Video Tagger."""

from .models import Video, Frame, ProcessingResult, TaskStatus
from .exceptions import (
    EmbyVideoTaggerError,
    EmbyAPIError,
    FrameExtractionError,
    VisionProcessingError,
    ConfigurationError
)

__all__ = [
    "Video",
    "Frame", 
    "ProcessingResult",
    "TaskStatus",
    "EmbyVideoTaggerError",
    "EmbyAPIError",
    "FrameExtractionError",
    "VisionProcessingError",
    "ConfigurationError"
]