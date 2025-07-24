"""Custom exceptions for Emby Video Tagger."""

from typing import Optional, Dict, Any


class EmbyVideoTaggerError(Exception):
    """Base exception for all custom errors in the application."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize exception with message and optional details."""
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(EmbyVideoTaggerError):
    """Raised when there are configuration issues."""
    pass


class EmbyAPIError(EmbyVideoTaggerError):
    """Raised when Emby API operations fail."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_body: Optional[str] = None, **kwargs):
        """Initialize Emby API error."""
        details = kwargs.get('details', {})
        if status_code:
            details['status_code'] = status_code
        if response_body:
            details['response_body'] = response_body
        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body


class FrameExtractionError(EmbyVideoTaggerError):
    """Raised when frame extraction fails."""
    
    def __init__(self, message: str, video_path: Optional[str] = None, **kwargs):
        """Initialize frame extraction error."""
        details = kwargs.get('details', {})
        if video_path:
            details['video_path'] = video_path
        super().__init__(message, details)
        self.video_path = video_path


class VisionProcessingError(EmbyVideoTaggerError):
    """Raised when AI vision processing fails."""
    
    def __init__(self, message: str, provider: Optional[str] = None, 
                 model: Optional[str] = None, **kwargs):
        """Initialize vision processing error."""
        details = kwargs.get('details', {})
        if provider:
            details['provider'] = provider
        if model:
            details['model'] = model
        super().__init__(message, details)
        self.provider = provider
        self.model = model


class DatabaseError(EmbyVideoTaggerError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        """Initialize database error."""
        details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        super().__init__(message, details)
        self.operation = operation


class PathMappingError(EmbyVideoTaggerError):
    """Raised when path mapping fails."""
    
    def __init__(self, message: str, source_path: Optional[str] = None, 
                 mapped_path: Optional[str] = None, **kwargs):
        """Initialize path mapping error."""
        details = kwargs.get('details', {})
        if source_path:
            details['source_path'] = source_path
        if mapped_path:
            details['mapped_path'] = mapped_path
        super().__init__(message, details)
        self.source_path = source_path
        self.mapped_path = mapped_path


class VideoValidationError(EmbyVideoTaggerError):
    """Raised when video validation fails."""
    
    def __init__(self, message: str, video_id: Optional[str] = None, 
                 reason: Optional[str] = None, **kwargs):
        """Initialize video validation error."""
        details = kwargs.get('details', {})
        if video_id:
            details['video_id'] = video_id
        if reason:
            details['reason'] = reason
        super().__init__(message, details)
        self.video_id = video_id
        self.reason = reason


class RetryableError(EmbyVideoTaggerError):
    """Base class for errors that can be retried."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        """Initialize retryable error."""
        details = kwargs.get('details', {})
        if retry_after:
            details['retry_after'] = retry_after
        super().__init__(message, details)
        self.retry_after = retry_after


class RateLimitError(RetryableError):
    """Raised when API rate limit is exceeded."""
    pass


class TemporaryError(RetryableError):
    """Raised for temporary failures that may succeed on retry."""
    pass