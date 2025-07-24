# API Contracts and Interfaces

This document defines the interfaces and contracts between components in the refactored Emby Video Tagger application.

## Core Interfaces

### 1. Video Model

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict

@dataclass
class Video:
    """Represents a video from Emby"""
    id: str
    name: str
    path: str
    date_created: datetime
    existing_tags: List[str]
    metadata: Dict[str, Any]
    
    @property
    def needs_processing(self) -> bool:
        """Check if video needs AI tag processing"""
        ai_indicators = ["ai-generated", "auto-tagged", "vision-analyzed"]
        return not any(
            indicator in tag.lower() 
            for tag in self.existing_tags 
            for indicator in ai_indicators
        )
```

### 2. Frame Model

```python
@dataclass
class Frame:
    """Represents an extracted video frame"""
    path: str
    frame_number: int
    timestamp: float
    video_id: str
    
    def cleanup(self) -> None:
        """Remove temporary frame file"""
        if Path(self.path).exists():
            Path(self.path).unlink()
```

### 3. Processing Result

```python
@dataclass
class ProcessingResult:
    """Result of video processing"""
    video_id: str
    status: TaskStatus
    tags: List[str]
    frames_processed: int
    processing_time: float
    error: Optional[str] = None
```

## Service Interfaces

### 1. Emby Service Interface

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class IEmbyService(ABC):
    """Interface for Emby API operations"""
    
    @abstractmethod
    async def get_recent_videos(
        self, 
        days_back: int = 7,
        limit: int = 100
    ) -> List[Video]:
        """Retrieve recently added videos from Emby"""
        pass
    
    @abstractmethod
    async def get_video_by_id(self, video_id: str) -> Optional[Video]:
        """Get specific video by ID"""
        pass
    
    @abstractmethod
    async def update_video_tags(
        self, 
        video_id: str, 
        tags: List[str]
    ) -> bool:
        """Update video tags in Emby"""
        pass
    
    @abstractmethod
    async def batch_update_tags(
        self, 
        updates: List[Tuple[str, List[str]]]
    ) -> Dict[str, bool]:
        """Batch update multiple videos' tags"""
        pass
```

### 2. Frame Extractor Interface

```python
class IFrameExtractor(ABC):
    """Interface for video frame extraction"""
    
    @abstractmethod
    async def extract_frames(
        self,
        video_path: str,
        max_frames: int = 10,
        strategy: str = "scene_detection"
    ) -> List[Frame]:
        """Extract representative frames from video"""
        pass
    
    @abstractmethod
    def validate_video(self, video_path: str) -> bool:
        """Check if video file is valid and accessible"""
        pass
    
    @abstractmethod
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata (duration, fps, resolution, etc.)"""
        pass
```

### 3. Vision Processor Interface

```python
class IVisionProcessor(ABC):
    """Interface for AI vision processing"""
    
    @abstractmethod
    async def analyze_frame(self, frame: Frame) -> Dict[str, List[str]]:
        """Analyze single frame and return categorized tags"""
        pass
    
    @abstractmethod
    async def analyze_frames_batch(
        self, 
        frames: List[Frame]
    ) -> List[Dict[str, List[str]]]:
        """Analyze multiple frames in batch"""
        pass
    
    @abstractmethod
    def merge_tags(
        self, 
        frame_results: List[Dict[str, List[str]]]
    ) -> List[str]:
        """Merge and deduplicate tags from multiple frames"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the AI model being used"""
        pass
```

### 4. Task Repository Interface

```python
from datetime import datetime
from typing import List, Optional

class ITaskRepository(ABC):
    """Interface for task persistence"""
    
    @abstractmethod
    async def create_task(
        self,
        video_id: str,
        file_path: str
    ) -> int:
        """Create new processing task"""
        pass
    
    @abstractmethod
    async def update_task_status(
        self,
        video_id: str,
        status: TaskStatus,
        tag_count: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update task status"""
        pass
    
    @abstractmethod
    async def get_task_by_video_id(
        self, 
        video_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get task by video ID"""
        pass
    
    @abstractmethod
    async def get_pending_tasks(
        self, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all pending tasks"""
        pass
    
    @abstractmethod
    async def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get processing statistics"""
        pass
```

### 5. Path Mapper Interface

```python
class IPathMapper(ABC):
    """Interface for path mapping between systems"""
    
    @abstractmethod
    def map_path(self, source_path: str) -> str:
        """Map source path to destination path"""
        pass
    
    @abstractmethod
    def validate_mapping(self, source_path: str) -> bool:
        """Check if path mapping exists and is valid"""
        pass
    
    @abstractmethod
    def add_mapping(
        self, 
        source_prefix: str, 
        dest_prefix: str
    ) -> None:
        """Add new path mapping"""
        pass
```

## Orchestrator Interface

```python
class IVideoTaggingOrchestrator(ABC):
    """Main orchestration interface"""
    
    @abstractmethod
    async def process_recent_videos(
        self,
        days_back: int = 5,
        max_concurrent: int = 3
    ) -> List[ProcessingResult]:
        """Process recently added videos"""
        pass
    
    @abstractmethod
    async def process_single_video(
        self,
        video_id: str
    ) -> ProcessingResult:
        """Process a single video by ID"""
        pass
    
    @abstractmethod
    async def process_videos_batch(
        self,
        video_ids: List[str],
        max_concurrent: int = 3
    ) -> List[ProcessingResult]:
        """Process multiple videos concurrently"""
        pass
    
    @abstractmethod
    async def reprocess_failed_videos(
        self,
        max_retries: int = 3
    ) -> List[ProcessingResult]:
        """Retry processing for failed videos"""
        pass
```

## Event System

```python
from enum import Enum
from typing import Any, Callable, Dict

class EventType(Enum):
    """System events"""
    VIDEO_PROCESSING_STARTED = "video_processing_started"
    VIDEO_PROCESSING_COMPLETED = "video_processing_completed"
    VIDEO_PROCESSING_FAILED = "video_processing_failed"
    FRAME_EXTRACTION_STARTED = "frame_extraction_started"
    FRAME_EXTRACTION_COMPLETED = "frame_extraction_completed"
    VISION_ANALYSIS_STARTED = "vision_analysis_started"
    VISION_ANALYSIS_COMPLETED = "vision_analysis_completed"
    TAGS_UPDATED = "tags_updated"

@dataclass
class Event:
    """System event"""
    type: EventType
    video_id: str
    timestamp: datetime
    data: Dict[str, Any]

class IEventBus(ABC):
    """Event bus interface for decoupled communication"""
    
    @abstractmethod
    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None]
    ) -> None:
        """Subscribe to an event type"""
        pass
    
    @abstractmethod
    def publish(self, event: Event) -> None:
        """Publish an event"""
        pass
```

## Configuration Interface

```python
from typing import Any, Dict, Optional

class IConfiguration(ABC):
    """Configuration interface"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass
    
    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section"""
        pass
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate configuration and return errors"""
        pass
    
    @abstractmethod
    def reload(self) -> None:
        """Reload configuration from source"""
        pass
```

## Error Handling

```python
from typing import Optional, Type

class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.handlers: Dict[Type[Exception], Callable] = {}
    
    def register_handler(
        self,
        exception_type: Type[Exception],
        handler: Callable[[Exception], None]
    ) -> None:
        """Register error handler for specific exception type"""
        self.handlers[exception_type] = handler
    
    def handle(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle error with appropriate handler"""
        handler = self.handlers.get(type(error))
        if handler:
            handler(error)
        else:
            self.logger.error(
                f"Unhandled error: {error}",
                exc_info=True,
                extra=context
            )
```

## Dependency Injection Container

```python
from typing import Any, Callable, Dict, Type

class DIContainer:
    """Simple dependency injection container"""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
    
    def register(
        self,
        interface: Type,
        implementation: Any = None,
        factory: Callable = None
    ) -> None:
        """Register service or factory"""
        if implementation:
            self._services[interface] = implementation
        elif factory:
            self._factories[interface] = factory
        else:
            raise ValueError("Must provide implementation or factory")
    
    def resolve(self, interface: Type) -> Any:
        """Resolve service"""
        if interface in self._services:
            return self._services[interface]
        elif interface in self._factories:
            return self._factories[interface]()
        else:
            raise ValueError(f"No registration for {interface}")
```

## Usage Example

```python
# Application setup
async def setup_application() -> DIContainer:
    """Configure and setup application"""
    container = DIContainer()
    
    # Load configuration
    config = AppConfig()
    container.register(IConfiguration, config)
    
    # Register services
    emby_service = EmbyService(
        config.emby.server_url,
        config.emby.api_key,
        config.emby.user_id
    )
    container.register(IEmbyService, emby_service)
    
    # Register frame extractor
    frame_extractor = IntelligentFrameExtractor()
    container.register(IFrameExtractor, frame_extractor)
    
    # Register vision processor
    vision_processor = VisionProcessorFactory.create(
        config.ai_provider,
        config.get_section(config.ai_provider)
    )
    container.register(IVisionProcessor, vision_processor)
    
    # Register repository
    db_url = config.get("database_url", "sqlite:///video_tasks.db")
    task_repository = TaskRepository(db_url)
    container.register(ITaskRepository, task_repository)
    
    # Register orchestrator
    orchestrator = VideoTaggingOrchestrator(
        emby_service=container.resolve(IEmbyService),
        frame_extractor=container.resolve(IFrameExtractor),
        vision_processor=container.resolve(IVisionProcessor),
        task_repository=container.resolve(ITaskRepository),
        config=container.resolve(IConfiguration)
    )
    container.register(IVideoTaggingOrchestrator, orchestrator)
    
    return container

# Main application entry
async def main():
    """Main application entry point"""
    container = await setup_application()
    orchestrator = container.resolve(IVideoTaggingOrchestrator)
    
    # Process recent videos
    results = await orchestrator.process_recent_videos(days_back=5)
    
    # Log results
    for result in results:
        if result.status == TaskStatus.COMPLETED:
            print(f"✓ {result.video_id}: {len(result.tags)} tags")
        else:
            print(f"✗ {result.video_id}: {result.error}")
```

## Testing Contracts

```python
# Example test using interfaces
import pytest
from unittest.mock import AsyncMock, Mock

@pytest.mark.asyncio
async def test_orchestrator_process_video():
    """Test video processing with mocked dependencies"""
    # Create mocks
    emby_service = AsyncMock(spec=IEmbyService)
    frame_extractor = AsyncMock(spec=IFrameExtractor)
    vision_processor = AsyncMock(spec=IVisionProcessor)
    task_repository = AsyncMock(spec=ITaskRepository)
    config = Mock(spec=IConfiguration)
    
    # Setup mock returns
    video = Video(
        id="123",
        name="Test Video",
        path="/path/to/video.mp4",
        date_created=datetime.now(),
        existing_tags=[],
        metadata={}
    )
    emby_service.get_video_by_id.return_value = video
    
    frames = [
        Frame(path="/tmp/frame1.jpg", frame_number=1, timestamp=1.0, video_id="123"),
        Frame(path="/tmp/frame2.jpg", frame_number=2, timestamp=2.0, video_id="123")
    ]
    frame_extractor.extract_frames.return_value = frames
    
    vision_processor.analyze_frames_batch.return_value = [
        {"subjects": ["person"], "activities": ["walking"]},
        {"subjects": ["car"], "activities": ["driving"]}
    ]
    vision_processor.merge_tags.return_value = ["person", "walking", "car", "driving"]
    
    emby_service.update_video_tags.return_value = True
    
    # Create orchestrator
    orchestrator = VideoTaggingOrchestrator(
        emby_service=emby_service,
        frame_extractor=frame_extractor,
        vision_processor=vision_processor,
        task_repository=task_repository,
        config=config
    )
    
    # Test
    result = await orchestrator.process_single_video("123")
    
    # Assertions
    assert result.status == TaskStatus.COMPLETED
    assert result.tags == ["person", "walking", "car", "driving"]
    assert result.frames_processed == 2
    
    # Verify calls
    emby_service.get_video_by_id.assert_called_once_with("123")
    frame_extractor.extract_frames.assert_called_once()
    vision_processor.analyze_frames_batch.assert_called_once_with(frames)
    emby_service.update_video_tags.assert_called_once_with(
        "123", 
        ["person", "walking", "car", "driving", "ai-generated"]
    )