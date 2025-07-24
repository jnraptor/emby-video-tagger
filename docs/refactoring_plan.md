# Emby Video Tagger Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring plan for the `emby_video_tagger.py` script to improve code readability, maintainability, and performance. The current monolithic 924-line script will be transformed into a modular, well-structured application following SOLID principles and Python best practices.

## Current State Analysis

### Issues Identified

1. **Monolithic Structure**
   - All functionality in a single 924-line file
   - Difficult to navigate and maintain
   - Hard to test individual components

2. **Mixed Responsibilities**
   - Classes handle multiple concerns (API calls, file I/O, logging, business logic)
   - Violation of Single Responsibility Principle

3. **Tight Coupling**
   - Direct dependencies between components
   - Difficult to mock for testing
   - Hard to swap implementations

4. **Performance Limitations**
   - Sequential processing of videos and frames
   - No concurrent execution
   - Inefficient resource utilization

5. **Configuration Management**
   - Configuration mixed with business logic
   - No validation of configuration values
   - Limited flexibility

## Proposed Architecture

### 1. Module Structure

```
emby_video_tagger/
├── __init__.py
├── __main__.py              # Entry point
├── cli.py                   # Command-line interface
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuration management
│   └── validators.py        # Configuration validation
├── core/
│   ├── __init__.py
│   ├── models.py           # Data models and enums
│   ├── exceptions.py       # Custom exceptions
│   └── interfaces.py       # Abstract base classes
├── services/
│   ├── __init__.py
│   ├── emby.py            # Emby API service
│   ├── frame_extractor.py # Video frame extraction
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── base.py        # Base vision processor
│   │   ├── lmstudio.py    # LMStudio implementation
│   │   ├── ollama.py      # Ollama implementation
│   │   └── factory.py     # Vision processor factory
│   └── orchestrator.py     # Main orchestration service
├── storage/
│   ├── __init__.py
│   ├── database.py         # Database operations
│   └── models.py           # SQLAlchemy models
├── utils/
│   ├── __init__.py
│   ├── logging.py          # Logging configuration
│   ├── path_mapper.py      # Path mapping utilities
│   └── decorators.py       # Utility decorators
├── scheduler/
│   ├── __init__.py
│   └── jobs.py             # Scheduled job definitions
└── tests/
    ├── __init__.py
    ├── unit/
    ├── integration/
    └── fixtures/
```

### 2. Key Design Improvements

#### A. Dependency Injection
```python
# Example: Orchestrator with injected dependencies
class VideoTaggingOrchestrator:
    def __init__(
        self,
        emby_service: EmbyService,
        frame_extractor: FrameExtractor,
        vision_processor: BaseVisionProcessor,
        task_repository: TaskRepository,
        config: Config
    ):
        self.emby_service = emby_service
        self.frame_extractor = frame_extractor
        self.vision_processor = vision_processor
        self.task_repository = task_repository
        self.config = config
```

#### B. Configuration Management
```python
# config/settings.py
from pydantic import BaseSettings, validator
from typing import Dict, Optional

class EmbyConfig(BaseSettings):
    server_url: str
    api_key: str
    user_id: str
    
    @validator('server_url')
    def validate_url(cls, v):
        # URL validation logic
        return v.rstrip('/')

class AppConfig(BaseSettings):
    emby: EmbyConfig
    ai_provider: str = "lmstudio"
    path_mappings: Dict[str, str] = {}
    days_back: int = 5
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
```

#### C. Error Handling Strategy
```python
# core/exceptions.py
class EmbyVideoTaggerError(Exception):
    """Base exception for all custom errors"""
    pass

class EmbyAPIError(EmbyVideoTaggerError):
    """Raised when Emby API operations fail"""
    pass

class FrameExtractionError(EmbyVideoTaggerError):
    """Raised when frame extraction fails"""
    pass

class VisionProcessingError(EmbyVideoTaggerError):
    """Raised when AI vision processing fails"""
    pass
```

#### D. Async/Concurrent Processing
```python
# services/orchestrator.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

class VideoTaggingOrchestrator:
    async def process_videos_async(self, videos: List[Video]):
        """Process multiple videos concurrently"""
        tasks = [self.process_single_video_async(video) for video in videos]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def process_single_video_async(self, video: Video):
        """Process single video with concurrent frame analysis"""
        # Extract frames
        frames = await self.extract_frames_async(video.path)
        
        # Analyze frames concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor, 
                    self.vision_processor.analyze_frame, 
                    frame
                )
                for frame in frames
            ]
            results = await asyncio.gather(*tasks)
        
        return self.merge_tags(results)
```

### 3. Performance Optimizations

1. **Concurrent Processing**
   - Process multiple videos in parallel
   - Analyze frames concurrently
   - Batch API requests where possible

2. **Caching**
   - Cache Emby API responses
   - Cache processed frame results
   - Implement smart cache invalidation

3. **Resource Management**
   - Connection pooling for API requests
   - Proper cleanup of temporary files
   - Memory-efficient frame processing

4. **Database Optimizations**
   - Use SQLAlchemy ORM for better query optimization
   - Implement database connection pooling
   - Add proper indexes for frequently queried fields

### 4. Testing Strategy

```python
# tests/unit/test_emby_service.py
import pytest
from unittest.mock import Mock, patch
from emby_video_tagger.services.emby import EmbyService

class TestEmbyService:
    @pytest.fixture
    def emby_service(self):
        return EmbyService(
            server_url="http://test.local",
            api_key="test-key",
            user_id="test-user"
        )
    
    @patch('requests.Session.get')
    def test_get_recent_videos(self, mock_get, emby_service):
        # Test implementation
        pass
```

### 5. Logging and Monitoring

```python
# utils/logging.py
import logging
import structlog
from typing import Dict, Any

def setup_logging(config: Dict[str, Any]):
    """Configure structured logging"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

### 6. Migration Plan

#### Phase 1: Setup and Infrastructure (Week 1)
- [ ] Create new package structure
- [ ] Set up configuration management with Pydantic
- [ ] Implement logging infrastructure
- [ ] Create base exceptions and interfaces

#### Phase 2: Core Services (Week 2)
- [ ] Extract and refactor EmbyService
- [ ] Extract and refactor FrameExtractor
- [ ] Extract and refactor Vision processors
- [ ] Implement dependency injection

#### Phase 3: Data Layer (Week 3)
- [ ] Implement SQLAlchemy models
- [ ] Create repository pattern for data access
- [ ] Migrate from raw SQL to ORM
- [ ] Add database migrations with Alembic

#### Phase 4: Orchestration and CLI (Week 4)
- [ ] Refactor orchestrator with async support
- [ ] Implement new CLI with Click
- [ ] Add comprehensive error handling
- [ ] Update scheduler implementation

#### Phase 5: Testing and Documentation (Week 5)
- [ ] Write unit tests (target 80% coverage)
- [ ] Write integration tests
- [ ] Update documentation
- [ ] Create API documentation

#### Phase 6: Performance and Polish (Week 6)
- [ ] Implement concurrent processing
- [ ] Add caching layer
- [ ] Performance testing and optimization
- [ ] Final code review and cleanup

## Benefits of Refactoring

1. **Improved Maintainability**
   - Clear separation of concerns
   - Easier to understand and modify
   - Better code organization

2. **Enhanced Testability**
   - Isolated components
   - Easy mocking and testing
   - Higher code coverage

3. **Better Performance**
   - Concurrent processing
   - Efficient resource utilization
   - Reduced processing time

4. **Increased Flexibility**
   - Easy to add new AI providers
   - Configurable and extensible
   - Plugin architecture ready

5. **Professional Code Quality**
   - Following Python best practices
   - Type hints throughout
   - Comprehensive documentation

## Next Steps

1. Review and approve this refactoring plan
2. Set up the new project structure
3. Begin incremental migration following the phases
4. Maintain backward compatibility during transition
5. Thoroughly test each component
6. Deploy and monitor the refactored application

## Conclusion

This refactoring will transform the current monolithic script into a professional, maintainable, and performant application. The modular architecture will make it easier to add features, fix bugs, and onboard new developers. The investment in refactoring will pay dividends in reduced maintenance costs and improved reliability.