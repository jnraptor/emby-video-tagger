# Emby Video Tagger

An automated video tagging system for Emby media server that uses AI vision analysis to generate intelligent tags for your video content. The system extracts representative frames from videos, analyzes them using local AI models, and automatically updates Emby metadata with descriptive tags.

## 🚀 New in Version 2.0

Version 2.0 brings a complete architectural overhaul with:

- **Modular Architecture**: Clean separation of concerns with dedicated modules for each component
- **Async/Concurrent Processing**: 3-5x faster processing with concurrent video and frame analysis
- **Enhanced Error Handling**: Comprehensive error handling with retry mechanisms
- **Type Safety**: Full type hints throughout the codebase
- **Better Configuration**: Pydantic-based configuration with validation
- **Improved CLI**: Rich terminal interface with progress indicators
- **Database Integration**: SQLAlchemy-based task tracking and statistics

## Features

- **Automated Video Processing**: Automatically processes recently added videos in your Emby library
- **AI-Powered Analysis**: Uses local AI models (LM Studio or Ollama) for intelligent content analysis
- **Smart Frame Extraction**: Intelligently extracts representative frames using scene detection
- **Flexible Configuration**: Support for multiple AI providers and customizable processing options
- **Robust Error Handling**: Comprehensive error handling with retry mechanisms
- **Task Tracking**: SQLite-based task tracking to prevent duplicate processing
- **Scheduled Processing**: Built-in scheduling for automated daily processing
- **Path Mapping**: Flexible path mapping for different server configurations
- **Rich CLI**: Beautiful command-line interface with progress tracking
- **Comprehensive Logging**: Structured logging with JSON output support

## Requirements

- Python 3.8+
- Emby Media Server
- LM Studio or Ollama (for AI processing)
- FFmpeg (for video frame extraction)

## Installation

### From Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/emby-video-tagger.git
   cd emby-video-tagger
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

   Or for development:
   ```bash
   pip install -e ".[dev]"
   ```

### Using pip

```bash
pip install emby-video-tagger
```

## Configuration

1. **Copy the example environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** with your configuration:

   ```env
   # Emby Server Configuration
   EMBY_SERVER_URL=http://localhost:8096
   EMBY_API_KEY=your-emby-api-key-here
   EMBY_USER_ID=your-emby-user-id-here

   # AI Provider Configuration (choose one)
   AI_PROVIDER=lmstudio  # or "ollama"

   # LM Studio Configuration (if using LM Studio)
   LMSTUDIO_MODEL_NAME=qwen2.5-vl-7b-instruct-abliterated

   # Ollama Configuration (if using Ollama)
   OLLAMA_MODEL_NAME=llava
   OLLAMA_BASE_URL=http://localhost:11434

   # Path Mappings
   # Format: /source_path:/destination_path
   PATH_MAPPINGS=/volume1/shows:/Volumes/shows,/volume1/movies:/Volumes/movies

   # Processing Configuration
   PROCESSING_DAYS_BACK=5
   PROCESSING_MAX_FRAMES_PER_VIDEO=10
   PROCESSING_SCENE_THRESHOLD=27.0
   PROCESSING_MAX_CONCURRENT_VIDEOS=3
   PROCESSING_MAX_CONCURRENT_FRAMES=5

   # Scheduler Configuration
   SCHEDULER_ENABLED=true
   SCHEDULER_HOUR=2
   SCHEDULER_MINUTE=0

   # Logging Configuration
   LOGGING_LEVEL=INFO
   LOGGING_FORMAT=json
   LOGGING_FILE=video_tagging.log
   ```

### Getting Emby Credentials

1. **API Key**: 
   - Go to Emby Dashboard → Advanced → Security → API Keys
   - Create a new API key for the application

2. **User ID**:
   - Go to Emby Dashboard → Users
   - Click on your user and copy the User ID from the URL

### AI Model Setup

**Option 1: LM Studio**
- Install [LM Studio](https://lmstudio.ai/)
- Download a vision-capable model (e.g., `qwen2.5-vl-7b-instruct`)
- Start the local server in LM Studio
- Set `AI_PROVIDER=lmstudio` in your `.env` file

**Option 2: Ollama**
- Install [Ollama](https://ollama.ai/)
- Download a vision model: `ollama pull llava`
- Start the Ollama server: `ollama serve`
- Set `AI_PROVIDER=ollama` in your `.env` file

## Usage

### Command Line Interface

The application provides a rich CLI with several commands:

#### Process Recent Videos

Process videos added in the last N days:

```bash
emby-video-tagger process --days-back 7 --max-concurrent 3
```

#### Process Specific Video

Process a single video by its Emby ID:

```bash
emby-video-tagger process-video VIDEO_ID
```

#### View Statistics

Display processing statistics:

```bash
emby-video-tagger stats
```

#### Retry Failed Videos

Reprocess videos that previously failed:

```bash
emby-video-tagger retry-failed
```

#### Run Scheduled Processing

Start the scheduler for automated daily processing:

```bash
emby-video-tagger schedule
```

### Python API

You can also use the package programmatically:

```python
import asyncio
from emby_video_tagger.config.settings import AppConfig
from emby_video_tagger.services.emby import EmbyService
from emby_video_tagger.services.orchestrator import VideoTaggingOrchestrator

async def main():
    # Load configuration
    config = AppConfig.load_config()
    
    # Initialize services
    emby_service = EmbyService(config.emby)
    # ... initialize other services
    
    # Create orchestrator
    orchestrator = VideoTaggingOrchestrator(
        emby_service=emby_service,
        # ... other services
    )
    
    # Process recent videos
    results = await orchestrator.process_recent_videos(days_back=5)
    
    for result in results:
        if result.is_successful:
            print(f"✓ {result.video_id}: {len(result.tags)} tags")
        else:
            print(f"✗ {result.video_id}: {result.error}")

asyncio.run(main())
```

## Architecture

The application follows a modular architecture with clear separation of concerns:

```
emby_video_tagger/
├── cli.py                   # Command-line interface
├── config/                  # Configuration management
│   ├── settings.py         # Pydantic configuration models
│   └── validators.py       # Configuration validators
├── core/                    # Core domain models and interfaces
│   ├── models.py           # Data models
│   ├── exceptions.py       # Custom exceptions
│   └── interfaces.py       # Abstract interfaces
├── services/                # Business logic services
│   ├── emby.py            # Emby API integration
│   ├── frame_extractor.py # Video frame extraction
│   ├── vision/            # AI vision processing
│   │   ├── base.py       # Base vision processor
│   │   ├── lmstudio.py   # LM Studio implementation
│   │   ├── ollama.py     # Ollama implementation
│   │   └── factory.py    # Vision processor factory
│   └── orchestrator.py    # Main workflow orchestration
├── storage/                # Data persistence
│   ├── database.py        # Database connection
│   ├── models.py          # SQLAlchemy models
│   └── repository.py      # Data access layer
├── scheduler/              # Task scheduling
│   └── jobs.py           # Scheduled job definitions
└── utils/                  # Utilities
    └── logging.py         # Logging configuration
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=emby_video_tagger

# Run specific test file
pytest tests/unit/test_emby_service.py
```

### Code Quality

```bash
# Format code
black emby_video_tagger

# Lint code
ruff check emby_video_tagger

# Type checking
mypy emby_video_tagger
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Troubleshooting

### Common Issues

1. **"Video file not found"**:
   - Check path mappings configuration
   - Ensure the processing system can access video files
   - Verify Emby path vs. local system path differences

2. **"Failed to extract frames"**:
   - Ensure FFmpeg is installed and accessible
   - Check video file permissions
   - Verify video file is not corrupted

3. **"API authentication failed"**:
   - Verify Emby API key is correct
   - Check Emby server URL is accessible
   - Ensure User ID is valid

4. **"AI model not responding"**:
   - Check that LM Studio/Ollama is running
   - Verify the model is loaded
   - Check the configured port numbers

### Debug Mode

Enable debug logging by setting `LOGGING_LEVEL=DEBUG` in your `.env` file.

## Performance Tips

1. **Concurrent Processing**: Adjust `PROCESSING_MAX_CONCURRENT_VIDEOS` based on your system resources
2. **Frame Count**: Lower `PROCESSING_MAX_FRAMES_PER_VIDEO` for faster processing
3. **Scene Threshold**: Adjust `PROCESSING_SCENE_THRESHOLD` to control scene detection sensitivity
4. **Database**: For large libraries, consider using PostgreSQL instead of SQLite

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Emby Media Server](https://emby.media/) for the excellent media server platform
- [LM Studio](https://lmstudio.ai/) and [Ollama](https://ollama.ai/) for local AI model hosting
- [PySceneDetect](https://scenedetect.com/) for intelligent frame extraction
- [Rich](https://github.com/Textualize/rich) for the beautiful CLI interface

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs in `video_tagging.log`
3. Open an issue on GitHub with detailed information about your setup and the problem

## Changelog

### Version 2.0.0
- Complete architectural overhaul with modular design
- Added async/concurrent processing for 3-5x performance improvement
- Implemented comprehensive error handling and retry mechanisms
- Added SQLAlchemy-based database integration
- Created rich CLI with progress tracking
- Added support for Ollama in addition to LM Studio
- Implemented proper dependency injection
- Added full type hints throughout the codebase
- Improved configuration management with Pydantic
- Enhanced logging with structured output support

### Version 1.0.0
- Initial release with basic functionality
- Support for LM Studio vision models
- Scene detection-based frame extraction
- Basic Emby integration