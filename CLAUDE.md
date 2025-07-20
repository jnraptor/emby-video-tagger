# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Copy environment template and configure
cp .env.example .env
# Edit .env with your Emby server details and LM Studio configuration
```

### Running the Application
```bash
# Run with automatic scheduling (daily at 2 AM)
python emby_video_tagger.py

# Process recent videos once without scheduling
python emby_video_tagger.py once

# Process a specific video by ID
python emby_video_tagger.py manual <video_id>

# Show processing statistics
python emby_video_tagger.py stats
```

### Testing and Development
```bash
# Run manual processing for debugging
python emby_video_tagger.py manual <video_id>

# Check logs for debugging
tail -f video_tagging.log
```

## Architecture Overview

This is an automated video tagging system for Emby media servers that uses AI vision analysis to generate intelligent tags. The system operates in a pipeline:

### Core Components

1. **EmbyVideoTagger** (`emby_video_tagger.py:57-133`): Handles Emby API interactions
   - Retrieves recently added videos via Emby REST API
   - Updates video metadata with generated tags
   - Manages API authentication and rate limiting

2. **IntelligentFrameExtractor** (`emby_video_tagger.py:135-253`): Video frame extraction
   - Uses PySceneDetect for intelligent scene detection
   - Extracts representative frames from different scenes
   - Falls back to uniform sampling if scene detection fails
   - Creates temporary directories for frame storage

3. **Vision Processing Architecture** (Factory Pattern):
   - **BaseVisionProcessor** (`emby_video_tagger.py:257-319`): Abstract base class defining common interface
   - **LMStudioVisionProcessor** (`emby_video_tagger.py:322-366`): LM Studio integration
   - **OllamaVisionProcessor** (`emby_video_tagger.py:369-425`): Ollama integration
   - **VisionProcessorFactory** (`emby_video_tagger.py:428-442`): Creates appropriate processor based on configuration
   - All processors handle JSON response parsing and tag category flattening

4. **VideoTaggingAutomation** (`emby_video_tagger.py:445-736`): Main orchestration class
   - Coordinates the entire tagging pipeline
   - Manages task tracking via SQLite database
   - Implements path remapping for cross-platform file access
   - Provides scheduled automation via APScheduler
   - Uses factory pattern to select AI provider at runtime

### Key Features

- **Path Remapping**: Handles different file paths between Emby server and processing system
- **Task Tracking**: SQLite database prevents duplicate processing and tracks status
- **Scheduled Processing**: Daily automated processing of new videos
- **Comprehensive Logging**: Detailed logging to `video_tagging.log`
- **Error Recovery**: Robust error handling with retry mechanisms
- **Temporary File Management**: Automatic cleanup of extracted frame files

### External Dependencies

- **Emby Media Server**: Source of video metadata and file information
- **AI Providers**: Local AI model hosting for vision analysis
  - **LM Studio**: Primary option with qwen2.5-vl models
  - **Ollama**: Alternative with llava/llama3.2-vision models
- **FFmpeg**: Video processing (via OpenCV) for frame extraction
- **PySceneDetect**: Intelligent scene detection for frame selection

### Configuration

Environment variables (via `.env` file):
- `EMBY_SERVER_URL`, `EMBY_API_KEY`, `EMBY_USER_ID`: Emby server connection
- `AI_PROVIDER`: Choose between "lmstudio" or "ollama" (default: "lmstudio")
- `LMSTUDIO_MODEL_NAME`: LM Studio model specification
- `OLLAMA_MODEL_NAME`, `OLLAMA_BASE_URL`: Ollama configuration
- `PATH_MAPPINGS`: Cross-platform path translation
- `DAYS_BACK`: Processing window for recent videos

### Database Schema

- `video_tasks.db`: Task tracking with status, timestamps, and error logging
- `jobs.sqlite`: APScheduler job persistence for automated scheduling

### Processing Flow

1. Query Emby for recently added videos (configurable timeframe)
2. Filter videos (skip already processed, check file existence/size)
3. Extract representative frames using scene detection
4. Analyze frames with selected AI provider (LM Studio or Ollama) to generate tags
5. Update Emby metadata with generated tags plus "ai-generated" marker
6. Log results and clean up temporary files

### AI Provider Selection

The system uses a factory pattern to select the appropriate AI processor:
- Configuration determines provider via `AI_PROVIDER` environment variable
- Factory creates either `LMStudioVisionProcessor` or `OllamaVisionProcessor`
- Both processors implement the same `BaseVisionProcessor` interface
- Switching providers requires only configuration changes, no code modifications