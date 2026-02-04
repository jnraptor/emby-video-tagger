# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Use virtual env
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Copy environment template and configure
cp .env.example .env
# Edit .env with your Emby server details and AI provider configuration
```

### Running the Application
```bash
# Run with automatic scheduling (daily at 2 AM)
python emby_video_tagger.py

# Process recent videos once without scheduling
python emby_video_tagger.py once

# Process recent videos including favorites
python emby_video_tagger.py once --include-favorites

# Process only favorite videos
python emby_video_tagger.py favorites

# Run frame extraction pass only (two-pass architecture)
python emby_video_tagger.py extract

# Run frame analysis pass only (two-pass architecture)
python emby_video_tagger.py analyze

# Process a specific video by ID (runs both passes)
python emby_video_tagger.py manual <video_id>

# Show processing statistics
python emby_video_tagger.py stats

# Consolidate duplicate tags (dry run by default)
python emby_video_tagger.py consolidate-tags
python emby_video_tagger.py consolidate-tags --use-llm  # Use LLM for semantic analysis
```

### Testing and Development
```bash
# Run manual processing for debugging
python emby_video_tagger.py manual <video_id>

# Check logs for debugging
tail -f video_tagging.log
```

## Architecture Overview

This is an automated video tagging system for Emby media servers that uses AI vision analysis to generate intelligent tags. The system operates using a **two-pass architecture**:

1. **Extraction Pass**: Frames are cached to disk for all pending videos (concurrent)
2. **Analysis Pass**: Cached frames are analyzed via AI and Emby is updated (concurrent)

This separation enables:
- Parallel extraction across multiple videos without waiting for AI analysis
- Ability to run extraction/analysis independently for debugging
- Recovery from failures by re-running either pass

### Core Components

1. **TaskStatus** (`emby_video_tagger.py:58-63`): Enum tracking processing states
   - `PENDING_EXTRACTION`, `PENDING_ANALYSIS`, `PROCESSING`, `COMPLETED`, `FAILED`

2. **EmbyVideoTagger** (`emby_video_tagger.py:66-258`): Handles Emby API interactions
   - Retrieves recently added videos via Emby REST API
   - Retrieves favorite videos using IsFavorite filter
   - Updates video metadata with generated tags
   - Retrieves/manages tags for consolidation operations
   - Manages API authentication and rate limiting

3. **IntelligentFrameExtractor** (`emby_video_tagger.py:261-437`): Video frame extraction
   - Uses PySceneDetect for intelligent scene detection with AdaptiveDetector
   - Resizes frames to stay within `MAX_PIXELS` limit (default 640k pixels)
   - Extracts representative frames from different scenes
   - Falls back to uniform sampling if scene detection fails
   - Uses PIL/LANCZOS for high-quality frame processing

4. **Vision Processing Architecture** (Factory Pattern):
   - **BaseVisionProcessor** (`emby_video_tagger.py:440-632`): Abstract base class with parallel processing, tag normalization against existing library tags, and JSON parsing from markdown
   - **LMStudioVisionProcessor** (`emby_video_tagger.py:634-692`): LM Studio with configurable parallel requests (default 2)
   - **OllamaVisionProcessor** (`emby_video_tagger.py:695-770`): Ollama with rate limiting and parallel processing (default 1)
   - **APIVisionProcessor** (`emby_video_tagger.py:773-889`): Z.AI API with parallel request handling (default 3)
   - **VisionProcessorFactory** (`emby_video_tagger.py:892-918`): Creates appropriate processor via `AI_PROVIDER` config

5. **TagConsolidator** (`emby_video_tagger.py:921-1100`): Tag deduplication and merging
   - Rule-based consolidation: normalizes and merges tag variants (action-shot → action shot)
   - Optional LLM-based semantic duplicate detection via `analyze_tags_for_consolidation()`
   - Updates Emby videos atomically when merging tags
   - Supports dry-run and interactive modes

6. **VideoTaggingAutomation** (`emby_video_tagger.py:1103-1700`): Main orchestration class
   - Two-pass processing: `run_extraction_pass()` then `run_analysis_pass()`
   - Task tracking via SQLite with status (PENDING_EXTRACTION/ANALYSIS, COMPLETED, FAILED) and source type (recent/favorites/manual)
   - Path remapping via `PATH_MAPPINGS` for cross-platform file access
   - Scheduled automation via APScheduler using MemoryJobStore
   - Favorite video copying to configurable `COPY_FAVORITES_TO` destination
   - Uses `concurrent.futures.ThreadPoolExecutor` for parallel video processing

### Key Features

- **Two-Pass Architecture**: Separate extraction and analysis passes allow concurrent video processing across both phases
- **Path Remapping**: Handles different file paths between Emby server and processing system
- **Task Tracking**: SQLite database prevents duplicate processing and tracks status (PENDING_EXTRACTION/ANALYSIS, COMPLETED, FAILED) and source type (recent/favorites/manual)
- **Tag Normalization**: Generated tags are normalized against existing Emby library tags, preferring established forms
- **Frame Resizing**: Automatically resizes extracted frames to stay within size limits (`MAX_PIXELS`) to optimize AI processing
- **Frame Caching**: Extracted frames cached to `FRAME_CACHE_PATH` directory, cleaned up after successful analysis
- **Scheduled Processing**: Daily automated processing of new videos at 2 AM with optional favorites inclusion
- **Favorites Processing**: Dedicated support for processing favorite videos separately or combined, with optional copying to `COPY_FAVORITES_TO` directory
- **Comprehensive Logging**: Detailed logging to `video_tagging.log` with source type identification
- **Error Recovery**: Failed tasks can be retried by re-running either pass independently
- **Tag Consolidation**: Built-in tool for merging duplicate/variant tags (rule-based and optional LLM semantic analysis)
- **Parallel Processing**: Configurable concurrent frame analysis and concurrent video processing
  - **LM Studio**: Default 2 concurrent requests (optimized for local processing)
  - **Ollama**: Default 1 concurrent request (respects built-in rate limiting)
  - **Z.AI API**: Default 3 concurrent requests (optimized for cloud API throughput)
  - **Concurrent Videos**: Configurable via `MAX_CONCURRENT_VIDEOS` (default 2) for parallel extraction/analysis

### External Dependencies

- **Emby Media Server**: Source of video metadata and file information
- **AI Providers**: Vision analysis through multiple provider options
  - **LM Studio**: Local option with qwen2.5-vl models
  - **Ollama**: Local alternative with llava/llama3.2-vision models
  - **Z.AI API**: Cloud-based option with glm-4.5v model
- **FFmpeg**: Video processing (via OpenCV) for frame extraction
- **PySceneDetect**: Intelligent scene detection for frame selection

### Configuration

Environment variables (via `.env` file):
- `EMBY_SERVER_URL`, `EMBY_API_KEY`, `EMBY_USER_ID`: Emby server connection
- `AI_PROVIDER`: Choose between "lmstudio", "ollama", or "api" (default: "lmstudio")
- `LMSTUDIO_MODEL_NAME`: LM Studio model specification (default: "qwen2.5-vl-7b-instruct-abliterated")
- `LMSTUDIO_MAX_CONCURRENT`: Maximum concurrent requests for LM Studio (default: 2)
- `OLLAMA_MODEL_NAME`, `OLLAMA_BASE_URL`: Ollama configuration
- `OLLAMA_MAX_CONCURRENT`: Maximum concurrent requests for Ollama (default: 1)
- `API_MODEL_NAME`, `API_BASE_URL`, `API_AUTH_TOKEN`: Z.AI API configuration
- `API_MAX_CONCURRENT`: Maximum concurrent requests for Z.AI API (default: 3)
- `PATH_MAPPINGS`: Cross-platform path translation (format: "/source:/dest,/source2:/dest2")
- `DAYS_BACK`: Processing window for recent videos (default: 5)
- `PROCESS_FAVORITES`: Include favorites in scheduled processing (default: false)
- `FAVORITES_ONLY`: Process only favorites, ignore recent videos (default: false)
- `COPY_FAVORITES_TO`: Destination directory to copy favorite videos (default: empty/disabled)
- `MAX_CONCURRENT_VIDEOS`: Number of videos to process concurrently (default: 2)
- `FRAME_CACHE_PATH`: Directory for temporary frame storage (default: "/tmp/frame_cache")
- `MAX_PIXELS`: Maximum frame pixel count for AI processing (default: 640000 = 800x800)

### Processing Flow

The system uses a two-pass architecture:

**Extraction Pass** (`run_extraction_pass()` or `extract` command):
1. Query Emby for videos based on configuration:
   - Recent videos (within `DAYS_BACK` window)
   - Favorite videos (using IsFavorite API filter)
   - Or both, depending on `PROCESS_FAVORITES` and `FAVORITES_ONLY` settings
2. Filter videos (skip already processed via "ai-generated" tag, check file existence/size)
3. For favorites with `COPY_FAVORITES_TO` configured: copy video to destination with sanitized filename
4. Extract representative frames to `FRAME_CACHE_PATH/<video_id>/` using scene detection
5. Update task status to `PENDING_ANALYSIS` in video_tasks.db

**Analysis Pass** (`run_analysis_pass()` or `analyze` command):
1. Fetch all tasks with status `PENDING_ANALYSIS` from video_tasks.db
2. Retrieve existing Emby library tags for normalization
3. Analyze cached frames with selected AI provider to generate tags
4. Normalize tags against existing library tags (prefer established forms)
5. Update Emby metadata with combined existing tags + new normalized tags + "ai-generated" marker
6. Clean up frame cache directory for successful completions
7. Update task status to `COMPLETED` or `FAILED`

### AI Provider Selection

The system uses a factory pattern to select the appropriate AI processor:
- Configuration determines provider via `AI_PROVIDER` environment variable
- Factory creates `LMStudioVisionProcessor`, `OllamaVisionProcessor`, or `APIVisionProcessor`
- All processors implement the same `BaseVisionProcessor` interface
- Switching providers requires only configuration changes, no code modifications

### Database Schema

- `video_tasks.db`: SQLite task tracking with columns:
  - `id`: Primary key (auto-increment)
  - `emby_id`: Emby video ID (UNIQUE constraint)
  - `file_path`: Local file path (after path remapping)
  - `status`: One of "pending_extraction", "pending_analysis", "processing", "completed", "failed"
  - `source_type`: "recent", "favorites", or "manual"
  - `created_at`: Timestamp when task was created
  - `completed_at`: Timestamp when task finished (NULL for incomplete)
  - `tag_count`: Number of tags generated (default 0)
  - `error_message`: Details on failure (NULL for successful tasks)

### Two-Pass Debugging

The two-pass architecture enables isolation and debugging:
```bash
# Run extraction pass only - frames saved to cache, tasks set to PENDING_ANALYSIS
python emby_video_tagger.py extract

# Check which videos are pending analysis
python emby_video_tagger.py stats

# Manually inspect extracted frames if needed
ls /tmp/frame_cache/<video_id>/

# Run analysis pass only - processes cached frames, sets tasks to COMPLETED/FAILED
python emby_video_tagger.py analyze

# Or combine: process specific video manually (runs both passes synchronously)
python emby_video_tagger.py manual <video_id>
```

If analysis fails for specific videos, you can:
1. Delete the failed task entry from video_tasks.db to allow re-processing
2. Re-run the extract/analyze commands
3. Or use `manual <video_id>` to process a single video end-to-end
