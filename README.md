# Auto Video Tagging for Emby

An automated video tagging system for Emby media server that uses AI vision analysis to generate intelligent tags for your video content. The system extracts representative frames from videos, analyzes them using local LM Studio models, and automatically updates Emby metadata with descriptive tags.

## Features

- **Automated Video Processing**: Automatically processes recently added videos in your Emby library
- **AI-Powered Analysis**: Uses local LM Studio models for intelligent content analysis
- **Smart Frame Extraction**: Intelligently extracts representative frames using scene detection
- **Flexible Configuration**: Support for local (LM Studio) AI models
- **Robust Error Handling**: Comprehensive error handling with retry mechanisms
- **Task Tracking**: SQLite-based task tracking to prevent duplicate processing
- **Scheduled Processing**: Built-in scheduling for automated daily processing
- **Path Mapping**: Flexible path mapping for different server configurations
- **Logging**: Comprehensive logging for monitoring and debugging

## Requirements

- Python 3.8+
- Emby Media Server
- LM Studio (for local processing)
- FFmpeg (for video frame extraction)

## Installation

You can run this application either with Docker (recommended) or directly with Python.

### Option 1: Docker Installation (Recommended)

Docker provides an isolated environment with all dependencies pre-configured, including OpenCV with CUDA support.

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd emby-video-tagger
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings (see Configuration section)
   ```

3. **Update docker-compose.yml**:
   - Edit the media volume path: `/path/to/your/media:/media:ro`
   - Uncomment GPU sections if you have NVIDIA GPU support
   - Remove ollama service if using external AI provider

4. **Build and run**:
   ```bash
   # Start with scheduling (runs daily at 2 AM)
   docker-compose up -d
   
   # Or run once and exit
   docker-compose run --rm emby-video-tagger once
   ```

### Option 2: Direct Python Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd emby-video-tagger
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg** (required for video processing):
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)

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
   DAYS_BACK=5
   ```

### Getting Emby Credentials

1. **API Key**: 
   - Go to Emby Dashboard → Advanced → Security → API Keys
   - Create a new API key for the application

2. **User ID**:
   - Go to Emby Dashboard → Users
   - Click on your user and copy the User ID from the URL

### AI Model Options

You can choose between two AI providers for video frame analysis:

**Option 1: LM Studio (Default)**
- Install [LM Studio](https://lmstudio.ai/)
- Download a vision-capable model (e.g., `qwen2.5-vl-7b-instruct`)
- Start the local server in LM Studio
- Set `AI_PROVIDER=lmstudio` and `LMSTUDIO_MODEL_NAME` in your `.env` file

**Option 2: Ollama**
- Install [Ollama](https://ollama.ai/)
- Download a vision model: `ollama pull llava` or `ollama pull llama3.2-vision`
- Start the Ollama server: `ollama serve`
- Set `AI_PROVIDER=ollama` and `OLLAMA_MODEL_NAME` in your `.env` file

## Usage

### Docker Usage

#### Automated Processing (Recommended)
```bash
# Start with scheduling (runs daily at 2 AM)
docker-compose up -d

# View logs
docker-compose logs -f emby-video-tagger
```

#### Manual Processing
```bash
# Process recent videos once
docker-compose run --rm emby-video-tagger once

# Process recent videos including favorites
docker-compose run --rm emby-video-tagger once --include-favorites

# Process only favorite videos
docker-compose run --rm emby-video-tagger favorites

# Process a specific video by ID
docker-compose run --rm emby-video-tagger manual <video_id>

# Show processing statistics
docker-compose run --rm emby-video-tagger stats
```

#### Management Commands
```bash
# Stop the service
docker-compose down

# Rebuild after code changes
docker-compose build

# View container status
docker-compose ps
```

### Direct Python Usage

#### Manual Processing

Process a specific video by ID:
```bash
python emby_video_tagger.py manual <video_id>
```

Process recent videos (last 5 days):
```bash
python emby_video_tagger.py once
```

Process recent videos including favorites:
```bash
python emby_video_tagger.py once --include-favorites
```

Process only favorite videos:
```bash
python emby_video_tagger.py favorites
```

Show processing statistics:
```bash
python emby_video_tagger.py stats
```

#### Automated Processing

The system can run as a scheduled service to automatically process new videos:

```bash
python emby_video_tagger.py
```

This will:
- Process videos added in the last 5 days (configurable)
- Run daily at 2 AM (configurable in code)
- Skip already processed videos
- Log all activities

### Configuration Options

The system supports various configuration options through environment variables:

- `DAYS_BACK`: Number of days to look back for new videos (default: 5)

## Path Mapping

If your Emby server and processing system have different path structures, configure path mappings in .env:

- PATH_MAPPINGS=/volume1/shows:/Volumes/shows,/volume1/movies:/Volumes/movies

## Logging

The system creates detailed logs in `video_tagging.log` including:
- Processing status for each video
- AI analysis results
- Error messages and stack traces
- Performance metrics

## Database

The system uses SQLite databases for:
- **Task tracking** (`video_tasks.db`): Prevents duplicate processing
- **Job scheduling** (`jobs.sqlite`): APScheduler job persistence

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

### Debug Mode

Enable debug logging by modifying the logging configuration in the code:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- [Emby Media Server](https://emby.media/) for the excellent media server platform
- [LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.ai/) for local AI model hosting
- [PySceneDetect](https://scenedetect.com/) for intelligent frame extraction

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs in `video_tagging.log`
3. Open an issue on GitHub with detailed information about your setup and the problem