# Build/Lint/Test Commands
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python emby_video_tagger.py

# Test specific functionality
python emby_video_tagger.py manual <video_id>
python emby_video_tagger.py once
python emby_video_tagger.py stats
```

# Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports with blank lines
- **Formatting**: Follow PEP 8, use 4-space indentation, max 100 char lines
- **Types**: Use type hints consistently (List, Dict, Optional, etc.)
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPER_CASE for constants
- **Error Handling**: Use try-except blocks with specific exception types and logging
- **Logging**: Use logging module with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **Classes**: Use factory patterns for provider selection, abstract base classes for interfaces