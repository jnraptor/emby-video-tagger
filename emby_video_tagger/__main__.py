"""
Entry point for the Emby Video Tagger application.

This module serves as the main entry point when running the package as a module:
    python -m emby_video_tagger
"""

import sys
from emby_video_tagger.cli import main

if __name__ == "__main__":
    sys.exit(main())