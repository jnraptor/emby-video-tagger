"""Logging configuration and utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from structlog.stdlib import LoggerFactory

from emby_video_tagger.config.settings import LoggingConfig


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """
    Configure structured logging for the application.
    
    Args:
        config: Logging configuration
        
    Returns:
        Configured logger instance
    """
    # Set up standard logging
    log_level = getattr(logging, config.level.upper(), logging.INFO)
    
    # Configure handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    handlers.append(console_handler)
    
    # File handler if configured
    if config.file:
        # Ensure log directory exists
        config.file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.file)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure structlog if JSON format is requested
    if config.format == "json":
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
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Return structlog logger
        return structlog.get_logger("emby_video_tagger")
    else:
        # Return standard logger
        return logging.getLogger("emby_video_tagger")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (defaults to emby_video_tagger)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"emby_video_tagger.{name}")
    return logging.getLogger("emby_video_tagger")


class LoggerAdapter:
    """
    Adapter to make different logger types compatible with our interfaces.
    """
    
    def __init__(self, logger):
        """Initialize adapter with a logger."""
        self.logger = logger
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        if hasattr(self.logger, 'debug'):
            self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        if hasattr(self.logger, 'info'):
            self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        if hasattr(self.logger, 'warning'):
            self.logger.warning(message, **kwargs)
        elif hasattr(self.logger, 'warn'):
            self.logger.warn(message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message."""
        if hasattr(self.logger, 'error'):
            self.logger.error(message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log critical message."""
        if hasattr(self.logger, 'critical'):
            self.logger.critical(message, exc_info=exc_info, **kwargs)
        elif hasattr(self.logger, 'error'):
            self.logger.error(message, exc_info=exc_info, **kwargs)