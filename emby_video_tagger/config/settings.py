"""
Configuration settings for the Emby Video Tagger application.

This module defines all configuration models for the application, including:
- Emby server configuration
- AI provider configuration
- Path mappings
- General application settings
"""

import os
from typing import Dict, Optional, Union, Any
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):  # type: ignore
        """Fallback if python-dotenv is not installed."""
        return True


class AIProvider(str, Enum):
    """Supported AI providers for vision processing."""
    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"


@dataclass
class EmbyConfig:
    """Emby server configuration."""
    server_url: str
    api_key: str
    user_id: str
    
    def __post_init__(self):
        """Validate and clean configuration."""
        self.server_url = self.server_url.rstrip('/')
    
    @classmethod
    def from_env(cls) -> "EmbyConfig":
        """Create config from environment variables."""
        return cls(
            server_url=os.getenv("EMBY_SERVER_URL", "http://localhost:8096"),
            api_key=os.getenv("EMBY_API_KEY", ""),
            user_id=os.getenv("EMBY_USER_ID", "")
        )


@dataclass
class LMStudioConfig:
    """LMStudio configuration."""
    model_name: str = "gemma-3-4b-it-abliterated"
    
    @classmethod
    def from_env(cls) -> "LMStudioConfig":
        """Create config from environment variables."""
        return cls(
            model_name=os.getenv("LMSTUDIO_MODEL_NAME", "gemma-3-4b-it-abliterated")
        )


@dataclass
class OllamaConfig:
    """Ollama configuration."""
    model_name: str = "qwen2.5vl:3b"
    base_url: str = "http://localhost:11434"
    
    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Create config from environment variables."""
        return cls(
            model_name=os.getenv("OLLAMA_MODEL_NAME", "qwen2.5vl:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    days_back: int = 5
    max_frames_per_video: int = 10
    scene_threshold: float = 27.0
    max_concurrent_videos: int = 3
    max_concurrent_frames: int = 5
    
    @classmethod
    def from_env(cls) -> "ProcessingConfig":
        """Create config from environment variables."""
        return cls(
            days_back=int(os.getenv("PROCESSING_DAYS_BACK", "5")),
            max_frames_per_video=int(os.getenv("PROCESSING_MAX_FRAMES_PER_VIDEO", "10")),
            scene_threshold=float(os.getenv("PROCESSING_SCENE_THRESHOLD", "27.0")),
            max_concurrent_videos=int(os.getenv("PROCESSING_MAX_CONCURRENT_VIDEOS", "3")),
            max_concurrent_frames=int(os.getenv("PROCESSING_MAX_CONCURRENT_FRAMES", "5"))
        )


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///video_tasks.db"
    echo: bool = False
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        return cls(
            url=os.getenv("DATABASE_URL", cls.url),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true"
        )


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    enabled: bool = True
    hour: int = 2
    minute: int = 0
    
    @classmethod
    def from_env(cls) -> "SchedulerConfig":
        """Create config from environment variables."""
        return cls(
            enabled=os.getenv("SCHEDULER_ENABLED", "true").lower() == "true",
            hour=int(os.getenv("SCHEDULER_HOUR", "2")),
            minute=int(os.getenv("SCHEDULER_MINUTE", "0"))
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    file: Optional[Path] = field(default_factory=lambda: Path("video_tagging.log"))
    
    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """Create config from environment variables."""
        log_file = os.getenv("LOGGING_FILE", "video_tagging.log")
        return cls(
            level=os.getenv("LOGGING_LEVEL", "INFO"),
            format=os.getenv("LOGGING_FORMAT", "json"),
            file=Path(log_file) if log_file else None
        )


@dataclass
class AIProviderConfig:
    """AI provider configuration."""
    provider: AIProvider
    lmstudio: Optional[LMStudioConfig] = None
    ollama: Optional[OllamaConfig] = None
    
    def __post_init__(self):
        """Initialize provider-specific config."""
        if self.provider == AIProvider.LMSTUDIO and self.lmstudio is None:
            self.lmstudio = LMStudioConfig.from_env()
        elif self.provider == AIProvider.OLLAMA and self.ollama is None:
            self.ollama = OllamaConfig.from_env()
    
    @classmethod
    def from_env(cls) -> "AIProviderConfig":
        """Create config from environment variables."""
        provider_str = os.getenv("AI_PROVIDER", "lmstudio").lower()
        provider = AIProvider(provider_str)
        return cls(provider=provider)


@dataclass
class AppConfig:
    """Main application configuration."""
    emby: EmbyConfig
    ai: AIProviderConfig
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    path_mappings: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def load_config(cls) -> "AppConfig":
        """Load configuration from environment variables and .env file."""
        # Load .env file if it exists
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        # Parse path mappings
        path_mappings = cls._parse_path_mappings(os.getenv("PATH_MAPPINGS", ""))
        
        return cls(
            emby=EmbyConfig.from_env(),
            ai=AIProviderConfig.from_env(),
            processing=ProcessingConfig.from_env(),
            database=DatabaseConfig.from_env(),
            scheduler=SchedulerConfig.from_env(),
            logging=LoggingConfig.from_env(),
            path_mappings=path_mappings
        )
    
    @staticmethod
    def _parse_path_mappings(mappings_str: str) -> Dict[str, str]:
        """Parse path mappings from environment variable format."""
        mappings = {}
        if mappings_str:
            for mapping in mappings_str.split(','):
                if ':' in mapping:
                    emby_path, local_path = mapping.strip().split(':', 1)
                    mappings[emby_path.strip()] = local_path.strip()
        return mappings
    
    def get_ai_config(self) -> Union[LMStudioConfig, OllamaConfig]:
        """Get the active AI provider configuration."""
        if self.ai.provider == AIProvider.LMSTUDIO:
            if self.ai.lmstudio is None:
                raise ValueError("LMStudio config not initialized")
            return self.ai.lmstudio
        elif self.ai.provider == AIProvider.OLLAMA:
            if self.ai.ollama is None:
                raise ValueError("Ollama config not initialized")
            return self.ai.ollama
        else:
            raise ValueError(f"Unknown AI provider: {self.ai.provider}")
    
    def validate_config(self) -> list[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        # Check required Emby configuration
        if not self.emby.api_key:
            errors.append("EMBY_API_KEY is required")
        if not self.emby.user_id:
            errors.append("EMBY_USER_ID is required")
        
        # Validate path mappings
        for emby_path, local_path in self.path_mappings.items():
            if not Path(local_path).exists():
                errors.append(f"Local path does not exist: {local_path}")
        
        return errors