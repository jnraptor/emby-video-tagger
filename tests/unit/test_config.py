"""Unit tests for configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from emby_video_tagger.config.settings import (
    AppConfig,
    EmbyConfig,
    AIProviderConfig,
    AIProvider,
    LMStudioConfig,
    OllamaConfig,
    ProcessingConfig,
    DatabaseConfig,
    SchedulerConfig,
    LoggingConfig,
)


class TestEmbyConfig:
    """Test EmbyConfig class."""
    
    def test_emby_config_creation(self):
        """Test creating EmbyConfig instance."""
        config = EmbyConfig(
            server_url="http://localhost:8096/",
            api_key="test-key",
            user_id="test-user"
        )
        
        assert config.server_url == "http://localhost:8096"  # Trailing slash removed
        assert config.api_key == "test-key"
        assert config.user_id == "test-user"
    
    def test_emby_config_from_env(self):
        """Test creating EmbyConfig from environment variables."""
        with patch.dict(os.environ, {
            "EMBY_SERVER_URL": "http://test.server:8096",
            "EMBY_API_KEY": "env-api-key",
            "EMBY_USER_ID": "env-user-id"
        }):
            config = EmbyConfig.from_env()
            
            assert config.server_url == "http://test.server:8096"
            assert config.api_key == "env-api-key"
            assert config.user_id == "env-user-id"
    
    def test_emby_config_defaults(self):
        """Test EmbyConfig default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = EmbyConfig.from_env()
            
            assert config.server_url == "http://localhost:8096"
            assert config.api_key == ""
            assert config.user_id == ""


class TestAIProviderConfig:
    """Test AIProviderConfig class."""
    
    def test_ai_provider_config_lmstudio(self):
        """Test AIProviderConfig with LMStudio."""
        config = AIProviderConfig(provider=AIProvider.LMSTUDIO)
        
        assert config.provider == AIProvider.LMSTUDIO
        assert config.lmstudio is not None
        assert config.lmstudio.model_name == "qwen2.5-vl-7b-instruct-abliterated"
    
    def test_ai_provider_config_ollama(self):
        """Test AIProviderConfig with Ollama."""
        config = AIProviderConfig(provider=AIProvider.OLLAMA)
        
        assert config.provider == AIProvider.OLLAMA
        assert config.ollama is not None
        assert config.ollama.model_name == "llava"
        assert config.ollama.base_url == "http://localhost:11434"
    
    def test_ai_provider_config_from_env(self):
        """Test creating AIProviderConfig from environment variables."""
        with patch.dict(os.environ, {
            "AI_PROVIDER": "ollama",
            "OLLAMA_MODEL_NAME": "custom-model",
            "OLLAMA_BASE_URL": "http://custom:11434"
        }):
            config = AIProviderConfig.from_env()
            
            assert config.provider == AIProvider.OLLAMA
            assert config.ollama.model_name == "custom-model"
            assert config.ollama.base_url == "http://custom:11434"


class TestProcessingConfig:
    """Test ProcessingConfig class."""
    
    def test_processing_config_defaults(self):
        """Test ProcessingConfig default values."""
        config = ProcessingConfig()
        
        assert config.days_back == 5
        assert config.max_frames_per_video == 10
        assert config.scene_threshold == 27.0
        assert config.max_concurrent_videos == 3
        assert config.max_concurrent_frames == 5
    
    def test_processing_config_from_env(self):
        """Test creating ProcessingConfig from environment variables."""
        with patch.dict(os.environ, {
            "PROCESSING_DAYS_BACK": "7",
            "PROCESSING_MAX_FRAMES_PER_VIDEO": "15",
            "PROCESSING_SCENE_THRESHOLD": "30.0",
            "PROCESSING_MAX_CONCURRENT_VIDEOS": "5",
            "PROCESSING_MAX_CONCURRENT_FRAMES": "10"
        }):
            config = ProcessingConfig.from_env()
            
            assert config.days_back == 7
            assert config.max_frames_per_video == 15
            assert config.scene_threshold == 30.0
            assert config.max_concurrent_videos == 5
            assert config.max_concurrent_frames == 10


class TestDatabaseConfig:
    """Test DatabaseConfig class."""
    
    def test_database_config_defaults(self):
        """Test DatabaseConfig default values."""
        config = DatabaseConfig()
        
        assert config.url == "sqlite:///video_tasks.db"
        assert config.echo is False
    
    def test_database_config_from_env(self):
        """Test creating DatabaseConfig from environment variables."""
        with patch.dict(os.environ, {
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "DATABASE_ECHO": "true"
        }):
            config = DatabaseConfig.from_env()
            
            assert config.url == "postgresql://user:pass@localhost/db"
            assert config.echo is True


class TestSchedulerConfig:
    """Test SchedulerConfig class."""
    
    def test_scheduler_config_defaults(self):
        """Test SchedulerConfig default values."""
        config = SchedulerConfig()
        
        assert config.enabled is True
        assert config.hour == 2
        assert config.minute == 0
    
    def test_scheduler_config_from_env(self):
        """Test creating SchedulerConfig from environment variables."""
        with patch.dict(os.environ, {
            "SCHEDULER_ENABLED": "false",
            "SCHEDULER_HOUR": "10",
            "SCHEDULER_MINUTE": "30"
        }):
            config = SchedulerConfig.from_env()
            
            assert config.enabled is False
            assert config.hour == 10
            assert config.minute == 30


class TestLoggingConfig:
    """Test LoggingConfig class."""
    
    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.file == Path("video_tagging.log")
    
    def test_logging_config_from_env(self):
        """Test creating LoggingConfig from environment variables."""
        with patch.dict(os.environ, {
            "LOGGING_LEVEL": "DEBUG",
            "LOGGING_FORMAT": "text",
            "LOGGING_FILE": "/var/log/app.log"
        }):
            config = LoggingConfig.from_env()
            
            assert config.level == "DEBUG"
            assert config.format == "text"
            assert config.file == Path("/var/log/app.log")


class TestAppConfig:
    """Test AppConfig class."""
    
    def test_app_config_creation(self, mock_config):
        """Test creating AppConfig instance."""
        assert mock_config.emby.server_url == "http://test.emby.local:8096"
        assert mock_config.ai.provider == AIProvider.LMSTUDIO
        assert mock_config.processing.days_back == 7
        assert mock_config.database.url == f"sqlite:///{mock_config.logging.file.parent}/test.db"
    
    def test_parse_path_mappings(self):
        """Test parsing path mappings."""
        mappings_str = "/remote/path:/local/path,/another/remote:/another/local"
        mappings = AppConfig._parse_path_mappings(mappings_str)
        
        assert mappings == {
            "/remote/path": "/local/path",
            "/another/remote": "/another/local"
        }
    
    def test_parse_path_mappings_empty(self):
        """Test parsing empty path mappings."""
        mappings = AppConfig._parse_path_mappings("")
        assert mappings == {}
    
    def test_get_ai_config_lmstudio(self, mock_config):
        """Test getting LMStudio config."""
        mock_config.ai.provider = AIProvider.LMSTUDIO
        ai_config = mock_config.get_ai_config()
        
        assert isinstance(ai_config, LMStudioConfig)
        assert ai_config.model_name == "test-model"
    
    def test_get_ai_config_ollama(self, mock_config):
        """Test getting Ollama config."""
        mock_config.ai.provider = AIProvider.OLLAMA
        ai_config = mock_config.get_ai_config()
        
        assert isinstance(ai_config, OllamaConfig)
        assert ai_config.model_name == "llava"
    
    def test_validate_config_missing_api_key(self, mock_config):
        """Test config validation with missing API key."""
        mock_config.emby.api_key = ""
        errors = mock_config.validate_config()
        
        assert "EMBY_API_KEY is required" in errors
    
    def test_validate_config_missing_user_id(self, mock_config):
        """Test config validation with missing user ID."""
        mock_config.emby.user_id = ""
        errors = mock_config.validate_config()
        
        assert "EMBY_USER_ID is required" in errors
    
    def test_validate_config_invalid_path_mapping(self, mock_config, temp_dir):
        """Test config validation with invalid path mapping."""
        mock_config.path_mappings = {
            "/remote/path": str(temp_dir),
            "/invalid/path": "/non/existent/path"
        }
        errors = mock_config.validate_config()
        
        assert any("Local path does not exist" in error for error in errors)
    
    @patch.dict(os.environ, {
        "EMBY_SERVER_URL": "http://env.server:8096",
        "EMBY_API_KEY": "env-key",
        "EMBY_USER_ID": "env-user",
        "AI_PROVIDER": "lmstudio",
        "PATH_MAPPINGS": "/remote:/local",
        "PROCESSING_DAYS_BACK": "10"
    })
    def test_load_config_from_env(self):
        """Test loading complete config from environment."""
        with patch("emby_video_tagger.config.settings.load_dotenv"):
            config = AppConfig.load_config()
            
            assert config.emby.server_url == "http://env.server:8096"
            assert config.emby.api_key == "env-key"
            assert config.emby.user_id == "env-user"
            assert config.ai.provider == AIProvider.LMSTUDIO
            assert config.processing.days_back == 10
            assert config.path_mappings == {"/remote": "/local"}