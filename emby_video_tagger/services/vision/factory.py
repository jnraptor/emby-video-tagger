"""Factory for creating vision processor instances."""

from typing import Union

from emby_video_tagger.config.settings import AIProvider, AIProviderConfig, LMStudioConfig, OllamaConfig
from emby_video_tagger.services.vision.base import BaseVisionProcessor
from emby_video_tagger.services.vision.lmstudio import LMStudioVisionProcessor
from emby_video_tagger.services.vision.ollama import OllamaVisionProcessor
from emby_video_tagger.core.exceptions import ConfigurationError


class VisionProcessorFactory:
    """Factory class for creating vision processor instances."""
    
    @staticmethod
    def create_processor(
        ai_config: AIProviderConfig,
        logger=None
    ) -> BaseVisionProcessor:
        """
        Create a vision processor based on the provider type.
        
        Args:
            ai_config: AI provider configuration
            logger: Optional logger instance
            
        Returns:
            Vision processor instance
            
        Raises:
            ConfigurationError: If provider is not supported or config is invalid
        """
        provider = ai_config.provider
        
        if provider == AIProvider.LMSTUDIO:
            if not ai_config.lmstudio:
                raise ConfigurationError("LMStudio configuration is missing")
            return LMStudioVisionProcessor(ai_config.lmstudio, logger)
            
        elif provider == AIProvider.OLLAMA:
            if not ai_config.ollama:
                raise ConfigurationError("Ollama configuration is missing")
            return OllamaVisionProcessor(ai_config.ollama, logger)
            
        else:
            raise ConfigurationError(
                f"Unsupported AI provider: {provider}. "
                f"Supported providers: {', '.join([p.value for p in AIProvider])}"
            )
    
    @staticmethod
    def create_from_config(
        provider: str,
        config: Union[LMStudioConfig, OllamaConfig],
        logger=None
    ) -> BaseVisionProcessor:
        """
        Create a vision processor from a specific provider config.
        
        Args:
            provider: Provider name (lmstudio or ollama)
            config: Provider-specific configuration
            logger: Optional logger instance
            
        Returns:
            Vision processor instance
            
        Raises:
            ConfigurationError: If provider is not supported
        """
        provider_lower = provider.lower()
        
        if provider_lower == "lmstudio" and isinstance(config, LMStudioConfig):
            return LMStudioVisionProcessor(config, logger)
            
        elif provider_lower == "ollama" and isinstance(config, OllamaConfig):
            return OllamaVisionProcessor(config, logger)
            
        else:
            raise ConfigurationError(
                f"Invalid provider '{provider}' or config type mismatch"
            )