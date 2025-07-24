"""Ollama vision processor implementation."""

import asyncio
from typing import Dict, List, Any
from ollama import AsyncClient

from emby_video_tagger.services.vision.base import BaseVisionProcessor
from emby_video_tagger.core.models import Frame, FrameAnalysis
from emby_video_tagger.core.exceptions import VisionProcessingError
from emby_video_tagger.config.settings import OllamaConfig


class OllamaVisionProcessor(BaseVisionProcessor):
    """Handles Ollama Vision API interactions for video frame analysis."""
    
    def __init__(self, config: OllamaConfig, logger=None):
        """Initialize Ollama vision processor."""
        super().__init__(config.model_name, logger)
        self.config = config
        self.base_url = config.base_url
        self.client = AsyncClient(host=str(config.base_url))
    
    async def analyze_frame(self, frame: Frame) -> FrameAnalysis:
        """Analyze single frame and return analysis."""
        try:
            # Encode image as base64
            image_data = self.encode_image(frame.path)
            
            # Create chat request with image
            response = await self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": self.tag_prompt,
                        "images": [image_data],
                    }
                ],
                options={
                    "num_ctx": 4096,
                    "temperature": 0.1
                }
            )
            
            # Extract response content
            analysis_text = response["message"]["content"]
            
            # Parse response
            parsed_response = self._extract_json_from_response(analysis_text)
            return self._parse_analysis_response(parsed_response, frame)
            
        except Exception as e:
            raise VisionProcessingError(
                f"Failed to analyze frame with Ollama: {str(e)}",
                provider="ollama",
                model=self.model_name
            )
    
    async def analyze_frames_batch(self, frames: List[Frame]) -> List[FrameAnalysis]:
        """Analyze multiple frames in batch."""
        # Ollama doesn't support true batch processing, so we'll process concurrently
        tasks = []
        
        # Limit concurrent requests to avoid overwhelming the server
        semaphore = asyncio.Semaphore(3)
        
        async def analyze_with_semaphore(frame: Frame) -> FrameAnalysis:
            async with semaphore:
                try:
                    # Add small delay between requests to avoid rate limiting
                    await asyncio.sleep(0.5)
                    return await self.analyze_frame(frame)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to analyze frame {frame.path}: {e}")
                    # Return empty analysis for failed frames
                    return FrameAnalysis(frame=frame)
        
        for frame in frames:
            task = analyze_with_semaphore(frame)
            tasks.append(task)
        
        analyses = await asyncio.gather(*tasks)
        return analyses
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Ollama model being used."""
        info = super().get_model_info()
        info.update({
            "provider": "Ollama",
            "config": {
                "model_name": self.config.model_name,
                "base_url": str(self.config.base_url)
            }
        })
        return info
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Ollama client doesn't need explicit cleanup
        pass