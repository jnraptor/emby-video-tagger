"""LMStudio vision processor implementation."""

import asyncio
from typing import Dict, List, Any
import lmstudio as lms

from emby_video_tagger.services.vision.base import BaseVisionProcessor
from emby_video_tagger.core.models import Frame, FrameAnalysis
from emby_video_tagger.core.exceptions import VisionProcessingError
from emby_video_tagger.config.settings import LMStudioConfig


class LMStudioVisionProcessor(BaseVisionProcessor):
    """Handles LMStudio Vision API interactions for video frame analysis."""
    
    def __init__(self, config: LMStudioConfig, logger=None):
        """Initialize LMStudio vision processor."""
        super().__init__(config.model_name, logger)
        self.config = config
    
    async def analyze_frame(self, frame: Frame) -> FrameAnalysis:
        """Analyze single frame and return analysis."""
        try:
            # Run LMStudio analysis in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._analyze_frame_sync,
                frame.path
            )
            
            # Parse response
            parsed_response = self._extract_json_from_response(response)
            return self._parse_analysis_response(parsed_response, frame)
            
        except Exception as e:
            raise VisionProcessingError(
                f"Failed to analyze frame with LMStudio: {str(e)}",
                provider="lmstudio",
                model=self.model_name
            )
    
    def _analyze_frame_sync(self, frame_path: str) -> str:
        """Synchronous frame analysis for LMStudio."""
        try:
            with lms.Client() as client:
                model = client.llm.model(self.model_name)
                image_handle = client.files.prepare_image(frame_path)
                
                chat = lms.Chat()
                chat.add_user_message(self.tag_prompt, images=[image_handle])
                prediction = model.respond(chat)
                
                return str(prediction)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"LMStudio analysis failed for {frame_path}: {e}")
            raise
    
    async def analyze_frames_batch(self, frames: List[Frame]) -> List[FrameAnalysis]:
        """Analyze multiple frames in batch."""
        # LMStudio doesn't support true batch processing, so we'll process concurrently
        tasks = []
        
        # Limit concurrent requests
        semaphore = asyncio.Semaphore(3)
        
        async def analyze_with_semaphore(frame: Frame) -> FrameAnalysis:
            async with semaphore:
                try:
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
        """Get information about the LMStudio model being used."""
        info = super().get_model_info()
        info.update({
            "provider": "LMStudio",
            "config": {
                "model_name": self.config.model_name
            }
        })
        return info