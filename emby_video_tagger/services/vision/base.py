"""Base vision processor implementation."""

import base64
import json
import re
from abc import ABC
from typing import List, Dict, Any
from pathlib import Path

from emby_video_tagger.core.interfaces import IVisionProcessor
from emby_video_tagger.core.models import Frame, FrameAnalysis
from emby_video_tagger.core.exceptions import VisionProcessingError


class BaseVisionProcessor(IVisionProcessor, ABC):
    """Abstract base class for AI vision processors."""
    
    def __init__(self, model_name: str, logger=None):
        """Initialize base vision processor."""
        self.model_name = model_name
        self.logger = logger
        self.tag_prompt = self._create_tagging_prompt()
    
    def _create_tagging_prompt(self) -> str:
        """Create the prompt for tag generation."""
        return """
        Analyze this video frame and generate descriptive tags for media organization.
        
        Focus on:
        - Main subjects (people, objects, animals)
        - Activities and actions
        - Setting and environment (indoor/outdoor, time of day)
        - Visual style and mood
        - Technical aspects (lighting, composition)
        
        Return results as JSON:
        {
            "subjects": ["person", "car", "building"],
            "activities": ["walking", "driving", "talking"], 
            "settings": ["urban", "outdoor", "daytime"],
            "styles": ["documentary", "handheld", "wide-shot"],
            "moods": ["energetic", "professional", "casual"]
        }
        """
    
    def encode_image(self, image_path: str) -> str:
        """Convert image to base64 for API submission."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            raise VisionProcessingError(
                f"Failed to encode image {image_path}: {str(e)}",
                provider=self.__class__.__name__
            )
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, List[str]]:
        """Extract JSON content from markdown code blocks or plain text."""
        # Try to find JSON within markdown code fences
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if match:
            json_str = match.group(1).strip()
        else:
            # Try to find JSON object without code fences
            json_object_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
            match = re.search(json_object_pattern, response_text, re.DOTALL)
            
            if match:
                json_str = match.group(0).strip()
            else:
                # If no JSON found, return empty dict
                if self.logger:
                    self.logger.warning(f"No JSON found in response: {response_text[:200]}...")
                return {}
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.error(f"Failed to parse JSON: {e}\nJSON string: {json_str}")
            return {}
    
    def merge_tags(self, analyses: List[FrameAnalysis]) -> List[str]:
        """Merge and deduplicate tags from multiple frame analyses."""
        all_tags = set()
        
        for analysis in analyses:
            all_tags.update(analysis.all_tags)
        
        # Convert to list and sort for consistency
        merged_tags = sorted(list(all_tags))
        
        if self.logger:
            self.logger.info(f"Merged {len(analyses)} analyses into {len(merged_tags)} unique tags")
        
        return merged_tags
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the AI model being used."""
        return {
            "provider": self.__class__.__name__,
            "model_name": self.model_name,
            "prompt_template": self.tag_prompt
        }
    
    def _parse_analysis_response(self, response: Dict[str, List[str]], frame: Frame) -> FrameAnalysis:
        """Parse API response into FrameAnalysis object."""
        return FrameAnalysis(
            frame=frame,
            subjects=response.get("subjects", []),
            activities=response.get("activities", []),
            settings=response.get("settings", response.get("setting", [])),  # Handle both keys
            styles=response.get("styles", response.get("style", [])),  # Handle both keys
            moods=response.get("moods", response.get("mood", []))  # Handle both keys
        )
    
    async def analyze_frames_batch(self, frames: List[Frame]) -> List[FrameAnalysis]:
        """Analyze multiple frames in batch."""
        # Default implementation: analyze frames one by one
        analyses = []
        
        for frame in frames:
            try:
                analysis = await self.analyze_frame(frame)
                analyses.append(analysis)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to analyze frame {frame.path}: {e}")
                # Create empty analysis for failed frames
                analyses.append(FrameAnalysis(frame=frame))
        
        return analyses