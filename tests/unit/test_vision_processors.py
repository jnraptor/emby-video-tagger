"""Unit tests for vision processing components."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from emby_video_tagger.services.vision.base import BaseVisionProcessor
from emby_video_tagger.services.vision.lmstudio import LMStudioVisionProcessor
from emby_video_tagger.services.vision.ollama import OllamaVisionProcessor
from emby_video_tagger.services.vision.factory import VisionProcessorFactory
from emby_video_tagger.config.settings import AIProvider, AIProviderConfig, LMStudioConfig, OllamaConfig
from emby_video_tagger.core.models import Frame, FrameAnalysis
from emby_video_tagger.core.exceptions import VisionProcessingError, ConfigurationError


class TestBaseVisionProcessor:
    """Test BaseVisionProcessor abstract class."""
    
    def test_cannot_instantiate_directly(self):
        """Test that BaseVisionProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVisionProcessor("test-model")
    
    def test_extract_json_from_response(self):
        """Test JSON extraction from various response formats."""
        # Create a concrete implementation for testing
        class TestProcessor(BaseVisionProcessor):
            async def analyze_frame(self, frame: Frame) -> FrameAnalysis:
                return FrameAnalysis(frame=frame)
        
        processor = TestProcessor("test-model")
        
        # Test JSON in markdown code block
        response = """Here's the analysis:
        ```json
        {
            "subjects": ["person", "car"],
            "activities": ["driving"],
            "settings": ["city", "street"]
        }
        ```
        """
        result = processor._extract_json_from_response(response)
        assert result["subjects"] == ["person", "car"]
        assert result["activities"] == ["driving"]
        assert result["settings"] == ["city", "street"]
        
        # Test plain JSON
        response = '{"subjects": ["dog"], "activities": ["running"]}'
        result = processor._extract_json_from_response(response)
        assert result["subjects"] == ["dog"]
        assert result["activities"] == ["running"]
        
        # Test no JSON found
        response = "This is just plain text with no JSON"
        result = processor._extract_json_from_response(response)
        assert result == {}
    
    def test_encode_image(self):
        """Test image encoding to base64."""
        class TestProcessor(BaseVisionProcessor):
            async def analyze_frame(self, frame: Frame) -> FrameAnalysis:
                return FrameAnalysis(frame=frame)
        
        processor = TestProcessor("test-model")
        
        # Test with mock file
        with patch("builtins.open", MagicMock()) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"fake image data"
            
            encoded = processor.encode_image("/path/to/image.jpg")
            
            assert encoded == "ZmFrZSBpbWFnZSBkYXRh"  # base64 of "fake image data"
    
    def test_encode_image_error(self):
        """Test image encoding error handling."""
        class TestProcessor(BaseVisionProcessor):
            async def analyze_frame(self, frame: Frame) -> FrameAnalysis:
                return FrameAnalysis(frame=frame)
        
        processor = TestProcessor("test-model")
        
        with pytest.raises(VisionProcessingError) as exc_info:
            processor.encode_image("/non/existent/image.jpg")
        
        assert "Failed to encode image" in str(exc_info.value)
    
    def test_merge_tags(self):
        """Test merging tags from multiple analyses."""
        class TestProcessor(BaseVisionProcessor):
            async def analyze_frame(self, frame: Frame) -> FrameAnalysis:
                return FrameAnalysis(frame=frame)
        
        processor = TestProcessor("test-model")
        
        # Create test analyses
        frame1 = Frame(path="frame1.jpg", frame_number=1, timestamp=1.0, video_id="test")
        frame2 = Frame(path="frame2.jpg", frame_number=2, timestamp=2.0, video_id="test")
        
        analysis1 = FrameAnalysis(
            frame=frame1,
            subjects=["person", "car"],
            activities=["driving"],
            settings=["city"]
        )
        
        analysis2 = FrameAnalysis(
            frame=frame2,
            subjects=["person", "dog"],  # Duplicate "person"
            activities=["walking"],
            settings=["park"]
        )
        
        merged = processor.merge_tags([analysis1, analysis2])
        
        # Should have unique, sorted tags
        assert "car" in merged
        assert "city" in merged
        assert "dog" in merged
        assert "driving" in merged
        assert "park" in merged
        assert "person" in merged
        assert "walking" in merged
        assert merged.count("person") == 1  # No duplicates
        assert merged == sorted(merged)  # Should be sorted
    
    def test_parse_analysis_response(self):
        """Test parsing API response into FrameAnalysis."""
        class TestProcessor(BaseVisionProcessor):
            async def analyze_frame(self, frame: Frame) -> FrameAnalysis:
                return FrameAnalysis(frame=frame)
        
        processor = TestProcessor("test-model")
        
        frame = Frame(path="test.jpg", frame_number=1, timestamp=1.0, video_id="test")
        
        # Test with all fields
        response = {
            "subjects": ["person"],
            "activities": ["walking"],
            "settings": ["park"],
            "styles": ["documentary"],
            "moods": ["peaceful"]
        }
        
        analysis = processor._parse_analysis_response(response, frame)
        
        assert analysis.frame == frame
        assert analysis.subjects == ["person"]
        assert analysis.activities == ["walking"]
        assert analysis.settings == ["park"]
        assert analysis.styles == ["documentary"]
        assert analysis.moods == ["peaceful"]
        
        # Test with alternative keys (setting vs settings)
        response = {
            "subjects": ["car"],
            "setting": ["street"],  # Alternative key
            "style": ["action"],    # Alternative key
            "mood": ["tense"]       # Alternative key
        }
        
        analysis = processor._parse_analysis_response(response, frame)
        
        assert analysis.settings == ["street"]
        assert analysis.styles == ["action"]
        assert analysis.moods == ["tense"]


@pytest.mark.asyncio
class TestLMStudioVisionProcessor:
    """Test LMStudioVisionProcessor class."""
    
    @pytest.fixture
    def lmstudio_config(self):
        """Create test LMStudio configuration."""
        return LMStudioConfig(
            model_name="test-vision-model"
        )
    
    @pytest.fixture
    def lmstudio_processor(self, lmstudio_config):
        """Create LMStudioVisionProcessor instance."""
        return LMStudioVisionProcessor(lmstudio_config)
    
    def test_init(self, lmstudio_processor, lmstudio_config):
        """Test LMStudioVisionProcessor initialization."""
        assert lmstudio_processor.config == lmstudio_config
        assert lmstudio_processor.model_name == "test-vision-model"
        assert lmstudio_processor.logger is None
    
    @patch("lmstudio.Client")
    async def test_analyze_frame_success(self, mock_client_class, lmstudio_processor, tmp_path):
        """Test successful frame analysis."""
        # Create test frame
        frame_path = tmp_path / "test_frame.jpg"
        frame_path.write_bytes(b"fake image data")
        frame = Frame(
            path=str(frame_path),
            frame_number=1,
            timestamp=1.0,
            video_id="test-video"
        )
        
        # Mock LMStudio client
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_client.llm.model.return_value = mock_model
        mock_client.files.prepare_image.return_value = "image_handle"
        
        # Mock response
        mock_response = """
        ```json
        {
            "subjects": ["person", "car"],
            "activities": ["driving"],
            "settings": ["city", "street"],
            "styles": ["action"],
            "moods": ["tense"]
        }
        ```
        """
        mock_model.respond.return_value = mock_response
        
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        # Analyze frame
        analysis = await lmstudio_processor.analyze_frame(frame)
        
        assert isinstance(analysis, FrameAnalysis)
        assert analysis.frame == frame
        assert "person" in analysis.subjects
        assert "car" in analysis.subjects
        assert "driving" in analysis.activities
        assert "city" in analysis.settings
    
    @patch("lmstudio.Client")
    async def test_analyze_frame_error(self, mock_client_class, lmstudio_processor, tmp_path):
        """Test frame analysis with error."""
        # Create test frame
        frame_path = tmp_path / "test_frame.jpg"
        frame_path.write_bytes(b"fake image data")
        frame = Frame(
            path=str(frame_path),
            frame_number=1,
            timestamp=1.0,
            video_id="test-video"
        )
        
        # Mock client to raise error
        mock_client_class.side_effect = Exception("LMStudio connection failed")
        
        with pytest.raises(VisionProcessingError) as exc_info:
            await lmstudio_processor.analyze_frame(frame)
        
        assert "Failed to analyze frame with LMStudio" in str(exc_info.value)
        assert exc_info.value.provider == "lmstudio"
        assert exc_info.value.model == "test-vision-model"
    
    @patch("lmstudio.Client")
    async def test_analyze_frames_batch(self, mock_client_class, lmstudio_processor, tmp_path):
        """Test batch frame analysis."""
        # Create test frames
        frames = []
        for i in range(3):
            frame_path = tmp_path / f"frame_{i}.jpg"
            frame_path.write_bytes(b"fake image data")
            frames.append(Frame(
                path=str(frame_path),
                frame_number=i,
                timestamp=i * 1.0,
                video_id="test-video"
            ))
        
        # Mock LMStudio client
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_client.llm.model.return_value = mock_model
        mock_client.files.prepare_image.return_value = "image_handle"
        
        # Mock response
        mock_response = '{"subjects": ["test"], "activities": ["testing"]}'
        mock_model.respond.return_value = mock_response
        
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        # Analyze frames
        analyses = await lmstudio_processor.analyze_frames_batch(frames)
        
        assert len(analyses) == 3
        assert all(isinstance(a, FrameAnalysis) for a in analyses)
    
    def test_get_model_info(self, lmstudio_processor):
        """Test getting model information."""
        info = lmstudio_processor.get_model_info()
        
        assert info["provider"] == "LMStudio"
        assert info["model_name"] == "test-vision-model"
        assert "config" in info
        assert info["config"]["model_name"] == "test-vision-model"


@pytest.mark.asyncio
class TestOllamaVisionProcessor:
    """Test OllamaVisionProcessor class."""
    
    @pytest.fixture
    def ollama_config(self):
        """Create test Ollama configuration."""
        return OllamaConfig(
            model_name="llava",
            base_url="http://localhost:11434"
        )
    
    @pytest.fixture
    def ollama_processor(self, ollama_config):
        """Create OllamaVisionProcessor instance."""
        return OllamaVisionProcessor(ollama_config)
    
    def test_init(self, ollama_processor, ollama_config):
        """Test OllamaVisionProcessor initialization."""
        assert ollama_processor.config == ollama_config
        assert ollama_processor.model_name == "llava"
        assert ollama_processor.base_url == "http://localhost:11434"


class TestVisionProcessorFactory:
    """Test VisionProcessorFactory class."""
    
    def test_create_processor_lmstudio(self):
        """Test creating LMStudio processor."""
        ai_config = AIProviderConfig(
            provider=AIProvider.LMSTUDIO,
            lmstudio=LMStudioConfig(model_name="test-model")
        )
        
        processor = VisionProcessorFactory.create_processor(ai_config)
        
        assert isinstance(processor, LMStudioVisionProcessor)
        assert processor.model_name == "test-model"
    
    def test_create_processor_ollama(self):
        """Test creating Ollama processor."""
        ai_config = AIProviderConfig(
            provider=AIProvider.OLLAMA,
            ollama=OllamaConfig(
                model_name="llava",
                base_url="http://localhost:11434"
            )
        )
        
        processor = VisionProcessorFactory.create_processor(ai_config)
        
        assert isinstance(processor, OllamaVisionProcessor)
        assert processor.model_name == "llava"
    
    def test_create_processor_missing_config(self):
        """Test creating processor with missing config."""
        ai_config = AIProviderConfig(
            provider=AIProvider.LMSTUDIO,
            lmstudio=None  # Missing config
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            VisionProcessorFactory.create_processor(ai_config)
        
        assert "LMStudio configuration is missing" in str(exc_info.value)
    
    def test_create_from_config_lmstudio(self):
        """Test creating processor from specific config."""
        config = LMStudioConfig(model_name="test-model")
        
        processor = VisionProcessorFactory.create_from_config(
            provider="lmstudio",
            config=config
        )
        
        assert isinstance(processor, LMStudioVisionProcessor)
        assert processor.model_name == "test-model"
    
    def test_create_from_config_ollama(self):
        """Test creating Ollama processor from specific config."""
        config = OllamaConfig(
            model_name="llava",
            base_url="http://localhost:11434"
        )
        
        processor = VisionProcessorFactory.create_from_config(
            provider="ollama",
            config=config
        )
        
        assert isinstance(processor, OllamaVisionProcessor)
        assert processor.model_name == "llava"
    
    def test_create_from_config_invalid_provider(self):
        """Test creating processor with invalid provider."""
        config = LMStudioConfig(model_name="test-model")
        
        with pytest.raises(ConfigurationError) as exc_info:
            VisionProcessorFactory.create_from_config(
                provider="invalid",
                config=config
            )
        
        assert "Invalid provider" in str(exc_info.value)
    
    def test_create_from_config_mismatched_config(self):
        """Test creating processor with mismatched config type."""
        # Ollama config with lmstudio provider
        config = OllamaConfig(
            model_name="llava",
            base_url="http://localhost:11434"
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            VisionProcessorFactory.create_from_config(
                provider="lmstudio",
                config=config
            )
        
        assert "Invalid provider" in str(exc_info.value) or "config type mismatch" in str(exc_info.value)