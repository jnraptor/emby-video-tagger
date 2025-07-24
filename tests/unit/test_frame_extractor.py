"""Unit tests for frame extractor service."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import cv2

from emby_video_tagger.services.frame_extractor import FrameExtractor
from emby_video_tagger.core.models import Frame
from emby_video_tagger.core.exceptions import FrameExtractionError


@pytest.mark.asyncio
class TestFrameExtractor:
    """Test FrameExtractor class."""
    
    @pytest.fixture
    def frame_extractor(self):
        """Create FrameExtractor instance."""
        return FrameExtractor(scene_threshold=30.0)
    
    @pytest.fixture
    def mock_video_path(self, tmp_path):
        """Create a mock video file path."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake video data")
        return str(video_path)
    
    def test_init(self, frame_extractor):
        """Test FrameExtractor initialization."""
        assert frame_extractor.scene_threshold == 30.0
        assert frame_extractor._temp_dirs == []
        assert frame_extractor.logger is None
    
    def test_init_with_logger(self):
        """Test FrameExtractor initialization with logger."""
        mock_logger = MagicMock()
        extractor = FrameExtractor(scene_threshold=25.0, logger=mock_logger)
        
        assert extractor.scene_threshold == 25.0
        assert extractor.logger == mock_logger
    
    async def test_extract_frames_file_not_found(self, frame_extractor):
        """Test extract_frames with non-existent file."""
        with pytest.raises(FrameExtractionError) as exc_info:
            await frame_extractor.extract_frames(
                video_path="/non/existent/video.mp4",
                max_frames=5
            )
        
        assert "Video file not found" in str(exc_info.value)
        assert exc_info.value.video_path == "/non/existent/video.mp4"
    
    @patch("emby_video_tagger.services.frame_extractor.detect")
    @patch("cv2.VideoCapture")
    async def test_extract_frames_scene_detection(self, mock_cv2_capture, mock_detect, frame_extractor, mock_video_path):
        """Test frame extraction using scene detection."""
        # Mock scene detection
        mock_scene = MagicMock()
        mock_scene[0].get_frames.return_value = 100
        mock_scene[1].get_frames.return_value = 200
        mock_detect.return_value = [mock_scene]
        
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0  # FPS
        mock_cap.read.return_value = (True, MagicMock())  # Successful frame read
        mock_cv2_capture.return_value = mock_cap
        
        # Mock cv2.imwrite
        with patch("cv2.imwrite", return_value=True):
            frames = await frame_extractor.extract_frames(
                video_path=mock_video_path,
                max_frames=5,
                strategy="scene_detection"
            )
        
        assert len(frames) == 1
        assert isinstance(frames[0], Frame)
        assert frames[0].frame_number == 150  # Middle of scene (100+200)/2
        assert frames[0].timestamp == 6.0  # 150 frames / 25 fps
        
        # Verify scene detection was called
        mock_detect.assert_called_once()
        
        # Verify video was released
        mock_cap.release.assert_called_once()
    
    @patch("emby_video_tagger.services.frame_extractor.detect")
    @patch("cv2.VideoCapture")
    async def test_extract_frames_no_scenes_fallback(self, mock_cv2_capture, mock_detect, frame_extractor, mock_video_path):
        """Test fallback to uniform extraction when no scenes detected."""
        # Mock no scenes detected
        mock_detect.return_value = []
        
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 1000,
            cv2.CAP_PROP_FPS: 25.0
        }.get(prop, 0)
        mock_cap.read.return_value = (True, MagicMock())
        mock_cv2_capture.return_value = mock_cap
        
        # Mock cv2.imwrite
        with patch("cv2.imwrite", return_value=True):
            frames = await frame_extractor.extract_frames(
                video_path=mock_video_path,
                max_frames=5,
                strategy="scene_detection"
            )
        
        # Should have used uniform extraction
        assert len(frames) <= 5
        assert all(isinstance(frame, Frame) for frame in frames)
    
    @patch("cv2.VideoCapture")
    async def test_extract_uniform_frames(self, mock_cv2_capture, frame_extractor, mock_video_path):
        """Test uniform frame extraction."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 1000,
            cv2.CAP_PROP_FPS: 25.0
        }.get(prop, 0)
        mock_cap.read.return_value = (True, MagicMock())
        mock_cv2_capture.return_value = mock_cap
        
        # Mock cv2.imwrite
        with patch("cv2.imwrite", return_value=True):
            frames = await frame_extractor.extract_frames(
                video_path=mock_video_path,
                max_frames=5,
                strategy="uniform"
            )
        
        assert len(frames) == 5
        assert all(isinstance(frame, Frame) for frame in frames)
        
        # Check frames are uniformly distributed
        frame_numbers = [f.frame_number for f in frames]
        assert frame_numbers[0] == 0
        assert all(frame_numbers[i] < frame_numbers[i+1] for i in range(len(frame_numbers)-1))
    
    async def test_extract_frames_invalid_strategy(self, frame_extractor, mock_video_path):
        """Test extract_frames with invalid strategy."""
        with pytest.raises(FrameExtractionError) as exc_info:
            await frame_extractor.extract_frames(
                video_path=mock_video_path,
                max_frames=5,
                strategy="invalid"
            )
        
        assert "Unknown extraction strategy" in str(exc_info.value)
    
    @patch("cv2.VideoCapture")
    async def test_extract_frames_video_open_error(self, mock_cv2_capture, frame_extractor, mock_video_path):
        """Test extract_frames when video cannot be opened."""
        # Mock video capture that fails to open
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2_capture.return_value = mock_cap
        
        with pytest.raises(FrameExtractionError) as exc_info:
            await frame_extractor.extract_frames(
                video_path=mock_video_path,
                max_frames=5,
                strategy="uniform"
            )
        
        assert "Cannot open video" in str(exc_info.value)
    
    def test_sample_scenes(self, frame_extractor):
        """Test scene sampling logic."""
        # Test with fewer scenes than max_frames
        scenes = list(range(3))
        sampled = frame_extractor._sample_scenes(scenes, max_frames=5)
        assert sampled == scenes
        
        # Test with more scenes than max_frames
        scenes = list(range(20))
        sampled = frame_extractor._sample_scenes(scenes, max_frames=5)
        assert len(sampled) == 5
        assert sampled[0] == 0
        assert sampled[-1] in scenes  # Last sample should be from the list
    
    @patch("cv2.VideoCapture")
    def test_validate_video_success(self, mock_cv2_capture, frame_extractor, mock_video_path):
        """Test successful video validation."""
        # Mock successful video open and read
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())
        mock_cv2_capture.return_value = mock_cap
        
        assert frame_extractor.validate_video(mock_video_path) is True
        mock_cap.release.assert_called_once()
    
    def test_validate_video_file_not_found(self, frame_extractor):
        """Test video validation with non-existent file."""
        assert frame_extractor.validate_video("/non/existent/video.mp4") is False
    
    @patch("cv2.VideoCapture")
    def test_validate_video_cannot_open(self, mock_cv2_capture, frame_extractor, mock_video_path):
        """Test video validation when file cannot be opened."""
        # Mock failed video open
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2_capture.return_value = mock_cap
        
        assert frame_extractor.validate_video(mock_video_path) is False
        mock_cap.release.assert_called_once()
    
    @patch("cv2.VideoCapture")
    def test_get_video_info_success(self, mock_cv2_capture, frame_extractor, mock_video_path):
        """Test getting video information."""
        # Mock video capture with properties
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 1500,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FOURCC: 1196444237  # 'MP4V'
        }.get(prop, 0)
        mock_cv2_capture.return_value = mock_cap
        
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=104857600)  # 100 MB
            
            info = frame_extractor.get_video_info(mock_video_path)
        
        assert info["path"] == mock_video_path
        assert info["frame_count"] == 1500
        assert info["fps"] == 30.0
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["duration_seconds"] == 50.0  # 1500 / 30
        assert info["file_size_mb"] == 100.0
        
        mock_cap.release.assert_called_once()
    
    @patch("cv2.VideoCapture")
    def test_get_video_info_cannot_open(self, mock_cv2_capture, frame_extractor, mock_video_path):
        """Test get_video_info when video cannot be opened."""
        # Mock failed video open
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2_capture.return_value = mock_cap
        
        with pytest.raises(FrameExtractionError) as exc_info:
            frame_extractor.get_video_info(mock_video_path)
        
        assert "Cannot open video" in str(exc_info.value)
    
    def test_fourcc_to_string(self, frame_extractor):
        """Test fourcc code conversion."""
        # Test common fourcc codes
        assert frame_extractor._fourcc_to_string(1196444237) == "MP4V"
        assert frame_extractor._fourcc_to_string(1145656920) == "XVID"
        assert frame_extractor._fourcc_to_string(875967080) == "H264"
    
    def test_cleanup_temp_dirs(self, frame_extractor, tmp_path):
        """Test cleanup of temporary directories."""
        # Create temp directories
        temp_dir1 = tmp_path / "temp1"
        temp_dir2 = tmp_path / "temp2"
        temp_dir1.mkdir()
        temp_dir2.mkdir()
        
        # Add some files
        (temp_dir1 / "frame1.jpg").write_bytes(b"data")
        (temp_dir2 / "frame2.jpg").write_bytes(b"data")
        
        # Add to extractor
        frame_extractor._temp_dirs = [temp_dir1, temp_dir2]
        
        # Cleanup
        frame_extractor.cleanup_temp_dirs()
        
        # Verify directories are removed
        assert not temp_dir1.exists()
        assert not temp_dir2.exists()
        assert frame_extractor._temp_dirs == []
    
    def test_cleanup_temp_dirs_with_error(self, frame_extractor, tmp_path):
        """Test cleanup handles errors gracefully."""
        # Create a temp directory that doesn't exist
        non_existent = tmp_path / "non_existent"
        frame_extractor._temp_dirs = [non_existent]
        
        # Should not raise exception
        frame_extractor.cleanup_temp_dirs()
        
        assert frame_extractor._temp_dirs == []
    
    def test_del_calls_cleanup(self, frame_extractor):
        """Test that __del__ calls cleanup."""
        with patch.object(frame_extractor, 'cleanup_temp_dirs') as mock_cleanup:
            frame_extractor.__del__()
            mock_cleanup.assert_called_once()