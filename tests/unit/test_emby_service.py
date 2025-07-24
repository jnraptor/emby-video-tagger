"""Unit tests for Emby service."""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import aiohttp

from emby_video_tagger.config.settings import EmbyConfig
from emby_video_tagger.services.emby import EmbyService
from emby_video_tagger.core.models import Video
from emby_video_tagger.core.exceptions import EmbyAPIError


@pytest.mark.asyncio
class TestEmbyService:
    """Test EmbyService class."""
    
    @pytest.fixture
    def emby_config(self):
        """Create test Emby configuration."""
        return EmbyConfig(
            server_url="http://test.emby.local:8096",
            api_key="test-api-key",
            user_id="test-user-id"
        )
    
    @pytest.fixture
    def emby_service(self, emby_config):
        """Create EmbyService instance."""
        return EmbyService(emby_config)
    
    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        return session
    
    async def test_init(self, emby_service, emby_config):
        """Test EmbyService initialization."""
        assert emby_service.config == emby_config
        assert emby_service.base_url == "http://test.emby.local:8096"
        assert emby_service.headers["X-Emby-Token"] == "test-api-key"
        assert emby_service.headers["Content-Type"] == "application/json"
        assert emby_service._session is None
    
    async def test_get_session_creates_new(self, emby_service):
        """Test _get_session creates new session."""
        session = await emby_service._get_session()
        
        assert session is not None
        assert emby_service._session == session
        assert isinstance(session, aiohttp.ClientSession)
        
        # Cleanup
        await emby_service.close()
    
    async def test_get_session_reuses_existing(self, emby_service):
        """Test _get_session reuses existing session."""
        session1 = await emby_service._get_session()
        session2 = await emby_service._get_session()
        
        assert session1 is session2
        
        # Cleanup
        await emby_service.close()
    
    async def test_close_with_session(self, emby_service):
        """Test closing service with active session."""
        # Create session
        await emby_service._get_session()
        assert emby_service._session is not None
        
        # Close
        await emby_service.close()
        assert emby_service._session is None
    
    async def test_close_without_session(self, emby_service):
        """Test closing service without session."""
        # Should not raise
        await emby_service.close()
        assert emby_service._session is None
    
    async def test_make_request_success(self, emby_service, mock_session, mock_http_response):
        """Test successful API request."""
        mock_http_response.json.return_value = {"result": "success"}
        mock_session.get.return_value.__aenter__.return_value = mock_http_response
        
        with patch.object(emby_service, "_get_session", return_value=mock_session):
            result = await emby_service._make_request("GET", "/test/endpoint")
        
        assert result == {"result": "success"}
        mock_session.get.assert_called_once_with(
            "http://test.emby.local:8096/test/endpoint",
            headers=emby_service.headers,
            params=None
        )
    
    async def test_make_request_with_params(self, emby_service, mock_session, mock_http_response):
        """Test API request with parameters."""
        mock_http_response.json.return_value = {"result": "success"}
        mock_session.get.return_value.__aenter__.return_value = mock_http_response
        
        params = {"limit": 10, "filter": "test"}
        
        with patch.object(emby_service, "_get_session", return_value=mock_session):
            await emby_service._make_request("GET", "/test/endpoint", params=params)
        
        mock_session.get.assert_called_once_with(
            "http://test.emby.local:8096/test/endpoint",
            headers=emby_service.headers,
            params=params
        )
    
    async def test_make_request_post_with_data(self, emby_service, mock_session, mock_http_response):
        """Test POST request with data."""
        mock_http_response.json.return_value = {"result": "success"}
        mock_session.post.return_value.__aenter__.return_value = mock_http_response
        
        data = {"name": "test", "value": 123}
        
        with patch.object(emby_service, "_get_session", return_value=mock_session):
            await emby_service._make_request("POST", "/test/endpoint", data=data)
        
        mock_session.post.assert_called_once_with(
            "http://test.emby.local:8096/test/endpoint",
            headers=emby_service.headers,
            json=data
        )
    
    async def test_make_request_authentication_error(self, emby_service, mock_session):
        """Test API request with authentication error."""
        mock_response = MagicMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Unauthorized")
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(emby_service, "_get_session", return_value=mock_session):
            with pytest.raises(EmbyAPIError) as exc_info:
                await emby_service._make_request("GET", "/test/endpoint")
        
        assert exc_info.value.status_code == 401
    
    async def test_make_request_connection_error(self, emby_service, mock_session):
        """Test API request with connection error."""
        mock_session.get.side_effect = aiohttp.ClientError("Connection failed")
        
        with patch.object(emby_service, "_get_session", return_value=mock_session):
            with pytest.raises(EmbyAPIError) as exc_info:
                await emby_service._make_request("GET", "/test/endpoint")
        
        assert "Connection failed" in str(exc_info.value)
    
    async def test_get_recent_videos_success(self, emby_service):
        """Test getting recent videos."""
        mock_response = {
            "Items": [
                {
                    "Id": "video1",
                    "Name": "Test Movie 1",
                    "Path": "/movies/test1.mp4",
                    "DateCreated": "2024-01-01T00:00:00Z",
                    "Tags": ["action"],
                    "Type": "Movie"
                },
                {
                    "Id": "video2",
                    "Name": "Test Movie 2",
                    "Path": "/movies/test2.mp4",
                    "DateCreated": "2024-01-02T00:00:00Z",
                    "Tags": [],
                    "Type": "Movie"
                }
            ]
        }
        
        with patch.object(emby_service, "_make_request", return_value=mock_response):
            videos = await emby_service.get_recent_videos(days_back=7)
        
        assert len(videos) == 2
        assert videos[0].id == "video1"
        assert videos[0].name == "Test Movie 1"
        assert videos[0].existing_tags == ["action"]
        assert videos[1].id == "video2"
        assert videos[1].existing_tags == []
    
    async def test_get_recent_videos_filters_old(self, emby_service):
        """Test filtering old videos."""
        # Create dates
        recent_date = (datetime.utcnow() - timedelta(days=2)).isoformat() + "Z"
        old_date = (datetime.utcnow() - timedelta(days=10)).isoformat() + "Z"
        
        mock_response = {
            "Items": [
                {
                    "Id": "recent",
                    "Name": "Recent Movie",
                    "Path": "/movies/recent.mp4",
                    "DateCreated": recent_date,
                    "Tags": []
                },
                {
                    "Id": "old",
                    "Name": "Old Movie",
                    "Path": "/movies/old.mp4",
                    "DateCreated": old_date,
                    "Tags": []
                }
            ]
        }
        
        with patch.object(emby_service, "_make_request", return_value=mock_response):
            videos = await emby_service.get_recent_videos(days_back=5)
        
        assert len(videos) == 1
        assert videos[0].id == "recent"
    
    async def test_get_recent_videos_empty(self, emby_service):
        """Test getting recent videos with empty result."""
        mock_response = {"Items": []}
        
        with patch.object(emby_service, "_make_request", return_value=mock_response):
            videos = await emby_service.get_recent_videos(days_back=7)
        
        assert videos == []
    
    async def test_get_video_by_id_success(self, emby_service):
        """Test getting video by ID."""
        mock_response = {
            "Id": "video123",
            "Name": "Test Movie",
            "Path": "/movies/test.mp4",
            "DateCreated": "2024-01-01T00:00:00Z",
            "Tags": ["action", "adventure"],
            "Type": "Movie"
        }
        
        with patch.object(emby_service, "_make_request", return_value=mock_response):
            video = await emby_service.get_video_by_id("video123")
        
        assert video is not None
        assert video.id == "video123"
        assert video.name == "Test Movie"
        assert len(video.existing_tags) == 2
    
    async def test_get_video_by_id_not_found(self, emby_service):
        """Test getting non-existent video."""
        with patch.object(emby_service, "_make_request", side_effect=EmbyAPIError("Not found", status_code=404)):
            video = await emby_service.get_video_by_id("nonexistent")
        
        assert video is None
    
    async def test_update_video_tags_add_new(self, emby_service):
        """Test adding new tags to video."""
        # Mock getting current video
        current_video = {
            "Id": "video123",
            "Name": "Test Movie",
            "Path": "/movies/test.mp4",
            "DateCreated": "2024-01-01T00:00:00Z",
            "Tags": ["existing"],
            "Type": "Movie"
        }
        
        with patch.object(emby_service, "_make_request") as mock_request:
            # First call gets current video, second updates tags
            mock_request.side_effect = [current_video, None]
            
            await emby_service.update_video_tags("video123", ["new1", "new2"])
            
            # Check the update call
            assert mock_request.call_count == 2
            update_call = mock_request.call_args_list[1]
            assert update_call[0][0] == "POST"  # Method
            assert "/Items/video123" in update_call[0][1]  # Endpoint
            
            # Check updated tags
            update_data = update_call[1]["data"]
            assert set(update_data["Tags"]) == {"existing", "new1", "new2"}
    
    async def test_update_video_tags_no_duplicates(self, emby_service):
        """Test updating tags doesn't create duplicates."""
        current_video = {
            "Id": "video123",
            "Name": "Test Movie",
            "Path": "/movies/test.mp4",
            "DateCreated": "2024-01-01T00:00:00Z",
            "Tags": ["action", "adventure"],
            "Type": "Movie"
        }
        
        with patch.object(emby_service, "_make_request") as mock_request:
            mock_request.side_effect = [current_video, None]
            
            await emby_service.update_video_tags("video123", ["action", "sci-fi"])
            
            update_call = mock_request.call_args_list[1]
            update_data = update_call[1]["data"]
            assert set(update_data["Tags"]) == {"action", "adventure", "sci-fi"}
            assert update_data["Tags"].count("action") == 1
    
    async def test_update_video_tags_error_handling(self, emby_service):
        """Test error handling in tag update."""
        with patch.object(emby_service, "_make_request", side_effect=EmbyAPIError("Update failed")):
            # Should not raise, but log error
            await emby_service.update_video_tags("video123", ["new"])
    
    async def test_parse_video_item(self, emby_service):
        """Test parsing video item from API response."""
        item = {
            "Id": "video123",
            "Name": "Test Movie",
            "Path": "/movies/test.mp4",
            "DateCreated": "2024-01-01T12:30:45Z",
            "Tags": ["action", "adventure"],
            "Type": "Movie",
            "RunTimeTicks": 7200000000,  # 2 hours
            "ProductionYear": 2024
        }
        
        video = emby_service._parse_video_item(item)
        
        assert video.id == "video123"
        assert video.name == "Test Movie"
        assert video.path == "/movies/test.mp4"
        assert video.existing_tags == ["action", "adventure"]
        assert video.metadata["Type"] == "Movie"
        assert video.metadata["RunTimeTicks"] == 7200000000
        assert video.metadata["ProductionYear"] == 2024
    
    async def test_parse_video_item_minimal(self, emby_service):
        """Test parsing video item with minimal data."""
        item = {
            "Id": "video123",
            "Name": "Test Movie",
            "Path": "/movies/test.mp4",
            "DateCreated": "2024-01-01T00:00:00Z"
        }
        
        video = emby_service._parse_video_item(item)
        
        assert video.id == "video123"
        assert video.name == "Test Movie"
        assert video.existing_tags == []
        assert video.metadata == item