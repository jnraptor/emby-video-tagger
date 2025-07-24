"""Emby API service implementation."""

import asyncio
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any
import aiohttp
from urllib.parse import urljoin

from emby_video_tagger.core.interfaces import IEmbyService
from emby_video_tagger.core.models import Video
from emby_video_tagger.core.exceptions import EmbyAPIError, ConfigurationError
from emby_video_tagger.config.settings import EmbyConfig


class EmbyService(IEmbyService):
    """Handles all Emby API interactions for video metadata management."""
    
    def __init__(self, config: EmbyConfig, logger=None):
        """Initialize Emby service with configuration."""
        self.config = config
        self.base_url = str(config.server_url).rstrip('/')
        self.api_key = config.api_key
        self.user_id = config.user_id
        self.logger = logger
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Headers for API requests
        self.headers = {
            "X-Emby-Token": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Validate configuration
        if not self.api_key:
            raise ConfigurationError("Emby API key is required")
        if not self.user_id:
            raise ConfigurationError("Emby user ID is required")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers
            )
        return self._session
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers
            )
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
    
    def _parse_emby_datetime(self, date_string: str) -> datetime:
        """
        Parse Emby datetime string, handling the case where microseconds have more than 6 digits.
        Emby sometimes returns datetime strings with 7 decimal places which Python can't parse.
        """
        try:
            # Replace Z with +00:00 for proper timezone handling
            date_str = date_string.replace('Z', '+00:00')
            
            # Handle microseconds with more than 6 decimal places
            pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d+)([\+\-]\d{2}:\d{2})$'
            match = re.match(pattern, date_str)
            
            if match:
                date_part, microseconds, timezone_part = match.groups()
                # Truncate microseconds to 6 digits if longer
                if len(microseconds) > 6:
                    microseconds = microseconds[:6]
                date_str = f"{date_part}.{microseconds}{timezone_part}"
            
            return datetime.fromisoformat(date_str)
        except ValueError as e:
            if self.logger:
                self.logger.error(f"Failed to parse datetime '{date_string}': {e}")
            # Return current time as fallback
            return datetime.now()
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the Emby API."""
        await self._ensure_session()
        
        url = urljoin(self.base_url, endpoint)
        
        try:
            async with self._session.request(method, url, **kwargs) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise EmbyAPIError(
                        f"Emby API request failed: {method} {url}",
                        status_code=response.status,
                        response_body=text
                    )
                
                # Handle empty responses
                if response.status == 204:
                    return {}
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise EmbyAPIError(f"Network error communicating with Emby: {str(e)}")
        except Exception as e:
            if isinstance(e, EmbyAPIError):
                raise
            raise EmbyAPIError(f"Unexpected error in Emby API request: {str(e)}")
    
    async def get_recent_videos(self, days_back: int = 7, limit: int = 100) -> List[Video]:
        """Retrieve recently added videos from Emby."""
        params = {
            "SortBy": "DateCreated",
            "SortOrder": "Descending",
            "Recursive": "true",
            "IncludeItemTypes": "Video",
            "Fields": "Tags,TagItems,Genres,ProviderIds,Path,DateCreated",
            "Limit": limit,
        }
        
        try:
            response = await self._make_request(
                "GET",
                f"/emby/Users/{self.user_id}/Items",
                params=params
            )
            
            items = response.get("Items", [])
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_videos = []
            
            for item in items:
                created_str = item.get("DateCreated", "")
                if created_str:
                    created_date = self._parse_emby_datetime(created_str)
                    if created_date.replace(tzinfo=None) >= cutoff_date:
                        video = self._item_to_video(item)
                        recent_videos.append(video)
            
            if self.logger:
                self.logger.info(f"Retrieved {len(recent_videos)} recent videos from Emby")
            
            return recent_videos
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to retrieve recent videos: {e}")
            raise
    
    async def get_video_by_id(self, video_id: str) -> Optional[Video]:
        """Get specific video by ID."""
        params = {
            "Fields": "Path,Tags,TagItems,ProviderIds,DateCreated"
        }
        
        try:
            response = await self._make_request(
                "GET",
                f"/emby/Items",
                params={"Ids": video_id, **params}
            )
            
            # Handle case where response is a single item (not wrapped in Items array)
            if isinstance(response, dict) and "Id" in response:
                return self._item_to_video(response)
            
            items = response.get("Items", [])
            if items:
                return self._item_to_video(items[0])
            
            return None
            
        except EmbyAPIError as e:
            if e.status_code == 404:
                return None
            if self.logger:
                self.logger.error(f"Failed to get video {video_id}: {e}")
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get video {video_id}: {e}")
            raise
    
    async def update_video_tags(self, video_id: str, tags: List[str]) -> bool:
        """Update video tags in Emby."""
        try:
            # First, get the current video data
            video_data = await self._make_request(
                "GET",
                f"/emby/Items",
                params={
                    "Ids": video_id,
                    "Fields": "Path,Tags,TagItems,ProviderIds"
                }
            )
            
            if not video_data.get("Items"):
                raise EmbyAPIError(f"Video {video_id} not found")
            
            item = video_data["Items"][0]
            
            # Get existing tags
            existing_tags = item.get("Tags", [])
            
            # Merge existing tags with new tags (no duplicates)
            all_tags = list(set(existing_tags + tags))
            
            # Update tags
            item["Tags"] = all_tags
            item["TagItems"] = [{"Name": tag} for tag in all_tags]
            
            if self.logger:
                self.logger.info(f"Updating tags for item {video_id}: {all_tags}")
            
            # Update the item
            await self._make_request(
                "POST",
                f"/emby/Items/{video_id}",
                data=item
            )
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to update tags for item {video_id}: {e}")
            return False
    
    async def batch_update_tags(self, updates: List[Tuple[str, List[str]]]) -> Dict[str, bool]:
        """Batch update multiple videos' tags."""
        results = {}
        
        # Process updates concurrently with a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
        
        async def update_with_semaphore(video_id: str, tags: List[str]):
            async with semaphore:
                success = await self.update_video_tags(video_id, tags)
                results[video_id] = success
        
        tasks = [
            update_with_semaphore(video_id, tags)
            for video_id, tags in updates
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def _item_to_video(self, item: Dict[str, Any]) -> Video:
        """Convert Emby item to Video model."""
        # Extract existing tags
        existing_tags = []
        
        # Try TagItems first (preferred)
        tag_items = item.get("TagItems", [])
        if tag_items:
            existing_tags = [tag["Name"] for tag in tag_items if isinstance(tag, dict) and "Name" in tag]
        # Fallback to Tags field
        elif "Tags" in item and isinstance(item["Tags"], list):
            existing_tags = item["Tags"]
        
        # Parse creation date
        created_str = item.get("DateCreated", "")
        created_date = self._parse_emby_datetime(created_str) if created_str else datetime.now()
        
        return Video(
            id=item["Id"],
            name=item.get("Name", "Unknown"),
            path=item.get("Path", ""),
            date_created=created_date,
            existing_tags=existing_tags,
            metadata={
                "type": item.get("Type"),
                "media_type": item.get("MediaType"),
                "provider_ids": item.get("ProviderIds", {}),
                "genres": item.get("Genres", []),
                "production_year": item.get("ProductionYear"),
                "runtime_ticks": item.get("RunTimeTicks"),
            }
        )
    
    def _parse_video_item(self, item: Dict[str, Any]) -> Video:
        """Parse video item from API response (alias for _item_to_video)."""
        return self._item_to_video(item)