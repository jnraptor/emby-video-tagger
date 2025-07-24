"""Main orchestration service for video tagging workflow."""

import asyncio
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from emby_video_tagger.core.interfaces import (
    IVideoTaggingOrchestrator,
    IEmbyService,
    IFrameExtractor,
    IVisionProcessor,
    ITaskRepository,
    IPathMapper
)
from emby_video_tagger.core.models import (
    Video,
    ProcessingResult,
    TaskStatus,
    Frame
)
from emby_video_tagger.core.exceptions import (
    EmbyVideoTaggerError,
    PathMappingError,
    VideoValidationError
)
from emby_video_tagger.config.settings import AppConfig


class PathMapper(IPathMapper):
    """Simple path mapper implementation."""
    
    def __init__(self, mappings: Dict[str, str], logger=None):
        """Initialize path mapper with mappings."""
        self.mappings = mappings
        self.logger = logger
    
    def map_path(self, source_path: str) -> str:
        """Map source path to destination path."""
        if not self.mappings:
            return source_path
        
        # Try each mapping to see if it matches the beginning of the path
        for emby_prefix, local_prefix in self.mappings.items():
            if source_path.startswith(emby_prefix):
                # Replace the emby prefix with the local prefix
                mapped_path = source_path.replace(emby_prefix, local_prefix, 1)
                if self.logger:
                    self.logger.debug(f"Mapped path: {source_path} -> {mapped_path}")
                return mapped_path
        
        # If no mapping found, return original path
        if self.logger:
            self.logger.warning(f"No path mapping found for: {source_path}")
        return source_path
    
    def validate_mapping(self, source_path: str) -> bool:
        """Check if path mapping exists and is valid."""
        mapped_path = self.map_path(source_path)
        return Path(mapped_path).exists()
    
    def add_mapping(self, source_prefix: str, dest_prefix: str) -> None:
        """Add new path mapping."""
        self.mappings[source_prefix] = dest_prefix


class VideoTaggingOrchestrator(IVideoTaggingOrchestrator):
    """Main orchestrator for the video tagging workflow."""
    
    def __init__(
        self,
        emby_service: IEmbyService,
        frame_extractor: IFrameExtractor,
        vision_processor: IVisionProcessor,
        task_repository: ITaskRepository,
        path_mapper: IPathMapper,
        config: AppConfig,
        logger=None
    ):
        """Initialize orchestrator with dependencies."""
        self.emby_service = emby_service
        self.frame_extractor = frame_extractor
        self.vision_processor = vision_processor
        self.task_repository = task_repository
        self.path_mapper = path_mapper
        self.config = config
        self.logger = logger
    
    async def process_recent_videos(
        self,
        days_back: int = 5,
        max_concurrent: int = 3
    ) -> List[ProcessingResult]:
        """Process recently added videos."""
        if self.logger:
            self.logger.info(f"Starting processing of videos from last {days_back} days")
        
        try:
            # Get recent videos from Emby
            recent_videos = await self.emby_service.get_recent_videos(days_back=days_back)
            if self.logger:
                self.logger.info(f"Found {len(recent_videos)} recent videos")
            
            # Filter videos that need processing
            videos_to_process = [v for v in recent_videos if self._should_process_video(v)]
            if self.logger:
                self.logger.info(f"{len(videos_to_process)} videos need processing")
            
            # Process videos concurrently
            return await self.process_videos_batch(
                [v.id for v in videos_to_process],
                max_concurrent=max_concurrent
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to process recent videos: {e}")
            raise
    
    async def process_single_video(self, video_id: str) -> ProcessingResult:
        """Process a single video by ID."""
        start_time = time.time()
        
        try:
            # Get video details
            video = await self.emby_service.get_video_by_id(video_id)
            if not video:
                raise VideoValidationError(
                    f"Video not found: {video_id}",
                    video_id=video_id,
                    reason="not_found"
                )
            
            if self.logger:
                self.logger.info(f"Processing video: {video.name} (ID: {video.id})")
            
            # Map video path
            mapped_path = self.path_mapper.map_path(video.path)
            
            # Validate video
            if not self.frame_extractor.validate_video(mapped_path):
                raise VideoValidationError(
                    f"Invalid video file: {mapped_path}",
                    video_id=video_id,
                    reason="invalid_file"
                )
            
            # Update task status to processing
            await self.task_repository.update_task_status(
                video_id,
                TaskStatus.PROCESSING
            )
            
            # Extract frames
            frames = await self.frame_extractor.extract_frames(
                mapped_path,
                max_frames=self.config.processing.max_frames_per_video
            )
            
            if not frames:
                raise EmbyVideoTaggerError(f"No frames extracted from video: {video.name}")
            
            if self.logger:
                self.logger.info(f"Extracted {len(frames)} frames from {video.name}")
            
            # Analyze frames
            analyses = await self.vision_processor.analyze_frames_batch(frames)
            
            # Merge tags
            tags = self.vision_processor.merge_tags(analyses)
            
            if self.logger:
                self.logger.info(f"Generated {len(tags)} tags for {video.name}")
            
            # Add AI marker tag
            all_tags = list(set(video.existing_tags + tags + ["ai-generated"]))
            
            # Update Emby with new tags
            success = await self.emby_service.update_video_tags(video_id, all_tags)
            
            if success:
                # Update task status
                await self.task_repository.update_task_status(
                    video_id,
                    TaskStatus.COMPLETED,
                    tag_count=len(tags)
                )
                
                processing_time = time.time() - start_time
                
                return ProcessingResult(
                    video_id=video_id,
                    status=TaskStatus.COMPLETED,
                    tags=tags,
                    frames_processed=len(frames),
                    processing_time=processing_time
                )
            else:
                raise EmbyVideoTaggerError(f"Failed to update tags in Emby for video: {video_id}")
                
        except Exception as e:
            # Update task status to failed
            await self.task_repository.update_task_status(
                video_id,
                TaskStatus.FAILED,
                error_message=str(e)
            )
            
            processing_time = time.time() - start_time
            
            if self.logger:
                self.logger.error(f"Failed to process video {video_id}: {e}")
            
            return ProcessingResult(
                video_id=video_id,
                status=TaskStatus.FAILED,
                processing_time=processing_time,
                error=str(e)
            )
        finally:
            # Cleanup temporary frame files
            if 'frames' in locals():
                for frame in frames:
                    frame.cleanup()
    
    async def process_videos_batch(
        self,
        video_ids: List[str],
        max_concurrent: int = 3
    ) -> List[ProcessingResult]:
        """Process multiple videos concurrently."""
        if not video_ids:
            return []
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(video_id: str) -> ProcessingResult:
            async with semaphore:
                return await self.process_single_video(video_id)
        
        # Process all videos concurrently
        tasks = [process_with_semaphore(video_id) for video_id in video_ids]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Log summary
        if self.logger:
            successful = sum(1 for r in results if r.status == TaskStatus.COMPLETED)
            failed = sum(1 for r in results if r.status == TaskStatus.FAILED)
            self.logger.info(
                f"Batch processing complete: {successful} successful, {failed} failed"
            )
        
        return results
    
    async def reprocess_failed_videos(self, max_retries: int = 3) -> List[ProcessingResult]:
        """Retry processing for failed videos."""
        if self.logger:
            self.logger.info("Starting reprocessing of failed videos")
        
        # Get failed tasks
        failed_tasks = await self.task_repository.get_failed_tasks(limit=100)
        
        if not failed_tasks:
            if self.logger:
                self.logger.info("No failed videos to reprocess")
            return []
        
        # Extract video IDs
        video_ids = [task["emby_id"] for task in failed_tasks]
        
        if self.logger:
            self.logger.info(f"Found {len(video_ids)} failed videos to reprocess")
        
        # Process failed videos
        return await self.process_videos_batch(video_ids, max_concurrent=2)
    
    def _should_process_video(self, video: Video) -> bool:
        """Determine if video needs processing."""
        # Check if already processed
        if not video.needs_processing:
            if self.logger:
                self.logger.debug(f"Skipping {video.name} - already processed")
            return False
        
        # Check file existence using mapped path
        mapped_path = self.path_mapper.map_path(video.path)
        if not Path(mapped_path).exists():
            if self.logger:
                self.logger.warning(
                    f"Video file not found: {mapped_path} (original: {video.path})"
                )
            return False
        
        # Skip very small files (likely not full videos)
        min_size_mb = 10
        if video.file_size_mb < min_size_mb:
            if self.logger:
                self.logger.debug(f"Skipping {video.name} - file too small")
            return False
        
        return True