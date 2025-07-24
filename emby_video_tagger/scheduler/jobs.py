"""Scheduled job definitions for automated video processing."""

import asyncio
from typing import Optional
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor

from emby_video_tagger.config.settings import AppConfig
from emby_video_tagger.services.emby import EmbyService
from emby_video_tagger.services.frame_extractor import FrameExtractor
from emby_video_tagger.services.vision.factory import VisionProcessorFactory
from emby_video_tagger.services.orchestrator import VideoTaggingOrchestrator, PathMapper
from emby_video_tagger.storage.database import Database
from emby_video_tagger.storage.repository import TaskRepository


def setup_scheduler(config: AppConfig, logger=None) -> BlockingScheduler:
    """
    Configure and setup the scheduler for automated processing.
    
    Args:
        config: Application configuration
        logger: Optional logger instance
        
    Returns:
        Configured scheduler instance
    """
    # Configure job stores and executors
    jobstores = {
        'default': MemoryJobStore()
    }
    
    executors = {
        'default': ThreadPoolExecutor(2)  # Limit concurrent processing
    }
    
    job_defaults = {
        'coalesce': True,
        'max_instances': 1
    }
    
    # Create scheduler
    scheduler = BlockingScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone='UTC'
    )
    
    # Add daily processing job
    scheduler.add_job(
        func=lambda: asyncio.run(run_daily_processing(config, logger)),
        trigger='cron',
        hour=config.scheduler.hour,
        minute=config.scheduler.minute,
        id='daily_video_tagging',
        name='Daily Video Tagging',
        replace_existing=True
    )
    
    if logger:
        logger.info(
            f"Scheduled daily video tagging at "
            f"{config.scheduler.hour:02d}:{config.scheduler.minute:02d} UTC"
        )
    
    return scheduler


async def run_daily_processing(config: AppConfig, logger=None):
    """
    Run the daily video processing task.
    
    Args:
        config: Application configuration
        logger: Optional logger instance
    """
    if logger:
        logger.info("Starting scheduled daily video processing")
    
    try:
        # Initialize services
        async with Database(config.database) as db:
            # Create services
            emby_service = EmbyService(config.emby, logger)
            frame_extractor = FrameExtractor(config.processing.scene_threshold, logger)
            vision_processor = VisionProcessorFactory.create_processor(config.ai, logger)
            task_repository = TaskRepository(db, logger)
            path_mapper = PathMapper(config.path_mappings, logger)
            
            # Create orchestrator
            orchestrator = VideoTaggingOrchestrator(
                emby_service=emby_service,
                frame_extractor=frame_extractor,
                vision_processor=vision_processor,
                task_repository=task_repository,
                path_mapper=path_mapper,
                config=config,
                logger=logger
            )
            
            # Process recent videos
            results = await orchestrator.process_recent_videos(
                days_back=config.processing.days_back,
                max_concurrent=config.processing.max_concurrent_videos
            )
            
            # Log results
            if logger:
                successful = sum(1 for r in results if r.is_successful)
                failed = len(results) - successful
                logger.info(
                    f"Daily processing completed: "
                    f"{successful} successful, {failed} failed"
                )
            
            # Cleanup
            await emby_service.close()
            
    except Exception as e:
        if logger:
            logger.error(f"Daily processing failed: {e}", exc_info=True)
        raise


def run_once_job(config: AppConfig, logger=None):
    """
    Run a one-time processing job.
    
    Args:
        config: Application configuration
        logger: Optional logger instance
    """
    asyncio.run(run_daily_processing(config, logger))