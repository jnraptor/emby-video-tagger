"""Command-line interface for Emby Video Tagger."""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from emby_video_tagger.config.settings import AppConfig
from emby_video_tagger.services.emby import EmbyService
from emby_video_tagger.services.frame_extractor import FrameExtractor
from emby_video_tagger.services.vision.factory import VisionProcessorFactory
from emby_video_tagger.services.orchestrator import VideoTaggingOrchestrator, PathMapper
from emby_video_tagger.storage.database import Database
from emby_video_tagger.storage.repository import TaskRepository
from emby_video_tagger.utils.logging import setup_logging
from emby_video_tagger.scheduler.jobs import setup_scheduler
from emby_video_tagger.core.models import TaskStatus


console = Console()


@click.group()
@click.pass_context
def cli(ctx):
    """Emby Video Tagger - Automated video tagging for Emby media server."""
    # Load configuration
    try:
        config = AppConfig.load_config()
        
        # Validate configuration
        errors = config.validate_config()
        if errors:
            console.print("[red]Configuration errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            sys.exit(1)
        
        # Store config in context
        ctx.ensure_object(dict)
        ctx.obj['config'] = config
        
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
@click.option('--days-back', '-d', default=5, help='Number of days back to look for videos')
@click.option('--max-concurrent', '-c', default=3, help='Maximum concurrent video processing')
def process(ctx, days_back: int, max_concurrent: int):
    """Process recently added videos."""
    config = ctx.obj['config']
    
    async def run_processing():
        # Setup logging
        logger = setup_logging(config.logging)
        
        # Initialize services
        async with Database(config.database) as db:
            emby_service = EmbyService(config.emby, logger)
            frame_extractor = FrameExtractor(config.processing.scene_threshold, logger)
            vision_processor = VisionProcessorFactory.create_processor(config.ai, logger)
            task_repository = TaskRepository(db, logger)
            path_mapper = PathMapper(config.path_mappings, logger)
            
            orchestrator = VideoTaggingOrchestrator(
                emby_service=emby_service,
                frame_extractor=frame_extractor,
                vision_processor=vision_processor,
                task_repository=task_repository,
                path_mapper=path_mapper,
                config=config,
                logger=logger
            )
            
            # Process videos
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Processing videos...", total=None)
                
                results = await orchestrator.process_recent_videos(
                    days_back=days_back,
                    max_concurrent=max_concurrent
                )
                
                progress.update(task, completed=True)
            
            # Display results
            display_results(results)
            
            # Cleanup
            await emby_service.close()
    
    # Run async function
    asyncio.run(run_processing())


@cli.command()
@click.pass_context
@click.argument('video_id')
def process_video(ctx, video_id: str):
    """Process a specific video by ID."""
    config = ctx.obj['config']
    
    async def run_single_processing():
        # Setup logging
        logger = setup_logging(config.logging)
        
        # Initialize services
        async with Database(config.database) as db:
            emby_service = EmbyService(config.emby, logger)
            frame_extractor = FrameExtractor(config.processing.scene_threshold, logger)
            vision_processor = VisionProcessorFactory.create_processor(config.ai, logger)
            task_repository = TaskRepository(db, logger)
            path_mapper = PathMapper(config.path_mappings, logger)
            
            orchestrator = VideoTaggingOrchestrator(
                emby_service=emby_service,
                frame_extractor=frame_extractor,
                vision_processor=vision_processor,
                task_repository=task_repository,
                path_mapper=path_mapper,
                config=config,
                logger=logger
            )
            
            # Process single video
            console.print(f"Processing video {video_id}...")
            result = await orchestrator.process_single_video(video_id)
            
            # Display result
            if result.is_successful:
                console.print(f"[green]✓ Successfully processed video {video_id}[/green]")
                console.print(f"  Generated {result.tag_count} tags in {result.processing_time:.2f}s")
                if result.tags:
                    console.print(f"  Tags: {', '.join(result.tags[:10])}")
                    if len(result.tags) > 10:
                        console.print(f"  ... and {len(result.tags) - 10} more")
            else:
                console.print(f"[red]✗ Failed to process video {video_id}[/red]")
                console.print(f"  Error: {result.error}")
            
            # Cleanup
            await emby_service.close()
    
    # Run async function
    asyncio.run(run_single_processing())


@cli.command()
@click.pass_context
def stats(ctx):
    """Show processing statistics."""
    config = ctx.obj['config']
    
    async def show_stats():
        # Setup logging
        logger = setup_logging(config.logging)
        
        # Get statistics
        async with Database(config.database) as db:
            task_repository = TaskRepository(db, logger)
            stats = await task_repository.get_statistics()
            
            # Display statistics
            console.print("\n[bold]Processing Statistics[/bold]\n")
            
            # Status breakdown
            table = Table(title="Task Status Breakdown")
            table.add_column("Status", style="cyan")
            table.add_column("Count", justify="right")
            
            for status, count in stats['by_status'].items():
                table.add_row(status.upper(), str(count))
            
            table.add_row("TOTAL", str(stats['total_tasks']), style="bold")
            console.print(table)
            
            # Other stats
            console.print(f"\n[bold]Total tags generated:[/bold] {stats['total_tags']:,}")
            
            if stats['avg_processing_time_seconds'] > 0:
                avg_time = stats['avg_processing_time_seconds']
                console.print(f"[bold]Average processing time:[/bold] {avg_time:.2f} seconds")
    
    # Run async function
    asyncio.run(show_stats())


@cli.command()
@click.pass_context
def retry_failed(ctx):
    """Retry processing for failed videos."""
    config = ctx.obj['config']
    
    async def run_retry():
        # Setup logging
        logger = setup_logging(config.logging)
        
        # Initialize services
        async with Database(config.database) as db:
            emby_service = EmbyService(config.emby, logger)
            frame_extractor = FrameExtractor(config.processing.scene_threshold, logger)
            vision_processor = VisionProcessorFactory.create_processor(config.ai, logger)
            task_repository = TaskRepository(db, logger)
            path_mapper = PathMapper(config.path_mappings, logger)
            
            orchestrator = VideoTaggingOrchestrator(
                emby_service=emby_service,
                frame_extractor=frame_extractor,
                vision_processor=vision_processor,
                task_repository=task_repository,
                path_mapper=path_mapper,
                config=config,
                logger=logger
            )
            
            # Retry failed videos
            console.print("Retrying failed videos...")
            results = await orchestrator.reprocess_failed_videos()
            
            # Display results
            display_results(results)
            
            # Cleanup
            await emby_service.close()
    
    # Run async function
    asyncio.run(run_retry())


@cli.command()
@click.pass_context
def schedule(ctx):
    """Run scheduled processing (blocks until interrupted)."""
    config = ctx.obj['config']
    
    if not config.scheduler.enabled:
        console.print("[yellow]Scheduler is disabled in configuration[/yellow]")
        return
    
    # Setup logging
    logger = setup_logging(config.logging)
    
    # Setup and run scheduler
    console.print(f"[green]Starting scheduler...[/green]")
    console.print(f"Daily processing scheduled at {config.scheduler.hour:02d}:{config.scheduler.minute:02d}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    
    scheduler = setup_scheduler(config, logger)
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]Scheduler stopped by user[/yellow]")
        scheduler.shutdown()


def display_results(results):
    """Display processing results in a table."""
    if not results:
        console.print("[yellow]No videos were processed[/yellow]")
        return
    
    # Create results table
    table = Table(title="Processing Results")
    table.add_column("Video ID", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Tags", justify="right")
    table.add_column("Frames", justify="right")
    table.add_column("Time (s)", justify="right")
    
    successful = 0
    failed = 0
    
    for result in results:
        if result.status == TaskStatus.COMPLETED:
            status = "[green]✓[/green]"
            successful += 1
        else:
            status = "[red]✗[/red]"
            failed += 1
        
        table.add_row(
            result.video_id,
            status,
            str(result.tag_count) if result.tags else "-",
            str(result.frames_processed) if result.frames_processed else "-",
            f"{result.processing_time:.2f}"
        )
    
    console.print(table)
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold] {successful} successful, {failed} failed")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()