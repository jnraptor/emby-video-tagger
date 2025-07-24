"""Repository for task persistence operations."""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, cast
from sqlalchemy import select, update, delete, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from emby_video_tagger.core.interfaces import ITaskRepository
from emby_video_tagger.core.models import TaskStatus, VideoTask
from emby_video_tagger.storage.models import VideoTaskModel
from emby_video_tagger.core.exceptions import DatabaseError


class TaskRepository(ITaskRepository):
    """Repository for video task persistence."""
    
    def __init__(self, session: AsyncSession, logger=None):
        """Initialize repository with database session."""
        self.session = session
        self.logger = logger
    
    async def create_task(self, task: VideoTask, file_path: str) -> VideoTask:
        """Create new processing task."""
        try:
            # Check if task already exists
            existing = await self.session.execute(
                select(VideoTaskModel).where(VideoTaskModel.emby_id == task.emby_id)
            )
            existing_task = existing.scalar_one_or_none()
            
            if existing_task:
                # Update existing task using update statement
                stmt = (
                    update(VideoTaskModel)
                    .where(VideoTaskModel.emby_id == task.emby_id)
                    .values(
                        file_path=file_path,
                        status=TaskStatus.PENDING,
                        created_at=datetime.utcnow(),
                        completed_at=None,
                        error_message=None
                    )
                )
                await self.session.execute(stmt)
                await self.session.commit()
                
                # Refresh the existing task
                await self.session.refresh(existing_task)
                return self._convert_db_to_model(existing_task)
            else:
                # Create new task
                db_task = VideoTaskModel(
                    emby_id=task.emby_id,
                    file_path=file_path,
                    status=TaskStatus.PENDING,
                    created_at=datetime.utcnow()
                )
                self.session.add(db_task)
                await self.session.commit()
                await self.session.refresh(db_task)
                return self._convert_db_to_model(db_task)
                    
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(
                f"Failed to create task for video {task.emby_id}: {str(e)}",
                operation="create_task"
            )
    
    async def update_task_status(
        self,
        video_id: str,
        status: TaskStatus,
        tag_count: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update task status."""
        try:
            # Build update values
            update_values: Dict[str, Any] = {"status": status}
            
            if status == TaskStatus.COMPLETED:
                update_values["completed_at"] = datetime.utcnow()
                if tag_count is not None:
                    update_values["tag_count"] = tag_count
                update_values["error_message"] = None
                    
            elif status == TaskStatus.FAILED:
                update_values["completed_at"] = datetime.utcnow()
                if error_message:
                    update_values["error_message"] = error_message
            
            # Execute update
            stmt = (
                update(VideoTaskModel)
                .where(VideoTaskModel.emby_id == video_id)
                .values(**update_values)
            )
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            return result.rowcount > 0
                
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(
                f"Failed to update task status for video {video_id}: {str(e)}",
                operation="update_task_status"
            )
    
    async def update_task(self, task: VideoTask) -> Optional[VideoTask]:
        """Update an existing task."""
        try:
            # Find the task
            result = await self.session.execute(
                select(VideoTaskModel).where(VideoTaskModel.emby_id == task.emby_id)
            )
            db_task = result.scalar_one_or_none()
            
            if not db_task:
                return None
            
            # Update the task
            stmt = (
                update(VideoTaskModel)
                .where(VideoTaskModel.emby_id == task.emby_id)
                .values(
                    status=task.status,
                    tag_count=task.tag_count,
                    error_message=task.error_message,
                    completed_at=task.completed_at
                )
            )
            await self.session.execute(stmt)
            await self.session.commit()
            
            # Refresh and return
            await self.session.refresh(db_task)
            return self._convert_db_to_model(db_task)
                
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(
                f"Failed to update task for video {task.emby_id}: {str(e)}",
                operation="update_task"
            )
    
    async def get_task_by_video_id(self, video_id: str) -> Optional[VideoTask]:
        """Get task by video ID."""
        try:
            result = await self.session.execute(
                select(VideoTaskModel).where(VideoTaskModel.emby_id == video_id)
            )
            task = result.scalar_one_or_none()
            
            if task:
                return self._convert_db_to_model(task)
            return None
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get task for video {video_id}: {str(e)}",
                operation="get_task_by_video_id"
            )
    
    async def get_pending_tasks(self, limit: int = 100) -> List[VideoTask]:
        """Get all pending tasks."""
        try:
            result = await self.session.execute(
                select(VideoTaskModel)
                .where(VideoTaskModel.status == TaskStatus.PENDING)
                .order_by(VideoTaskModel.created_at)
                .limit(limit)
            )
            tasks = result.scalars().all()
            
            return [self._convert_db_to_model(task) for task in tasks]
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get pending tasks: {str(e)}",
                operation="get_pending_tasks"
            )
    
    async def get_failed_tasks(self, limit: int = 100) -> List[VideoTask]:
        """Get failed tasks for retry."""
        try:
            result = await self.session.execute(
                select(VideoTaskModel)
                .where(VideoTaskModel.status == TaskStatus.FAILED)
                .order_by(VideoTaskModel.created_at.desc())
                .limit(limit)
            )
            tasks = result.scalars().all()
            
            return [self._convert_db_to_model(task) for task in tasks]
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get failed tasks: {str(e)}",
                operation="get_failed_tasks"
            )
    
    async def get_recent_tasks(self, hours: int = 24, limit: int = 100) -> List[VideoTask]:
        """Get recent tasks."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            result = await self.session.execute(
                select(VideoTaskModel)
                .where(VideoTaskModel.created_at >= cutoff_time)
                .order_by(VideoTaskModel.created_at.desc())
                .limit(limit)
            )
            tasks = result.scalars().all()
            
            return [self._convert_db_to_model(task) for task in tasks]
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get recent tasks: {str(e)}",
                operation="get_recent_tasks"
            )
    
    async def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get processing statistics."""
        try:
            # Total tasks
            total_result = await self.session.execute(
                select(func.count(VideoTaskModel.id))
            )
            total_tasks = total_result.scalar_one()
            
            # Successful tasks
            success_result = await self.session.execute(
                select(func.count(VideoTaskModel.id))
                .where(VideoTaskModel.status == TaskStatus.COMPLETED)
            )
            successful_tasks = success_result.scalar_one()
            
            # Failed tasks
            failed_result = await self.session.execute(
                select(func.count(VideoTaskModel.id))
                .where(VideoTaskModel.status == TaskStatus.FAILED)
            )
            failed_tasks = failed_result.scalar_one()
            
            # Total tags generated
            tags_result = await self.session.execute(
                select(func.sum(VideoTaskModel.tag_count))
                .where(VideoTaskModel.status == TaskStatus.COMPLETED)
            )
            total_tags = tags_result.scalar_one() or 0
            
            # Average processing time (for completed tasks)
            avg_time_result = await self.session.execute(
                select(
                    func.avg(
                        func.julianday(VideoTaskModel.completed_at) - 
                        func.julianday(VideoTaskModel.created_at)
                    ) * 24 * 60 * 60  # Convert to seconds
                )
                .where(
                    and_(
                        VideoTaskModel.status == TaskStatus.COMPLETED,
                        VideoTaskModel.completed_at.isnot(None)
                    )
                )
            )
            avg_processing_time = avg_time_result.scalar_one() or 0.0
            
            # Calculate success rate
            success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0.0
            
            return {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "total_tags_generated": int(total_tags),
                "average_processing_time": float(avg_processing_time),
                "success_rate": success_rate
            }
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get statistics: {str(e)}",
                operation="get_statistics"
            )
    
    async def delete_old_tasks(self, days: int = 30) -> int:
        """Delete old tasks."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            stmt = delete(VideoTaskModel).where(VideoTaskModel.created_at < cutoff_date)
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            return result.rowcount
                
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(
                f"Failed to delete old tasks: {str(e)}",
                operation="delete_old_tasks"
            )
    
    async def task_exists(self, video_id: str) -> bool:
        """Check if task exists for video ID."""
        try:
            result = await self.session.execute(
                select(VideoTaskModel.id).where(VideoTaskModel.emby_id == video_id)
            )
            task_id = result.scalar_one_or_none()
            
            return task_id is not None
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to check task existence for video {video_id}: {str(e)}",
                operation="task_exists"
            )
    
    async def get_tasks_by_status(self, status: TaskStatus, limit: int = 100) -> List[VideoTask]:
        """Get tasks by status."""
        try:
            result = await self.session.execute(
                select(VideoTaskModel)
                .where(VideoTaskModel.status == status)
                .order_by(VideoTaskModel.created_at.desc())
                .limit(limit)
            )
            tasks = result.scalars().all()
            
            return [self._convert_db_to_model(task) for task in tasks]
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get tasks by status {status}: {str(e)}",
                operation="get_tasks_by_status"
            )
    
    def _convert_db_to_model(self, db_task: VideoTaskModel) -> VideoTask:
        """Convert database model to domain model."""
        return VideoTask(
            id=db_task.id,  # type: ignore
            emby_id=db_task.emby_id,  # type: ignore
            file_path=db_task.file_path,  # type: ignore
            status=db_task.status,  # type: ignore
            created_at=db_task.created_at,  # type: ignore
            completed_at=db_task.completed_at,  # type: ignore
            tag_count=db_task.tag_count or 0,  # type: ignore
            error_message=db_task.error_message  # type: ignore
        )
    
    def _convert_model_to_db(self, task: VideoTask) -> VideoTaskModel:
        """Convert domain model to database model."""
        return VideoTaskModel(
            emby_id=task.emby_id,
            file_path=task.file_path,
            status=task.status,
            created_at=task.created_at,
            completed_at=task.completed_at,
            tag_count=task.tag_count,
            error_message=task.error_message
        )