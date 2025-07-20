#!/usr/bin/env python3
"""
Automated Video Tagging for Emby Media Server

This script automatically generates tags for videos in an Emby media server using:
1. Emby API to retrieve recently added videos
2. OpenCV for intelligent video frame extraction
3. LMStudio Vision API for scene analysis and tag generation
4. Automated scheduling for daily processing
5. Path remapping for cross-platform file access

Requirements:
- Emby server with API access
- LMStudio server running with vision model
- Python packages: opencv-python, requests, lmstudio, apscheduler, scenedetect, python-dotenv

Environment Variables:
- EMBY_SERVER_URL: Emby server URL
- EMBY_API_KEY: Emby API key
- EMBY_USER_ID: Emby user ID
- LMSTUDIO_MODEL_NAME: LMStudio model name (optional)
- PATH_MAPPINGS: Path mappings in format "emby_path1:local_path1,emby_path2:local_path2" (optional)
- DAYS_BACK: Number of days back to look for recent videos (optional, default: 5)
"""

import requests
import cv2
import lmstudio as lms
import ollama
import base64
import json
import time
import logging
import sqlite3
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from enum import Enum
from datetime import datetime, timedelta
from scenedetect import detect, ContentDetector
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
import psutil
import os
import tempfile
import shutil
from dotenv import load_dotenv
from abc import ABC, abstractmethod


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EmbyVideoTagger:
    """Handles all Emby API interactions for video metadata management"""

    def __init__(self, server_url: str, api_key: str, user_id: str):
        self.base_url = server_url.rstrip("/")
        self.api_key = api_key
        self.user_id = user_id
        self.session = self._create_session()
        self.logger = logging.getLogger(__name__)

    def _create_session(self):
        session = requests.Session()
        session.headers.update(
            {"X-Emby-Token": self.api_key, "Content-Type": "application/json"}
        )
        return session

    def get_recent_videos(self, days_back: int = 7) -> List[Dict]:
        """Retrieve recently added videos from Emby"""
        url = f"{self.base_url}/emby/Users/{self.user_id}/Items"
        params = {
            "SortBy": "DateCreated",
            "SortOrder": "Descending",
            "Recursive": "true",
            "IncludeItemTypes": "Video",
            "Fields": "Tags,TagItems,Genres,ProviderIds,Path,DateCreated",
            "Limit": 100,
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            items = response.json().get("Items", [])

            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_items = []

            for item in items:
                created_str = item.get("DateCreated", "")
                if created_str:
                    created_date = datetime.fromisoformat(
                        created_str.replace("Z", "+00:00")
                    )
                    if created_date.replace(tzinfo=None) >= cutoff_date:
                        recent_items.append(item)

            return recent_items

        except Exception as e:
            self.logger.error(f"Failed to retrieve recent videos: {e}")
            return []

    def update_video_tags(self, item_id: str, new_tags: List[str]) -> bool:
        """Update video tags in Emby"""
        updateUrl = f"{self.base_url}/emby/Items/{item_id}"

        url = f"{self.base_url}/emby/Items?Ids={item_id}&Fields=Path,Tags,TagItems,ProviderIds"
        response = self.session.get(url)
        response.raise_for_status()
        video = response.json()
        item = video["Items"][0]
        item["Tags"] = item.get("Tags", []) + new_tags
        item["TagItems"] = item.get("TagItems", []) + new_tags
        self.logger.info(f"Updating tags for item {item_id}: {item['TagItems']}")
        # self.logger.info(item)

        try:
            time.sleep(0.1)  # Basic rate limiting
            response = self.session.post(updateUrl, json=item, timeout=30)
            response.raise_for_status()
            self.logger.info(response.text)
            return response.status_code == 204
        except Exception as e:
            self.logger.error(f"Failed to update tags for item {item_id}: {e}")
            return False


class IntelligentFrameExtractor:
    """Extracts representative frames from videos using scene detection"""

    def __init__(self, scene_threshold: float = 27.0):
        self.scene_threshold = scene_threshold
        self.logger = logging.getLogger(__name__)

    def extract_representative_frames(
        self, video_path: str, max_frames: int = 10
    ) -> List[Tuple[str, int]]:
        """Extract key frames using scene detection"""

        if not Path(video_path).exists():
            self.logger.error(f"Video file not found: {video_path}")
            return []

        try:
            # Detect scenes
            scene_list = detect(
                video_path, ContentDetector(threshold=self.scene_threshold)
            )

            if not scene_list:
                self.logger.warning(
                    f"No scenes detected in {video_path}, using fallback"
                )
                return self._fallback_extraction(video_path, max_frames)

            return self._extract_scene_frames(video_path, scene_list, max_frames)

        except Exception as e:
            self.logger.error(f"Scene detection failed for {video_path}: {e}")
            return self._fallback_extraction(video_path, max_frames)

    def _extract_scene_frames(
        self, video_path: str, scene_list: List, max_frames: int
    ) -> List[Tuple[str, int]]:
        """Extract frames from detected scenes"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # Default to 25 FPS if unavailable

        extracted_frames = []
        output_dir = Path(tempfile.mkdtemp(prefix="frames_"))

        # Sample scenes evenly if too many detected
        scenes_to_process = (
            scene_list[:max_frames]
            if len(scene_list) <= max_frames
            else [
                scene_list[i]
                for i in range(0, len(scene_list), len(scene_list) // max_frames)
            ]
        )

        for i, scene in enumerate(scenes_to_process):
            try:
                # Extract middle frame of each scene
                start_frame = scene[0].get_frames()
                end_frame = (
                    scene[1].get_frames()
                    if len(scene) > 1
                    else start_frame + int(fps * 2)
                )
                middle_frame = (start_frame + end_frame) // 2

                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                ret, frame = cap.read()

                if ret:
                    filename = str(
                        output_dir / f"scene_{i:03d}_frame_{middle_frame:06d}.jpg"
                    )
                    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    extracted_frames.append((filename, middle_frame))

            except Exception as e:
                self.logger.warning(f"Failed to extract frame from scene {i}: {e}")
                continue

        cap.release()
        return extracted_frames

    def _fallback_extraction(
        self, video_path: str, max_frames: int
    ) -> List[Tuple[str, int]]:
        """Fallback to uniform sampling if scene detection fails"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        frame_step = max(1, total_frames // max_frames)

        extracted_frames = []
        output_dir = Path(tempfile.mkdtemp(prefix="frames_"))

        for i in range(0, total_frames, frame_step):
            if len(extracted_frames) >= max_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if ret:
                filename = str(output_dir / f"uniform_frame_{i:06d}.jpg")
                cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                extracted_frames.append((filename, i))

        cap.release()
        return extracted_frames


class BaseVisionProcessor(ABC):
    """Abstract base class for AI vision processors"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tag_prompt = self._create_tagging_prompt()
        self.logger = logging.getLogger(__name__)

    def _create_tagging_prompt(self) -> str:
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
            "setting": ["urban", "outdoor", "daytime"],
            "style": ["documentary", "handheld", "wide-shot"],
            "mood": ["energetic", "professional", "casual"]
        }
        """

    def encode_image(self, image_path: str) -> str:
        """Convert image to base64 for API submission"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Failed to encode image {image_path}: {e}")
            return ""

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON content from markdown code blocks or plain text"""
        import re

        # Try to find JSON within markdown code fences
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, response_text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Try to find JSON object without code fences
        json_object_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        match = re.search(json_object_pattern, response_text, re.DOTALL)

        if match:
            return match.group(0).strip()

        # If no JSON found, return the original text
        return response_text.strip()

    @abstractmethod
    def analyze_frames_sync(self, frame_paths: List[str]) -> List[str]:
        """Synchronous frame analysis for immediate processing"""
        pass


class LMStudioVisionProcessor(BaseVisionProcessor):
    """Handles LMStudio Vision API interactions for video frame analysis"""

    def __init__(self, model_name: str = "qwen2.5-vl-7b-instruct-abliterated"):
        super().__init__(model_name)

    def analyze_frames_sync(self, frame_paths: List[str]) -> List[str]:
        """Synchronous frame analysis for immediate processing"""
        all_tags = []

        for frame_path in frame_paths:
            try:
                with lms.Client() as client:
                    model = client.llm.model(self.model_name)
                    image_handle = client.files.prepare_image(frame_path)

                    chat = lms.Chat()
                    chat.add_user_message(self.tag_prompt, images=[image_handle])
                    prediction = model.respond(chat)

                    analysis = str(prediction)

                try:
                    # Extract JSON from markdown code blocks if present
                    json_content = self._extract_json_from_response(analysis)
                    parsed_tags = json.loads(json_content)

                    # Flatten all tag categories into a single list
                    frame_tags = []
                    for category, tags in parsed_tags.items():
                        if isinstance(tags, list):
                            frame_tags.extend(tags)
                    all_tags.extend(frame_tags)

                except json.JSONDecodeError as e:
                    self.logger.warning(
                        f"Failed to parse JSON response for {frame_path}: {analysis}"
                    )
                    self.logger.debug(f"JSON decode error: {e}")

            except Exception as e:
                self.logger.error(f"Failed to analyze frame {frame_path}: {e}")

        # Remove duplicates and return unique tags
        return list(set(all_tags))


class OllamaVisionProcessor(BaseVisionProcessor):
    """Handles Ollama Vision API interactions for video frame analysis"""

    def __init__(
        self, model_name: str = "llava", base_url: str = "http://localhost:11434"
    ):
        super().__init__(model_name)
        self.base_url = base_url

    def analyze_frames_sync(self, frame_paths: List[str]) -> List[str]:
        """Synchronous frame analysis for immediate processing using Ollama"""
        all_tags = []

        for frame_path in frame_paths:
            try:
                # Encode image as base64
                image_data = self.encode_image(frame_path)
                if not image_data:
                    continue

                # Create chat request with image
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": self.tag_prompt,
                            "images": [image_data],
                        }
                    ],
                    options={"num_ctx": 4096, "temperature": 0.1},
                )

                analysis = response["message"]["content"]

                try:
                    # Extract JSON from markdown code blocks if present
                    json_content = self._extract_json_from_response(analysis)
                    parsed_tags = json.loads(json_content)

                    # Flatten all tag categories into a single list
                    frame_tags = []
                    for category, tags in parsed_tags.items():
                        if isinstance(tags, list):
                            frame_tags.extend(tags)
                    all_tags.extend(frame_tags)

                except json.JSONDecodeError as e:
                    self.logger.warning(
                        f"Failed to parse JSON response for {frame_path}: {analysis}"
                    )
                    self.logger.debug(f"JSON decode error: {e}")

            except Exception as e:
                self.logger.error(
                    f"Failed to analyze frame {frame_path} with Ollama: {e}"
                )

        # Remove duplicates and return unique tags
        return list(set(all_tags))


class VisionProcessorFactory:
    """Factory class for creating vision processor instances"""

    @staticmethod
    def create_processor(provider: str, **config) -> BaseVisionProcessor:
        """Create a vision processor based on the provider type"""
        if provider.lower() == "lmstudio":
            model_name = config.get("model_name", "qwen2.5-vl-7b-instruct-abliterated")
            return LMStudioVisionProcessor(model_name)
        elif provider.lower() == "ollama":
            model_name = config.get("model_name", "llava")
            base_url = config.get("base_url", "http://localhost:11434")
            return OllamaVisionProcessor(model_name, base_url)
        else:
            raise ValueError(
                f"Unsupported AI provider: {provider}. Supported providers: lmstudio, ollama"
            )


class VideoTaggingAutomation:
    """Main automation class that orchestrates the entire video tagging process"""

    def __init__(self, config: Dict):
        self.config = config
        self.emby_client = EmbyVideoTagger(
            config["emby"]["server_url"],
            config["emby"]["api_key"],
            config["emby"]["user_id"],
        )
        self.frame_extractor = IntelligentFrameExtractor()

        # Create vision processor using factory pattern
        ai_provider = config.get("ai_provider", "lmstudio")
        processor_config = config.get(ai_provider, {})
        self.vision_processor = VisionProcessorFactory.create_processor(
            ai_provider, **processor_config
        )
        self.task_tracker = self._setup_task_tracking()
        self.logger = self._setup_logging()
        self.path_mappings = config.get("path_mappings", {})
        self.days_back = config.get("days_back", 5)

    def _setup_logging(self):
        """Configure logging for the application"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("video_tagging.log"),
                logging.StreamHandler(),
            ],
        )
        return logging.getLogger(__name__)

    def _setup_task_tracking(self):
        """Initialize SQLite database for tracking processing tasks"""
        db_path = "video_tasks.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emby_id TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                tag_count INTEGER DEFAULT 0,
                error_message TEXT
            )
        """
        )
        conn.commit()
        conn.close()
        return db_path

    def _setup_scheduler(self):
        """Configure APScheduler for automated processing"""
        jobstores = {"default": SQLAlchemyJobStore(url="sqlite:///jobs.sqlite")}
        executors = {"default": ThreadPoolExecutor(2)}  # Limit concurrent processing
        job_defaults = {"coalesce": True, "max_instances": 1}

        # Check if job already exists in the database before creating scheduler
        job_id = "daily_video_tagging"
        job_exists = self._check_job_exists_in_db(job_id)

        scheduler = BlockingScheduler(
            jobstores=jobstores, executors=executors, job_defaults=job_defaults
        )

        if not job_exists:
            # Job doesn't exist, add it
            scheduler.add_job(
                self.process_daily_videos,
                "cron",
                args=[self.days_back],
                hour=2,
                minute=0,
                id=job_id,
            )
            self.logger.info(f"Added new job '{job_id}'")
        else:
            self.logger.info(f"Job '{job_id}' already exists in database, skipping add")

        return scheduler

    def _check_job_exists_in_db(self, job_id: str) -> bool:
        """Check if a job exists in the APScheduler SQLite database"""
        import sqlite3

        db_path = "jobs.sqlite"
        if not Path(db_path).exists():
            return False

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='apscheduler_jobs'"
            )
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                conn.close()
                return False

            cursor = conn.execute(
                "SELECT COUNT(*) FROM apscheduler_jobs WHERE id = ?", (job_id,)
            )
            count = cursor.fetchone()[0]
            conn.close()

            return count > 0

        except Exception as e:
            self.logger.warning(f"Failed to check job existence in database: {e}")
            return False

    def _remap_video_path(self, emby_path: str) -> str:
        """Remap Emby server path to local machine path"""
        if not self.path_mappings:
            return emby_path

        # Try each mapping to see if it matches the beginning of the path
        for emby_prefix, local_prefix in self.path_mappings.items():
            if emby_path.startswith(emby_prefix):
                # Replace the emby prefix with the local prefix
                remapped_path = emby_path.replace(emby_prefix, local_prefix, 1)
                self.logger.debug(f"Remapped path: {emby_path} -> {remapped_path}")
                return remapped_path

        # If no mapping found, return original path
        self.logger.warning(f"No path mapping found for: {emby_path}")
        return emby_path

    def process_daily_videos(self, days_back: int = 5):
        """Main automation function - processes recent videos"""
        self.logger.info("Starting daily video processing")

        try:
            # Get recently added videos
            recent_videos = self.emby_client.get_recent_videos(days_back=days_back)
            self.logger.info(f"Found {len(recent_videos)} recent videos")

            successful_processes = 0

            for video in recent_videos:
                self.logger.info(
                    f"Processing video: {video.get('Name', 'Unknown')} with id {video['Id']}"
                )
                try:
                    if self._should_process_video(video):
                        success = self._process_single_video(video)
                        if success:
                            successful_processes += 1

                        # Add delay between videos to avoid overwhelming the system
                        time.sleep(2)

                except Exception as e:
                    self.logger.error(
                        f"Failed to process video {video.get('Name', 'Unknown')}: {str(e)}"
                    )
                    self._update_task_status(
                        video["Id"], TaskStatus.FAILED, error=str(e)
                    )

            self.logger.info(
                f"Completed daily processing: {successful_processes}/{len(recent_videos)} successful"
            )

        except Exception as e:
            self.logger.error(f"Daily processing failed: {str(e)}")

    def _process_single_video(self, video: Dict) -> bool:
        """Process individual video with comprehensive error handling"""
        video_id = video["Id"]
        emby_video_path = video["Path"]
        video_path = self._remap_video_path(emby_video_path)
        video_name = video.get("Name", "Unknown")

        self.logger.info(f"Processing video: {video_name}")
        self.logger.info(f"Original path: {emby_video_path}")
        self.logger.info(f"Remapped path: {video_path}")
        self._update_task_status(video_id, TaskStatus.PROCESSING, file_path=video_path)

        temp_dirs = []

        try:
            # Extract frames
            frame_paths = self.frame_extractor.extract_representative_frames(
                video_path, max_frames=5
            )
            if not frame_paths:
                raise ValueError("No frames extracted from video")

            # Track temp directories for cleanup
            temp_dirs.extend([Path(fp[0]).parent for fp in frame_paths])

            # Analyze frames using LMStudio
            tags = self.vision_processor.analyze_frames_sync(
                [fp[0] for fp in frame_paths]
            )

            if tags:
                # Get existing tags and merge with new ones
                existing_tags = video.get("Tags", [])
                all_tags = list(
                    set(existing_tags + tags + ["ai-generated"])
                )  # Add marker tag

                # Update Emby with new tags
                success = self.emby_client.update_video_tags(video_id, all_tags)
                if success:
                    self.logger.info(f"Updated {video_name} with {len(tags)} new tags")
                    self._update_task_status(
                        video_id, TaskStatus.COMPLETED, tag_count=len(tags)
                    )
                else:
                    raise ValueError("Failed to update tags in Emby")
            else:
                raise ValueError("No tags generated from analysis")

            return True

        except Exception as e:
            self.logger.error(f"Processing failed for {video_name}: {str(e)}")
            self._update_task_status(video_id, TaskStatus.FAILED, error=str(e))
            return False

        finally:
            # Clean up temporary files
            for temp_dir in temp_dirs:
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to clean up temp directory {temp_dir}: {e}"
                    )

    def _should_process_video(self, video: Dict) -> bool:
        """Determine if video needs processing"""
        # Check if already processed
        existing_tags = [x["Name"] for x in video.get("TagItems", [])]
        ai_tag_indicators = ["ai-generated", "auto-tagged", "vision-analyzed"]
        if any(
            indicator in tag.lower()
            for tag in existing_tags
            for indicator in ai_tag_indicators
        ):
            self.logger.info(
                f"Skipping {video.get('Name', 'Unknown')} - already processed"
            )
            return False

        # Check file existence and size using remapped path
        emby_video_path = video["Path"]
        video_path = Path(self._remap_video_path(emby_video_path))
        if not video_path.exists():
            self.logger.warning(
                f"Video file not found: {video_path} (original: {emby_video_path})"
            )
            return False

        # Skip very small files (likely not full videos)
        min_size = 10 * 1024 * 1024  # 10MB
        if video_path.stat().st_size < min_size:
            self.logger.info(
                f"Skipping {video.get('Name', 'Unknown')} - file too small"
            )
            return False

        return True

    def _update_task_status(
        self,
        video_id: str,
        status: TaskStatus,
        tag_count: int = 0,
        error: Optional[str] = None,
        file_path: Optional[str] = None,
    ):
        """Update task processing status in database"""
        conn = sqlite3.connect(self.task_tracker)

        try:
            if status == TaskStatus.PROCESSING and file_path:
                conn.execute(
                    "INSERT OR REPLACE INTO video_tasks (emby_id, file_path, status) VALUES (?, ?, ?)",
                    (video_id, file_path, status.value),
                )
            elif status == TaskStatus.COMPLETED:
                conn.execute(
                    "UPDATE video_tasks SET status = ?, completed_at = CURRENT_TIMESTAMP, tag_count = ? WHERE emby_id = ?",
                    (status.value, tag_count, video_id),
                )
            elif status == TaskStatus.FAILED:
                conn.execute(
                    "UPDATE video_tasks SET status = ?, error_message = ? WHERE emby_id = ?",
                    (status.value, error, video_id),
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO video_tasks (emby_id, status) VALUES (?, ?)",
                    (video_id, status.value),
                )

            conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to update task status: {e}")
        finally:
            conn.close()

    def process_single_video_manual(self, video_id: str) -> bool:
        """Manually process a single video by ID"""
        try:
            # Get video details from Emby
            url = f"{self.emby_client.base_url}/emby/Items?Ids={video_id}&Fields=Path,Tags"
            response = self.emby_client.session.get(url)
            response.raise_for_status()
            video = response.json()
            self.logger.info(video["Items"][0])

            return self._process_single_video(video["Items"][0])

        except Exception as e:
            self.logger.error(f"Failed to manually process video {video_id}: {e}")
            return False

    def get_processing_stats(self) -> Dict:
        """Get statistics about processing tasks"""
        conn = sqlite3.connect(self.task_tracker)

        try:
            stats = {}

            # Count by status
            cursor = conn.execute(
                "SELECT status, COUNT(*) FROM video_tasks GROUP BY status"
            )
            stats["by_status"] = dict(cursor.fetchall())

            # Recent activity
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM video_tasks 
                WHERE created_at >= datetime('now', '-7 days')
            """
            )
            stats["last_7_days"] = cursor.fetchone()[0]

            # Total tags generated
            cursor = conn.execute(
                "SELECT SUM(tag_count) FROM video_tasks WHERE status = 'completed'"
            )
            result = cursor.fetchone()[0]
            stats["total_tags"] = result if result else 0

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}
        finally:
            conn.close()

    def run_once(self):
        """Run processing once without scheduling"""
        self.logger.info("Running video tagging once")
        self.process_daily_videos(self.days_back)

    def run_scheduler(self):
        """Start the automation scheduler"""
        scheduler = self._setup_scheduler()
        self.logger.info("Starting video tagging automation scheduler")

        try:
            scheduler.start()
        except KeyboardInterrupt:
            self.logger.info("Automation stopped by user")
            scheduler.shutdown()


def main():
    """Main function to configure and run the video tagging automation"""

    # Load environment variables from .env file
    load_dotenv()

    # Parse path mappings from environment variable
    path_mappings = {}
    path_mappings_env = os.getenv("PATH_MAPPINGS", "")
    if path_mappings_env:
        # Format: "emby_path1:local_path1,emby_path2:local_path2"
        for mapping in path_mappings_env.split(","):
            if ":" in mapping:
                emby_path, local_path = mapping.strip().split(":", 1)
                path_mappings[emby_path.strip()] = local_path.strip()

    # Configuration - Update these with your actual values
    ai_provider = os.getenv("AI_PROVIDER", "lmstudio").lower()

    config = {
        "emby": {
            "server_url": os.getenv("EMBY_SERVER_URL", "http://localhost:8096"),
            "api_key": os.getenv("EMBY_API_KEY", "your-emby-api-key"),
            "user_id": os.getenv("EMBY_USER_ID", "your-user-id"),
        },
        "ai_provider": ai_provider,
        "lmstudio": {
            "model_name": os.getenv(
                "LMSTUDIO_MODEL_NAME", "qwen2.5-vl-7b-instruct-abliterated"
            )
        },
        "ollama": {
            "model_name": os.getenv("OLLAMA_MODEL_NAME", "llava"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        },
        "path_mappings": path_mappings,
        "days_back": int(os.getenv("DAYS_BACK", "5")),
    }

    # Validate configuration
    required_vars = ["EMBY_API_KEY", "EMBY_USER_ID"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(
            f"Error: Missing required environment variables: {', '.join(missing_vars)}"
        )
        print("Please set these environment variables before running the script.")
        return

    # Create automation instance
    automation = VideoTaggingAutomation(config)

    # Check command line arguments
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "once":
            # Run once without scheduling
            automation.run_once()
        elif command == "stats":
            # Show processing statistics
            stats = automation.get_processing_stats()
            print("\nProcessing Statistics:")
            print(f"Status counts: {stats.get('by_status', {})}")
            print(f"Videos processed in last 7 days: {stats.get('last_7_days', 0)}")
            print(f"Total tags generated: {stats.get('total_tags', 0)}")
        elif command == "manual" and len(sys.argv) > 2:
            # Manually process a specific video ID
            video_id = sys.argv[2]
            success = automation.process_single_video_manual(video_id)
            print(
                f"Manual processing {'succeeded' if success else 'failed'} for video {video_id}"
            )
        else:
            print("Usage:")
            print("  python emby_video_tagger.py          # Run scheduled automation")
            print(
                "  python emby_video_tagger.py once     # Run once without scheduling"
            )
            print("  python emby_video_tagger.py stats    # Show processing statistics")
            print(
                "  python emby_video_tagger.py manual <video_id>  # Process specific video"
            )
    else:
        # Run with scheduling
        automation.run_scheduler()


if __name__ == "__main__":
    main()
