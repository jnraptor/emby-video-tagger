"""Scheduler module for automated video processing."""

from .jobs import setup_scheduler, run_daily_processing

__all__ = ["setup_scheduler", "run_daily_processing"]