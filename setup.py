"""Setup configuration for Emby Video Tagger."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="emby-video-tagger",
    version="2.0.0",
    author="KiloCode",
    description="Automated video tagging for Emby media server using AI vision analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/emby-video-tagger",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.32.0",
        "aiohttp>=3.9.0",
        "opencv-python>=4.12.0",
        "scenedetect>=0.6.6",
        "numpy>=2.2.0",
        "lmstudio>=1.4.1",
        "ollama>=0.5.1",
        "sqlalchemy>=2.0.0",
        "aiosqlite>=0.19.0",
        "apscheduler>=3.11.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "structlog>=24.0.0",
        "psutil>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=24.0.0",
            "ruff>=0.1.0",
            "mypy>=1.8.0",
            "types-aiofiles",
            "types-python-dateutil",
        ],
        "test": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "emby-video-tagger=emby_video_tagger.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "emby_video_tagger": ["py.typed"],
    },
)