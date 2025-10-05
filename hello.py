#!/usr/bin/env python3
"""
Simple Hello World script for the emby-video-tagger project.

Usage:
    python hello.py            # prints "Hello, World!"
    python hello.py Alice      # prints "Hello, Alice!"
"""

from __future__ import annotations
import sys

def hello(name: str = "World") -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}!"

def sum(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Return the difference of two integers."""
    return a - b

def main(argv: list[str] | None = None) -> None:
    """Entry point for the script."""
    argv = argv if argv is not None else sys.argv[1:]
    name = argv[0] if argv else "World"
    print(hello(name))

if __name__ == "__main__":
    main()
