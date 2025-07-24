#!/usr/bin/env python
"""Test runner script for Emby Video Tagger."""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run the test suite with coverage."""
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    
    # Test command
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "--cov=emby_video_tagger",
        "--cov-report=term-missing",
        "--cov-report=html",
        "tests/"
    ]
    
    print("Running tests with coverage...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run tests
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
        print("\nCoverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

def run_specific_test(test_path):
    """Run a specific test file or test case."""
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        test_path
    ]
    
    print(f"Running specific test: {test_path}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode != 0:
        sys.exit(1)

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Run specific test
        run_specific_test(sys.argv[1])
    else:
        # Run all tests
        run_tests()

if __name__ == "__main__":
    main()