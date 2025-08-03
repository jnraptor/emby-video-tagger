# Video Copy Implementation Plan

## Overview
Add functionality to copy favorite videos to a destination folder when `_process_single_video` is called with `source_type="favorites"`.

## Environment Variable
- `COPY_FAVORITES_TO`: Path to destination folder (empty means disabled)
- Flat structure: all files copied directly to destination folder

## Implementation Steps

### 1. Configuration Changes

**File: `emby_video_tagger.py`**

**Location: Around line 972 (in main function config dictionary)**
```python
# Add after line 972:
"copy_favorites_to": os.getenv("COPY_FAVORITES_TO", "").strip(),
```

**Location: Around line 527 (in VideoTaggingAutomation.__init__)**
```python
# Add after line 527:
self.copy_favorites_to = config.get("copy_favorites_to", "")
```

### 2. Copy Method Implementation

**Location: Add new method to VideoTaggingAutomation class (around line 744)**
```python
def _copy_favorite_video(self, video_path: str, video_name: str) -> bool:
    """Copy favorite video to destination folder if configured"""
    if not self.copy_favorites_to:
        return True  # No copy destination configured, skip silently
    
    try:
        # Ensure destination directory exists
        dest_dir = Path(self.copy_favorites_to)
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Get source file info
        source_path = Path(video_path)
        if not source_path.exists():
            self.logger.warning(f"Source video file not found for copy: {video_path}")
            return False
        
        # Create destination path (flat structure)
        dest_path = dest_dir / source_path.name
        
        # Check if file already exists
        if dest_path.exists():
            # Compare file sizes to determine if it's the same file
            if dest_path.stat().st_size == source_path.stat().st_size:
                self.logger.info(f"Video already exists in destination, skipping copy: {dest_path}")
                return True
            else:
                self.logger.warning(f"File exists but different size, overwriting: {dest_path}")
        
        # Copy the file
        self.logger.info(f"Copying favorite video: {source_path} -> {dest_path}")
        shutil.copy2(source_path, dest_path)
        self.logger.info(f"Successfully copied favorite video to: {dest_path}")
        return True
        
    except Exception as e:
        self.logger.error(f"Failed to copy favorite video {video_name}: {str(e)}")
        return False
```

### 3. Integration in _process_single_video

**Location: Around line 727 (after successful tag update)**
```python
# Add after line 727 (after successful tag update):
                    
                    # Copy favorite video if configured and source_type is favorites
                    if source_type == "favorites":
                        copy_success = self._copy_favorite_video(video_path, video_name)
                        if not copy_success:
                            self.logger.warning(f"Tag update succeeded but copy failed for {video_name}")
```

### 4. Update .env.example

**File: `.env.example`**

**Location: Add at the end**
```env
# Favorites Copy Configuration
# Copy favorite videos to this folder (leave empty to disable)
COPY_FAVORITES_TO=
```

## Error Handling Considerations

1. **File System Errors**: Handle permissions, disk space, network issues
2. **Path Issues**: Handle invalid paths, missing directories
3. **Duplicate Files**: Check file existence and size comparison
4. **Logging**: Comprehensive logging for debugging and monitoring
5. **Non-blocking**: Copy failures should not prevent tag processing

## Testing Scenarios

1. **Basic Copy**: Favorite video copied successfully to destination
2. **Duplicate Handling**: Same file already exists (skip)
3. **Different File Same Name**: Different size file exists (overwrite)
4. **Missing Destination**: Destination folder doesn't exist (create)
5. **Copy Disabled**: COPY_FAVORITES_TO not set (skip silently)
6. **Copy Failure**: Handle various error conditions gracefully
7. **Non-Favorites**: Regular videos should not be copied

## Implementation Notes

- Use `shutil.copy2()` to preserve file metadata
- Use `Path` objects for cross-platform compatibility
- Flat directory structure as requested
- Copy happens after successful tag processing
- Copy failures are logged but don't affect tag processing success
- File size comparison for duplicate detection (simple but effective)