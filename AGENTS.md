# Build/Lint/Test Commands
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python emby_video_tagger.py

# Test specific functionality
python emby_video_tagger.py manual <video_id>
python emby_video_tagger.py once
python emby_video_tagger.py stats
```

# Linting / Type Checking (LSP)
This project uses `ruff` (linting) and `ty` (type checking) as the LSP toolchain. Both are installed system-wide at `~/.local/bin/` and require no venv install. Run them before committing changes; both must report zero errors.

```bash
# Lint check (unused imports, bare except, f-string warnings, etc.)
ruff check emby_video_tagger.py

# Type check (PIL attribute lookups, parameter defaults, return types, etc.)
ty check emby_video_tagger.py
```

Conventions enforced by the LSP setup:
- **No unused imports** (ruff F401)
- **No bare `except:`** (ruff E722) — use `except ValueError:` or another specific type
- **No f-strings without placeholders** (ruff F541) — drop the `f` prefix on static strings
- **Optional defaults typed explicitly** — use `Optional[List[str]] = None`, not `List[str] = None` (ty `invalid-parameter-default`)
- **Pillow 10+ attribute path** — use `Image.Resampling.LANCZOS`, not `Image.LANCZOS` (ty `unresolved-attribute`)
- **Typing `Any` is uppercase** — `Dict[str, Any]`, not `Dict[str, any]` (ty `invalid-type-form`)

# Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports with blank lines
- **Formatting**: Follow PEP 8, use 4-space indentation, max 100 char lines
- **Types**: Use type hints consistently (List, Dict, Optional, etc.)
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPER_CASE for constants
- **Error Handling**: Use try-except blocks with specific exception types and logging
- **Logging**: Use logging module with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **Classes**: Use factory patterns for provider selection, abstract base classes for interfaces