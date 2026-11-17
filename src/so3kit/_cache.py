import os
from pathlib import Path


def cache_dir() -> Path:
    """Directory for precomputed index tables (override with SO3KIT_CACHE)."""
    path = Path(os.environ.get("SO3KIT_CACHE", Path.home() / ".cache" / "so3kit"))
    path.mkdir(parents=True, exist_ok=True)
    return path
