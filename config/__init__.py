"""
config package
Exposes settings and logging setup at package level.
"""

from config.settings import settings
from config.logging_config import setup_logging

__all__ = ["settings", "setup_logging"]
