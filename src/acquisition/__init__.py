"""Content acquisition engine for extracting product documentation from websites."""

from src.acquisition.browser import BrowserManager
from src.acquisition.engine import ContentAcquisitionEngine
from src.acquisition.site_analyzer import SiteAnalyzer
from src.acquisition.spa_handler import SPAHandler

__all__ = [
    "BrowserManager",
    "ContentAcquisitionEngine",
    "SiteAnalyzer",
    "SPAHandler",
]
