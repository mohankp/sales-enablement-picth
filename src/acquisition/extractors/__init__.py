"""Content extractors for different types of web content."""

from src.acquisition.extractors.text import TextExtractor
from src.acquisition.extractors.images import ImageExtractor
from src.acquisition.extractors.tables import TableExtractor
from src.acquisition.extractors.base import BaseExtractor

__all__ = [
    "BaseExtractor",
    "TextExtractor",
    "ImageExtractor",
    "TableExtractor",
]
