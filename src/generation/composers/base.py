"""Base classes for output composers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ...models.pitch import Pitch


class ThemeColor(str, Enum):
    """Pre-defined color themes for presentations."""

    CORPORATE_BLUE = "corporate_blue"
    MODERN_DARK = "modern_dark"
    CLEAN_LIGHT = "clean_light"
    PROFESSIONAL_GREEN = "professional_green"
    EXECUTIVE_GRAY = "executive_gray"


# Color palettes for themes (primary, secondary, accent, background, text)
THEME_PALETTES: dict[ThemeColor, dict[str, str]] = {
    ThemeColor.CORPORATE_BLUE: {
        "primary": "#0066CC",
        "secondary": "#004C99",
        "accent": "#FF6600",
        "background": "#FFFFFF",
        "text": "#333333",
        "text_light": "#666666",
        "highlight": "#E6F2FF",
    },
    ThemeColor.MODERN_DARK: {
        "primary": "#1A1A2E",
        "secondary": "#16213E",
        "accent": "#E94560",
        "background": "#0F0F1A",
        "text": "#FFFFFF",
        "text_light": "#CCCCCC",
        "highlight": "#2D2D44",
    },
    ThemeColor.CLEAN_LIGHT: {
        "primary": "#2C3E50",
        "secondary": "#34495E",
        "accent": "#3498DB",
        "background": "#FAFAFA",
        "text": "#2C3E50",
        "text_light": "#7F8C8D",
        "highlight": "#ECF0F1",
    },
    ThemeColor.PROFESSIONAL_GREEN: {
        "primary": "#1E5631",
        "secondary": "#2E7D32",
        "accent": "#4CAF50",
        "background": "#FFFFFF",
        "text": "#1E3A1E",
        "text_light": "#558855",
        "highlight": "#E8F5E9",
    },
    ThemeColor.EXECUTIVE_GRAY: {
        "primary": "#37474F",
        "secondary": "#455A64",
        "accent": "#FF5722",
        "background": "#FFFFFF",
        "text": "#263238",
        "text_light": "#607D8B",
        "highlight": "#ECEFF1",
    },
}


@dataclass
class ComposerConfig:
    """Base configuration for output composers."""

    # Output settings
    output_path: Optional[Path] = None

    # Theme settings
    theme: ThemeColor = ThemeColor.CORPORATE_BLUE

    # Branding
    company_logo_path: Optional[Path] = None
    footer_text: Optional[str] = None

    # Content options
    include_speaker_notes: bool = True
    include_visual_assets: bool = True
    include_metadata: bool = False

    # Quality settings
    image_quality: int = 85  # JPEG quality for embedded images

    def get_palette(self) -> dict[str, str]:
        """Get the color palette for the current theme."""
        return THEME_PALETTES[self.theme]


@dataclass
class ComposerResult:
    """Result of a composition operation."""

    success: bool
    output_path: Optional[Path] = None
    file_size_bytes: int = 0
    page_count: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseComposer(ABC):
    """Abstract base class for output composers."""

    def __init__(self, config: ComposerConfig):
        self.config = config

    @abstractmethod
    def compose(self, pitch: Pitch, output_path: Optional[Path] = None) -> ComposerResult:
        """
        Compose the pitch into the target format.

        Args:
            pitch: The Pitch object to compose
            output_path: Override output path (uses config.output_path if not provided)

        Returns:
            ComposerResult with status and output information
        """
        pass

    def _resolve_output_path(self, output_path: Optional[Path], extension: str) -> Path:
        """Resolve the output path, creating directories if needed."""
        if output_path:
            path = Path(output_path)
        elif self.config.output_path:
            path = Path(self.config.output_path)
        else:
            path = Path(f"pitch_output{extension}")

        # Ensure correct extension
        if path.suffix.lower() != extension.lower():
            path = path.with_suffix(extension)

        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)

        return path

    def _hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore

    def _load_image_if_exists(self, path: Optional[str]) -> Optional[Path]:
        """Load an image if the path exists."""
        if not path:
            return None
        img_path = Path(path)
        if img_path.exists():
            return img_path
        return None
