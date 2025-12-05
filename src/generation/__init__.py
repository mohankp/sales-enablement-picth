"""Pitch generation module for creating sales pitches from processed content."""

from .generator import PitchGenerator, GenerationConfig, GenerationResult
from .templates import (
    SectionTemplate,
    SECTION_TEMPLATES,
    TONE_GUIDELINES,
    LENGTH_GUIDELINES,
    get_section_template,
    get_tone_guidelines,
    get_length_config,
    get_sections_for_length,
)
from .visual_matcher import SectionVisualMatcher

__all__ = [
    # Generator
    "PitchGenerator",
    "GenerationConfig",
    "GenerationResult",
    # Templates
    "SectionTemplate",
    "SECTION_TEMPLATES",
    "TONE_GUIDELINES",
    "LENGTH_GUIDELINES",
    "get_section_template",
    "get_tone_guidelines",
    "get_length_config",
    "get_sections_for_length",
    # Visual matching
    "SectionVisualMatcher",
]
