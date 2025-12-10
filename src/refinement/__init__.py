"""
Refinement Engine for iterative pitch improvement.

This module provides a conversational interface for refining generated pitches
through natural language feedback, with support for:
- Section-specific refinements
- Tone and style adjustments
- Length modifications
- Audience adaptation
- Undo/redo with persistent history

Usage:
    from src.refinement import RefinementEngine, RefinementConfig, RefinementRequest

    config = RefinementConfig()
    async with RefinementEngine(config) as engine:
        result = await engine.refine(
            pitch,
            RefinementRequest(instruction="make it more technical")
        )
        print(result.changes_summary)
"""

from .models import (
    RefinementConfig,
    RefinementRequest,
    RefinementResult,
    RefinementType,
)
from .engine import RefinementEngine
from .history import RefinementHistory, RefinementHistoryEntry

__all__ = [
    # Core
    "RefinementEngine",
    "RefinementConfig",
    # Request/Response
    "RefinementRequest",
    "RefinementResult",
    "RefinementType",
    # History
    "RefinementHistory",
    "RefinementHistoryEntry",
]
