"""Refinement history tracking with persistence."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.models.pitch import Pitch

from .models import RefinementRequest, RefinementResult, RefinementType

logger = logging.getLogger(__name__)


class RefinementHistoryEntry(BaseModel):
    """A single entry in the refinement history."""

    # Request info
    instruction: str
    refinement_type: RefinementType
    target_description: str  # e.g., "section 'hook'" or "overall pitch"

    # Result summary
    success: bool
    changes_summary: list[str]
    rationale: str

    # Pitch snapshot (for undo)
    pitch_before: Pitch
    pitch_after: Pitch

    # Metrics
    tokens_used: int = 0
    cost_usd: float = 0.0

    # Timestamps
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # State tracking
    is_undone: bool = False  # True if this refinement was undone

    def to_summary(self) -> str:
        """Get a one-line summary of this entry."""
        status = "(undone)" if self.is_undone else ""
        changes = len(self.changes_summary)
        return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] {self.refinement_type.value}: \"{self.instruction[:50]}...\" ({changes} changes) {status}".strip()


class RefinementHistory(BaseModel):
    """
    Manages refinement history with undo/redo support.

    History is persisted to a JSON file alongside the pitch file.
    """

    # Identity
    pitch_id: str = Field(description="ID of the original pitch")
    product_name: str = ""

    # History entries
    entries: list[RefinementHistoryEntry] = Field(default_factory=list)

    # Current position in history (for undo/redo)
    current_index: int = Field(
        default=-1,
        description="Index of current state (-1 means original, 0+ means after that many refinements)",
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # File path (set when loaded/saved)
    _file_path: Optional[Path] = None

    @classmethod
    def auto_history_path(cls, pitch_path: Path) -> Path:
        """Get the history file path for a pitch file."""
        if pitch_path.suffix == ".json":
            return pitch_path.with_suffix(".history.json")
        return Path(str(pitch_path) + ".history.json")

    @classmethod
    def load(cls, path: Path) -> Optional["RefinementHistory"]:
        """Load history from a file."""
        if not path.exists():
            logger.debug(f"No history file found at {path}")
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            history = cls.model_validate(data)
            history._file_path = path
            logger.info(f"Loaded refinement history with {len(history.entries)} entries")
            return history
        except Exception as e:
            logger.warning(f"Failed to load history from {path}: {e}")
            return None

    @classmethod
    def load_for_pitch(cls, pitch_path: Path) -> Optional["RefinementHistory"]:
        """Load history for a pitch file."""
        history_path = cls.auto_history_path(pitch_path)
        return cls.load(history_path)

    @classmethod
    def create_for_pitch(cls, pitch: Pitch, pitch_path: Optional[Path] = None) -> "RefinementHistory":
        """Create a new history for a pitch."""
        history = cls(
            pitch_id=pitch.pitch_id,
            product_name=pitch.product_name,
        )
        if pitch_path:
            history._file_path = cls.auto_history_path(pitch_path)
        return history

    def save(self, path: Optional[Path] = None) -> Path:
        """Save history to a file."""
        save_path = path or self._file_path
        if not save_path:
            raise ValueError("No path specified and no default path set")

        self.updated_at = datetime.now(timezone.utc)

        with open(save_path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

        self._file_path = save_path
        logger.info(f"Saved refinement history to {save_path}")
        return save_path

    def add(self, request: RefinementRequest, result: RefinementResult) -> None:
        """Add a new refinement to history."""
        # If we're not at the end, truncate future entries
        if self.current_index < len(self.entries) - 1:
            self.entries = self.entries[: self.current_index + 1]

        entry = RefinementHistoryEntry(
            instruction=request.instruction,
            refinement_type=request.refinement_type,
            target_description=request.get_target_description(),
            success=result.success,
            changes_summary=result.changes_summary,
            rationale=result.refinement_rationale,
            pitch_before=result.original_pitch,
            pitch_after=result.refined_pitch,
            tokens_used=result.tokens_used,
            cost_usd=result.cost_usd,
        )

        self.entries.append(entry)
        self.current_index = len(self.entries) - 1
        self.updated_at = datetime.now(timezone.utc)

        logger.debug(f"Added history entry: {entry.to_summary()}")

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self.current_index >= 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self.current_index < len(self.entries) - 1

    def undo(self) -> Optional[Pitch]:
        """
        Undo the last refinement.

        Returns:
            The pitch state before the undone refinement, or None if can't undo
        """
        if not self.can_undo():
            logger.debug("Nothing to undo")
            return None

        # Mark current entry as undone
        current_entry = self.entries[self.current_index]
        current_entry.is_undone = True

        # Move back in history
        self.current_index -= 1

        # Return the pitch before the undone refinement
        pitch = current_entry.pitch_before
        logger.info(f"Undone: {current_entry.instruction[:50]}...")

        return pitch

    def redo(self) -> Optional[Pitch]:
        """
        Redo the last undone refinement.

        Returns:
            The pitch state after the redone refinement, or None if can't redo
        """
        if not self.can_redo():
            logger.debug("Nothing to redo")
            return None

        # Move forward in history
        self.current_index += 1

        # Mark entry as no longer undone
        entry = self.entries[self.current_index]
        entry.is_undone = False

        # Return the pitch after the redone refinement
        pitch = entry.pitch_after
        logger.info(f"Redone: {entry.instruction[:50]}...")

        return pitch

    def get_current_pitch(self, original_pitch: Pitch) -> Pitch:
        """
        Get the current pitch state based on history position.

        Args:
            original_pitch: The original pitch (before any refinements)

        Returns:
            The current pitch state
        """
        if self.current_index < 0:
            return original_pitch

        return self.entries[self.current_index].pitch_after

    def get_context_for_llm(self, max_entries: int = 5) -> str:
        """
        Get a summary of recent history for LLM context.

        Args:
            max_entries: Maximum number of entries to include

        Returns:
            A text summary of recent refinements
        """
        if not self.entries:
            return "No previous refinements."

        # Get recent active (not undone) entries
        active_entries = [e for e in self.entries if not e.is_undone]
        recent = active_entries[-max_entries:]

        if not recent:
            return "No active refinements (all have been undone)."

        lines = ["Previous refinements:"]
        for i, entry in enumerate(recent, 1):
            lines.append(f"{i}. [{entry.refinement_type.value}] {entry.instruction}")
            if entry.changes_summary:
                for change in entry.changes_summary[:2]:
                    lines.append(f"   - {change}")

        return "\n".join(lines)

    def get_summary(self) -> dict:
        """Get a summary of the history state."""
        active_count = sum(1 for e in self.entries if not e.is_undone)
        total_tokens = sum(e.tokens_used for e in self.entries)
        total_cost = sum(e.cost_usd for e in self.entries)

        return {
            "total_entries": len(self.entries),
            "active_entries": active_count,
            "undone_entries": len(self.entries) - active_count,
            "current_index": self.current_index,
            "can_undo": self.can_undo(),
            "can_redo": self.can_redo(),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
        }

    def list_entries(self) -> list[str]:
        """Get a list of entry summaries."""
        return [entry.to_summary() for entry in self.entries]

    def clear(self) -> None:
        """Clear all history."""
        self.entries = []
        self.current_index = -1
        self.updated_at = datetime.now(timezone.utc)
        logger.info("Cleared refinement history")
