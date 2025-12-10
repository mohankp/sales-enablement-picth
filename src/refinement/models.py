"""Data models for the refinement engine."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.models.pitch import Pitch, PitchSection, PitchTone, SectionType
from src.llm import ProviderType


class RefinementType(str, Enum):
    """Types of refinement operations."""

    TONE = "tone"  # Change overall tone
    SECTION = "section"  # Modify specific section
    LENGTH = "length"  # Expand/condense content
    AUDIENCE = "audience"  # Adjust for different audience
    FEATURE = "feature"  # Highlight/modify feature emphasis
    STYLE = "style"  # Adjust writing style
    CUSTOM = "custom"  # Free-form refinement


class LengthDirection(str, Enum):
    """Direction for length refinement."""

    EXPAND = "expand"
    CONDENSE = "condense"
    MAINTAIN = "maintain"


class RefinementRequest(BaseModel):
    """A request to refine a pitch."""

    # Core instruction
    instruction: str = Field(
        description="Natural language instruction for the refinement"
    )

    # Refinement type (auto-detected if not specified)
    refinement_type: RefinementType = Field(
        default=RefinementType.CUSTOM,
        description="Type of refinement operation",
    )

    # Target (for section-specific refinements)
    target_section: Optional[SectionType] = Field(
        default=None,
        description="Specific section to refine (for section-type refinements)",
    )

    # Tone adjustment
    target_tone: Optional[PitchTone] = Field(
        default=None,
        description="Target tone for tone refinements",
    )

    # Length adjustment
    length_direction: Optional[LengthDirection] = Field(
        default=None,
        description="Direction for length adjustments",
    )

    # Audience adjustment
    target_audience: Optional[str] = Field(
        default=None,
        description="Target audience for audience refinements",
    )

    # Constraints
    preserve_sections: list[SectionType] = Field(
        default_factory=list,
        description="Sections to leave unchanged",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Additional constraints for the refinement",
    )

    # Metadata
    request_id: str = Field(
        default="",
        description="Unique identifier for this request",
    )

    def get_target_description(self) -> str:
        """Get a human-readable description of the refinement target."""
        if self.target_section:
            return f"section '{self.target_section.value}'"
        elif self.refinement_type == RefinementType.TONE:
            return f"tone to {self.target_tone.value if self.target_tone else 'new style'}"
        elif self.refinement_type == RefinementType.AUDIENCE:
            return f"audience to {self.target_audience or 'new audience'}"
        elif self.refinement_type == RefinementType.LENGTH:
            direction = self.length_direction.value if self.length_direction else "adjust"
            return f"length ({direction})"
        else:
            return "overall pitch"


class SectionChange(BaseModel):
    """Describes a change to a specific section."""

    section_type: SectionType
    field_changed: str  # e.g., "content", "key_points", "title"
    change_description: str
    original_value: Optional[str] = None
    new_value: Optional[str] = None


class RefinementResult(BaseModel):
    """Result of a refinement operation."""

    # Success status
    success: bool = True
    error: Optional[str] = None

    # Pitches
    original_pitch: Pitch
    refined_pitch: Pitch

    # Change tracking
    changes_summary: list[str] = Field(
        default_factory=list,
        description="List of changes made",
    )
    section_changes: list[SectionChange] = Field(
        default_factory=list,
        description="Detailed section-level changes",
    )
    refinement_rationale: str = Field(
        default="",
        description="Explanation of why changes were made",
    )

    # Metadata
    refinement_type: RefinementType = RefinementType.CUSTOM
    instruction: str = ""

    # Metrics
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    # Timestamps
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def get_changed_sections(self) -> list[SectionType]:
        """Get list of sections that were modified."""
        return list(set(c.section_type for c in self.section_changes))

    def has_changes(self) -> bool:
        """Check if any changes were made."""
        return len(self.changes_summary) > 0 or len(self.section_changes) > 0


class RefinementConfig(BaseModel):
    """Configuration for the refinement engine."""

    # Provider settings
    provider: ProviderType = Field(
        default=ProviderType.GEMINI,
        description="LLM provider to use",
    )
    model: str = Field(
        default="gemini-2.0-flash",
        description="Model to use for refinements",
    )

    # Generation settings
    max_tokens: int = Field(
        default=8192,
        description="Maximum tokens for LLM responses",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )

    # History settings
    max_history: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum refinement history entries to keep",
    )
    auto_save_history: bool = Field(
        default=True,
        description="Automatically save history after each refinement",
    )

    # Behavior settings
    preserve_structure: bool = Field(
        default=True,
        description="Preserve section order and structure",
    )
    preserve_metadata: bool = Field(
        default=True,
        description="Preserve pitch metadata (IDs, timestamps, etc.)",
    )

    # Rate limiting
    requests_per_minute: Optional[int] = Field(
        default=None,
        description="Rate limit for LLM requests",
    )
    timeout_seconds: float = Field(
        default=120.0,
        description="Timeout for LLM requests",
    )

    # Logging
    verbose: bool = False
    log_requests: bool = False
    log_responses: bool = False


@dataclass
class RefinementContext:
    """Internal context for refinement operations."""

    pitch: Pitch
    request: RefinementRequest
    config: RefinementConfig
    history_context: str = ""

    def to_prompt_context(self) -> dict[str, Any]:
        """Convert to dictionary for prompt template substitution."""
        return {
            "product_name": self.pitch.product_name,
            "current_tone": self.pitch.config.tone.value,
            "current_audience": self.pitch.config.target_audience or "general",
            "instruction": self.request.instruction,
            "refinement_type": self.request.refinement_type.value,
            "target_section": (
                self.request.target_section.value
                if self.request.target_section
                else None
            ),
            "target_tone": (
                self.request.target_tone.value if self.request.target_tone else None
            ),
            "target_audience": self.request.target_audience,
            "constraints": "\n".join(self.request.constraints) if self.request.constraints else "None",
            "preserve_sections": ", ".join(
                s.value for s in self.request.preserve_sections
            ) if self.request.preserve_sections else "None",
            "history_context": self.history_context,
        }
