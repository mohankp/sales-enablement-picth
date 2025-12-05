"""Pitch output models for generated sales content."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class PitchTone(str, Enum):
    """Tone options for pitch content."""

    PROFESSIONAL = "professional"  # Formal, business-focused
    CONVERSATIONAL = "conversational"  # Friendly, approachable
    TECHNICAL = "technical"  # Detail-oriented, spec-focused
    EXECUTIVE = "executive"  # High-level, ROI-focused
    ENTHUSIASTIC = "enthusiastic"  # Energetic, compelling
    CONSULTATIVE = "consultative"  # Problem-solving focused


class PitchLength(str, Enum):
    """Length options for pitch content."""

    ELEVATOR = "elevator"  # 30-second pitch (~100 words)
    SHORT = "short"  # 2-minute pitch (~300 words)
    STANDARD = "standard"  # 5-minute pitch (~750 words)
    DETAILED = "detailed"  # 10-minute pitch (~1500 words)
    COMPREHENSIVE = "comprehensive"  # Full presentation (~3000+ words)


class SectionType(str, Enum):
    """Types of pitch sections."""

    HOOK = "hook"  # Opening attention-grabber
    PROBLEM = "problem"  # Problem statement
    SOLUTION = "solution"  # Solution overview
    FEATURES = "features"  # Feature highlights
    BENEFITS = "benefits"  # Customer benefits
    USE_CASES = "use_cases"  # Use case examples
    DIFFERENTIATORS = "differentiators"  # Competitive advantages
    SOCIAL_PROOF = "social_proof"  # Testimonials, case studies
    PRICING = "pricing"  # Pricing overview
    CTA = "cta"  # Call to action
    OBJECTION_HANDLING = "objection_handling"  # Common objections
    TECHNICAL = "technical"  # Technical details
    ROI = "roi"  # ROI/value proposition
    CLOSING = "closing"  # Closing statement


# ============================================================================
# Section Models
# ============================================================================


class PitchSection(BaseModel):
    """A single section of the pitch."""

    section_type: SectionType
    title: str = Field(description="Section heading")
    content: str = Field(description="Main section content")
    key_points: list[str] = Field(
        default_factory=list,
        description="Bullet points for this section",
    )
    talking_points: list[str] = Field(
        default_factory=list,
        description="Speaker notes/talking points",
    )
    visual_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested visuals or slides",
    )
    duration_seconds: Optional[int] = Field(
        default=None,
        description="Suggested duration for this section",
    )
    order: int = Field(default=0, description="Section order in pitch")

    def word_count(self) -> int:
        """Get word count for this section."""
        words = len(self.content.split())
        words += sum(len(point.split()) for point in self.key_points)
        return words


class FeatureHighlight(BaseModel):
    """A highlighted feature for the pitch."""

    name: str
    headline: str = Field(description="Catchy one-liner about the feature")
    description: str = Field(description="Brief description")
    benefit: str = Field(description="Key benefit statement")
    proof_point: Optional[str] = Field(
        default=None,
        description="Supporting evidence or metric",
    )


class BenefitStatement(BaseModel):
    """A benefit statement for the pitch."""

    headline: str = Field(description="Benefit headline")
    description: str = Field(description="Expanded benefit description")
    supporting_feature: Optional[str] = Field(
        default=None,
        description="Feature that enables this benefit",
    )
    target_audience: Optional[str] = Field(
        default=None,
        description="Who this benefit resonates with most",
    )


class CompetitivePoint(BaseModel):
    """A competitive differentiation point."""

    claim: str = Field(description="Differentiation claim")
    explanation: str = Field(description="Why this matters")
    compared_to: Optional[str] = Field(
        default=None,
        description="What we're comparing against",
    )


class CallToAction(BaseModel):
    """Call to action for the pitch."""

    primary_cta: str = Field(description="Main call to action")
    secondary_cta: Optional[str] = Field(
        default=None,
        description="Alternative/softer CTA",
    )
    urgency_statement: Optional[str] = Field(
        default=None,
        description="Why act now",
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="Clear next steps",
    )


# ============================================================================
# Pitch Configuration
# ============================================================================


class PitchConfig(BaseModel):
    """Configuration for pitch generation."""

    # Target settings
    target_audience: Optional[str] = Field(
        default=None,
        description="Primary target audience for this pitch",
    )
    tone: PitchTone = PitchTone.PROFESSIONAL
    length: PitchLength = PitchLength.STANDARD

    # Content settings
    include_pricing: bool = True
    include_technical: bool = False
    include_competitors: bool = True
    max_features: int = Field(default=5, ge=1, le=10)
    max_benefits: int = Field(default=5, ge=1, le=10)
    max_use_cases: int = Field(default=3, ge=1, le=5)

    # Section selection (None means use length-based defaults)
    sections_to_include: Optional[list[SectionType]] = Field(
        default=None,
        description="Custom section selection (None uses length-based defaults)",
    )

    # Style settings
    use_statistics: bool = True
    use_questions: bool = True  # Rhetorical questions
    use_stories: bool = False  # Narrative elements
    formality_level: int = Field(
        default=3,
        ge=1,
        le=5,
        description="1=very casual, 5=very formal",
    )

    # Custom elements
    company_name: Optional[str] = Field(
        default=None,
        description="Prospect company name for personalization",
    )
    industry: Optional[str] = Field(
        default=None,
        description="Prospect industry for relevance",
    )
    pain_points: list[str] = Field(
        default_factory=list,
        description="Known pain points to address",
    )


# ============================================================================
# Main Pitch Model
# ============================================================================


class Pitch(BaseModel):
    """
    Complete generated sales pitch.

    Contains all sections, highlights, and metadata for a
    sales pitch document.
    """

    # Identity
    product_name: str
    product_url: str
    pitch_id: str = Field(default="")

    # Configuration used
    config: PitchConfig = Field(default_factory=PitchConfig)

    # Core content
    title: str = Field(description="Pitch title/headline")
    subtitle: Optional[str] = Field(
        default=None,
        description="Pitch subtitle or tagline",
    )
    executive_summary: str = Field(
        description="Brief pitch summary (1-2 sentences)",
    )

    # Sections
    sections: list[PitchSection] = Field(default_factory=list)

    # Structured highlights
    feature_highlights: list[FeatureHighlight] = Field(default_factory=list)
    benefit_statements: list[BenefitStatement] = Field(default_factory=list)
    competitive_points: list[CompetitivePoint] = Field(default_factory=list)
    call_to_action: Optional[CallToAction] = None

    # Quick reference
    elevator_pitch: str = Field(
        default="",
        description="30-second elevator pitch version",
    )
    one_liner: str = Field(
        default="",
        description="Single sentence product description",
    )
    key_messages: list[str] = Field(
        default_factory=list,
        description="3-5 key messages to remember",
    )

    # Objection handling
    common_objections: dict[str, str] = Field(
        default_factory=dict,
        description="Objection -> Response mapping",
    )

    # Metadata
    source_processed_id: str = Field(
        default="",
        description="ID of source ProcessedContent",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    generation_duration_ms: int = 0
    total_llm_tokens_used: int = 0
    total_llm_cost_usd: float = 0.0

    # Quality
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)

    def get_section(self, section_type: SectionType) -> Optional[PitchSection]:
        """Get a specific section by type."""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None

    def get_full_content(self) -> str:
        """Get all sections as a single text document."""
        parts = [f"# {self.title}"]
        if self.subtitle:
            parts.append(f"*{self.subtitle}*")
        parts.append("")
        parts.append(self.executive_summary)
        parts.append("")

        for section in sorted(self.sections, key=lambda s: s.order):
            parts.append(f"## {section.title}")
            parts.append(section.content)
            if section.key_points:
                parts.append("")
                for point in section.key_points:
                    parts.append(f"- {point}")
            parts.append("")

        return "\n".join(parts)

    def word_count(self) -> int:
        """Get total word count."""
        count = len(self.executive_summary.split())
        for section in self.sections:
            count += section.word_count()
        return count

    def estimated_duration_minutes(self) -> float:
        """Estimate presentation duration (150 words/minute)."""
        return self.word_count() / 150

    def to_presentation_outline(self) -> list[dict[str, Any]]:
        """Convert to presentation slide outline."""
        slides = []

        # Title slide
        slides.append({
            "slide_type": "title",
            "title": self.title,
            "subtitle": self.subtitle or self.executive_summary,
        })

        # Content slides
        for section in sorted(self.sections, key=lambda s: s.order):
            slides.append({
                "slide_type": "content",
                "title": section.title,
                "bullets": section.key_points or [section.content],
                "notes": section.talking_points,
                "visuals": section.visual_suggestions,
            })

        # CTA slide
        if self.call_to_action:
            slides.append({
                "slide_type": "cta",
                "title": "Next Steps",
                "primary_cta": self.call_to_action.primary_cta,
                "next_steps": self.call_to_action.next_steps,
            })

        return slides


class PitchVariant(BaseModel):
    """A variant of a pitch for a specific audience."""

    audience: str
    pitch: Pitch
    customizations: list[str] = Field(
        default_factory=list,
        description="What was customized for this audience",
    )


class PitchSet(BaseModel):
    """Collection of pitch variants for different audiences."""

    product_name: str
    base_pitch: Pitch
    variants: list[PitchVariant] = Field(default_factory=list)

    def get_variant(self, audience: str) -> Optional[Pitch]:
        """Get pitch variant for specific audience."""
        for variant in self.variants:
            if variant.audience.lower() == audience.lower():
                return variant.pitch
        return None
