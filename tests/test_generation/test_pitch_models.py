"""Tests for pitch output models."""

import pytest
from datetime import datetime, timezone

from src.models.pitch import (
    BenefitStatement,
    CallToAction,
    CompetitivePoint,
    FeatureHighlight,
    Pitch,
    PitchConfig,
    PitchLength,
    PitchSection,
    PitchSet,
    PitchTone,
    PitchVariant,
    SectionType,
)


class TestPitchTone:
    """Tests for PitchTone enum."""

    def test_all_tones_are_strings(self):
        """All tones are string enums."""
        for tone in PitchTone:
            assert isinstance(tone.value, str)

    def test_tone_values(self):
        """Verify expected tone values."""
        assert PitchTone.PROFESSIONAL.value == "professional"
        assert PitchTone.TECHNICAL.value == "technical"
        assert PitchTone.EXECUTIVE.value == "executive"


class TestPitchLength:
    """Tests for PitchLength enum."""

    def test_all_lengths_are_strings(self):
        """All lengths are string enums."""
        for length in PitchLength:
            assert isinstance(length.value, str)

    def test_length_values(self):
        """Verify expected length values."""
        assert PitchLength.ELEVATOR.value == "elevator"
        assert PitchLength.STANDARD.value == "standard"
        assert PitchLength.COMPREHENSIVE.value == "comprehensive"


class TestSectionType:
    """Tests for SectionType enum."""

    def test_has_core_sections(self):
        """Core section types exist."""
        assert hasattr(SectionType, "HOOK")
        assert hasattr(SectionType, "PROBLEM")
        assert hasattr(SectionType, "SOLUTION")
        assert hasattr(SectionType, "CTA")

    def test_has_content_sections(self):
        """Content section types exist."""
        assert hasattr(SectionType, "FEATURES")
        assert hasattr(SectionType, "BENEFITS")
        assert hasattr(SectionType, "USE_CASES")
        assert hasattr(SectionType, "DIFFERENTIATORS")


class TestPitchSection:
    """Tests for PitchSection model."""

    def test_create_section(self):
        """Create a pitch section."""
        section = PitchSection(
            section_type=SectionType.HOOK,
            title="Opening Hook",
            content="Attention-grabbing content here.",
            key_points=["Point 1", "Point 2"],
            order=1,
        )
        assert section.section_type == SectionType.HOOK
        assert section.title == "Opening Hook"
        assert len(section.key_points) == 2
        assert section.order == 1

    def test_section_word_count(self):
        """Word count calculation."""
        section = PitchSection(
            section_type=SectionType.SOLUTION,
            title="Solution",
            content="This is a five word sentence.",  # 6 words
            key_points=["Three word point", "Four more words here"],  # 3 + 4 = 7 words
            order=3,
        )
        assert section.word_count() == 13  # 6 + 7

    def test_section_with_optional_fields(self):
        """Section with all optional fields."""
        section = PitchSection(
            section_type=SectionType.FEATURES,
            title="Features",
            content="Content",
            key_points=["Point 1"],
            talking_points=["Talk about this", "Mention that"],
            visual_suggestions=["Show diagram", "Display screenshot"],
            duration_seconds=120,
            order=4,
        )
        assert len(section.talking_points) == 2
        assert len(section.visual_suggestions) == 2
        assert section.duration_seconds == 120


class TestFeatureHighlight:
    """Tests for FeatureHighlight model."""

    def test_create_feature_highlight(self):
        """Create a feature highlight."""
        feature = FeatureHighlight(
            name="AI Assistant",
            headline="Your intelligent co-pilot",
            description="AI-powered assistant for everyday tasks",
            benefit="Save hours of manual work",
            proof_point="95% accuracy rate",
        )
        assert feature.name == "AI Assistant"
        assert feature.proof_point == "95% accuracy rate"

    def test_feature_without_proof_point(self):
        """Feature without proof point."""
        feature = FeatureHighlight(
            name="Feature",
            headline="Headline",
            description="Description",
            benefit="Benefit",
        )
        assert feature.proof_point is None


class TestBenefitStatement:
    """Tests for BenefitStatement model."""

    def test_create_benefit_statement(self):
        """Create a benefit statement."""
        benefit = BenefitStatement(
            headline="Increase productivity by 50%",
            description="Streamlined workflows and automation",
            supporting_feature="AI Automation",
            target_audience="Operations teams",
        )
        assert benefit.headline == "Increase productivity by 50%"
        assert benefit.supporting_feature == "AI Automation"


class TestCompetitivePoint:
    """Tests for CompetitivePoint model."""

    def test_create_competitive_point(self):
        """Create a competitive point."""
        point = CompetitivePoint(
            claim="Only platform with real-time collaboration",
            explanation="Competitors require manual syncing",
            compared_to="Traditional project management tools",
        )
        assert "real-time" in point.claim
        assert point.compared_to is not None


class TestCallToAction:
    """Tests for CallToAction model."""

    def test_create_cta(self):
        """Create a call to action."""
        cta = CallToAction(
            primary_cta="Start your free trial",
            secondary_cta="Schedule a demo",
            urgency_statement="Limited time offer",
            next_steps=["Sign up", "Configure settings", "Invite team"],
        )
        assert cta.primary_cta == "Start your free trial"
        assert len(cta.next_steps) == 3

    def test_cta_minimal(self):
        """CTA with only primary action."""
        cta = CallToAction(primary_cta="Learn more")
        assert cta.primary_cta == "Learn more"
        assert cta.secondary_cta is None
        assert cta.urgency_statement is None
        assert len(cta.next_steps) == 0


class TestPitchConfig:
    """Tests for PitchConfig model."""

    def test_default_config(self):
        """Default configuration values."""
        config = PitchConfig()
        assert config.tone == PitchTone.PROFESSIONAL
        assert config.length == PitchLength.STANDARD
        assert config.include_pricing is True
        assert config.max_features == 5
        assert config.sections_to_include is None  # None means use length-based defaults

    def test_custom_config(self):
        """Custom configuration."""
        config = PitchConfig(
            target_audience="CTOs",
            tone=PitchTone.EXECUTIVE,
            length=PitchLength.ELEVATOR,
            include_pricing=False,
            max_features=3,
        )
        assert config.target_audience == "CTOs"
        assert config.tone == PitchTone.EXECUTIVE
        assert config.include_pricing is False

    def test_config_with_sections(self):
        """Configuration with custom sections."""
        config = PitchConfig(
            sections_to_include=[
                SectionType.HOOK,
                SectionType.SOLUTION,
                SectionType.CTA,
            ]
        )
        assert len(config.sections_to_include) == 3

    def test_config_with_pain_points(self):
        """Configuration with custom pain points."""
        config = PitchConfig(
            pain_points=["Manual processes", "Data silos", "Slow reporting"]
        )
        assert len(config.pain_points) == 3


class TestPitch:
    """Tests for Pitch model."""

    @pytest.fixture
    def sample_pitch(self):
        """Create a sample pitch."""
        return Pitch(
            product_name="TestProduct",
            product_url="https://example.com/product",
            pitch_id="pitch-123",
            title="TestProduct: Transform Your Workflow",
            subtitle="Work smarter, not harder",
            executive_summary="TestProduct helps teams automate workflows.",
            sections=[
                PitchSection(
                    section_type=SectionType.HOOK,
                    title="Opening",
                    content="What if you could save 10 hours a week?",
                    key_points=["Time savings", "Automation"],
                    order=1,
                ),
                PitchSection(
                    section_type=SectionType.SOLUTION,
                    title="The Solution",
                    content="TestProduct automates your repetitive tasks.",
                    key_points=["AI-powered", "Easy setup"],
                    order=3,
                ),
            ],
            elevator_pitch="TestProduct automates workflows so you can focus on what matters.",
            one_liner="AI-powered workflow automation",
            key_messages=["Save time", "Reduce errors", "Scale easily"],
        )

    def test_pitch_creation(self, sample_pitch):
        """Create a pitch."""
        assert sample_pitch.product_name == "TestProduct"
        assert len(sample_pitch.sections) == 2
        assert len(sample_pitch.key_messages) == 3

    def test_get_section(self, sample_pitch):
        """Get a specific section by type."""
        hook = sample_pitch.get_section(SectionType.HOOK)
        assert hook is not None
        assert hook.section_type == SectionType.HOOK

        # Non-existent section
        pricing = sample_pitch.get_section(SectionType.PRICING)
        assert pricing is None

    def test_get_full_content(self, sample_pitch):
        """Get full content as markdown."""
        content = sample_pitch.get_full_content()
        assert "# TestProduct: Transform Your Workflow" in content
        assert "## Opening" in content
        assert "## The Solution" in content
        assert "- Time savings" in content

    def test_word_count(self, sample_pitch):
        """Calculate total word count."""
        count = sample_pitch.word_count()
        assert count > 0
        # Executive summary + section content + key points
        assert count >= 15

    def test_estimated_duration(self, sample_pitch):
        """Estimate presentation duration."""
        duration = sample_pitch.estimated_duration_minutes()
        assert duration > 0
        assert duration < 5  # Sample pitch is short

    def test_to_presentation_outline(self, sample_pitch):
        """Convert to presentation outline."""
        slides = sample_pitch.to_presentation_outline()

        # Should have title slide + content slides
        assert len(slides) >= 3

        # First slide is title
        assert slides[0]["slide_type"] == "title"
        assert slides[0]["title"] == sample_pitch.title

        # Content slides
        content_slides = [s for s in slides if s["slide_type"] == "content"]
        assert len(content_slides) == 2


class TestPitchVariant:
    """Tests for PitchVariant model."""

    def test_create_variant(self):
        """Create a pitch variant."""
        pitch = Pitch(
            product_name="Product",
            product_url="https://example.com",
            title="Title",
            executive_summary="Summary",
        )
        variant = PitchVariant(
            audience="CTOs",
            pitch=pitch,
            customizations=["Added ROI focus", "Shortened features section"],
        )
        assert variant.audience == "CTOs"
        assert len(variant.customizations) == 2


class TestPitchSet:
    """Tests for PitchSet model."""

    def test_create_pitch_set(self):
        """Create a pitch set with variants."""
        base_pitch = Pitch(
            product_name="Product",
            product_url="https://example.com",
            title="Base Title",
            executive_summary="Summary",
        )
        cto_pitch = Pitch(
            product_name="Product",
            product_url="https://example.com",
            title="CTO Title",
            executive_summary="Summary for CTOs",
        )
        dev_pitch = Pitch(
            product_name="Product",
            product_url="https://example.com",
            title="Developer Title",
            executive_summary="Summary for developers",
        )

        pitch_set = PitchSet(
            product_name="Product",
            base_pitch=base_pitch,
            variants=[
                PitchVariant(audience="CTO", pitch=cto_pitch, customizations=[]),
                PitchVariant(audience="Developer", pitch=dev_pitch, customizations=[]),
            ],
        )

        assert pitch_set.product_name == "Product"
        assert len(pitch_set.variants) == 2

    def test_get_variant(self):
        """Get variant by audience."""
        base = Pitch(
            product_name="P",
            product_url="https://example.com",
            title="T",
            executive_summary="S",
        )
        cto = Pitch(
            product_name="P",
            product_url="https://example.com",
            title="CTO",
            executive_summary="For CTOs",
        )

        pitch_set = PitchSet(
            product_name="P",
            base_pitch=base,
            variants=[PitchVariant(audience="CTO", pitch=cto, customizations=[])],
        )

        found = pitch_set.get_variant("CTO")
        assert found is not None
        assert found.title == "CTO"

        # Case insensitive
        found_lower = pitch_set.get_variant("cto")
        assert found_lower is not None

        # Not found
        not_found = pitch_set.get_variant("Developer")
        assert not_found is None
