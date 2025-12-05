"""Tests for the pitch generator module."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.generation.generator import (
    GenerationConfig,
    GenerationResult,
    PitchGenerator,
    SectionResult,
)
from src.models.pitch import (
    Pitch,
    PitchConfig,
    PitchLength,
    PitchSection,
    PitchTone,
    SectionType,
)
from src.models.processed import (
    AudienceAnalysis,
    AudienceSegment,
    AudienceType,
    BenefitSet,
    CompetitiveAnalysis,
    ContentSummary,
    CustomerBenefit,
    Differentiator,
    FeatureCategory,
    FeatureSet,
    PricingInfo,
    PricingModel,
    ProcessedContent,
    ProductFeature,
    TechnicalSpecs,
    UseCase,
    UseCaseSet,
)


@pytest.fixture
def sample_processed_content():
    """Create sample processed content for testing."""
    return ProcessedContent(
        product_name="TestProduct",
        product_url="https://example.com/product",
        processing_id="test-123",
        summary=ContentSummary(
            executive_summary="TestProduct is an AI-powered solution for enterprise teams.",
            detailed_summary="A comprehensive platform for managing workflows.",
            comprehensive_summary="Full summary here.",
            key_points=["Fast deployment", "Enterprise security", "Easy integration"],
            product_category="Enterprise Software",
            tagline="Work smarter, not harder",
            value_proposition="Reduce manual work by 80%",
        ),
        features=FeatureSet(
            features=[
                ProductFeature(
                    name="AI Automation",
                    description="Automate repetitive tasks with AI",
                    category=FeatureCategory.AUTOMATION,
                    benefits=["Save time", "Reduce errors"],
                    is_flagship=True,
                ),
                ProductFeature(
                    name="Real-time Collaboration",
                    description="Work together in real-time",
                    category=FeatureCategory.COLLABORATION,
                    benefits=["Better teamwork"],
                    is_flagship=True,
                ),
                ProductFeature(
                    name="Enterprise Security",
                    description="SOC2 and GDPR compliant",
                    category=FeatureCategory.SECURITY,
                    benefits=["Peace of mind"],
                    is_flagship=False,
                ),
            ],
            flagship_features=["AI Automation", "Real-time Collaboration"],
        ),
        benefits=BenefitSet(
            benefits=[
                CustomerBenefit(
                    headline="Save 10+ hours per week",
                    description="Automation handles the busywork",
                    target_audience=[AudienceType.BUSINESS],
                    supporting_features=["AI Automation"],
                ),
                CustomerBenefit(
                    headline="Improve team productivity",
                    description="Real-time collaboration keeps everyone aligned",
                    target_audience=[AudienceType.EXECUTIVE],
                    supporting_features=["Real-time Collaboration"],
                ),
            ],
            top_benefits=["Save 10+ hours per week", "Improve team productivity"],
        ),
        use_cases=UseCaseSet(
            use_cases=[
                UseCase(
                    title="Project Management",
                    scenario="Managing complex projects with multiple stakeholders",
                    problem_solved="Lack of visibility and coordination",
                    solution_approach="Centralized dashboard with automated updates",
                    target_audience=[AudienceType.BUSINESS],
                ),
            ],
            primary_use_cases=["Project Management"],
        ),
        competitive_analysis=CompetitiveAnalysis(
            differentiators=[
                Differentiator(
                    claim="Only solution with AI-native design",
                    explanation="Built from ground up with AI, not bolted on",
                    strength="strong",
                ),
            ],
            unique_capabilities=["AI-native architecture"],
            mentioned_competitors=["CompetitorA", "CompetitorB"],
        ),
        audience_analysis=AudienceAnalysis(
            segments=[
                AudienceSegment(
                    segment_type=AudienceType.BUSINESS,
                    name="Operations Teams",
                    description="Teams managing daily operations",
                    pain_points=["Manual processes", "Lack of visibility"],
                    goals=["Efficiency", "Better reporting"],
                ),
            ],
            primary_audience="Operations Teams",
        ),
        pricing=PricingInfo(
            pricing_model=PricingModel.TIERED,
            has_free_tier=False,
            has_free_trial=True,
            trial_duration="14 days",
        ),
        technical_specs=TechnicalSpecs(
            platforms_supported=["Web", "iOS", "Android"],
            api_available=True,
            api_type="REST",
        ),
    )


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_default_config(self):
        """Default configuration values."""
        config = GenerationConfig()
        assert config.max_concurrent_sections == 3
        assert config.retry_failed_sections is True
        assert config.enable_caching is True

    def test_custom_config(self):
        """Custom configuration values."""
        config = GenerationConfig(
            max_concurrent_sections=5,
            retry_failed_sections=False,
            verbose=True,
        )
        assert config.max_concurrent_sections == 5
        assert config.retry_failed_sections is False
        assert config.verbose is True


class TestSectionResult:
    """Tests for SectionResult."""

    def test_successful_result(self):
        """Create a successful section result."""
        section = PitchSection(
            section_type=SectionType.HOOK,
            title="Opening Hook",
            content="Attention grabber here",
            order=1,
        )
        result = SectionResult(
            section_type=SectionType.HOOK,
            section=section,
            success=True,
            tokens_used=100,
            cost_usd=0.001,
        )
        assert result.success is True
        assert result.section is not None
        assert result.error is None

    def test_failed_result(self):
        """Create a failed section result."""
        result = SectionResult(
            section_type=SectionType.PRICING,
            success=False,
            error="LLM timeout",
        )
        assert result.success is False
        assert result.section is None
        assert result.error == "LLM timeout"


class TestGenerationResult:
    """Tests for GenerationResult."""

    def test_successful_generation(self, sample_processed_content):
        """Create a successful generation result."""
        pitch = Pitch(
            product_name="TestProduct",
            product_url="https://example.com",
            title="TestProduct Pitch",
            executive_summary="Great product",
        )
        result = GenerationResult(
            pitch=pitch,
            success=True,
            total_tokens_used=1000,
            total_cost_usd=0.01,
        )
        assert result.success is True
        assert result.pitch is not None
        assert len(result.errors) == 0

    def test_generation_with_errors(self, sample_processed_content):
        """Generation result with errors."""
        pitch = Pitch(
            product_name="TestProduct",
            product_url="https://example.com",
            title="TestProduct Pitch",
            executive_summary="Great product",
        )
        result = GenerationResult(
            pitch=pitch,
            success=False,
            errors=["Section generation failed"],
        )
        assert result.success is False
        assert len(result.errors) == 1


class TestPitchGeneratorContextBuilding:
    """Tests for context building in PitchGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return PitchGenerator(GenerationConfig())

    def test_build_context_includes_product_info(
        self, generator, sample_processed_content
    ):
        """Context includes product information."""
        config = PitchConfig()
        context = generator._build_generation_context(sample_processed_content, config)

        assert context["product_name"] == "TestProduct"
        assert "AI-powered" in context["executive_summary"]
        assert context["tagline"] == "Work smarter, not harder"

    def test_build_context_includes_features(
        self, generator, sample_processed_content
    ):
        """Context includes formatted features."""
        config = PitchConfig()
        context = generator._build_generation_context(sample_processed_content, config)

        assert "AI Automation" in context["features_list"]
        assert "AI Automation" in context["key_features"]

    def test_build_context_includes_benefits(
        self, generator, sample_processed_content
    ):
        """Context includes formatted benefits."""
        config = PitchConfig()
        context = generator._build_generation_context(sample_processed_content, config)

        assert "Save 10+ hours" in context["benefits_list"]
        assert "Save 10+ hours" in context["key_benefits"]

    def test_build_context_includes_tone(
        self, generator, sample_processed_content
    ):
        """Context includes tone settings."""
        config = PitchConfig(tone=PitchTone.EXECUTIVE)
        context = generator._build_generation_context(sample_processed_content, config)

        assert context["tone"] == "executive"
        assert "tone_guidelines" in context

    def test_build_context_includes_audience(
        self, generator, sample_processed_content
    ):
        """Context includes target audience."""
        config = PitchConfig(target_audience="CTOs")
        context = generator._build_generation_context(sample_processed_content, config)

        assert context["target_audience"] == "CTOs"

    def test_build_context_uses_default_audience(
        self, generator, sample_processed_content
    ):
        """Context uses processed content's primary audience as default."""
        config = PitchConfig()
        context = generator._build_generation_context(sample_processed_content, config)

        assert context["target_audience"] == "Operations Teams"


class TestPitchGeneratorSectionSelection:
    """Tests for section selection logic."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return PitchGenerator(GenerationConfig())

    def test_elevator_sections(self, generator):
        """Elevator pitch has minimal sections."""
        config = PitchConfig(length=PitchLength.ELEVATOR)
        sections = generator._get_sections_to_generate(config)

        assert len(sections) <= 3
        assert SectionType.HOOK in sections
        assert SectionType.CTA in sections

    def test_standard_sections(self, generator):
        """Standard pitch has core sections."""
        config = PitchConfig(length=PitchLength.STANDARD)
        sections = generator._get_sections_to_generate(config)

        assert SectionType.HOOK in sections
        assert SectionType.PROBLEM in sections
        assert SectionType.SOLUTION in sections
        assert SectionType.FEATURES in sections
        assert SectionType.BENEFITS in sections
        assert SectionType.CTA in sections

    def test_excludes_pricing_when_disabled(self, generator):
        """Pricing section excluded when disabled."""
        config = PitchConfig(
            length=PitchLength.DETAILED,
            include_pricing=False,
        )
        sections = generator._get_sections_to_generate(config)

        assert SectionType.PRICING not in sections

    def test_excludes_technical_when_disabled(self, generator):
        """Technical section excluded when disabled."""
        config = PitchConfig(
            length=PitchLength.COMPREHENSIVE,
            include_technical=False,
        )
        sections = generator._get_sections_to_generate(config)

        assert SectionType.TECHNICAL not in sections

    def test_excludes_competitors_when_disabled(self, generator):
        """Differentiators section excluded when disabled."""
        config = PitchConfig(
            length=PitchLength.STANDARD,
            include_competitors=False,
        )
        sections = generator._get_sections_to_generate(config)

        assert SectionType.DIFFERENTIATORS not in sections

    def test_custom_section_selection(self, generator):
        """Custom section selection overrides length defaults."""
        config = PitchConfig(
            sections_to_include=[
                SectionType.HOOK,
                SectionType.SOLUTION,
                SectionType.CTA,
            ]
        )
        sections = generator._get_sections_to_generate(config)

        assert len(sections) == 3
        assert SectionType.HOOK in sections
        assert SectionType.SOLUTION in sections
        assert SectionType.CTA in sections


class TestPitchGeneratorResponseParsing:
    """Tests for LLM response parsing."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return PitchGenerator(GenerationConfig())

    def test_parse_valid_json(self, generator):
        """Parse valid JSON response."""
        response = '{"title": "Test", "content": "Content here", "key_points": ["Point 1"]}'
        result = generator._parse_section_response(response)

        assert result["title"] == "Test"
        assert result["content"] == "Content here"
        assert result["key_points"] == ["Point 1"]

    def test_parse_json_with_markdown(self, generator):
        """Parse JSON wrapped in markdown code blocks."""
        response = """```json
{"title": "Test", "content": "Content", "key_points": []}
```"""
        result = generator._parse_section_response(response)

        assert result["title"] == "Test"
        assert result["content"] == "Content"

    def test_parse_json_with_extra_text(self, generator):
        """Extract JSON from response with extra text."""
        response = """Here is the section:
{"title": "Extracted", "content": "Test content", "key_points": []}
That's the section."""
        result = generator._parse_section_response(response)

        assert result["title"] == "Extracted"

    def test_parse_invalid_json_returns_raw(self, generator):
        """Invalid JSON returns raw content."""
        response = "This is just plain text, not JSON."
        result = generator._parse_section_response(response)

        assert result["content"] == response
        assert result["title"] == "Section"


class TestPitchGeneratorAssembly:
    """Tests for pitch assembly."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return PitchGenerator(GenerationConfig())

    def test_assemble_pitch_with_sections(
        self, generator, sample_processed_content
    ):
        """Assemble pitch with generated sections."""
        config = PitchConfig()
        section_results = [
            SectionResult(
                section_type=SectionType.HOOK,
                section=PitchSection(
                    section_type=SectionType.HOOK,
                    title="Opening",
                    content="Hook content",
                    order=1,
                ),
                success=True,
            ),
            SectionResult(
                section_type=SectionType.CTA,
                section=PitchSection(
                    section_type=SectionType.CTA,
                    title="Next Steps",
                    content="CTA content",
                    key_points=["Start free trial", "Schedule demo"],
                    order=13,
                ),
                success=True,
            ),
        ]

        pitch = generator._assemble_pitch(
            processed_content=sample_processed_content,
            pitch_config=config,
            section_results=section_results,
            elevator_pitch="30 second pitch",
            one_liner="Short description",
            key_messages=["Message 1", "Message 2"],
            objections={"Too expensive": "Value justification"},
        )

        assert pitch.product_name == "TestProduct"
        assert len(pitch.sections) == 2
        assert pitch.elevator_pitch == "30 second pitch"
        assert pitch.one_liner == "Short description"
        assert len(pitch.key_messages) == 2
        assert "Too expensive" in pitch.common_objections

    def test_assemble_pitch_calculates_confidence(
        self, generator, sample_processed_content
    ):
        """Confidence calculated from section success rate."""
        config = PitchConfig()
        section_results = [
            SectionResult(section_type=SectionType.HOOK, success=True, section=PitchSection(
                section_type=SectionType.HOOK, title="", content="", order=1
            )),
            SectionResult(section_type=SectionType.PROBLEM, success=True, section=PitchSection(
                section_type=SectionType.PROBLEM, title="", content="", order=2
            )),
            SectionResult(section_type=SectionType.SOLUTION, success=False, error="Failed"),
            SectionResult(section_type=SectionType.CTA, success=True, section=PitchSection(
                section_type=SectionType.CTA, title="", content="", order=13
            )),
        ]

        pitch = generator._assemble_pitch(
            processed_content=sample_processed_content,
            pitch_config=config,
            section_results=section_results,
            elevator_pitch="",
            one_liner="",
            key_messages=[],
            objections={},
        )

        # 3 of 4 sections succeeded = 75% confidence
        assert pitch.overall_confidence == 0.75

    def test_assemble_pitch_includes_feature_highlights(
        self, generator, sample_processed_content
    ):
        """Pitch includes feature highlights from processed content."""
        config = PitchConfig(max_features=3)
        pitch = generator._assemble_pitch(
            processed_content=sample_processed_content,
            pitch_config=config,
            section_results=[],
            elevator_pitch="",
            one_liner="",
            key_messages=[],
            objections={},
        )

        assert len(pitch.feature_highlights) > 0
        feature_names = [f.name for f in pitch.feature_highlights]
        assert "AI Automation" in feature_names

    def test_assemble_pitch_includes_benefit_statements(
        self, generator, sample_processed_content
    ):
        """Pitch includes benefit statements from processed content."""
        config = PitchConfig(max_benefits=2)
        pitch = generator._assemble_pitch(
            processed_content=sample_processed_content,
            pitch_config=config,
            section_results=[],
            elevator_pitch="",
            one_liner="",
            key_messages=[],
            objections={},
        )

        assert len(pitch.benefit_statements) > 0
        headlines = [b.headline for b in pitch.benefit_statements]
        assert "Save 10+ hours per week" in headlines


class TestPitchGeneratorIntegration:
    """Integration tests for PitchGenerator (requires mocking LLM)."""

    @pytest.mark.asyncio
    async def test_generate_full_pitch(self, sample_processed_content):
        """Generate a complete pitch with mocked LLM."""
        # Different responses for different prompt types
        section_response = MagicMock()
        section_response.content = json.dumps({
            "title": "Test Section",
            "content": "Test content",
            "key_points": ["Point 1"],
            "talking_points": ["Talk about this"],
        })
        section_response.usage = MagicMock()
        section_response.usage.total_tokens = 100
        section_response.cost_usd = 0.001
        section_response.latency_ms = 500

        # Response for key messages (JSON array)
        key_messages_response = MagicMock()
        key_messages_response.content = '["Message 1", "Message 2"]'
        key_messages_response.usage = MagicMock()
        key_messages_response.usage.total_tokens = 50
        key_messages_response.cost_usd = 0.0005
        key_messages_response.latency_ms = 200

        # Response for objections (JSON object)
        objections_response = MagicMock()
        objections_response.content = '{"Too expensive": "Value justification"}'
        objections_response.usage = MagicMock()
        objections_response.usage.total_tokens = 50
        objections_response.cost_usd = 0.0005
        objections_response.latency_ms = 200

        # Response for elevator pitch and one-liner (plain text)
        text_response = MagicMock()
        text_response.content = "This is a test pitch."
        text_response.usage = MagicMock()
        text_response.usage.total_tokens = 30
        text_response.cost_usd = 0.0003
        text_response.latency_ms = 100

        with patch("src.generation.generator.AnthropicClient") as MockClient:
            mock_client = AsyncMock()
            # Return different responses based on call order
            # First 5 calls are sections, then elevator, one-liner, key_messages, objections
            mock_client.complete = AsyncMock(side_effect=[
                section_response,  # HOOK
                section_response,  # PROBLEM
                section_response,  # SOLUTION
                section_response,  # BENEFITS
                section_response,  # CTA
                text_response,     # elevator_pitch
                text_response,     # one_liner
                key_messages_response,  # key_messages
                objections_response,    # objections
            ])
            MockClient.return_value = mock_client

            generator = PitchGenerator(GenerationConfig())
            generator._client = mock_client

            config = PitchConfig(
                tone=PitchTone.PROFESSIONAL,
                length=PitchLength.SHORT,
            )
            result = await generator.generate(sample_processed_content, config)

            assert result.pitch is not None
            assert result.pitch.product_name == "TestProduct"
            assert len(result.section_results) > 0

    @pytest.mark.asyncio
    async def test_generator_context_manager(self):
        """Generator works as async context manager."""
        with patch("src.generation.generator.AnthropicClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.start = AsyncMock()
            mock_client.stop = AsyncMock()
            MockClient.return_value = mock_client

            async with PitchGenerator() as generator:
                assert generator._client is not None

            mock_client.stop.assert_called_once()
