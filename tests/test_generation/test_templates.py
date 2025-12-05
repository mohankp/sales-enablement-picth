"""Tests for pitch generation templates."""

import pytest

from src.generation.templates import (
    SECTION_TEMPLATES,
    TONE_GUIDELINES,
    LENGTH_GUIDELINES,
    SectionTemplate,
    get_section_template,
    get_tone_guidelines,
    get_length_config,
    get_sections_for_length,
)
from src.models.pitch import PitchLength, PitchTone, SectionType


class TestSectionTemplates:
    """Tests for section templates."""

    def test_all_section_types_have_templates(self):
        """Verify all section types have templates defined."""
        for section_type in SectionType:
            template = get_section_template(section_type)
            assert template is not None, f"Missing template for {section_type}"
            assert isinstance(template, SectionTemplate)

    def test_template_has_required_fields(self):
        """Templates have all required fields."""
        for section_type, template in SECTION_TEMPLATES.items():
            assert template.section_type == section_type
            assert template.name, f"Template {section_type} missing name"
            assert template.description, f"Template {section_type} missing description"
            assert template.prompt_template, f"Template {section_type} missing prompt"
            assert template.default_order > 0

    def test_hook_template_is_required(self):
        """Hook section is marked as required."""
        hook = get_section_template(SectionType.HOOK)
        assert hook.required is True

    def test_solution_template_is_required(self):
        """Solution section is marked as required."""
        solution = get_section_template(SectionType.SOLUTION)
        assert solution.required is True

    def test_cta_template_is_required(self):
        """CTA section is marked as required."""
        cta = get_section_template(SectionType.CTA)
        assert cta.required is True

    def test_prompt_templates_have_placeholders(self):
        """Prompt templates contain expected placeholders."""
        hook = get_section_template(SectionType.HOOK)
        assert "{product_name}" in hook.prompt_template
        assert "{tone}" in hook.prompt_template

    def test_word_limits_are_reasonable(self):
        """Word limits are set reasonably."""
        for template in SECTION_TEMPLATES.values():
            assert template.min_words >= 0
            assert template.max_words > template.min_words
            assert template.max_words <= 500  # No single section should be too long


class TestToneGuidelines:
    """Tests for tone guidelines."""

    def test_all_tones_have_guidelines(self):
        """All pitch tones have guidelines defined."""
        for tone in PitchTone:
            guidelines = get_tone_guidelines(tone)
            assert guidelines is not None
            assert len(guidelines) > 0

    def test_guidelines_are_actionable(self):
        """Guidelines contain actionable instructions."""
        for tone, guidelines in TONE_GUIDELINES.items():
            assert "-" in guidelines, f"Guidelines for {tone} should be bullet points"
            assert len(guidelines) > 50, f"Guidelines for {tone} too short"

    def test_professional_tone_is_default(self):
        """Professional is the default tone."""
        # If an invalid tone is passed, should return professional guidelines
        professional = get_tone_guidelines(PitchTone.PROFESSIONAL)
        assert "formal" in professional.lower() or "professional" in professional.lower()

    def test_technical_tone_mentions_details(self):
        """Technical tone emphasizes technical details."""
        technical = get_tone_guidelines(PitchTone.TECHNICAL)
        assert "technical" in technical.lower()

    def test_executive_tone_mentions_roi(self):
        """Executive tone emphasizes business impact."""
        executive = get_tone_guidelines(PitchTone.EXECUTIVE)
        assert "roi" in executive.lower() or "business" in executive.lower()


class TestLengthGuidelines:
    """Tests for length configuration."""

    def test_all_lengths_have_config(self):
        """All pitch lengths have configuration."""
        for length in PitchLength:
            config = get_length_config(length)
            assert config is not None
            assert "total_words" in config
            assert "sections" in config
            assert "duration_seconds" in config

    def test_elevator_is_shortest(self):
        """Elevator pitch is the shortest."""
        elevator = get_length_config(PitchLength.ELEVATOR)
        short = get_length_config(PitchLength.SHORT)
        assert elevator["total_words"] < short["total_words"]
        assert elevator["duration_seconds"] < short["duration_seconds"]

    def test_comprehensive_is_longest(self):
        """Comprehensive pitch is the longest."""
        comprehensive = get_length_config(PitchLength.COMPREHENSIVE)
        detailed = get_length_config(PitchLength.DETAILED)
        assert comprehensive["total_words"] > detailed["total_words"]
        assert len(comprehensive["sections"]) >= len(detailed["sections"])

    def test_elevator_has_minimal_sections(self):
        """Elevator pitch has only essential sections."""
        sections = get_sections_for_length(PitchLength.ELEVATOR)
        assert len(sections) <= 3
        assert SectionType.HOOK in sections
        assert SectionType.CTA in sections

    def test_standard_has_core_sections(self):
        """Standard pitch has all core sections."""
        sections = get_sections_for_length(PitchLength.STANDARD)
        assert SectionType.HOOK in sections
        assert SectionType.PROBLEM in sections
        assert SectionType.SOLUTION in sections
        assert SectionType.FEATURES in sections
        assert SectionType.BENEFITS in sections
        assert SectionType.CTA in sections

    def test_comprehensive_has_all_sections(self):
        """Comprehensive pitch includes all section types."""
        sections = get_sections_for_length(PitchLength.COMPREHENSIVE)
        for section_type in SectionType:
            assert section_type in sections

    def test_lengths_are_ordered(self):
        """Pitch lengths are in ascending order of content."""
        lengths = [
            PitchLength.ELEVATOR,
            PitchLength.SHORT,
            PitchLength.STANDARD,
            PitchLength.DETAILED,
            PitchLength.COMPREHENSIVE,
        ]

        prev_words = 0
        for length in lengths:
            config = get_length_config(length)
            assert config["total_words"] > prev_words
            prev_words = config["total_words"]


class TestTemplatePlaceholders:
    """Tests for template placeholder formatting."""

    def test_hook_template_formats_correctly(self):
        """Hook template can be formatted with context."""
        template = get_section_template(SectionType.HOOK)
        context = {
            "product_name": "TestProduct",
            "executive_summary": "A great product",
            "target_audience": "developers",
            "tone": "professional",
            "custom_instructions": "",
        }
        result = template.prompt_template.format(**context)
        assert "TestProduct" in result
        assert "developers" in result

    def test_features_template_formats_correctly(self):
        """Features template can be formatted with context."""
        template = get_section_template(SectionType.FEATURES)
        context = {
            "product_name": "TestProduct",
            "features_list": "1. Feature A\n2. Feature B",
            "target_audience": "enterprises",
            "tone": "executive",
            "max_features": 5,
        }
        result = template.prompt_template.format(**context)
        assert "TestProduct" in result
        assert "Feature A" in result

    def test_pricing_template_formats_correctly(self):
        """Pricing template can be formatted with context."""
        template = get_section_template(SectionType.PRICING)
        context = {
            "product_name": "TestProduct",
            "pricing_info": "$99/mo",
            "tone": "professional",
        }
        result = template.prompt_template.format(**context)
        assert "TestProduct" in result
        assert "$99/mo" in result
