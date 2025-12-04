"""Tests for the structured output parser."""

import pytest
from pydantic import BaseModel

from src.llm.parser import (
    ParseError,
    ContentAnalysis,
    ExtractedEntity,
    ExtractedFeature,
    SectionContent,
    extract_json,
    parse_json,
    parse_model,
    parse_list,
    parse_markdown_sections,
    parse_bullet_points,
    parse_numbered_list,
    parse_key_value_pairs,
    get_json_schema,
    create_output_instructions,
)


class TestExtractJson:
    """Tests for extract_json function."""

    def test_extract_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        text = '''Here is the result:
```json
{"name": "test", "value": 42}
```
'''
        result = extract_json(text)
        assert result == '{"name": "test", "value": 42}'

    def test_extract_from_generic_code_block(self):
        """Test extracting JSON from generic code block."""
        text = '''```
{"items": [1, 2, 3]}
```'''
        result = extract_json(text)
        assert result == '{"items": [1, 2, 3]}'

    def test_extract_raw_json_object(self):
        """Test extracting raw JSON object."""
        text = 'The result is {"key": "value"} as shown.'
        result = extract_json(text)
        assert result == '{"key": "value"}'

    def test_extract_raw_json_array(self):
        """Test extracting raw JSON array."""
        text = 'Items: [1, 2, 3]'
        result = extract_json(text)
        assert result == '[1, 2, 3]'

    def test_no_json_returns_none(self):
        """Test that no JSON returns None."""
        text = "This is just plain text with no JSON."
        result = extract_json(text)
        assert result is None


class TestParseJson:
    """Tests for parse_json function."""

    def test_parse_direct_json(self):
        """Test parsing direct JSON string."""
        text = '{"name": "test"}'
        result = parse_json(text)
        assert result == {"name": "test"}

    def test_parse_json_in_text(self):
        """Test parsing JSON embedded in text."""
        text = 'Result: {"count": 5}'
        result = parse_json(text)
        assert result == {"count": 5}

    def test_parse_invalid_json_strict(self):
        """Test parsing invalid JSON in strict mode."""
        text = "No JSON here"
        with pytest.raises(ParseError):
            parse_json(text, strict=True)

    def test_parse_invalid_json_non_strict(self):
        """Test parsing invalid JSON in non-strict mode."""
        text = "No JSON here"
        result = parse_json(text, strict=False)
        assert result is None


class TestParseModel:
    """Tests for parse_model function."""

    def test_parse_valid_model(self):
        """Test parsing valid model data."""
        text = '{"name": "Entity1", "type": "product", "confidence": 0.9}'
        result = parse_model(text, ExtractedEntity)
        assert result is not None
        assert result.name == "Entity1"
        assert result.type == "product"
        assert result.confidence == 0.9

    def test_parse_model_with_defaults(self):
        """Test parsing model with default values."""
        text = '{"name": "Entity2", "type": "feature"}'
        result = parse_model(text, ExtractedEntity)
        assert result is not None
        assert result.confidence == 1.0  # Default value

    def test_parse_invalid_model_strict(self):
        """Test parsing invalid model in strict mode."""
        text = '{"invalid_field": "value"}'
        with pytest.raises(ParseError):
            parse_model(text, ExtractedEntity, strict=True)

    def test_parse_invalid_model_non_strict(self):
        """Test parsing invalid model in non-strict mode."""
        text = '{"invalid_field": "value"}'
        result = parse_model(text, ExtractedEntity, strict=False)
        assert result is None


class TestParseList:
    """Tests for parse_list function."""

    def test_parse_simple_list(self):
        """Test parsing simple list."""
        text = '[1, 2, 3, 4, 5]'
        result = parse_list(text)
        assert result == [1, 2, 3, 4, 5]

    def test_parse_list_of_objects(self):
        """Test parsing list of objects."""
        text = '[{"a": 1}, {"a": 2}]'
        result = parse_list(text)
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_parse_list_with_model(self):
        """Test parsing list with model validation."""
        text = '''[
            {"name": "E1", "type": "product"},
            {"name": "E2", "type": "feature"}
        ]'''
        result = parse_list(text, ExtractedEntity)
        assert len(result) == 2
        assert all(isinstance(item, ExtractedEntity) for item in result)

    def test_parse_single_object_as_list(self):
        """Test that single object is wrapped in list in non-strict mode."""
        text = '{"name": "E1", "type": "product"}'
        result = parse_list(text, ExtractedEntity, strict=False)
        assert len(result) == 1


class TestParseMarkdownSections:
    """Tests for parse_markdown_sections function."""

    def test_parse_sections(self):
        """Test parsing markdown sections."""
        text = """# Introduction
This is the intro.

## Features
- Feature 1
- Feature 2

## Benefits
Great benefits here."""
        sections = parse_markdown_sections(text)
        assert "introduction" in sections
        assert "features" in sections
        assert "benefits" in sections
        assert "Feature 1" in sections["features"]

    def test_no_headers(self):
        """Test text with no headers."""
        text = "Just some plain text."
        sections = parse_markdown_sections(text)
        assert "introduction" in sections
        assert sections["introduction"] == "Just some plain text."


class TestParseBulletPoints:
    """Tests for parse_bullet_points function."""

    def test_parse_dash_bullets(self):
        """Test parsing dash bullet points."""
        text = """
- First item
- Second item
- Third item
"""
        points = parse_bullet_points(text)
        assert len(points) == 3
        assert "First item" in points

    def test_parse_asterisk_bullets(self):
        """Test parsing asterisk bullet points."""
        text = """
* Item A
* Item B
"""
        points = parse_bullet_points(text)
        assert len(points) == 2

    def test_mixed_content(self):
        """Test parsing bullets from mixed content."""
        text = """
Some intro text.
- Bullet 1
Regular paragraph.
- Bullet 2
"""
        points = parse_bullet_points(text)
        assert len(points) == 2


class TestParseNumberedList:
    """Tests for parse_numbered_list function."""

    def test_parse_numbered_list(self):
        """Test parsing numbered list."""
        text = """
1. First
2. Second
3. Third
"""
        items = parse_numbered_list(text)
        assert len(items) == 3
        assert items[0] == "First"

    def test_parse_with_parenthesis(self):
        """Test parsing numbered list with parenthesis."""
        text = """
1) Option A
2) Option B
"""
        items = parse_numbered_list(text)
        assert len(items) == 2


class TestParseKeyValuePairs:
    """Tests for parse_key_value_pairs function."""

    def test_parse_colon_pairs(self):
        """Test parsing colon-separated pairs."""
        text = """
Name: Test Product
Version: 1.0
Status: Active
"""
        pairs = parse_key_value_pairs(text)
        assert pairs["name"] == "Test Product"
        assert pairs["version"] == "1.0"
        assert pairs["status"] == "Active"

    def test_parse_bold_keys(self):
        """Test parsing bold markdown keys."""
        text = """
**Product Name**: Widget
**Price**: $99
"""
        pairs = parse_key_value_pairs(text)
        assert pairs["product_name"] == "Widget"
        assert pairs["price"] == "$99"


class TestResponseModels:
    """Tests for built-in response models."""

    def test_extracted_entity(self):
        """Test ExtractedEntity model."""
        entity = ExtractedEntity(
            name="Claude",
            type="product",
            confidence=0.95,
            context="AI assistant",
        )
        assert entity.name == "Claude"
        assert entity.confidence == 0.95

    def test_extracted_feature(self):
        """Test ExtractedFeature model."""
        feature = ExtractedFeature(
            name="Natural Language",
            description="Understands natural language",
            benefits=["Easy to use", "No coding required"],
            use_cases=["Customer support", "Content creation"],
        )
        assert feature.name == "Natural Language"
        assert len(feature.benefits) == 2

    def test_content_analysis(self):
        """Test ContentAnalysis model."""
        analysis = ContentAnalysis(
            product_name="Test Product",
            summary="A great product",
            key_features=["Feature 1", "Feature 2"],
            target_audience=["Developers"],
        )
        assert analysis.product_name == "Test Product"
        assert len(analysis.key_features) == 2

    def test_section_content(self):
        """Test SectionContent model."""
        section = SectionContent(
            title="Introduction",
            content="Welcome to our product...",
            key_points=["Point 1", "Point 2"],
        )
        assert section.title == "Introduction"


class TestSchemaGeneration:
    """Tests for schema generation utilities."""

    def test_get_json_schema(self):
        """Test getting JSON schema from model."""
        schema = get_json_schema(ExtractedEntity)
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "type" in schema["properties"]

    def test_create_output_instructions(self):
        """Test creating output instructions."""
        instructions = create_output_instructions(ExtractedEntity)
        assert "JSON" in instructions
        assert "name" in instructions
        assert "type" in instructions
