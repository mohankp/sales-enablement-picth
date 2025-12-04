"""Tests for the prompt management system."""

import pytest
from pathlib import Path
import tempfile
import json

from src.llm.prompts import (
    PromptTemplate,
    PromptVariable,
    PromptRegistry,
    get_registry,
    render_prompt,
)
from src.llm.config import PromptConfig


class TestPromptVariable:
    """Tests for PromptVariable."""

    def test_required_variable(self):
        """Test required variable."""
        var = PromptVariable(
            name="content",
            description="The content to analyze",
            required=True,
        )
        assert var.name == "content"
        assert var.required is True
        assert var.default is None

    def test_optional_variable(self):
        """Test optional variable with default."""
        var = PromptVariable(
            name="tone",
            description="Writing tone",
            required=False,
            default="professional",
        )
        assert var.required is False
        assert var.default == "professional"


class TestPromptTemplate:
    """Tests for PromptTemplate."""

    def test_basic_variable_substitution(self):
        """Test basic variable substitution."""
        template = PromptTemplate(
            name="test",
            template="Hello, {{name}}!",
        )
        result = template.render({"name": "World"})
        assert result == "Hello, World!"

    def test_multiple_variables(self):
        """Test multiple variable substitution."""
        template = PromptTemplate(
            name="test",
            template="{{greeting}}, {{name}}! Welcome to {{place}}.",
        )
        result = template.render({
            "greeting": "Hello",
            "name": "Alice",
            "place": "Wonderland",
        })
        assert result == "Hello, Alice! Welcome to Wonderland."

    def test_conditional_section_true(self):
        """Test conditional section when condition is true."""
        template = PromptTemplate(
            name="test",
            template="Start {{#if include_extra}}Extra content{{/if}} End",
        )
        result = template.render({"include_extra": True})
        assert "Extra content" in result

    def test_conditional_section_false(self):
        """Test conditional section when condition is false."""
        template = PromptTemplate(
            name="test",
            template="Start {{#if include_extra}}Extra content{{/if}} End",
        )
        result = template.render({"include_extra": False})
        assert "Extra content" not in result

    def test_each_loop(self):
        """Test each loop iteration."""
        template = PromptTemplate(
            name="test",
            template="Items:\n{{#each items}}- {{item}}\n{{/each}}",
        )
        result = template.render({"items": ["apple", "banana", "cherry"]})
        assert "- apple" in result
        assert "- banana" in result
        assert "- cherry" in result

    def test_each_loop_with_objects(self):
        """Test each loop with object items."""
        template = PromptTemplate(
            name="test",
            template="{{#each features}}{{item.name}}: {{item.desc}}\n{{/each}}",
        )
        result = template.render({
            "features": [
                {"name": "Feature1", "desc": "Description1"},
                {"name": "Feature2", "desc": "Description2"},
            ]
        })
        assert "Feature1: Description1" in result
        assert "Feature2: Description2" in result

    def test_missing_required_variable_strict(self):
        """Test error on missing required variable in strict mode."""
        template = PromptTemplate(
            name="test",
            template="Hello, {{name}}!",
        )
        with pytest.raises(ValueError) as exc_info:
            template.render({}, strict=True)
        assert "Missing required variables" in str(exc_info.value)

    def test_missing_variable_non_strict(self):
        """Test missing variable in non-strict mode."""
        template = PromptTemplate(
            name="test",
            template="Hello, {{name}}!",
        )
        result = template.render({}, strict=False)
        assert "{{name}}" in result

    def test_get_required_variables(self):
        """Test extracting required variables."""
        template = PromptTemplate(
            name="test",
            template="{{greeting}}, {{name}}! {{#if show}}{{message}}{{/if}}",
        )
        vars = template.get_required_variables()
        assert "greeting" in vars
        assert "name" in vars
        assert "show" in vars
        assert "message" in vars

    def test_hash_property(self):
        """Test hash property."""
        template = PromptTemplate(
            name="test",
            template="Hello",
            version="1.0.0",
        )
        assert len(template.hash) == 12
        # Same content should produce same hash
        template2 = PromptTemplate(
            name="test",
            template="Hello",
            version="1.0.0",
        )
        assert template.hash == template2.hash

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        template = PromptTemplate(
            name="test",
            template="Hello, {{name}}!",
            description="Test template",
            version="1.2.0",
            tags=["test", "greeting"],
        )
        data = template.to_dict()
        restored = PromptTemplate.from_dict(data)
        assert restored.name == template.name
        assert restored.template == template.template
        assert restored.version == template.version
        assert restored.tags == template.tags


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving templates."""
        registry = PromptRegistry()
        template = PromptTemplate(
            name="custom",
            template="Custom: {{value}}",
        )
        registry.register(template)
        retrieved = registry.get("custom")
        assert retrieved is not None
        assert retrieved.name == "custom"

    def test_builtin_templates(self):
        """Test that builtin templates are loaded."""
        registry = PromptRegistry()
        assert registry.get("content_analysis") is not None
        assert registry.get("entity_extraction") is not None
        assert registry.get("summarize") is not None
        assert registry.get("feature_extraction") is not None
        assert registry.get("pitch_section") is not None

    def test_render_template(self):
        """Test rendering template through registry."""
        registry = PromptRegistry()
        result = registry.render("summarize", {
            "content": "Test content here",
            "length": "brief",
        })
        assert "Test content here" in result
        assert "brief" in result

    def test_list_templates(self):
        """Test listing templates."""
        registry = PromptRegistry()
        templates = registry.list_templates()
        assert len(templates) >= 5
        assert "content_analysis" in templates

    def test_list_templates_by_tag(self):
        """Test listing templates by tag."""
        registry = PromptRegistry()
        extraction_templates = registry.list_templates(tag="extraction")
        assert "entity_extraction" in extraction_templates
        assert "feature_extraction" in extraction_templates

    def test_get_template_info(self):
        """Test getting template info."""
        registry = PromptRegistry()
        info = registry.get_template_info("content_analysis")
        assert info is not None
        assert info["name"] == "content_analysis"
        assert "variables" in info
        assert "hash" in info

    def test_version_history(self):
        """Test version history tracking."""
        registry = PromptRegistry()
        # Register first version
        v1 = PromptTemplate(
            name="versioned",
            template="V1: {{x}}",
            version="1.0.0",
        )
        registry.register(v1)

        # Register second version
        v2 = PromptTemplate(
            name="versioned",
            template="V2: {{x}}",
            version="2.0.0",
        )
        registry.register(v2)

        # Current should be v2
        current = registry.get("versioned")
        assert current.version == "2.0.0"

        # Should be able to get v1
        old = registry.get("versioned", version="1.0.0")
        assert old.version == "1.0.0"

    def test_save_and_load(self):
        """Test saving and loading templates."""
        registry = PromptRegistry()
        template = PromptTemplate(
            name="saveable",
            template="Save me: {{data}}",
            version="1.0.0",
        )
        registry.register(template)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            registry.save_to_file(path, names=["saveable"])

            # Create new registry and load
            new_registry = PromptRegistry()
            new_registry.load_from_file(path)

            loaded = new_registry.get("saveable")
            assert loaded is not None
            assert loaded.template == "Save me: {{data}}"
        finally:
            path.unlink()


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry(self):
        """Test getting global registry."""
        registry = get_registry()
        assert registry is not None
        assert isinstance(registry, PromptRegistry)

    def test_render_prompt(self):
        """Test render_prompt convenience function."""
        result = render_prompt("summarize", {
            "content": "Some content",
            "length": "detailed",
        })
        assert "Some content" in result
        assert "detailed" in result
