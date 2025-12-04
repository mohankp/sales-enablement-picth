"""Prompt management system with templates, variables, and versioning."""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

from .config import PromptConfig

logger = logging.getLogger(__name__)


class PromptVariable(BaseModel):
    """Definition of a variable in a prompt template."""

    name: str
    description: str
    required: bool = True
    default: Optional[Any] = None
    validator: Optional[str] = None  # Regex pattern for validation


@dataclass
class PromptTemplate:
    """
    A prompt template with variable substitution and versioning.

    Supports:
    - Variable substitution: {{variable_name}}
    - Optional sections: {{#if condition}}...{{/if}}
    - Loops: {{#each items}}...{{/each}}
    """

    name: str
    template: str
    description: str = ""
    version: str = "1.0.0"
    variables: list[PromptVariable] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    # Variable pattern: {{variable_name}}
    VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")
    # Conditional pattern: {{#if condition}}...{{/if}}
    IF_PATTERN = re.compile(r"\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}", re.DOTALL)
    # Loop pattern: {{#each items}}...{{/each}}
    EACH_PATTERN = re.compile(r"\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}", re.DOTALL)

    @property
    def hash(self) -> str:
        """Get a hash of the template content for versioning."""
        content = f"{self.name}:{self.version}:{self.template}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def get_required_variables(self) -> set[str]:
        """Extract all variable names from the template."""
        variables = set(self.VAR_PATTERN.findall(self.template))

        # Also extract from conditionals
        for match in self.IF_PATTERN.finditer(self.template):
            variables.add(match.group(1))
            variables.update(self.VAR_PATTERN.findall(match.group(2)))

        # Extract from loops
        for match in self.EACH_PATTERN.finditer(self.template):
            variables.add(match.group(1))

        return variables

    def render(
        self,
        variables: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> str:
        """
        Render the template with provided variables.

        Args:
            variables: Dictionary of variable values
            strict: If True, raise error for missing required variables

        Returns:
            Rendered prompt string
        """
        variables = variables or {}
        result = self.template

        # Process conditionals first
        def replace_if(match: re.Match) -> str:
            condition_var = match.group(1)
            content = match.group(2)
            if variables.get(condition_var):
                return content.strip()
            return ""

        result = self.IF_PATTERN.sub(replace_if, result)

        # Process loops
        def replace_each(match: re.Match) -> str:
            list_var = match.group(1)
            item_template = match.group(2)
            items = variables.get(list_var, [])

            if not isinstance(items, (list, tuple)):
                return ""

            rendered_items = []
            for i, item in enumerate(items):
                item_result = item_template
                if isinstance(item, dict):
                    for key, value in item.items():
                        item_result = item_result.replace(f"{{{{item.{key}}}}}", str(value))
                else:
                    item_result = item_result.replace("{{item}}", str(item))
                item_result = item_result.replace("{{index}}", str(i))
                rendered_items.append(item_result.strip())

            return "\n".join(rendered_items)

        result = self.EACH_PATTERN.sub(replace_each, result)

        # Process simple variable substitution
        required_vars = self.get_required_variables()
        missing_vars = []

        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in variables:
                return str(variables[var_name])
            elif strict and var_name in required_vars:
                missing_vars.append(var_name)
                return match.group(0)
            else:
                # Try to find a default
                for var_def in self.variables:
                    if var_def.name == var_name and var_def.default is not None:
                        return str(var_def.default)
                return match.group(0)

        result = self.VAR_PATTERN.sub(replace_var, result)

        if strict and missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")

        # Clean up extra whitespace
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def to_dict(self) -> dict[str, Any]:
        """Serialize template to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "description": self.description,
            "version": self.version,
            "variables": [v.model_dump() for v in self.variables],
            "tags": self.tags,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        """Deserialize template from dictionary."""
        variables = [PromptVariable(**v) for v in data.get("variables", [])]
        return cls(
            name=data["name"],
            template=data["template"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            variables=variables,
            tags=data.get("tags", []),
        )


class PromptRegistry:
    """
    Registry for managing and versioning prompt templates.

    Usage:
        registry = PromptRegistry()
        registry.register(my_template)
        prompt = registry.render("my_template", {"var": "value"})
    """

    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()
        self._templates: dict[str, PromptTemplate] = {}
        self._template_history: dict[str, list[PromptTemplate]] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self) -> None:
        """Load built-in prompt templates."""
        # Content analysis template
        self.register(
            PromptTemplate(
                name="content_analysis",
                description="Analyze extracted content to identify key information",
                template="""Analyze the following content extracted from a product documentation website.

## Content to Analyze
{{content}}

## Analysis Tasks
1. Identify the main product/service being described
2. Extract key features and capabilities
3. Identify target audience and use cases
4. Note any technical specifications
5. Find pricing information if available
6. Identify competitive advantages

{{#if focus_areas}}
## Focus Areas
Pay special attention to: {{focus_areas}}
{{/if}}

Provide a structured analysis with clear sections for each category found.""",
                variables=[
                    PromptVariable(name="content", description="The extracted content to analyze", required=True),
                    PromptVariable(name="focus_areas", description="Specific areas to focus on", required=False),
                ],
                tags=["analysis", "content"],
            )
        )

        # Entity extraction template
        self.register(
            PromptTemplate(
                name="entity_extraction",
                description="Extract named entities from content",
                template="""Extract all named entities from the following content.

## Content
{{content}}

## Entity Types to Extract
- Products: Product names, versions, editions
- Features: Specific feature names and capabilities
- Technologies: Technical terms, APIs, integrations
- Companies: Company names, partners, competitors
- People: Names, titles, roles mentioned
- Metrics: Numbers, statistics, performance claims
- Dates: Release dates, timelines, deadlines

Return the extracted entities as a JSON object with entity types as keys and arrays of found entities as values.""",
                variables=[
                    PromptVariable(name="content", description="Content to extract entities from", required=True),
                ],
                tags=["extraction", "entities"],
            )
        )

        # Summarization template
        self.register(
            PromptTemplate(
                name="summarize",
                description="Summarize content to specified length",
                template="""Summarize the following content in {{length}} format.

## Content
{{content}}

## Requirements
- Capture the key points and main message
- Maintain factual accuracy
- Use clear, professional language
{{#if audience}}
- Target audience: {{audience}}
{{/if}}
{{#if style}}
- Writing style: {{style}}
{{/if}}

Provide the summary:""",
                variables=[
                    PromptVariable(name="content", description="Content to summarize", required=True),
                    PromptVariable(name="length", description="Length specification (brief/detailed/bullet points)", required=True, default="brief"),
                    PromptVariable(name="audience", description="Target audience", required=False),
                    PromptVariable(name="style", description="Writing style", required=False),
                ],
                tags=["summarization"],
            )
        )

        # Feature extraction template
        self.register(
            PromptTemplate(
                name="feature_extraction",
                description="Extract product features with details",
                template="""Extract all product features from the following content.

## Content
{{content}}

For each feature, provide:
1. **Feature Name**: Clear, concise name
2. **Description**: What the feature does
3. **Benefits**: Why it matters to users
4. **Technical Details**: Any specifications or requirements
5. **Use Cases**: When/how to use this feature

{{#if product_name}}
Product: {{product_name}}
{{/if}}

Return features as a JSON array with objects containing: name, description, benefits, technical_details, use_cases.""",
                variables=[
                    PromptVariable(name="content", description="Content to extract features from", required=True),
                    PromptVariable(name="product_name", description="Name of the product", required=False),
                ],
                tags=["extraction", "features"],
            )
        )

        # Pitch section generator
        self.register(
            PromptTemplate(
                name="pitch_section",
                description="Generate a section for a sales pitch",
                template="""Generate a {{section_type}} section for a sales pitch.

## Source Material
{{source_content}}

## Section Requirements
- Section Type: {{section_type}}
- Target Length: {{target_length}} words
- Tone: {{tone}}

{{#if key_points}}
## Key Points to Include
{{#each key_points}}
- {{item}}
{{/each}}
{{/if}}

{{#if audience}}
## Target Audience
{{audience}}
{{/if}}

Write a compelling {{section_type}} section that would work well in a sales presentation:""",
                variables=[
                    PromptVariable(name="source_content", description="Source material to base section on", required=True),
                    PromptVariable(name="section_type", description="Type of section (intro, features, benefits, etc.)", required=True),
                    PromptVariable(name="target_length", description="Target word count", required=True, default="150"),
                    PromptVariable(name="tone", description="Writing tone", required=True, default="professional"),
                    PromptVariable(name="key_points", description="Key points to include", required=False),
                    PromptVariable(name="audience", description="Target audience description", required=False),
                ],
                tags=["generation", "pitch"],
            )
        )

    def register(self, template: PromptTemplate) -> None:
        """Register a new template or update existing."""
        existing = self._templates.get(template.name)

        if existing:
            # Store in history
            if template.name not in self._template_history:
                self._template_history[template.name] = []
            self._template_history[template.name].append(existing)

        self._templates[template.name] = template
        logger.debug(f"Registered template: {template.name} (v{template.version})")

    def get(self, name: str, version: Optional[str] = None) -> Optional[PromptTemplate]:
        """Get a template by name, optionally a specific version."""
        if version:
            # Look in history for specific version
            history = self._template_history.get(name, [])
            for template in history:
                if template.version == version:
                    return template
            # Check current
            current = self._templates.get(name)
            if current and current.version == version:
                return current
            return None

        return self._templates.get(name)

    def render(
        self,
        name: str,
        variables: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> str:
        """
        Render a template by name with provided variables.

        Args:
            name: Template name
            variables: Variable values
            strict: Override strict mode from config

        Returns:
            Rendered prompt string
        """
        template = self.get(name)
        if not template:
            raise ValueError(f"Template not found: {name}")

        # Merge with default variables
        merged_vars = {**self.config.default_variables}
        if variables:
            merged_vars.update(variables)

        strict_mode = strict if strict is not None else self.config.strict_variables
        return template.render(merged_vars, strict=strict_mode)

    def list_templates(self, tag: Optional[str] = None) -> list[str]:
        """List all registered template names, optionally filtered by tag."""
        if tag:
            return [
                name
                for name, template in self._templates.items()
                if tag in template.tags
            ]
        return list(self._templates.keys())

    def get_template_info(self, name: str) -> Optional[dict[str, Any]]:
        """Get information about a template."""
        template = self.get(name)
        if not template:
            return None

        return {
            "name": template.name,
            "description": template.description,
            "version": template.version,
            "tags": template.tags,
            "variables": [v.model_dump() for v in template.variables],
            "hash": template.hash,
            "history_count": len(self._template_history.get(name, [])),
        }

    def load_from_file(self, path: Path) -> None:
        """Load templates from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                self.register(PromptTemplate.from_dict(item))
        else:
            self.register(PromptTemplate.from_dict(data))

    def save_to_file(self, path: Path, names: Optional[list[str]] = None) -> None:
        """Save templates to a JSON file."""
        templates_to_save = []

        if names:
            for name in names:
                template = self.get(name)
                if template:
                    templates_to_save.append(template.to_dict())
        else:
            templates_to_save = [t.to_dict() for t in self._templates.values()]

        with open(path, "w") as f:
            json.dump(templates_to_save, f, indent=2, default=str)


# Global registry instance
_default_registry: Optional[PromptRegistry] = None


def get_registry() -> PromptRegistry:
    """Get the global prompt registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = PromptRegistry()
    return _default_registry


def render_prompt(name: str, variables: Optional[dict[str, Any]] = None) -> str:
    """Convenience function to render a prompt from the global registry."""
    return get_registry().render(name, variables)
