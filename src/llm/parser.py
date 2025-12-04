"""Structured output parsing for LLM responses."""

import json
import logging
import re
from typing import Any, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ParseError(Exception):
    """Error parsing LLM output."""

    def __init__(self, message: str, raw_content: str, errors: Optional[list[Any]] = None):
        super().__init__(message)
        self.raw_content = raw_content
        self.errors = errors or []


def extract_json(text: str) -> Optional[str]:
    """
    Extract JSON from text that may contain markdown code blocks or other content.

    Handles:
    - ```json ... ``` blocks
    - ``` ... ``` blocks
    - Raw JSON objects/arrays
    - JSON embedded in text
    """
    # Try to find JSON in code blocks first
    code_block_patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, text)
        if match:
            content = match.group(1).strip()
            if content.startswith(("{", "[")):
                return content

    # Try to find raw JSON object or array
    # Look for complete JSON structures
    json_patterns = [
        r"(\{[\s\S]*\})",  # Object
        r"(\[[\s\S]*\])",  # Array
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue

    return None


def parse_json(text: str, strict: bool = True) -> Any:
    """
    Parse JSON from LLM output.

    Args:
        text: Raw LLM output text
        strict: If True, raise error on parse failure

    Returns:
        Parsed JSON object

    Raises:
        ParseError: If JSON cannot be parsed
    """
    # First try direct parsing
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON
    json_str = extract_json(text)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if strict:
                raise ParseError(
                    f"Invalid JSON structure: {e}",
                    raw_content=text,
                    errors=[str(e)],
                )
            return None

    if strict:
        raise ParseError(
            "No valid JSON found in response",
            raw_content=text,
        )
    return None


def parse_model(text: str, model_class: Type[T], strict: bool = True) -> Optional[T]:
    """
    Parse LLM output into a Pydantic model.

    Args:
        text: Raw LLM output text
        model_class: Pydantic model class to parse into
        strict: If True, raise error on parse failure

    Returns:
        Parsed model instance

    Raises:
        ParseError: If parsing or validation fails
    """
    data = parse_json(text, strict=strict)
    if data is None:
        return None

    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        if strict:
            raise ParseError(
                f"Validation failed for {model_class.__name__}: {e}",
                raw_content=text,
                errors=e.errors(),
            )
        return None


def parse_list(text: str, item_model: Optional[Type[T]] = None, strict: bool = True) -> list[Any]:
    """
    Parse LLM output as a list, optionally validating items against a model.

    Args:
        text: Raw LLM output text
        item_model: Optional Pydantic model for list items
        strict: If True, raise error on parse failure

    Returns:
        List of parsed items
    """
    data = parse_json(text, strict=strict)
    if data is None:
        return []

    if not isinstance(data, list):
        if strict:
            raise ParseError(
                "Expected JSON array but got object",
                raw_content=text,
            )
        # Try to wrap single object in list
        data = [data]

    if item_model is None:
        return data

    results = []
    errors = []
    for i, item in enumerate(data):
        try:
            results.append(item_model.model_validate(item))
        except ValidationError as e:
            errors.append({"index": i, "errors": e.errors()})
            if not strict:
                continue
            raise ParseError(
                f"Validation failed for item {i}: {e}",
                raw_content=text,
                errors=errors,
            )

    return results


def parse_markdown_sections(text: str) -> dict[str, str]:
    """
    Parse markdown-formatted response into sections.

    Args:
        text: Markdown text with headers

    Returns:
        Dictionary mapping section headers to content
    """
    sections = {}
    current_section = "introduction"
    current_content = []

    for line in text.split("\n"):
        # Check for headers (## Header or # Header)
        header_match = re.match(r"^#{1,3}\s+(.+)$", line)
        if header_match:
            # Save previous section
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = header_match.group(1).lower().strip()
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def parse_bullet_points(text: str) -> list[str]:
    """
    Extract bullet points from text.

    Args:
        text: Text containing bullet points

    Returns:
        List of bullet point strings (without bullets)
    """
    points = []
    bullet_pattern = re.compile(r"^\s*[-*•]\s*(.+)$", re.MULTILINE)

    for match in bullet_pattern.finditer(text):
        point = match.group(1).strip()
        if point:
            points.append(point)

    return points


def parse_numbered_list(text: str) -> list[str]:
    """
    Extract numbered list items from text.

    Args:
        text: Text containing numbered list

    Returns:
        List of item strings (without numbers)
    """
    items = []
    number_pattern = re.compile(r"^\s*\d+[.)]\s*(.+)$", re.MULTILINE)

    for match in number_pattern.finditer(text):
        item = match.group(1).strip()
        if item:
            items.append(item)

    return items


def parse_key_value_pairs(text: str) -> dict[str, str]:
    """
    Extract key-value pairs from text.

    Handles formats like:
    - Key: Value
    - **Key**: Value
    - Key = Value

    Args:
        text: Text containing key-value pairs

    Returns:
        Dictionary of extracted pairs
    """
    pairs = {}

    patterns = [
        r"^\*\*([^*]+)\*\*:\s*(.+)$",  # **Key**: Value
        r"^([^:=]+):\s*(.+)$",  # Key: Value
        r"^([^:=]+)=\s*(.+)$",  # Key = Value
    ]

    for line in text.split("\n"):
        line = line.strip()
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                if key and value:
                    pairs[key.lower().replace(" ", "_")] = value
                break

    return pairs


# Common response models for structured output

class ExtractedEntity(BaseModel):
    """A single extracted entity."""

    name: str
    type: str
    confidence: float = 1.0
    context: Optional[str] = None


class ExtractedFeature(BaseModel):
    """An extracted product feature."""

    name: str
    description: str
    benefits: list[str] = []
    technical_details: Optional[str] = None
    use_cases: list[str] = []


class ContentAnalysis(BaseModel):
    """Analysis result for content."""

    product_name: Optional[str] = None
    summary: str
    key_features: list[str] = []
    target_audience: list[str] = []
    use_cases: list[str] = []
    technical_specs: dict[str, str] = {}
    pricing_info: Optional[str] = None
    competitive_advantages: list[str] = []


class SectionContent(BaseModel):
    """Generated section content."""

    title: str
    content: str
    key_points: list[str] = []
    suggested_visuals: list[str] = []


def get_json_schema(model_class: Type[BaseModel]) -> dict[str, Any]:
    """
    Get JSON schema for a Pydantic model.

    Args:
        model_class: Pydantic model class

    Returns:
        JSON schema dictionary
    """
    return model_class.model_json_schema()


def create_output_instructions(model_class: Type[BaseModel]) -> str:
    """
    Create output format instructions for an LLM based on a Pydantic model.

    Args:
        model_class: Pydantic model class

    Returns:
        Instruction string for the LLM
    """
    schema = get_json_schema(model_class)
    schema_str = json.dumps(schema, indent=2)

    return f"""Respond with a valid JSON object matching this schema:

```json
{schema_str}
```

Important:
- Return ONLY the JSON object, no additional text
- Ensure all required fields are included
- Use proper JSON formatting"""
