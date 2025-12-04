"""
LLM Integration Module for Sales Enablement Pitch Generator.

This module provides a comprehensive interface for working with Claude and other LLMs:
- Async client with retry logic and streaming support
- Prompt management with templates and versioning
- Structured output parsing with Pydantic validation
- Cost tracking and token counting

Usage:
    from src.llm import AnthropicClient, LLMConfig, render_prompt

    config = LLMConfig()
    async with AnthropicClient(config) as client:
        response = await client.complete("Hello, world!")
        print(response.content)

    # Using prompt templates
    prompt = render_prompt("content_analysis", {"content": extracted_text})
    response = await client.complete(prompt)
"""

from .client import (
    AnthropicClient,
    CostTracker,
    LLMResponse,
    StreamChunk,
    TokenUsage,
    quick_complete,
)
from .config import (
    LLMConfig,
    ModelCapability,
    ModelName,
    ModelSettings,
    PromptConfig,
    RetryConfig,
    TaskModelMapping,
    MODEL_CONTEXT_WINDOWS,
    MODEL_PRICING,
)
from .parser import (
    ContentAnalysis,
    ExtractedEntity,
    ExtractedFeature,
    ParseError,
    SectionContent,
    create_output_instructions,
    extract_json,
    get_json_schema,
    parse_bullet_points,
    parse_json,
    parse_key_value_pairs,
    parse_list,
    parse_markdown_sections,
    parse_model,
    parse_numbered_list,
)
from .prompts import (
    PromptRegistry,
    PromptTemplate,
    PromptVariable,
    get_registry,
    render_prompt,
)

__all__ = [
    # Client
    "AnthropicClient",
    "LLMResponse",
    "StreamChunk",
    "TokenUsage",
    "CostTracker",
    "quick_complete",
    # Config
    "LLMConfig",
    "ModelSettings",
    "ModelName",
    "ModelCapability",
    "RetryConfig",
    "PromptConfig",
    "TaskModelMapping",
    "MODEL_PRICING",
    "MODEL_CONTEXT_WINDOWS",
    # Parser
    "parse_json",
    "parse_model",
    "parse_list",
    "parse_markdown_sections",
    "parse_bullet_points",
    "parse_numbered_list",
    "parse_key_value_pairs",
    "extract_json",
    "get_json_schema",
    "create_output_instructions",
    "ParseError",
    "ExtractedEntity",
    "ExtractedFeature",
    "ContentAnalysis",
    "SectionContent",
    # Prompts
    "PromptTemplate",
    "PromptVariable",
    "PromptRegistry",
    "get_registry",
    "render_prompt",
]
