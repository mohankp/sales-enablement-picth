"""
LLM Integration Module for Sales Enablement Pitch Generator.

This module provides a comprehensive interface for working with Claude and other LLMs:
- Multi-provider support (Anthropic Claude, Google Gemini)
- Async client with retry logic and streaming support
- Prompt management with templates and versioning
- Structured output parsing with Pydantic validation
- Cost tracking and token counting

Usage:
    from src.llm import create_llm_provider, ProviderType, LLMConfig

    # Using the provider factory (recommended)
    provider = create_llm_provider(ProviderType.ANTHROPIC)
    async with provider:
        response = await provider.complete("Hello, world!")
        print(response.content)

    # Using Gemini
    provider = create_llm_provider(ProviderType.GEMINI)
    async with provider:
        response = await provider.complete("Hello, world!")

    # Legacy usage (still supported)
    from src.llm import AnthropicClient, LLMConfig
    config = LLMConfig()
    async with AnthropicClient(config) as client:
        response = await client.complete("Hello, world!")
"""

# Provider system (recommended)
from .providers import (
    LLMProvider,
    ProviderType,
    ProviderCapabilities,
    AnthropicProvider,
    GeminiProvider,
    create_llm_provider,
    get_provider_capabilities,
)

# Response types (shared across providers)
from .providers.base import (
    CostTracker,
    LLMResponse,
    StreamChunk,
    TokenUsage,
)

# Legacy client (for backward compatibility)
from .client import (
    AnthropicClient,
    quick_complete,
)

# Configuration
from .config import (
    LLMConfig,
    ModelCapability,
    ModelName,
    GeminiModelName,
    ModelSettings,
    PromptConfig,
    RetryConfig,
    TaskModelMapping,
    MODEL_CONTEXT_WINDOWS,
    MODEL_PRICING,
    GEMINI_PRICING,
    GEMINI_CONTEXT_WINDOWS,
    ProviderType as ConfigProviderType,  # Also export from config for convenience
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
    # Provider system (recommended)
    "LLMProvider",
    "ProviderType",
    "ProviderCapabilities",
    "AnthropicProvider",
    "GeminiProvider",
    "create_llm_provider",
    "get_provider_capabilities",
    # Response types
    "LLMResponse",
    "StreamChunk",
    "TokenUsage",
    "CostTracker",
    # Legacy client (backward compatibility)
    "AnthropicClient",
    "quick_complete",
    # Config
    "LLMConfig",
    "ModelSettings",
    "ModelName",
    "GeminiModelName",
    "ModelCapability",
    "RetryConfig",
    "PromptConfig",
    "TaskModelMapping",
    "MODEL_PRICING",
    "MODEL_CONTEXT_WINDOWS",
    "GEMINI_PRICING",
    "GEMINI_CONTEXT_WINDOWS",
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
