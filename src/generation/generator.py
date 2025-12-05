"""Pitch generator orchestrator for creating sales pitches from processed content."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel

from src.llm.client import AnthropicClient, LLMResponse
from src.llm.config import LLMConfig, ModelSettings
from src.models.pitch import (
    BenefitStatement,
    CallToAction,
    CompetitivePoint,
    FeatureHighlight,
    Pitch,
    PitchConfig,
    PitchLength,
    PitchSection,
    PitchTone,
    SectionType,
)
from src.models.processed import ProcessedContent

from .templates import (
    ELEVATOR_PITCH_PROMPT,
    KEY_MESSAGES_PROMPT,
    OBJECTION_RESPONSES_PROMPT,
    ONE_LINER_PROMPT,
    SECTION_TEMPLATES,
    TONE_GUIDELINES,
    get_length_config,
    get_section_template,
    get_sections_for_length,
    get_tone_guidelines,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for pitch generation."""

    # LLM settings
    llm_config: Optional[LLMConfig] = None
    model_settings: Optional[ModelSettings] = None

    # Generation settings
    max_concurrent_sections: int = 3
    retry_failed_sections: bool = True
    max_section_retries: int = 2

    # Caching
    enable_caching: bool = True
    cache_dir: Optional[str] = None

    # Output settings
    include_raw_responses: bool = False
    verbose: bool = False


@dataclass
class SectionResult:
    """Result of generating a single section."""

    section_type: SectionType
    section: Optional[PitchSection] = None
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass
class GenerationResult:
    """Complete result of pitch generation."""

    pitch: Pitch
    success: bool = True
    section_results: list[SectionResult] = field(default_factory=list)
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PitchGenerator:
    """
    Orchestrates pitch generation from processed content.

    Uses LLM to generate each section based on templates, then
    assembles them into a complete pitch document.

    Usage:
        generator = PitchGenerator()
        async with generator:
            result = await generator.generate(
                processed_content,
                config=PitchConfig(tone=PitchTone.PROFESSIONAL),
            )
            print(result.pitch.get_full_content())
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self._client: Optional[AnthropicClient] = None
        self._cache: Optional[Any] = None

    async def __aenter__(self) -> "PitchGenerator":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Initialize resources."""
        llm_config = self.config.llm_config or LLMConfig()
        self._client = AnthropicClient(llm_config)
        await self._client.start()

        if self.config.enable_caching and self.config.cache_dir:
            from src.processing.cache import ProcessingCache

            self._cache = ProcessingCache(cache_dir=self.config.cache_dir)

        logger.info("PitchGenerator initialized")

    async def stop(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.stop()
            self._client = None
        logger.info("PitchGenerator stopped")

    def _ensure_client(self) -> AnthropicClient:
        """Ensure LLM client is available."""
        if self._client is None:
            raise RuntimeError(
                "Generator not started. Use 'async with PitchGenerator()' or call start()."
            )
        return self._client

    async def generate(
        self,
        processed_content: ProcessedContent,
        pitch_config: Optional[PitchConfig] = None,
    ) -> GenerationResult:
        """
        Generate a complete sales pitch from processed content.

        Args:
            processed_content: The processed product content
            pitch_config: Configuration for the pitch (tone, length, sections, etc.)

        Returns:
            GenerationResult with the complete pitch and metadata
        """
        start_time = time.time()
        client = self._ensure_client()

        pitch_config = pitch_config or PitchConfig()
        context = self._build_generation_context(processed_content, pitch_config)

        # Determine sections to generate
        sections_to_generate = self._get_sections_to_generate(pitch_config)

        logger.info(
            f"Generating pitch for {processed_content.product_name} "
            f"with {len(sections_to_generate)} sections"
        )

        # Generate sections concurrently with semaphore
        section_results = await self._generate_sections(
            sections_to_generate, context, pitch_config
        )

        # Generate supporting content
        elevator_pitch, one_liner, key_messages, objections = await asyncio.gather(
            self._generate_elevator_pitch(context),
            self._generate_one_liner(context),
            self._generate_key_messages(context),
            self._generate_objection_responses(context),
        )

        # Assemble pitch
        pitch = self._assemble_pitch(
            processed_content=processed_content,
            pitch_config=pitch_config,
            section_results=section_results,
            elevator_pitch=elevator_pitch,
            one_liner=one_liner,
            key_messages=key_messages,
            objections=objections,
        )

        # Calculate totals
        total_tokens = sum(r.tokens_used for r in section_results)
        total_cost = sum(r.cost_usd for r in section_results)
        total_duration = (time.time() - start_time) * 1000

        # Update pitch metadata
        pitch.generation_duration_ms = int(total_duration)
        pitch.total_llm_tokens_used = total_tokens
        pitch.total_llm_cost_usd = total_cost

        warnings = []
        errors = []
        for result in section_results:
            if not result.success:
                errors.append(f"Section {result.section_type.value}: {result.error}")

        result = GenerationResult(
            pitch=pitch,
            success=len(errors) == 0,
            section_results=section_results,
            total_tokens_used=total_tokens,
            total_cost_usd=total_cost,
            total_duration_ms=total_duration,
            warnings=warnings,
            errors=errors,
        )

        logger.info(
            f"Pitch generation complete: {len(pitch.sections)} sections, "
            f"{total_tokens} tokens, ${total_cost:.4f}"
        )

        return result

    def _get_sections_to_generate(self, pitch_config: PitchConfig) -> list[SectionType]:
        """Determine which sections to generate based on config."""
        # Start with sections for the selected length
        length_sections = get_sections_for_length(pitch_config.length)

        # Apply custom section selection if provided
        if pitch_config.sections_to_include:
            sections = pitch_config.sections_to_include
        else:
            sections = length_sections

        # Filter based on config flags
        if not pitch_config.include_pricing:
            sections = [s for s in sections if s != SectionType.PRICING]
        if not pitch_config.include_technical:
            sections = [s for s in sections if s != SectionType.TECHNICAL]
        if not pitch_config.include_competitors:
            sections = [s for s in sections if s != SectionType.DIFFERENTIATORS]

        return sections

    def _build_generation_context(
        self,
        processed_content: ProcessedContent,
        pitch_config: PitchConfig,
    ) -> dict[str, Any]:
        """Build the context dictionary for template variable substitution."""
        # Get base context from processed content
        context = processed_content.to_pitch_context()

        # Add pitch config values
        context["tone"] = pitch_config.tone.value
        context["tone_guidelines"] = get_tone_guidelines(pitch_config.tone)
        context["target_audience"] = (
            pitch_config.target_audience
            or processed_content.audience_analysis.primary_audience
            or "general business audience"
        )
        context["industry"] = pitch_config.industry or "general"
        context["company_name"] = pitch_config.company_name or "your company"
        context["pain_points"] = ", ".join(pitch_config.pain_points) if pitch_config.pain_points else "common industry challenges"
        context["max_features"] = pitch_config.max_features
        context["max_benefits"] = pitch_config.max_benefits

        # Format lists for prompts
        context["features_list"] = self._format_features_for_prompt(processed_content)
        context["benefits_list"] = self._format_benefits_for_prompt(processed_content)
        context["use_cases_list"] = self._format_use_cases_for_prompt(processed_content)
        context["differentiators_list"] = self._format_differentiators_for_prompt(
            processed_content
        )
        context["pricing_info"] = self._format_pricing_for_prompt(processed_content)
        context["technical_specs"] = self._format_technical_for_prompt(processed_content)
        context["competitors"] = ", ".join(
            processed_content.competitive_analysis.mentioned_competitors
        ) or "competitors"
        context["proof_points"] = self._format_proof_points_for_prompt(processed_content)
        context["objections"] = "price, implementation time, switching costs"
        context["has_free_trial"] = "Yes" if processed_content.pricing.has_free_trial else "No"
        context["pricing_model"] = processed_content.pricing.pricing_model.value
        context["product_category"] = processed_content.summary.product_category or "software"
        context["key_features"] = ", ".join(
            processed_content.features.flagship_features[:5]
        )
        context["key_benefits"] = ", ".join(processed_content.benefits.top_benefits[:5])
        context["differentiators"] = ", ".join(
            d.claim for d in processed_content.competitive_analysis.differentiators[:5]
        )
        context["custom_instructions"] = ""

        return context

    def _format_features_for_prompt(self, content: ProcessedContent) -> str:
        """Format features for prompt injection."""
        lines = []
        for i, feature in enumerate(content.features.features[:10], 1):
            benefits = ", ".join(feature.benefits[:2]) if feature.benefits else ""
            lines.append(f"{i}. {feature.name}: {feature.description}")
            if benefits:
                lines.append(f"   Benefits: {benefits}")
        return "\n".join(lines) if lines else "No specific features available"

    def _format_benefits_for_prompt(self, content: ProcessedContent) -> str:
        """Format benefits for prompt injection."""
        lines = []
        for i, benefit in enumerate(content.benefits.benefits[:10], 1):
            lines.append(f"{i}. {benefit.headline}: {benefit.description}")
        return "\n".join(lines) if lines else "No specific benefits available"

    def _format_use_cases_for_prompt(self, content: ProcessedContent) -> str:
        """Format use cases for prompt injection."""
        lines = []
        for i, use_case in enumerate(content.use_cases.use_cases[:5], 1):
            lines.append(f"{i}. {use_case.title}: {use_case.scenario}")
        return "\n".join(lines) if lines else "No specific use cases available"

    def _format_differentiators_for_prompt(self, content: ProcessedContent) -> str:
        """Format differentiators for prompt injection."""
        lines = []
        for i, diff in enumerate(content.competitive_analysis.differentiators[:5], 1):
            lines.append(f"{i}. {diff.claim}: {diff.explanation}")
        return "\n".join(lines) if lines else "No specific differentiators available"

    def _format_pricing_for_prompt(self, content: ProcessedContent) -> str:
        """Format pricing for prompt injection."""
        pricing = content.pricing
        lines = [f"Model: {pricing.pricing_model.value}"]
        if pricing.has_free_tier:
            lines.append("Free tier: Available")
        if pricing.has_free_trial:
            lines.append(f"Free trial: {pricing.trial_duration or 'Available'}")
        for tier in pricing.tiers[:4]:
            price_str = tier.price or "Contact sales"
            lines.append(f"- {tier.name}: {price_str}")
        return "\n".join(lines)

    def _format_technical_for_prompt(self, content: ProcessedContent) -> str:
        """Format technical specs for prompt injection."""
        specs = content.technical_specs
        lines = []
        if specs.platforms_supported:
            lines.append(f"Platforms: {', '.join(specs.platforms_supported)}")
        if specs.deployment_options:
            lines.append(f"Deployment: {', '.join(specs.deployment_options)}")
        if specs.api_available:
            lines.append(f"API: {specs.api_type or 'Available'}")
        if specs.security_certifications:
            lines.append(f"Security: {', '.join(specs.security_certifications)}")
        return "\n".join(lines) if lines else "Technical details available upon request"

    def _format_proof_points_for_prompt(self, content: ProcessedContent) -> str:
        """Format social proof for prompt injection."""
        lines = []
        for benefit in content.benefits.benefits:
            lines.extend(benefit.proof_points[:2])
        return "\n".join(lines[:5]) if lines else "Customer success stories available"

    async def _generate_sections(
        self,
        sections: list[SectionType],
        context: dict[str, Any],
        pitch_config: PitchConfig,
    ) -> list[SectionResult]:
        """Generate all sections with concurrency control."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_sections)

        async def generate_with_semaphore(section_type: SectionType) -> SectionResult:
            async with semaphore:
                return await self._generate_section(section_type, context, pitch_config)

        results = await asyncio.gather(
            *[generate_with_semaphore(s) for s in sections],
            return_exceptions=True,
        )

        section_results = []
        for section_type, result in zip(sections, results):
            if isinstance(result, Exception):
                section_results.append(
                    SectionResult(
                        section_type=section_type,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                section_results.append(result)

        return section_results

    async def _generate_section(
        self,
        section_type: SectionType,
        context: dict[str, Any],
        pitch_config: PitchConfig,
    ) -> SectionResult:
        """Generate a single pitch section."""
        client = self._ensure_client()
        template = get_section_template(section_type)

        if not template:
            return SectionResult(
                section_type=section_type,
                success=False,
                error=f"No template found for section type: {section_type}",
            )

        # Build prompt from template
        try:
            prompt = template.prompt_template.format(**context)
        except KeyError as e:
            return SectionResult(
                section_type=section_type,
                success=False,
                error=f"Missing context variable: {e}",
            )

        system_prompt = f"""You are a sales pitch writer creating compelling content.
{context.get('tone_guidelines', '')}

Write in a {context.get('tone', 'professional')} tone.
Target audience: {context.get('target_audience', 'business professionals')}

Return valid JSON only. No markdown formatting."""

        try:
            response = await client.complete(
                prompt,
                system=system_prompt,
                settings=self.config.model_settings,
            )

            # Parse response
            section_data = self._parse_section_response(response.content)

            section = PitchSection(
                section_type=section_type,
                title=section_data.get("title", template.name),
                content=section_data.get("content", ""),
                key_points=section_data.get("key_points", []),
                talking_points=section_data.get("talking_points", []),
                visual_suggestions=section_data.get("visual_suggestions", []),
                order=template.default_order,
            )

            return SectionResult(
                section_type=section_type,
                section=section,
                success=True,
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
            )

        except Exception as e:
            logger.error(f"Error generating section {section_type}: {e}")
            return SectionResult(
                section_type=section_type,
                success=False,
                error=str(e),
            )

    def _parse_section_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response for section data."""
        # Clean up common JSON issues
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re

            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Return basic structure with raw content
            return {
                "title": "Section",
                "content": content,
                "key_points": [],
                "talking_points": [],
            }

    async def _generate_elevator_pitch(self, context: dict[str, Any]) -> str:
        """Generate a 30-second elevator pitch."""
        client = self._ensure_client()
        prompt = ELEVATOR_PITCH_PROMPT.format(**context)

        try:
            response = await client.complete(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Error generating elevator pitch: {e}")
            return context.get("executive_summary", "")

    async def _generate_one_liner(self, context: dict[str, Any]) -> str:
        """Generate a single-sentence product description."""
        client = self._ensure_client()
        prompt = ONE_LINER_PROMPT.format(**context)

        try:
            response = await client.complete(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Error generating one-liner: {e}")
            return context.get("tagline", "")

    async def _generate_key_messages(self, context: dict[str, Any]) -> list[str]:
        """Generate key messages for sales reps."""
        client = self._ensure_client()
        prompt = KEY_MESSAGES_PROMPT.format(**context)

        try:
            response = await client.complete(prompt)
            messages = json.loads(response.content.strip())
            if isinstance(messages, list):
                return messages[:5]
        except Exception as e:
            logger.warning(f"Error generating key messages: {e}")

        return context.get("key_points", [])[:5]

    async def _generate_objection_responses(
        self, context: dict[str, Any]
    ) -> dict[str, str]:
        """Generate responses to common objections."""
        client = self._ensure_client()
        prompt = OBJECTION_RESPONSES_PROMPT.format(**context)

        try:
            response = await client.complete(prompt)
            objections = json.loads(response.content.strip())
            if isinstance(objections, dict):
                return objections
        except Exception as e:
            logger.warning(f"Error generating objection responses: {e}")

        return {}

    def _assemble_pitch(
        self,
        processed_content: ProcessedContent,
        pitch_config: PitchConfig,
        section_results: list[SectionResult],
        elevator_pitch: str,
        one_liner: str,
        key_messages: list[str],
        objections: dict[str, str],
    ) -> Pitch:
        """Assemble all generated content into a Pitch object."""
        # Collect successful sections
        sections = []
        for result in section_results:
            if result.success and result.section:
                sections.append(result.section)

        # Sort by order
        sections.sort(key=lambda s: s.order)

        # Build feature highlights
        feature_highlights = [
            FeatureHighlight(
                name=f.name,
                headline=f.description[:100],
                description=f.description,
                benefit=f.benefits[0] if f.benefits else "",
                proof_point=None,
            )
            for f in processed_content.features.features[:pitch_config.max_features]
        ]

        # Build benefit statements
        benefit_statements = [
            BenefitStatement(
                headline=b.headline,
                description=b.description,
                supporting_feature=b.supporting_features[0] if b.supporting_features else None,
                target_audience=", ".join(a.value for a in b.target_audience[:2]),
            )
            for b in processed_content.benefits.benefits[:pitch_config.max_benefits]
        ]

        # Build competitive points
        competitive_points = [
            CompetitivePoint(
                claim=d.claim,
                explanation=d.explanation,
                compared_to=", ".join(d.compared_to) if d.compared_to else None,
            )
            for d in processed_content.competitive_analysis.differentiators[:5]
        ]

        # Build CTA
        cta_section = next(
            (s for s in sections if s.section_type == SectionType.CTA), None
        )
        call_to_action = None
        if cta_section:
            call_to_action = CallToAction(
                primary_cta=cta_section.key_points[0] if cta_section.key_points else "Learn more",
                secondary_cta=cta_section.key_points[1] if len(cta_section.key_points) > 1 else None,
                urgency_statement=None,
                next_steps=cta_section.key_points[2:] if len(cta_section.key_points) > 2 else [],
            )

        # Calculate confidence based on section success rate
        total_sections = len(section_results)
        successful_sections = sum(1 for r in section_results if r.success)
        confidence = successful_sections / total_sections if total_sections > 0 else 0.5

        # Collect warnings
        warnings = [r.error for r in section_results if not r.success and r.error]

        return Pitch(
            product_name=processed_content.product_name,
            product_url=processed_content.product_url,
            pitch_id=str(uuid.uuid4()),
            config=pitch_config,
            title=f"{processed_content.product_name}: {one_liner}" if one_liner else processed_content.product_name,
            subtitle=processed_content.summary.tagline,
            executive_summary=processed_content.summary.executive_summary,
            sections=sections,
            feature_highlights=feature_highlights,
            benefit_statements=benefit_statements,
            competitive_points=competitive_points,
            call_to_action=call_to_action,
            elevator_pitch=elevator_pitch,
            one_liner=one_liner,
            key_messages=key_messages,
            common_objections=objections,
            source_processed_id=processed_content.processing_id,
            overall_confidence=confidence,
            warnings=warnings,
        )

    async def generate_variant(
        self,
        base_pitch: Pitch,
        audience: str,
        tone: Optional[PitchTone] = None,
    ) -> Pitch:
        """
        Generate a pitch variant for a specific audience.

        Args:
            base_pitch: The base pitch to adapt
            audience: Target audience description
            tone: Optional tone override

        Returns:
            Adapted Pitch for the audience
        """
        # Create a new config for this variant
        variant_config = PitchConfig(
            target_audience=audience,
            tone=tone or base_pitch.config.tone,
            length=base_pitch.config.length,
            include_pricing=base_pitch.config.include_pricing,
            include_technical=base_pitch.config.include_technical,
            include_competitors=base_pitch.config.include_competitors,
        )

        # For now, return a copy with updated config
        # In a full implementation, this would re-generate sections
        variant = base_pitch.model_copy(deep=True)
        variant.pitch_id = str(uuid.uuid4())
        variant.config = variant_config

        return variant
