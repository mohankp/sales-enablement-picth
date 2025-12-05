"""Main content processor orchestrating LLM-based analysis."""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.llm import (
    AnthropicClient,
    LLMConfig,
    ModelName,
    ModelSettings,
    parse_json,
    parse_model,
    ParseError,
)
from src.models.content import ExtractedContent
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
    Integration,
    PricingInfo,
    PricingModel,
    PricingTier,
    ProcessedContent,
    ProductFeature,
    TechnicalSpecs,
    UseCase,
    UseCaseSet,
    VisualInventory,
)
from .chunker import ContentChunker, ChunkingStrategy, ContentChunk
from .cache import ProcessingCache
from .visuals import VisualInventoryBuilder

logger = logging.getLogger(__name__)


class ProcessingConfig(BaseModel):
    """Configuration for content processing."""

    # LLM Settings
    llm_config: Optional[LLMConfig] = None
    default_model: ModelName = ModelName.SONNET
    analysis_model: ModelName = ModelName.SONNET  # For deep analysis
    extraction_model: ModelName = ModelName.SONNET  # For data extraction
    summarization_model: ModelName = ModelName.HAIKU  # For summaries

    # Processing options
    enable_features: bool = True
    enable_benefits: bool = True
    enable_use_cases: bool = True
    enable_competitive: bool = True
    enable_audience: bool = True
    enable_pricing: bool = True
    enable_technical: bool = True
    enable_visuals: bool = True  # Build visual inventory from extraction

    # Chunking
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    max_chunk_chars: int = 50_000

    # Caching
    enable_cache: bool = True
    cache_dir: Optional[str] = "data/cache/processing"
    cache_ttl_seconds: int = 86400  # 24 hours

    # Concurrency
    max_concurrent_requests: int = 3

    # Quality settings
    min_confidence_threshold: float = 0.5
    retry_on_low_confidence: bool = True


@dataclass
class ProcessingResult:
    """Result of a processing operation."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class ContentProcessor:
    """
    Orchestrates LLM-based content analysis.

    Takes extracted website content and produces structured
    product information suitable for pitch generation.

    Usage:
        config = ProcessingConfig()
        async with ContentProcessor(config) as processor:
            result = await processor.process(extracted_content)
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self._client: Optional[AnthropicClient] = None
        self._chunker = ContentChunker(
            strategy=self.config.chunking_strategy,
            max_chars=self.config.max_chunk_chars,
        )
        self._cache: Optional[ProcessingCache] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._visual_builder = VisualInventoryBuilder()

        # Initialize cache if enabled
        if self.config.enable_cache:
            self._cache = ProcessingCache(
                cache_dir=self.config.cache_dir,
                default_ttl_seconds=self.config.cache_ttl_seconds,
            )

    async def __aenter__(self) -> "ContentProcessor":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Initialize the processor and LLM client."""
        llm_config = self.config.llm_config or LLMConfig()
        self._client = AnthropicClient(llm_config)
        await self._client.start()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        logger.info("ContentProcessor initialized")

    async def stop(self) -> None:
        """Shutdown the processor."""
        if self._client:
            await self._client.stop()
            self._client = None
        logger.info("ContentProcessor stopped")

    def _ensure_client(self) -> AnthropicClient:
        """Ensure client is initialized."""
        if self._client is None:
            raise RuntimeError(
                "Processor not initialized. Use 'async with ContentProcessor()' or call start()."
            )
        return self._client

    async def process(
        self,
        content: ExtractedContent,
        product_name: Optional[str] = None,
    ) -> ProcessedContent:
        """
        Process extracted content into structured product information.

        Args:
            content: Extracted website content
            product_name: Override product name (auto-detected if not provided)

        Returns:
            ProcessedContent with all analyzed information
        """
        start_time = time.time()
        client = self._ensure_client()

        # Generate processing ID
        processing_id = hashlib.md5(
            f"{content.product_url}:{time.time()}".encode()
        ).hexdigest()[:12]

        logger.info(f"Starting content processing: {processing_id}")

        # Chunk the content
        chunks = self._chunker.chunk_extracted_content(content)
        logger.info(f"Content split into {len(chunks)} chunks")

        # Prepare combined content for analysis
        combined_content = self._combine_chunks_for_analysis(chunks)

        # Detect product name if not provided
        if not product_name:
            product_name = content.product_name or await self._detect_product_name(
                combined_content
            )

        # Compute content hash for caching
        content_hash = content.content_hash or hashlib.md5(
            combined_content.encode()
        ).hexdigest()[:16]

        # Run analyses in parallel where possible
        tasks = []

        if self.config.enable_features:
            tasks.append(("features", self._cached_extract(
                "features", content_hash, combined_content, product_name,
                self._extract_features, FeatureSet
            )))

        if self.config.enable_benefits:
            tasks.append(("benefits", self._cached_extract(
                "benefits", content_hash, combined_content, product_name,
                self._extract_benefits, BenefitSet
            )))

        if self.config.enable_use_cases:
            tasks.append(("use_cases", self._cached_extract(
                "use_cases", content_hash, combined_content, product_name,
                self._extract_use_cases, UseCaseSet
            )))

        if self.config.enable_competitive:
            tasks.append(("competitive", self._cached_extract(
                "competitive", content_hash, combined_content, product_name,
                self._analyze_competitive, CompetitiveAnalysis
            )))

        if self.config.enable_audience:
            tasks.append(("audience", self._cached_extract(
                "audience", content_hash, combined_content, product_name,
                self._analyze_audience, AudienceAnalysis
            )))

        if self.config.enable_pricing:
            tasks.append(("pricing", self._cached_extract(
                "pricing", content_hash, combined_content, product_name,
                self._extract_pricing, PricingInfo
            )))

        if self.config.enable_technical:
            tasks.append(("technical", self._cached_extract(
                "technical", content_hash, combined_content, product_name,
                self._extract_technical, TechnicalSpecs
            )))

        # Always generate summary
        tasks.append(("summary", self._cached_extract(
            "summary", content_hash, combined_content, product_name,
            self._generate_summary, ContentSummary
        )))

        # Execute with concurrency control
        results: dict[str, ProcessingResult] = {}
        async with asyncio.TaskGroup() as tg:
            async def run_task(name: str, coro):
                async with self._semaphore:
                    results[name] = await coro

            for name, coro in tasks:
                tg.create_task(run_task(name, coro))

        # Assemble results
        processing_duration = int((time.time() - start_time) * 1000)
        total_tokens = sum(r.tokens_used for r in results.values())
        total_cost = sum(r.cost_usd for r in results.values())

        # Collect warnings and errors
        warnings = []
        errors = []
        for name, result in results.items():
            if not result.success:
                errors.append(f"{name}: {result.error}")
            elif result.data is None:
                warnings.append(f"{name}: No data extracted")

        processed = ProcessedContent(
            product_name=product_name or "Unknown Product",
            product_url=content.product_url,
            processing_id=processing_id,
            summary=results.get("summary", ProcessingResult(success=False)).data or ContentSummary(
                executive_summary="",
                detailed_summary="",
                comprehensive_summary="",
            ),
            features=results.get("features", ProcessingResult(success=False)).data or FeatureSet(),
            benefits=results.get("benefits", ProcessingResult(success=False)).data or BenefitSet(),
            use_cases=results.get("use_cases", ProcessingResult(success=False)).data or UseCaseSet(),
            competitive_analysis=results.get("competitive", ProcessingResult(success=False)).data or CompetitiveAnalysis(),
            audience_analysis=results.get("audience", ProcessingResult(success=False)).data or AudienceAnalysis(),
            pricing=results.get("pricing", ProcessingResult(success=False)).data or PricingInfo(),
            technical_specs=results.get("technical", ProcessingResult(success=False)).data or TechnicalSpecs(),
            source_extraction_id=content.extraction_id,
            source_content_hash=content.content_hash,
            processed_at=datetime.now(timezone.utc),
            processing_duration_ms=processing_duration,
            total_llm_tokens_used=total_tokens,
            total_llm_cost_usd=total_cost,
            warnings=warnings,
            errors=errors,
        )

        # Build visual inventory if enabled
        if self.config.enable_visuals:
            try:
                visual_inventory = self._visual_builder.build_inventory(content)
                processed.visual_inventory = visual_inventory
                logger.info(
                    f"Built visual inventory: {visual_inventory.total_count} assets "
                    f"({len(visual_inventory.images)} images, "
                    f"{len(visual_inventory.tables)} tables, "
                    f"{len(visual_inventory.videos)} videos)"
                )

                # Map visuals to features using LLM if we have both
                if visual_inventory.total_count > 0 and processed.features.features:
                    feature_visuals = await self._map_visuals_to_features(
                        processed.features,
                        visual_inventory,
                        product_name or "Unknown Product",
                    )
                    processed.feature_visuals = feature_visuals

            except Exception as e:
                logger.warning(f"Failed to build visual inventory: {e}")
                warnings.append(f"Visual inventory: {str(e)}")

        # Compute statistics
        processed.compute_stats()

        # Calculate overall confidence
        successful = sum(1 for r in results.values() if r.success and r.data)
        processed.overall_confidence = successful / len(results) if results else 0.0

        logger.info(
            f"Processing complete: {processing_id} "
            f"({processing_duration}ms, {total_tokens} tokens, ${total_cost:.4f})"
        )

        return processed

    def _combine_chunks_for_analysis(self, chunks: list[ContentChunk]) -> str:
        """Combine chunks into a single string for analysis."""
        parts = []
        for chunk in chunks:
            if chunk.section_headers:
                parts.append(f"[Sections: {', '.join(chunk.section_headers)}]")
            parts.append(chunk.content)

        return "\n\n---\n\n".join(parts)

    async def _cached_extract(
        self,
        aspect: str,
        content_hash: str,
        content: str,
        product_name: str,
        extractor_func,
        model_class: type,
    ) -> ProcessingResult:
        """
        Extract with caching support.

        Checks cache first, calls extractor on miss, caches result.
        """
        # Try cache first
        if self._cache:
            cached_data = self._cache.get(aspect, content_hash, model_class)
            if cached_data is not None:
                logger.info(f"Cache hit for {aspect}")
                return ProcessingResult(
                    success=True,
                    data=cached_data,
                    tokens_used=0,
                    cost_usd=0.0,
                    latency_ms=0.0,
                )

        # Cache miss - call extractor
        result = await extractor_func(content, product_name)

        # Cache successful results
        if result.success and result.data and self._cache:
            self._cache.set(aspect, content_hash, result.data)

        return result

    def get_cache_stats(self) -> Optional[dict[str, Any]]:
        """Get cache statistics if caching is enabled."""
        if self._cache:
            return self._cache.get_stats().to_dict()
        return None

    def clear_cache(self) -> None:
        """Clear the processing cache."""
        if self._cache:
            self._cache.clear()

    async def _detect_product_name(self, content: str) -> str:
        """Auto-detect the product name from content."""
        client = self._ensure_client()

        prompt = f"""Analyze this product documentation and identify the product name.
Return ONLY the product name, nothing else.

Content (first 2000 chars):
{content[:2000]}

Product name:"""

        try:
            settings = ModelSettings(
                model=ModelName.HAIKU,
                max_tokens=50,
                temperature=0.0,
            )
            response = await client.complete(prompt, settings=settings)
            return response.content.strip().strip('"').strip("'")
        except Exception as e:
            logger.warning(f"Failed to detect product name: {e}")
            return "Unknown Product"

    async def _extract_features(
        self, content: str, product_name: str
    ) -> ProcessingResult:
        """Extract product features from content."""
        client = self._ensure_client()
        start_time = time.time()

        prompt = f"""Analyze this product documentation for {product_name} and extract ALL product features.

## Content
{content[:80000]}

## Instructions
For each feature found, provide:
1. name: Clear, concise feature name
2. description: What the feature does
3. category: One of [core, integration, security, performance, usability, analytics, collaboration, automation, customization, support, other]
4. benefits: List of customer benefits
5. use_cases: When/how to use it
6. technical_details: Any specs or requirements
7. is_flagship: true if this is a headline/major feature
8. is_unique: true if this appears to be unique to this product

Return a JSON object with this structure:
{{
  "features": [
    {{
      "name": "Feature Name",
      "description": "Description",
      "category": "core",
      "benefits": ["benefit1", "benefit2"],
      "use_cases": ["use case 1"],
      "technical_details": "optional tech details",
      "is_flagship": false,
      "is_unique": false
    }}
  ]
}}

Extract ALL features you can find. Be thorough."""

        try:
            settings = ModelSettings(
                model=self.config.extraction_model,
                max_tokens=8000,
                temperature=0.2,
            )
            response = await client.complete(prompt, settings=settings)

            data = parse_json(response.content)
            features_data = data.get("features", [])

            features = []
            for f in features_data:
                try:
                    category = FeatureCategory(f.get("category", "other"))
                except ValueError:
                    category = FeatureCategory.OTHER

                features.append(
                    ProductFeature(
                        name=f.get("name", ""),
                        description=f.get("description", ""),
                        category=category,
                        benefits=f.get("benefits", []),
                        use_cases=f.get("use_cases", []),
                        technical_details=f.get("technical_details"),
                        is_flagship=f.get("is_flagship", False),
                        is_unique=f.get("is_unique", False),
                    )
                )

            feature_set = FeatureSet(features=features)
            feature_set.compute_stats()

            return ProcessingResult(
                success=True,
                data=feature_set,
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return ProcessingResult(success=False, error=str(e))

    async def _extract_benefits(
        self, content: str, product_name: str
    ) -> ProcessingResult:
        """Extract customer benefits from content."""
        client = self._ensure_client()
        start_time = time.time()

        prompt = f"""Analyze this product documentation for {product_name} and identify the KEY CUSTOMER BENEFITS.

## Content
{content[:60000]}

## Instructions
Focus on BENEFITS (what customers gain), not just features (what the product does).
Transform technical capabilities into business/user value.

For each benefit:
1. headline: Short, impactful benefit statement (action-oriented)
2. description: Expanded explanation
3. target_audience: Who benefits most [technical, business, executive, end_user, enterprise, smb, startup]
4. supporting_features: Which features enable this benefit
5. proof_points: Evidence or metrics if available
6. business_impact: ROI or business impact if mentioned

Return JSON:
{{
  "benefits": [
    {{
      "headline": "Reduce deployment time by 80%",
      "description": "Automated CI/CD pipelines eliminate manual deployment steps...",
      "target_audience": ["technical", "business"],
      "supporting_features": ["Auto-deployment", "CI/CD Integration"],
      "proof_points": ["Case study: Company X reduced deployment from 4 hours to 45 minutes"],
      "business_impact": "Faster time-to-market, reduced operational costs"
    }}
  ]
}}

Focus on the most compelling benefits."""

        try:
            settings = ModelSettings(
                model=self.config.extraction_model,
                max_tokens=6000,
                temperature=0.3,
            )
            response = await client.complete(prompt, settings=settings)

            data = parse_json(response.content)
            benefits_data = data.get("benefits", [])

            benefits = []
            for b in benefits_data:
                audiences = []
                for aud in b.get("target_audience", []):
                    try:
                        audiences.append(AudienceType(aud))
                    except ValueError:
                        pass

                benefits.append(
                    CustomerBenefit(
                        headline=b.get("headline", ""),
                        description=b.get("description", ""),
                        target_audience=audiences,
                        supporting_features=b.get("supporting_features", []),
                        proof_points=b.get("proof_points", []),
                        business_impact=b.get("business_impact"),
                    )
                )

            benefit_set = BenefitSet(benefits=benefits)
            benefit_set.compute_stats()

            return ProcessingResult(
                success=True,
                data=benefit_set,
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Benefit extraction failed: {e}")
            return ProcessingResult(success=False, error=str(e))

    async def _extract_use_cases(
        self, content: str, product_name: str
    ) -> ProcessingResult:
        """Extract use cases from content."""
        client = self._ensure_client()
        start_time = time.time()

        prompt = f"""Analyze this product documentation for {product_name} and identify USE CASES.

## Content
{content[:60000]}

## Instructions
Identify real-world scenarios where this product is used.

For each use case:
1. title: Descriptive title
2. scenario: The situation/context
3. problem_solved: What problem this addresses
4. solution_approach: How the product solves it
5. target_audience: Who this use case is for
6. industry_vertical: Specific industry if applicable
7. key_features_used: Features required
8. expected_outcomes: Results/benefits

Return JSON:
{{
  "use_cases": [
    {{
      "title": "Enterprise Data Migration",
      "scenario": "Large organizations migrating from legacy systems...",
      "problem_solved": "Complex, risky data migrations that take months",
      "solution_approach": "Automated migration with validation and rollback",
      "target_audience": ["enterprise", "technical"],
      "industry_vertical": "Financial Services",
      "key_features_used": ["Data Mapping", "Validation Engine"],
      "expected_outcomes": ["50% faster migrations", "Zero data loss"]
    }}
  ]
}}"""

        try:
            settings = ModelSettings(
                model=self.config.extraction_model,
                max_tokens=6000,
                temperature=0.3,
            )
            response = await client.complete(prompt, settings=settings)

            data = parse_json(response.content)
            use_cases_data = data.get("use_cases", [])

            use_cases = []
            for uc in use_cases_data:
                audiences = []
                for aud in uc.get("target_audience", []):
                    try:
                        audiences.append(AudienceType(aud))
                    except ValueError:
                        pass

                use_cases.append(
                    UseCase(
                        title=uc.get("title", ""),
                        scenario=uc.get("scenario", ""),
                        problem_solved=uc.get("problem_solved", ""),
                        solution_approach=uc.get("solution_approach", ""),
                        target_audience=audiences,
                        industry_vertical=uc.get("industry_vertical"),
                        key_features_used=uc.get("key_features_used", []),
                        expected_outcomes=uc.get("expected_outcomes", []),
                    )
                )

            use_case_set = UseCaseSet(use_cases=use_cases)
            use_case_set.compute_stats()

            return ProcessingResult(
                success=True,
                data=use_case_set,
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Use case extraction failed: {e}")
            return ProcessingResult(success=False, error=str(e))

    async def _analyze_competitive(
        self, content: str, product_name: str
    ) -> ProcessingResult:
        """Analyze competitive positioning."""
        client = self._ensure_client()
        start_time = time.time()

        prompt = f"""Analyze this product documentation for {product_name} and identify COMPETITIVE DIFFERENTIATORS.

## Content
{content[:60000]}

## Instructions
Look for:
- Unique capabilities not common in the market
- Explicit comparisons to competitors
- Claims of being "first", "only", "best"
- Market positioning statements

Return JSON:
{{
  "differentiators": [
    {{
      "claim": "Only solution with real-time sync",
      "explanation": "Why this matters to customers",
      "evidence": ["Proof point 1", "Proof point 2"],
      "compared_to": ["Competitor A", "market alternatives"],
      "strength": "strong"
    }}
  ],
  "unique_capabilities": ["Capability 1", "Capability 2"],
  "market_position": "Leader in X market segment",
  "mentioned_competitors": ["Competitor A", "Competitor B"],
  "competitive_advantages": ["Advantage 1", "Advantage 2"]
}}

strength can be: strong, moderate, weak"""

        try:
            settings = ModelSettings(
                model=self.config.analysis_model,
                max_tokens=4000,
                temperature=0.3,
            )
            response = await client.complete(prompt, settings=settings)

            data = parse_json(response.content)

            differentiators = []
            for d in data.get("differentiators", []):
                differentiators.append(
                    Differentiator(
                        claim=d.get("claim", ""),
                        explanation=d.get("explanation", ""),
                        evidence=d.get("evidence", []),
                        compared_to=d.get("compared_to", []),
                        strength=d.get("strength", "moderate"),
                    )
                )

            analysis = CompetitiveAnalysis(
                differentiators=differentiators,
                unique_capabilities=data.get("unique_capabilities", []),
                market_position=data.get("market_position"),
                mentioned_competitors=data.get("mentioned_competitors", []),
                competitive_advantages=data.get("competitive_advantages", []),
            )

            return ProcessingResult(
                success=True,
                data=analysis,
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Competitive analysis failed: {e}")
            return ProcessingResult(success=False, error=str(e))

    async def _analyze_audience(
        self, content: str, product_name: str
    ) -> ProcessingResult:
        """Analyze target audiences."""
        client = self._ensure_client()
        start_time = time.time()

        prompt = f"""Analyze this product documentation for {product_name} and identify TARGET AUDIENCES.

## Content
{content[:60000]}

## Instructions
Identify distinct audience segments and their needs.

For each segment:
1. segment_type: [technical, business, executive, end_user, enterprise, smb, startup]
2. name: Specific persona name (e.g., "DevOps Engineers")
3. description: Who they are
4. pain_points: Their challenges
5. goals: What they want to achieve
6. relevant_features: Features most relevant to them
7. relevant_benefits: Benefits that resonate
8. messaging_tone: Recommended tone (technical, business, casual, etc.)
9. key_messages: Key messages for this segment

Return JSON:
{{
  "segments": [...],
  "primary_audience": "DevOps Engineers",
  "secondary_audiences": ["Engineering Managers", "CTOs"],
  "buyer_vs_user": "Technical users evaluate, executives approve budget"
}}"""

        try:
            settings = ModelSettings(
                model=self.config.analysis_model,
                max_tokens=5000,
                temperature=0.3,
            )
            response = await client.complete(prompt, settings=settings)

            data = parse_json(response.content)

            segments = []
            for s in data.get("segments", []):
                try:
                    segment_type = AudienceType(s.get("segment_type", "end_user"))
                except ValueError:
                    segment_type = AudienceType.END_USER

                segments.append(
                    AudienceSegment(
                        segment_type=segment_type,
                        name=s.get("name", ""),
                        description=s.get("description", ""),
                        pain_points=s.get("pain_points", []),
                        goals=s.get("goals", []),
                        relevant_features=s.get("relevant_features", []),
                        relevant_benefits=s.get("relevant_benefits", []),
                        messaging_tone=s.get("messaging_tone"),
                        key_messages=s.get("key_messages", []),
                    )
                )

            analysis = AudienceAnalysis(
                segments=segments,
                primary_audience=data.get("primary_audience"),
                secondary_audiences=data.get("secondary_audiences", []),
                buyer_vs_user=data.get("buyer_vs_user"),
            )

            return ProcessingResult(
                success=True,
                data=analysis,
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Audience analysis failed: {e}")
            return ProcessingResult(success=False, error=str(e))

    async def _extract_pricing(
        self, content: str, product_name: str
    ) -> ProcessingResult:
        """Extract pricing information."""
        client = self._ensure_client()
        start_time = time.time()

        prompt = f"""Analyze this product documentation for {product_name} and extract PRICING INFORMATION.

## Content
{content[:40000]}

## Instructions
Look for:
- Pricing tiers/plans
- Prices and billing periods
- Features per tier
- Free trial information
- Enterprise/custom pricing

Return JSON:
{{
  "pricing_model": "tiered",
  "has_free_tier": true,
  "has_free_trial": true,
  "trial_duration": "14 days",
  "tiers": [
    {{
      "name": "Free",
      "price": "$0",
      "billing_period": "monthly",
      "features_included": ["Feature 1", "Feature 2"],
      "limitations": ["Up to 3 users"],
      "is_popular": false,
      "is_enterprise": false
    }}
  ],
  "currency": "USD",
  "notes": ["Volume discounts available"]
}}

pricing_model options: free, freemium, subscription, usage_based, per_seat, tiered, enterprise, contact_sales, one_time

If no pricing found, return {{"pricing_model": "contact_sales", "notes": ["Pricing not publicly available"]}}"""

        try:
            settings = ModelSettings(
                model=self.config.extraction_model,
                max_tokens=4000,
                temperature=0.1,
            )
            response = await client.complete(prompt, settings=settings)

            data = parse_json(response.content)

            try:
                pricing_model = PricingModel(data.get("pricing_model", "contact_sales"))
            except ValueError:
                pricing_model = PricingModel.CONTACT_SALES

            tiers = []
            for t in data.get("tiers", []):
                tiers.append(
                    PricingTier(
                        name=t.get("name", ""),
                        price=t.get("price"),
                        billing_period=t.get("billing_period"),
                        features_included=t.get("features_included", []),
                        limitations=t.get("limitations", []),
                        target_audience=t.get("target_audience"),
                        is_popular=t.get("is_popular", False),
                        is_enterprise=t.get("is_enterprise", False),
                    )
                )

            pricing = PricingInfo(
                pricing_model=pricing_model,
                has_free_tier=data.get("has_free_tier", False),
                has_free_trial=data.get("has_free_trial", False),
                trial_duration=data.get("trial_duration"),
                tiers=tiers,
                currency=data.get("currency", "USD"),
                notes=data.get("notes", []),
            )

            return ProcessingResult(
                success=True,
                data=pricing,
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Pricing extraction failed: {e}")
            return ProcessingResult(success=False, error=str(e))

    async def _extract_technical(
        self, content: str, product_name: str
    ) -> ProcessingResult:
        """Extract technical specifications."""
        client = self._ensure_client()
        start_time = time.time()

        prompt = f"""Analyze this product documentation for {product_name} and extract TECHNICAL SPECIFICATIONS.

## Content
{content[:50000]}

## Instructions
Look for:
- Supported platforms/browsers
- Deployment options
- Integrations and APIs
- Security and compliance
- System requirements
- Performance specs

Return JSON:
{{
  "platforms_supported": ["Windows", "macOS", "Linux", "Web"],
  "deployment_options": ["Cloud (SaaS)", "On-premise", "Hybrid"],
  "integrations": [
    {{
      "name": "Slack",
      "type": "native",
      "description": "Native Slack integration for notifications"
    }}
  ],
  "api_available": true,
  "api_type": "REST",
  "security_certifications": ["SOC 2", "ISO 27001"],
  "compliance_standards": ["GDPR", "HIPAA"],
  "system_requirements": {{"RAM": "4GB minimum", "Storage": "10GB"}},
  "performance_metrics": {{"uptime": "99.9%", "latency": "<100ms"}},
  "data_residency_options": ["US", "EU", "APAC"]
}}"""

        try:
            settings = ModelSettings(
                model=self.config.extraction_model,
                max_tokens=4000,
                temperature=0.1,
            )
            response = await client.complete(prompt, settings=settings)

            data = parse_json(response.content)

            integrations = []
            for i in data.get("integrations", []):
                integrations.append(
                    Integration(
                        name=i.get("name", ""),
                        type=i.get("type", ""),
                        description=i.get("description"),
                        documentation_url=i.get("documentation_url"),
                    )
                )

            specs = TechnicalSpecs(
                platforms_supported=data.get("platforms_supported", []),
                deployment_options=data.get("deployment_options", []),
                integrations=integrations,
                api_available=data.get("api_available", False),
                api_type=data.get("api_type"),
                security_certifications=data.get("security_certifications", []),
                compliance_standards=data.get("compliance_standards", []),
                system_requirements=data.get("system_requirements", {}),
                performance_metrics=data.get("performance_metrics", {}),
                data_residency_options=data.get("data_residency_options", []),
            )

            return ProcessingResult(
                success=True,
                data=specs,
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Technical extraction failed: {e}")
            return ProcessingResult(success=False, error=str(e))

    async def _generate_summary(
        self, content: str, product_name: str
    ) -> ProcessingResult:
        """Generate multi-level summaries."""
        client = self._ensure_client()
        start_time = time.time()

        prompt = f"""Analyze this product documentation for {product_name} and create SUMMARIES.

## Content
{content[:80000]}

## Instructions
Create multiple summary levels:

1. executive_summary: 1-2 sentences for executives
2. detailed_summary: 1 paragraph (4-6 sentences)
3. comprehensive_summary: Full summary with sections (features, benefits, use cases)
4. key_points: 5-7 bullet points of most important info
5. product_category: What type of product is this?
6. tagline: The product's tagline if present
7. value_proposition: Core value prop in one sentence

Return JSON:
{{
  "executive_summary": "...",
  "detailed_summary": "...",
  "comprehensive_summary": "...",
  "key_points": ["point 1", "point 2"],
  "product_category": "Developer Tools",
  "tagline": "Build faster, ship sooner",
  "value_proposition": "..."
}}"""

        try:
            settings = ModelSettings(
                model=self.config.summarization_model,
                max_tokens=4000,
                temperature=0.4,
            )
            response = await client.complete(prompt, settings=settings)

            data = parse_json(response.content)

            summary = ContentSummary(
                executive_summary=data.get("executive_summary", ""),
                detailed_summary=data.get("detailed_summary", ""),
                comprehensive_summary=data.get("comprehensive_summary", ""),
                key_points=data.get("key_points", []),
                product_category=data.get("product_category"),
                tagline=data.get("tagline"),
                value_proposition=data.get("value_proposition"),
            )

            return ProcessingResult(
                success=True,
                data=summary,
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return ProcessingResult(success=False, error=str(e))

    async def process_single_aspect(
        self,
        content: ExtractedContent,
        aspect: str,
        product_name: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Process only a single aspect of the content.

        Args:
            content: Extracted content
            aspect: One of: features, benefits, use_cases, competitive, audience, pricing, technical, summary
            product_name: Product name override

        Returns:
            ProcessingResult with the extracted data
        """
        chunks = self._chunker.chunk_extracted_content(content)
        combined = self._combine_chunks_for_analysis(chunks)
        name = product_name or content.product_name or await self._detect_product_name(combined)

        aspect_map = {
            "features": self._extract_features,
            "benefits": self._extract_benefits,
            "use_cases": self._extract_use_cases,
            "competitive": self._analyze_competitive,
            "audience": self._analyze_audience,
            "pricing": self._extract_pricing,
            "technical": self._extract_technical,
            "summary": self._generate_summary,
        }

        if aspect not in aspect_map:
            return ProcessingResult(
                success=False,
                error=f"Unknown aspect: {aspect}. Valid: {list(aspect_map.keys())}",
            )

        return await aspect_map[aspect](combined, name)

    async def _map_visuals_to_features(
        self,
        features: FeatureSet,
        inventory: VisualInventory,
        product_name: str,
    ) -> dict[str, list[str]]:
        """
        Use LLM to map visual assets to features based on metadata.

        Args:
            features: Extracted features
            inventory: Visual asset inventory
            product_name: Product name for context

        Returns:
            Dict mapping feature names to lists of asset IDs
        """
        client = self._ensure_client()

        # Format visual metadata for LLM
        visual_descriptions = []
        for asset in inventory.images + inventory.tables:
            desc = asset.get_description_for_matching()
            visual_descriptions.append(f"- {asset.asset_id}: {desc}")

        if not visual_descriptions:
            return {}

        # Format features
        feature_names = [f.name for f in features.features]

        prompt = f"""Match visual assets to product features for {product_name}.

## Features
{chr(10).join(f"- {name}" for name in feature_names)}

## Visual Assets
{chr(10).join(visual_descriptions[:30])}

## Instructions
For each feature, identify which visual assets (by ID) would best demonstrate or support it.
Only match visuals that are clearly relevant based on their metadata.
Not every feature needs a visual, and not every visual needs to be matched.

Return JSON:
{{
  "mappings": {{
    "Feature Name": ["asset_id_1", "asset_id_2"],
    "Another Feature": ["asset_id_3"]
  }}
}}

Only include features that have relevant visuals."""

        try:
            settings = ModelSettings(
                model=ModelName.HAIKU,  # Use fast model for this
                max_tokens=2000,
                temperature=0.1,
            )
            response = await client.complete(prompt, settings=settings)

            data = parse_json(response.content)
            mappings = data.get("mappings", {})

            # Validate that asset IDs exist
            valid_ids = {
                a.asset_id for a in inventory.images + inventory.tables + inventory.videos
            }
            validated_mappings: dict[str, list[str]] = {}
            for feature, asset_ids in mappings.items():
                valid_asset_ids = [aid for aid in asset_ids if aid in valid_ids]
                if valid_asset_ids:
                    validated_mappings[feature] = valid_asset_ids

            logger.info(f"Mapped visuals to {len(validated_mappings)} features")
            return validated_mappings

        except Exception as e:
            logger.warning(f"Feature-visual mapping failed: {e}")
            return {}
