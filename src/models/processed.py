"""Processed content models for LLM-analyzed product information."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AudienceType(str, Enum):
    """Target audience categories."""

    TECHNICAL = "technical"  # Developers, engineers, IT
    BUSINESS = "business"  # Business users, managers
    EXECUTIVE = "executive"  # C-suite, decision makers
    END_USER = "end_user"  # General users
    ENTERPRISE = "enterprise"  # Enterprise buyers
    SMB = "smb"  # Small/medium business
    STARTUP = "startup"  # Startups


class FeatureCategory(str, Enum):
    """Categories for product features."""

    CORE = "core"  # Core functionality
    INTEGRATION = "integration"  # Integrations & APIs
    SECURITY = "security"  # Security features
    PERFORMANCE = "performance"  # Performance & scalability
    USABILITY = "usability"  # UX & ease of use
    ANALYTICS = "analytics"  # Reporting & analytics
    COLLABORATION = "collaboration"  # Team features
    AUTOMATION = "automation"  # Automation capabilities
    CUSTOMIZATION = "customization"  # Customization options
    SUPPORT = "support"  # Support & services
    OTHER = "other"


class PricingModel(str, Enum):
    """Types of pricing models."""

    FREE = "free"
    FREEMIUM = "freemium"
    SUBSCRIPTION = "subscription"
    USAGE_BASED = "usage_based"
    PER_SEAT = "per_seat"
    TIERED = "tiered"
    ENTERPRISE = "enterprise"
    CONTACT_SALES = "contact_sales"
    ONE_TIME = "one_time"


# ============================================================================
# Feature Models
# ============================================================================


class ProductFeature(BaseModel):
    """A single product feature with full analysis."""

    name: str = Field(description="Clear, concise feature name")
    description: str = Field(description="What the feature does")
    category: FeatureCategory = FeatureCategory.OTHER
    benefits: list[str] = Field(
        default_factory=list,
        description="Customer benefits from this feature",
    )
    use_cases: list[str] = Field(
        default_factory=list,
        description="When/how to use this feature",
    )
    technical_details: Optional[str] = Field(
        default=None,
        description="Technical specifications or requirements",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations or constraints",
    )
    related_features: list[str] = Field(
        default_factory=list,
        description="Names of related features",
    )
    is_unique: bool = Field(
        default=False,
        description="Is this a unique/differentiating feature",
    )
    is_flagship: bool = Field(
        default=False,
        description="Is this a flagship/headline feature",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in extraction accuracy",
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description="URLs where this feature was found",
    )


class FeatureSet(BaseModel):
    """Collection of all extracted features."""

    features: list[ProductFeature] = Field(default_factory=list)
    flagship_features: list[str] = Field(
        default_factory=list,
        description="Names of flagship/headline features",
    )
    feature_count_by_category: dict[str, int] = Field(default_factory=dict)

    def compute_stats(self) -> None:
        """Compute feature statistics."""
        self.flagship_features = [f.name for f in self.features if f.is_flagship]
        category_counts: dict[str, int] = {}
        for feature in self.features:
            cat = feature.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        self.feature_count_by_category = category_counts


# ============================================================================
# Benefit Models
# ============================================================================


class CustomerBenefit(BaseModel):
    """A customer-facing benefit derived from features."""

    headline: str = Field(description="Short, impactful benefit statement")
    description: str = Field(description="Expanded explanation of the benefit")
    target_audience: list[AudienceType] = Field(
        default_factory=list,
        description="Who benefits most",
    )
    supporting_features: list[str] = Field(
        default_factory=list,
        description="Features that enable this benefit",
    )
    proof_points: list[str] = Field(
        default_factory=list,
        description="Evidence or metrics supporting this benefit",
    )
    emotional_appeal: Optional[str] = Field(
        default=None,
        description="Emotional aspect of the benefit",
    )
    business_impact: Optional[str] = Field(
        default=None,
        description="Business/ROI impact",
    )


class BenefitSet(BaseModel):
    """Collection of customer benefits."""

    benefits: list[CustomerBenefit] = Field(default_factory=list)
    top_benefits: list[str] = Field(
        default_factory=list,
        description="Headlines of top 3-5 benefits",
    )
    benefits_by_audience: dict[str, list[str]] = Field(default_factory=dict)

    def compute_stats(self) -> None:
        """Compute benefit statistics."""
        self.top_benefits = [b.headline for b in self.benefits[:5]]
        audience_benefits: dict[str, list[str]] = {}
        for benefit in self.benefits:
            for audience in benefit.target_audience:
                key = audience.value
                if key not in audience_benefits:
                    audience_benefits[key] = []
                audience_benefits[key].append(benefit.headline)
        self.benefits_by_audience = audience_benefits


# ============================================================================
# Use Case Models
# ============================================================================


class UseCase(BaseModel):
    """A specific use case or application scenario."""

    title: str = Field(description="Use case title")
    scenario: str = Field(description="Description of the scenario")
    problem_solved: str = Field(description="What problem this addresses")
    solution_approach: str = Field(description="How the product solves it")
    target_audience: list[AudienceType] = Field(default_factory=list)
    industry_vertical: Optional[str] = Field(
        default=None,
        description="Specific industry if applicable",
    )
    key_features_used: list[str] = Field(
        default_factory=list,
        description="Features required for this use case",
    )
    expected_outcomes: list[str] = Field(
        default_factory=list,
        description="Expected results/benefits",
    )
    complexity: Optional[str] = Field(
        default=None,
        description="Implementation complexity (simple/moderate/complex)",
    )


class UseCaseSet(BaseModel):
    """Collection of use cases."""

    use_cases: list[UseCase] = Field(default_factory=list)
    primary_use_cases: list[str] = Field(
        default_factory=list,
        description="Titles of primary use cases",
    )
    industries_served: list[str] = Field(default_factory=list)

    def compute_stats(self) -> None:
        """Compute use case statistics."""
        self.primary_use_cases = [uc.title for uc in self.use_cases[:5]]
        industries = set()
        for uc in self.use_cases:
            if uc.industry_vertical:
                industries.add(uc.industry_vertical)
        self.industries_served = list(industries)


# ============================================================================
# Competitive Analysis Models
# ============================================================================


class Differentiator(BaseModel):
    """A competitive differentiator."""

    claim: str = Field(description="The differentiating claim")
    explanation: str = Field(description="Why this matters")
    evidence: list[str] = Field(
        default_factory=list,
        description="Supporting evidence",
    )
    compared_to: list[str] = Field(
        default_factory=list,
        description="Competitors this differentiates from",
    )
    strength: str = Field(
        default="moderate",
        description="Strength of differentiation (strong/moderate/weak)",
    )


class CompetitiveAnalysis(BaseModel):
    """Competitive positioning analysis."""

    differentiators: list[Differentiator] = Field(default_factory=list)
    unique_capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities unique to this product",
    )
    market_position: Optional[str] = Field(
        default=None,
        description="Overall market positioning statement",
    )
    mentioned_competitors: list[str] = Field(
        default_factory=list,
        description="Competitors mentioned in content",
    )
    competitive_advantages: list[str] = Field(
        default_factory=list,
        description="Key competitive advantages",
    )
    potential_weaknesses: list[str] = Field(
        default_factory=list,
        description="Potential areas of weakness (inferred)",
    )


# ============================================================================
# Audience Models
# ============================================================================


class AudienceSegment(BaseModel):
    """A target audience segment."""

    segment_type: AudienceType
    name: str = Field(description="Segment name (e.g., 'DevOps Engineers')")
    description: str = Field(description="Who they are")
    pain_points: list[str] = Field(
        default_factory=list,
        description="Their challenges",
    )
    goals: list[str] = Field(
        default_factory=list,
        description="What they want to achieve",
    )
    relevant_features: list[str] = Field(
        default_factory=list,
        description="Features most relevant to them",
    )
    relevant_benefits: list[str] = Field(
        default_factory=list,
        description="Benefits that resonate with them",
    )
    messaging_tone: Optional[str] = Field(
        default=None,
        description="Recommended messaging tone",
    )
    key_messages: list[str] = Field(
        default_factory=list,
        description="Key messages for this segment",
    )


class AudienceAnalysis(BaseModel):
    """Complete audience analysis."""

    segments: list[AudienceSegment] = Field(default_factory=list)
    primary_audience: Optional[str] = Field(
        default=None,
        description="Primary target audience",
    )
    secondary_audiences: list[str] = Field(default_factory=list)
    buyer_vs_user: Optional[str] = Field(
        default=None,
        description="Notes on buyer vs user distinction",
    )


# ============================================================================
# Pricing Models
# ============================================================================


class PricingTier(BaseModel):
    """A single pricing tier."""

    name: str = Field(description="Tier name (e.g., 'Pro', 'Enterprise')")
    price: Optional[str] = Field(
        default=None,
        description="Price (e.g., '$99/mo', 'Contact Sales')",
    )
    billing_period: Optional[str] = Field(
        default=None,
        description="Billing period (monthly/annual)",
    )
    features_included: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    target_audience: Optional[str] = Field(default=None)
    is_popular: bool = Field(default=False, description="Marked as popular/recommended")
    is_enterprise: bool = Field(default=False)


class PricingInfo(BaseModel):
    """Complete pricing information."""

    pricing_model: PricingModel = PricingModel.CONTACT_SALES
    has_free_tier: bool = False
    has_free_trial: bool = False
    trial_duration: Optional[str] = Field(default=None)
    tiers: list[PricingTier] = Field(default_factory=list)
    currency: str = "USD"
    pricing_page_url: Optional[str] = Field(default=None)
    notes: list[str] = Field(
        default_factory=list,
        description="Additional pricing notes",
    )


# ============================================================================
# Technical Specs Models
# ============================================================================


class Integration(BaseModel):
    """An integration or API capability."""

    name: str
    type: str = Field(description="Type (API, webhook, native, etc.)")
    description: Optional[str] = None
    documentation_url: Optional[str] = None


class TechnicalSpecs(BaseModel):
    """Technical specifications and requirements."""

    platforms_supported: list[str] = Field(default_factory=list)
    deployment_options: list[str] = Field(
        default_factory=list,
        description="Cloud, on-premise, hybrid, etc.",
    )
    integrations: list[Integration] = Field(default_factory=list)
    api_available: bool = False
    api_type: Optional[str] = Field(
        default=None,
        description="REST, GraphQL, etc.",
    )
    security_certifications: list[str] = Field(default_factory=list)
    compliance_standards: list[str] = Field(default_factory=list)
    system_requirements: dict[str, str] = Field(default_factory=dict)
    performance_metrics: dict[str, str] = Field(default_factory=dict)
    data_residency_options: list[str] = Field(default_factory=list)


# ============================================================================
# Summary Models
# ============================================================================


class ContentSummary(BaseModel):
    """Multi-level content summaries."""

    executive_summary: str = Field(
        description="1-2 sentence executive summary",
    )
    detailed_summary: str = Field(
        description="1 paragraph detailed summary",
    )
    comprehensive_summary: str = Field(
        description="Full summary with sections",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Bullet points of key information",
    )
    product_category: Optional[str] = Field(
        default=None,
        description="Product category/type",
    )
    tagline: Optional[str] = Field(
        default=None,
        description="Product tagline if found",
    )
    value_proposition: Optional[str] = Field(
        default=None,
        description="Core value proposition",
    )


# ============================================================================
# Main Processed Content Model
# ============================================================================


class ProcessedContent(BaseModel):
    """
    Complete processed content from LLM analysis.

    This is the main output of the content processing pipeline,
    containing all analyzed and structured information about a product.
    """

    # Identity
    product_name: str
    product_url: str
    processing_id: str = Field(default="")

    # Summaries
    summary: ContentSummary

    # Analysis Results
    features: FeatureSet = Field(default_factory=FeatureSet)
    benefits: BenefitSet = Field(default_factory=BenefitSet)
    use_cases: UseCaseSet = Field(default_factory=UseCaseSet)
    competitive_analysis: CompetitiveAnalysis = Field(
        default_factory=CompetitiveAnalysis
    )
    audience_analysis: AudienceAnalysis = Field(default_factory=AudienceAnalysis)
    pricing: PricingInfo = Field(default_factory=PricingInfo)
    technical_specs: TechnicalSpecs = Field(default_factory=TechnicalSpecs)

    # Metadata
    source_extraction_id: str = Field(
        default="",
        description="ID of the source ExtractedContent",
    )
    source_content_hash: Optional[str] = Field(
        default=None,
        description="Hash of source content for cache validation",
    )
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    processing_duration_ms: int = 0
    total_llm_tokens_used: int = 0
    total_llm_cost_usd: float = 0.0

    # Quality Indicators
    overall_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in processing quality",
    )
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def compute_stats(self) -> None:
        """Compute all statistics for nested models."""
        self.features.compute_stats()
        self.benefits.compute_stats()
        self.use_cases.compute_stats()

    def to_pitch_context(self) -> dict[str, Any]:
        """
        Convert to a context dictionary suitable for pitch generation.

        Returns a flattened structure with the most important information
        for generating sales pitches.
        """
        return {
            "product_name": self.product_name,
            "tagline": self.summary.tagline,
            "value_proposition": self.summary.value_proposition,
            "executive_summary": self.summary.executive_summary,
            "key_points": self.summary.key_points,
            "flagship_features": self.features.flagship_features,
            "all_features": [f.model_dump() for f in self.features.features],
            "top_benefits": self.benefits.top_benefits,
            "all_benefits": [b.model_dump() for b in self.benefits.benefits],
            "primary_use_cases": self.use_cases.primary_use_cases,
            "differentiators": [
                d.claim for d in self.competitive_analysis.differentiators
            ],
            "target_audiences": [
                s.name for s in self.audience_analysis.segments
            ],
            "pricing_model": self.pricing.pricing_model.value,
            "has_free_trial": self.pricing.has_free_trial,
        }
