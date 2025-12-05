"""Pitch generation templates and prompts."""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.models.pitch import PitchConfig, PitchLength, PitchTone, SectionType


@dataclass
class SectionTemplate:
    """Template for generating a pitch section."""

    section_type: SectionType
    name: str
    description: str
    prompt_template: str
    default_order: int
    min_words: int = 50
    max_words: int = 200
    required: bool = False


# ============================================================================
# Section Templates
# ============================================================================

SECTION_TEMPLATES: dict[SectionType, SectionTemplate] = {
    SectionType.HOOK: SectionTemplate(
        section_type=SectionType.HOOK,
        name="Opening Hook",
        description="Attention-grabbing opening statement",
        default_order=1,
        min_words=20,
        max_words=50,
        required=True,
        prompt_template="""Create a compelling opening hook for a sales pitch about {product_name}.

Product Summary: {executive_summary}
Target Audience: {target_audience}
Tone: {tone}

The hook should:
- Immediately grab attention
- Create curiosity or address a pain point
- Be memorable and quotable
- Set up the problem/solution narrative

{custom_instructions}

Return a JSON object:
{{
    "title": "Opening Hook",
    "content": "The hook statement (1-2 sentences)",
    "key_points": [],
    "talking_points": ["Speaker note 1", "Speaker note 2"]
}}""",
    ),

    SectionType.PROBLEM: SectionTemplate(
        section_type=SectionType.PROBLEM,
        name="Problem Statement",
        description="Define the problem being solved",
        default_order=2,
        min_words=50,
        max_words=150,
        prompt_template="""Create a problem statement section for a sales pitch about {product_name}.

Product Summary: {executive_summary}
Target Audience: {target_audience}
Known Pain Points: {pain_points}
Tone: {tone}

The problem statement should:
- Clearly articulate the challenge customers face
- Make the audience feel understood
- Create urgency around solving this problem
- Use specific, relatable examples

Return a JSON object:
{{
    "title": "The Challenge",
    "content": "Problem description paragraph",
    "key_points": ["Pain point 1", "Pain point 2", "Pain point 3"],
    "talking_points": ["Expand on this...", "Ask audience if they relate..."]
}}""",
    ),

    SectionType.SOLUTION: SectionTemplate(
        section_type=SectionType.SOLUTION,
        name="Solution Overview",
        description="Present the product as the solution",
        default_order=3,
        min_words=75,
        max_words=200,
        required=True,
        prompt_template="""Create a solution overview section for {product_name}.

Product Summary: {executive_summary}
Value Proposition: {value_proposition}
Key Features: {key_features}
Tone: {tone}

The solution section should:
- Position the product as the answer to the stated problem
- Provide a high-level overview without too much detail
- Convey confidence and capability
- Transition smoothly from problem to features/benefits

Return a JSON object:
{{
    "title": "The Solution: {product_name}",
    "content": "Solution overview paragraph",
    "key_points": ["Key capability 1", "Key capability 2"],
    "talking_points": ["Emphasize transformation...", "Connect back to pain points..."],
    "visual_suggestions": ["Product screenshot", "Architecture diagram"]
}}""",
    ),

    SectionType.FEATURES: SectionTemplate(
        section_type=SectionType.FEATURES,
        name="Key Features",
        description="Highlight top product features",
        default_order=4,
        min_words=100,
        max_words=300,
        prompt_template="""Create a features section for {product_name}.

Features to highlight:
{features_list}

Target Audience: {target_audience}
Tone: {tone}
Max features to include: {max_features}

For each feature:
- Lead with the benefit, not the feature name
- Keep descriptions concise and impactful
- Include proof points where available

Return a JSON object:
{{
    "title": "Key Features",
    "content": "Brief intro to features",
    "key_points": ["Feature 1: benefit statement", "Feature 2: benefit statement"],
    "talking_points": ["Demo suggestion 1", "Demo suggestion 2"],
    "visual_suggestions": ["Feature screenshot 1", "Feature comparison table"]
}}""",
    ),

    SectionType.BENEFITS: SectionTemplate(
        section_type=SectionType.BENEFITS,
        name="Customer Benefits",
        description="Focus on customer outcomes",
        default_order=5,
        min_words=100,
        max_words=250,
        prompt_template="""Create a benefits section for {product_name}.

Benefits to highlight:
{benefits_list}

Target Audience: {target_audience}
Tone: {tone}

Focus on:
- Tangible outcomes customers achieve
- Emotional benefits (peace of mind, confidence)
- Business impact (ROI, efficiency, growth)
- Quantify where possible

Return a JSON object:
{{
    "title": "What You'll Achieve",
    "content": "Benefits overview",
    "key_points": ["Benefit 1 with impact", "Benefit 2 with impact"],
    "talking_points": ["Share customer success story...", "Quantify the impact..."]
}}""",
    ),

    SectionType.USE_CASES: SectionTemplate(
        section_type=SectionType.USE_CASES,
        name="Use Cases",
        description="Real-world application examples",
        default_order=6,
        min_words=75,
        max_words=200,
        prompt_template="""Create a use cases section for {product_name}.

Use cases:
{use_cases_list}

Target Audience: {target_audience}
Industry: {industry}
Tone: {tone}

Present:
- Relatable scenarios
- Before/after transformation
- Specific outcomes achieved

Return a JSON object:
{{
    "title": "How Teams Use {product_name}",
    "content": "Use cases introduction",
    "key_points": ["Use case 1 summary", "Use case 2 summary"],
    "talking_points": ["Customize based on prospect industry..."]
}}""",
    ),

    SectionType.DIFFERENTIATORS: SectionTemplate(
        section_type=SectionType.DIFFERENTIATORS,
        name="Why Choose Us",
        description="Competitive differentiation",
        default_order=7,
        min_words=75,
        max_words=200,
        prompt_template="""Create a differentiation section for {product_name}.

Differentiators:
{differentiators_list}

Competitors mentioned: {competitors}
Tone: {tone}

Guidelines:
- Focus on unique strengths, not competitor weaknesses
- Be confident but not aggressive
- Highlight what only this product can do
- Support claims with evidence

Return a JSON object:
{{
    "title": "Why {product_name}",
    "content": "Differentiation overview",
    "key_points": ["Unique advantage 1", "Unique advantage 2"],
    "talking_points": ["If asked about competitor X, say..."]
}}""",
    ),

    SectionType.SOCIAL_PROOF: SectionTemplate(
        section_type=SectionType.SOCIAL_PROOF,
        name="Social Proof",
        description="Testimonials and case studies",
        default_order=8,
        min_words=50,
        max_words=150,
        prompt_template="""Create a social proof section for {product_name}.

Available proof points:
{proof_points}

Target Audience: {target_audience}
Tone: {tone}

Include:
- Customer testimonials or quotes
- Case study highlights
- Metrics and results achieved
- Logos or notable customers if available

Return a JSON object:
{{
    "title": "Trusted By",
    "content": "Social proof narrative",
    "key_points": ["Customer result 1", "Customer result 2"],
    "visual_suggestions": ["Customer logos", "Testimonial quote cards"]
}}""",
    ),

    SectionType.PRICING: SectionTemplate(
        section_type=SectionType.PRICING,
        name="Pricing Overview",
        description="Pricing and packaging",
        default_order=9,
        min_words=50,
        max_words=150,
        prompt_template="""Create a pricing section for {product_name}.

Pricing information:
{pricing_info}

Tone: {tone}

Present pricing:
- Lead with value, not cost
- Highlight what's included
- Mention free trial if available
- Frame as investment, not expense

Return a JSON object:
{{
    "title": "Investment",
    "content": "Pricing overview",
    "key_points": ["Tier 1 summary", "Tier 2 summary"],
    "talking_points": ["Address budget concerns...", "ROI justification..."]
}}""",
    ),

    SectionType.TECHNICAL: SectionTemplate(
        section_type=SectionType.TECHNICAL,
        name="Technical Details",
        description="Technical specifications",
        default_order=10,
        min_words=75,
        max_words=200,
        prompt_template="""Create a technical details section for {product_name}.

Technical specs:
{technical_specs}

Target Audience: {target_audience}
Tone: {tone}

Cover:
- Integration capabilities
- Security and compliance
- Platform support
- Performance characteristics

Return a JSON object:
{{
    "title": "Technical Overview",
    "content": "Technical summary",
    "key_points": ["Tech highlight 1", "Tech highlight 2"],
    "talking_points": ["For technical buyers, emphasize..."]
}}""",
    ),

    SectionType.ROI: SectionTemplate(
        section_type=SectionType.ROI,
        name="ROI / Value",
        description="Return on investment analysis",
        default_order=11,
        min_words=75,
        max_words=200,
        prompt_template="""Create an ROI section for {product_name}.

Benefits: {benefits_list}
Pricing: {pricing_info}
Tone: {tone}

Calculate/present:
- Time savings
- Cost reductions
- Revenue impact
- Payback period

Return a JSON object:
{{
    "title": "The Value",
    "content": "ROI narrative",
    "key_points": ["ROI metric 1", "ROI metric 2"],
    "talking_points": ["Help prospect calculate their specific ROI..."]
}}""",
    ),

    SectionType.OBJECTION_HANDLING: SectionTemplate(
        section_type=SectionType.OBJECTION_HANDLING,
        name="Addressing Concerns",
        description="Common objections and responses",
        default_order=12,
        min_words=100,
        max_words=250,
        prompt_template="""Create objection handling content for {product_name}.

Product info: {executive_summary}
Common concerns in this space: {objections}
Tone: {tone}

Address:
- Price objections
- Switching costs
- Implementation concerns
- Competition comparisons

Return a JSON object:
{{
    "title": "Common Questions",
    "content": "Transition to addressing concerns",
    "key_points": ["Concern 1 addressed", "Concern 2 addressed"],
    "talking_points": ["If they say X, respond with..."]
}}""",
    ),

    SectionType.CTA: SectionTemplate(
        section_type=SectionType.CTA,
        name="Call to Action",
        description="Clear next steps",
        default_order=13,
        min_words=30,
        max_words=100,
        required=True,
        prompt_template="""Create a call to action for {product_name}.

Available actions:
- Free trial: {has_free_trial}
- Demo: available
- Pricing model: {pricing_model}

Tone: {tone}
Target Audience: {target_audience}

The CTA should:
- Be clear and specific
- Create urgency without being pushy
- Offer a low-friction next step
- Include alternative options

Return a JSON object:
{{
    "title": "Get Started",
    "content": "CTA paragraph",
    "key_points": ["Primary action", "Alternative action"],
    "talking_points": ["Ask for the meeting...", "Offer to send materials..."]
}}""",
    ),

    SectionType.CLOSING: SectionTemplate(
        section_type=SectionType.CLOSING,
        name="Closing Statement",
        description="Memorable closing",
        default_order=14,
        min_words=20,
        max_words=75,
        prompt_template="""Create a closing statement for a pitch about {product_name}.

Key message: {value_proposition}
Tone: {tone}

The closing should:
- Reinforce the main value proposition
- Be memorable
- Leave the audience wanting to learn more
- Circle back to the opening hook

Return a JSON object:
{{
    "title": "In Summary",
    "content": "Closing statement",
    "key_points": [],
    "talking_points": ["End with confidence...", "Pause for questions..."]
}}""",
    ),
}


# ============================================================================
# Tone Guidelines
# ============================================================================

TONE_GUIDELINES: dict[PitchTone, str] = {
    PitchTone.PROFESSIONAL: """
- Use formal but accessible language
- Avoid slang and overly casual expressions
- Focus on facts and business outcomes
- Maintain credibility and trustworthiness
- Use third-person perspective when appropriate
""",
    PitchTone.CONVERSATIONAL: """
- Use friendly, approachable language
- Include rhetorical questions to engage
- Use "you" and "your" frequently
- Feel like a helpful conversation
- Include relatable examples
""",
    PitchTone.TECHNICAL: """
- Include specific technical details
- Use industry terminology appropriately
- Focus on architecture, integration, and specs
- Provide precise metrics and benchmarks
- Address technical concerns directly
""",
    PitchTone.EXECUTIVE: """
- Lead with business impact and ROI
- Keep content high-level and strategic
- Focus on competitive advantage
- Use executive-friendly language
- Emphasize time-to-value and risk mitigation
""",
    PitchTone.ENTHUSIASTIC: """
- Use energetic, dynamic language
- Include powerful action verbs
- Create excitement about possibilities
- Use superlatives appropriately
- Convey genuine passion for the solution
""",
    PitchTone.CONSULTATIVE: """
- Position as a trusted advisor
- Ask questions and explore needs
- Provide insights and recommendations
- Focus on solving specific problems
- Demonstrate deep understanding
""",
}


# ============================================================================
# Length Guidelines
# ============================================================================

LENGTH_GUIDELINES: dict[PitchLength, dict[str, Any]] = {
    PitchLength.ELEVATOR: {
        "total_words": 100,
        "sections": [SectionType.HOOK, SectionType.SOLUTION, SectionType.CTA],
        "duration_seconds": 30,
    },
    PitchLength.SHORT: {
        "total_words": 300,
        "sections": [
            SectionType.HOOK,
            SectionType.PROBLEM,
            SectionType.SOLUTION,
            SectionType.BENEFITS,
            SectionType.CTA,
        ],
        "duration_seconds": 120,
    },
    PitchLength.STANDARD: {
        "total_words": 750,
        "sections": [
            SectionType.HOOK,
            SectionType.PROBLEM,
            SectionType.SOLUTION,
            SectionType.FEATURES,
            SectionType.BENEFITS,
            SectionType.DIFFERENTIATORS,
            SectionType.CTA,
        ],
        "duration_seconds": 300,
    },
    PitchLength.DETAILED: {
        "total_words": 1500,
        "sections": [
            SectionType.HOOK,
            SectionType.PROBLEM,
            SectionType.SOLUTION,
            SectionType.FEATURES,
            SectionType.BENEFITS,
            SectionType.USE_CASES,
            SectionType.DIFFERENTIATORS,
            SectionType.SOCIAL_PROOF,
            SectionType.PRICING,
            SectionType.CTA,
        ],
        "duration_seconds": 600,
    },
    PitchLength.COMPREHENSIVE: {
        "total_words": 3000,
        "sections": list(SectionType),
        "duration_seconds": 1200,
    },
}


# ============================================================================
# Master Prompts
# ============================================================================

ELEVATOR_PITCH_PROMPT = """Create a 30-second elevator pitch for {product_name}.

Product Summary: {executive_summary}
Value Proposition: {value_proposition}
Key Benefits: {key_benefits}
Target Audience: {target_audience}

The elevator pitch should:
- Be exactly 2-3 sentences (under 100 words)
- Immediately convey what the product does
- Highlight the primary benefit
- Be memorable and repeatable
- Work in any context (meeting, networking, etc.)

Return ONLY the elevator pitch text, no JSON formatting."""


ONE_LINER_PROMPT = """Create a single-sentence description of {product_name}.

Product Summary: {executive_summary}
Category: {product_category}

The one-liner should:
- Be under 20 words
- Clearly state what the product does
- Be suitable for a website tagline or intro
- Not use jargon

Return ONLY the one-liner text."""


KEY_MESSAGES_PROMPT = """Create 3-5 key messages for {product_name} that sales reps should remember.

Product Summary: {executive_summary}
Key Features: {key_features}
Key Benefits: {key_benefits}
Differentiators: {differentiators}

Each key message should:
- Be concise (under 15 words)
- Convey a single important point
- Be easy to remember and repeat
- Support the overall value proposition

Return a JSON array of strings:
["Key message 1", "Key message 2", "Key message 3"]"""


OBJECTION_RESPONSES_PROMPT = """Create responses to common objections for {product_name}.

Product Info: {executive_summary}
Pricing: {pricing_info}
Competitors: {competitors}

Common objections to address:
1. "It's too expensive"
2. "We're already using [competitor]"
3. "We don't have time to implement something new"
4. "We need to think about it"
5. "Can you prove the ROI?"

For each objection, provide a concise, empathetic response that:
- Acknowledges the concern
- Reframes the perspective
- Provides evidence or reassurance
- Moves toward next steps

Return a JSON object mapping objection to response:
{{
    "It's too expensive": "Response...",
    "We're already using a competitor": "Response...",
    ...
}}"""


def get_section_template(section_type: SectionType) -> SectionTemplate:
    """Get the template for a specific section type."""
    return SECTION_TEMPLATES.get(section_type)


def get_tone_guidelines(tone: PitchTone) -> str:
    """Get writing guidelines for a specific tone."""
    return TONE_GUIDELINES.get(tone, TONE_GUIDELINES[PitchTone.PROFESSIONAL])


def get_length_config(length: PitchLength) -> dict[str, Any]:
    """Get configuration for a specific pitch length."""
    return LENGTH_GUIDELINES.get(length, LENGTH_GUIDELINES[PitchLength.STANDARD])


def get_sections_for_length(length: PitchLength) -> list[SectionType]:
    """Get the sections appropriate for a pitch length."""
    config = get_length_config(length)
    return config.get("sections", [])
