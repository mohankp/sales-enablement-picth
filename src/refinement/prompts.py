"""Prompt templates for refinement operations."""

from dataclasses import dataclass
from typing import Optional

from src.models.pitch import PitchTone, SectionType

from .models import RefinementType


@dataclass
class RefinementPrompt:
    """A prompt template for refinement."""

    name: str
    refinement_type: RefinementType
    system_prompt: str
    user_prompt_template: str
    description: str = ""


# System prompt base for all refinements
REFINEMENT_SYSTEM_BASE = """You are an expert sales pitch editor and consultant.
Your task is to refine and improve sales pitch content based on user feedback.

Key principles:
1. Preserve the core message and value proposition unless explicitly asked to change
2. Maintain consistency across sections
3. Keep the same structure unless restructuring is requested
4. Ensure smooth transitions between sections
5. Match the requested tone throughout

You will receive the current pitch content in JSON format and instructions for refinement.
Return the refined pitch in the same JSON structure."""


# Tone refinement prompts
TONE_REFINEMENT_SYSTEM = REFINEMENT_SYSTEM_BASE + """

For tone adjustments:
- Adjust word choice, sentence structure, and phrasing to match the target tone
- Preserve key facts, features, and benefits
- Modify emotional appeal and formality level
- Update talking points and speaker notes to match the new tone

Tone guidelines:
- PROFESSIONAL: Formal, business-focused, precise language
- CONVERSATIONAL: Friendly, approachable, casual but credible
- TECHNICAL: Detail-oriented, specification-focused, precise terminology
- EXECUTIVE: High-level, ROI-focused, concise, strategic
- ENTHUSIASTIC: Energetic, compelling, action-oriented
- CONSULTATIVE: Problem-solving focused, advisory, empathetic"""

TONE_REFINEMENT_PROMPT = """Current pitch for "{product_name}":
```json
{pitch_json}
```

Refinement instruction: {instruction}

Target tone: {target_tone}
Current tone: {current_tone}

Constraints: {constraints}
Sections to preserve unchanged: {preserve_sections}

Refine the pitch to match the target tone while preserving key content.

Return the refined pitch as valid JSON with this structure:
{{
    "refined_sections": [
        {{
            "section_type": "hook",
            "title": "...",
            "content": "...",
            "key_points": ["..."],
            "talking_points": ["..."]
        }}
    ],
    "elevator_pitch": "...",
    "one_liner": "...",
    "key_messages": ["..."],
    "changes_summary": ["List of changes made"],
    "rationale": "Brief explanation of the refinement approach"
}}"""


# Section-specific refinement prompts
SECTION_REFINEMENT_SYSTEM = REFINEMENT_SYSTEM_BASE + """

For section-specific refinements:
- Focus changes on the targeted section
- Ensure the refined section integrates smoothly with surrounding content
- Maintain overall pitch coherence
- Update related elements (talking points, visuals) as needed"""

SECTION_REFINEMENT_PROMPT = """Current pitch for "{product_name}":
```json
{pitch_json}
```

Refinement instruction: {instruction}

Target section: {target_section}

Current section content:
```json
{section_json}
```

Constraints: {constraints}

Refine ONLY the specified section according to the instruction.

Return the refined section as valid JSON:
{{
    "section_type": "{target_section}",
    "title": "...",
    "content": "...",
    "key_points": ["..."],
    "talking_points": ["..."],
    "visual_suggestions": ["..."],
    "changes_summary": ["List of changes made"],
    "rationale": "Brief explanation of the refinement"
}}"""


# Length refinement prompts
LENGTH_REFINEMENT_SYSTEM = REFINEMENT_SYSTEM_BASE + """

For length adjustments:
- EXPAND: Add more detail, examples, context, and supporting points
- CONDENSE: Tighten language, remove redundancy, prioritize key messages

When expanding:
- Add concrete examples and use cases
- Include additional proof points and statistics
- Elaborate on benefits and features
- Extend talking points for speakers

When condensing:
- Remove filler words and redundant phrases
- Combine related points
- Focus on highest-impact content
- Keep the most compelling arguments"""

LENGTH_REFINEMENT_PROMPT = """Current pitch for "{product_name}":
```json
{pitch_json}
```

Refinement instruction: {instruction}

Length direction: {length_direction}
Current word count: {current_word_count}

Constraints: {constraints}
Sections to preserve: {preserve_sections}

{direction_specific_instruction}

Return the refined pitch as valid JSON:
{{
    "refined_sections": [...],
    "elevator_pitch": "...",
    "one_liner": "...",
    "changes_summary": ["List of changes made"],
    "rationale": "Brief explanation of the refinement"
}}"""


# Audience refinement prompts
AUDIENCE_REFINEMENT_SYSTEM = REFINEMENT_SYSTEM_BASE + """

For audience adjustments:
- Tailor language and terminology to the target audience
- Emphasize benefits most relevant to this audience
- Adjust technical depth appropriately
- Use examples and use cases that resonate with this audience
- Modify objection handling for audience-specific concerns"""

AUDIENCE_REFINEMENT_PROMPT = """Current pitch for "{product_name}":
```json
{pitch_json}
```

Refinement instruction: {instruction}

Target audience: {target_audience}
Current audience: {current_audience}

Consider:
- What does this audience care about most?
- What terminology resonates with them?
- What objections might they have?
- What proof points would be most compelling?

Constraints: {constraints}

Return the refined pitch as valid JSON:
{{
    "refined_sections": [...],
    "elevator_pitch": "...",
    "one_liner": "...",
    "key_messages": ["..."],
    "common_objections": {{"objection": "response", ...}},
    "changes_summary": ["List of changes made"],
    "rationale": "Brief explanation of the audience adaptation"
}}"""


# Custom/free-form refinement prompts
CUSTOM_REFINEMENT_SYSTEM = REFINEMENT_SYSTEM_BASE + """

For custom refinements:
- Carefully analyze the instruction to understand the intent
- Apply changes as specifically as possible
- If the instruction is ambiguous, make reasonable interpretations
- Explain your interpretation in the rationale"""

CUSTOM_REFINEMENT_PROMPT = """Current pitch for "{product_name}":
```json
{pitch_json}
```

Refinement instruction: {instruction}

Previous refinements context:
{history_context}

Constraints: {constraints}
Sections to preserve: {preserve_sections}

Apply the refinement as instructed. Be creative but stay true to the product.

Return the refined pitch as valid JSON:
{{
    "refined_sections": [
        {{
            "section_type": "...",
            "title": "...",
            "content": "...",
            "key_points": ["..."],
            "talking_points": ["..."]
        }}
    ],
    "elevator_pitch": "...",
    "one_liner": "...",
    "key_messages": ["..."],
    "changes_summary": ["List of specific changes made"],
    "rationale": "Explanation of how the instruction was interpreted and applied"
}}"""


# Classification prompt for auto-detecting refinement type
CLASSIFICATION_PROMPT = """Analyze this refinement instruction and classify it.

Instruction: "{instruction}"

Classify the refinement type as one of:
- TONE: Changes to overall tone/voice (e.g., "make it more formal", "sound friendlier")
- SECTION: Changes to a specific section (e.g., "update the hook", "rewrite benefits")
- LENGTH: Expand or condense content (e.g., "make it shorter", "add more detail")
- AUDIENCE: Adjust for different audience (e.g., "for technical users", "for executives")
- FEATURE: Emphasize specific features (e.g., "highlight the AI features")
- STYLE: Writing style changes (e.g., "use more bullet points", "add statistics")
- CUSTOM: Other refinements that don't fit above categories

Also identify:
- Target section (if section-specific): hook, problem, solution, features, benefits, etc.
- Target tone (if tone change): professional, conversational, technical, executive, enthusiastic, consultative
- Length direction (if length change): expand, condense
- Target audience (if audience change): the audience description

Return as JSON:
{{
    "refinement_type": "...",
    "target_section": null or "section_name",
    "target_tone": null or "tone_name",
    "length_direction": null or "expand/condense",
    "target_audience": null or "audience description",
    "confidence": 0.0-1.0
}}"""


# Prompt registry
REFINEMENT_PROMPTS: dict[RefinementType, RefinementPrompt] = {
    RefinementType.TONE: RefinementPrompt(
        name="Tone Refinement",
        refinement_type=RefinementType.TONE,
        system_prompt=TONE_REFINEMENT_SYSTEM,
        user_prompt_template=TONE_REFINEMENT_PROMPT,
        description="Adjust the overall tone and voice of the pitch",
    ),
    RefinementType.SECTION: RefinementPrompt(
        name="Section Refinement",
        refinement_type=RefinementType.SECTION,
        system_prompt=SECTION_REFINEMENT_SYSTEM,
        user_prompt_template=SECTION_REFINEMENT_PROMPT,
        description="Refine a specific section of the pitch",
    ),
    RefinementType.LENGTH: RefinementPrompt(
        name="Length Refinement",
        refinement_type=RefinementType.LENGTH,
        system_prompt=LENGTH_REFINEMENT_SYSTEM,
        user_prompt_template=LENGTH_REFINEMENT_PROMPT,
        description="Expand or condense the pitch content",
    ),
    RefinementType.AUDIENCE: RefinementPrompt(
        name="Audience Refinement",
        refinement_type=RefinementType.AUDIENCE,
        system_prompt=AUDIENCE_REFINEMENT_SYSTEM,
        user_prompt_template=AUDIENCE_REFINEMENT_PROMPT,
        description="Adapt the pitch for a different audience",
    ),
    RefinementType.CUSTOM: RefinementPrompt(
        name="Custom Refinement",
        refinement_type=RefinementType.CUSTOM,
        system_prompt=CUSTOM_REFINEMENT_SYSTEM,
        user_prompt_template=CUSTOM_REFINEMENT_PROMPT,
        description="Apply a custom refinement based on instructions",
    ),
}


def get_refinement_prompt(refinement_type: RefinementType) -> RefinementPrompt:
    """Get the prompt template for a refinement type."""
    return REFINEMENT_PROMPTS.get(
        refinement_type,
        REFINEMENT_PROMPTS[RefinementType.CUSTOM],
    )


def get_classification_prompt() -> str:
    """Get the instruction classification prompt."""
    return CLASSIFICATION_PROMPT
