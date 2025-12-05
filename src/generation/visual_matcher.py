"""Visual asset matcher for pitch section generation."""

import logging
from typing import Optional

from src.llm import AnthropicClient, ModelSettings, ModelName, parse_json
from src.models.pitch import SectionType, SectionVisualAsset
from src.models.processed import VisualAssetReference, VisualInventory

logger = logging.getLogger(__name__)


class SectionVisualMatcher:
    """
    Matches visual assets to pitch sections using LLM.

    Uses intelligent matching based on section content and visual metadata
    to determine which visuals best support each section.
    """

    # Default max visuals per section type
    DEFAULT_MAX_VISUALS: dict[SectionType, int] = {
        SectionType.HOOK: 1,
        SectionType.PROBLEM: 1,
        SectionType.SOLUTION: 2,
        SectionType.FEATURES: 3,
        SectionType.BENEFITS: 2,
        SectionType.USE_CASES: 2,
        SectionType.DIFFERENTIATORS: 2,
        SectionType.SOCIAL_PROOF: 3,
        SectionType.PRICING: 1,
        SectionType.TECHNICAL: 2,
        SectionType.ROI: 1,
        SectionType.OBJECTION_HANDLING: 1,
        SectionType.CTA: 1,
        SectionType.CLOSING: 1,
    }

    # Section type to preferred visual types mapping
    SECTION_VISUAL_PREFERENCES: dict[SectionType, list[str]] = {
        SectionType.HOOK: ["screenshot", "hero"],
        SectionType.SOLUTION: ["screenshot", "diagram"],
        SectionType.FEATURES: ["screenshot", "diagram"],
        SectionType.BENEFITS: ["screenshot", "diagram"],
        SectionType.USE_CASES: ["screenshot"],
        SectionType.DIFFERENTIATORS: ["comparison_table"],
        SectionType.SOCIAL_PROOF: ["logo"],
        SectionType.PRICING: ["pricing_table"],
        SectionType.TECHNICAL: ["diagram"],
        SectionType.ROI: ["diagram", "comparison_table"],
    }

    def __init__(self, llm_client: AnthropicClient):
        """
        Initialize the visual matcher.

        Args:
            llm_client: Anthropic API client for LLM calls
        """
        self.client = llm_client

    async def match_visuals_to_section(
        self,
        section_type: SectionType,
        section_content: str,
        section_title: str,
        visual_inventory: VisualInventory,
        feature_visuals: dict[str, list[str]],
        max_visuals: Optional[int] = None,
    ) -> list[SectionVisualAsset]:
        """
        Match best visuals for a section.

        Args:
            section_type: Type of the pitch section
            section_content: Generated content of the section
            section_title: Title of the section
            visual_inventory: Available visual assets
            feature_visuals: Pre-computed feature-to-visual mappings
            max_visuals: Maximum visuals to return (uses default if None)

        Returns:
            List of SectionVisualAsset objects for the section
        """
        if not visual_inventory or visual_inventory.total_count == 0:
            return []

        max_count = max_visuals or self.DEFAULT_MAX_VISUALS.get(section_type, 2)

        # Get candidate visuals based on section type
        candidates = self._get_candidates_for_section(section_type, visual_inventory)

        if not candidates:
            return []

        # For simple cases, use rule-based matching
        if len(candidates) <= max_count:
            return self._convert_to_section_assets(
                candidates[:max_count],
                section_type,
                "Direct match based on visual type",
            )

        # Use LLM for intelligent matching
        matched_assets = await self._llm_match_visuals(
            section_type=section_type,
            section_content=section_content,
            section_title=section_title,
            candidates=candidates,
            max_visuals=max_count,
        )

        return matched_assets

    def _get_candidates_for_section(
        self,
        section_type: SectionType,
        inventory: VisualInventory,
    ) -> list[VisualAssetReference]:
        """
        Get candidate visuals based on section type.

        Args:
            section_type: Type of pitch section
            inventory: Visual asset inventory

        Returns:
            List of candidate visual assets
        """
        candidates: list[VisualAssetReference] = []

        # Use the inventory's built-in method for type-based filtering
        section_key = section_type.value.lower()
        type_candidates = inventory.get_candidates_for_section(section_key)

        if type_candidates:
            candidates.extend(type_candidates)
        else:
            # Fallback: use section-specific preferences
            preferences = self.SECTION_VISUAL_PREFERENCES.get(section_type, [])

            for pref in preferences:
                if pref == "screenshot":
                    candidates.extend(
                        inventory.get_assets_by_ids(inventory.screenshots)
                    )
                elif pref == "diagram":
                    candidates.extend(
                        inventory.get_assets_by_ids(inventory.diagrams)
                    )
                elif pref == "logo":
                    candidates.extend(
                        inventory.get_assets_by_ids(inventory.logos)
                    )
                elif pref == "comparison_table":
                    candidates.extend(
                        inventory.get_assets_by_ids(inventory.comparison_tables)
                    )
                elif pref == "pricing_table":
                    candidates.extend(
                        inventory.get_assets_by_ids(inventory.pricing_tables)
                    )

            # Add generic images if no specific matches
            if not candidates and inventory.images:
                candidates.extend([
                    img for img in inventory.images
                    if not img.is_logo and not img.is_icon
                ][:10])

        # Deduplicate by asset_id
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.asset_id not in seen:
                seen.add(c.asset_id)
                unique_candidates.append(c)

        return unique_candidates

    async def _llm_match_visuals(
        self,
        section_type: SectionType,
        section_content: str,
        section_title: str,
        candidates: list[VisualAssetReference],
        max_visuals: int,
    ) -> list[SectionVisualAsset]:
        """
        Use LLM to select best visuals from candidates.

        Args:
            section_type: Type of pitch section
            section_content: Section content text
            section_title: Section title
            candidates: Candidate visual assets
            max_visuals: Maximum number to select

        Returns:
            List of matched visual assets
        """
        # Format candidates for LLM
        candidate_descriptions = []
        for i, c in enumerate(candidates[:15]):  # Limit to 15 candidates
            desc = c.get_description_for_matching()
            candidate_descriptions.append(f"{i+1}. [{c.asset_id}] {desc}")

        prompt = f"""Select the best visual assets for this pitch section.

## Section Type: {section_type.value}
## Section Title: {section_title}

## Section Content:
{section_content[:2000]}

## Available Visual Assets:
{chr(10).join(candidate_descriptions)}

## Instructions
Select up to {max_visuals} visual(s) that would best support this section.
Consider:
- Relevance to the section content
- Visual type appropriateness (screenshots for features, diagrams for technical, etc.)
- Information value for the audience

Return JSON:
{{
  "selections": [
    {{
      "asset_id": "...",
      "placement": "hero|inline|sidebar|comparison",
      "reason": "Brief explanation of why this visual fits"
    }}
  ]
}}

placement guide:
- hero: Main visual at top of section
- inline: Within the content flow
- sidebar: Supporting visual to the side
- comparison: For tables/comparisons"""

        try:
            settings = ModelSettings(
                model=ModelName.HAIKU,
                max_tokens=1000,
                temperature=0.1,
            )
            response = await self.client.complete(prompt, settings=settings)

            data = parse_json(response.content)
            selections = data.get("selections", [])

            # Convert to SectionVisualAsset objects
            result = []
            candidate_map = {c.asset_id: c for c in candidates}

            for sel in selections[:max_visuals]:
                asset_id = sel.get("asset_id", "")
                if asset_id in candidate_map:
                    asset = candidate_map[asset_id]
                    result.append(
                        SectionVisualAsset(
                            asset_id=asset.asset_id,
                            asset_type=asset.asset_type,
                            url=asset.url,
                            local_path=asset.local_path,
                            caption=asset.caption,
                            alt_text=asset.alt_text,
                            placement=sel.get("placement", "inline"),
                            relevance_reason=sel.get("reason", ""),
                            table_markdown=asset.table_markdown,
                        )
                    )

            return result

        except Exception as e:
            logger.warning(f"LLM visual matching failed: {e}")
            # Fallback to simple selection
            return self._convert_to_section_assets(
                candidates[:max_visuals],
                section_type,
                "Fallback selection",
            )

    def _convert_to_section_assets(
        self,
        assets: list[VisualAssetReference],
        section_type: SectionType,
        reason: str,
    ) -> list[SectionVisualAsset]:
        """
        Convert VisualAssetReference objects to SectionVisualAsset objects.

        Args:
            assets: Visual asset references
            section_type: Section type for context
            reason: Reason for selection

        Returns:
            List of SectionVisualAsset objects
        """
        result = []
        for i, asset in enumerate(assets):
            # Determine placement based on position and type
            if i == 0 and section_type in [SectionType.HOOK, SectionType.SOLUTION]:
                placement = "hero"
            elif asset.asset_type == "table":
                placement = "comparison"
            else:
                placement = "inline"

            result.append(
                SectionVisualAsset(
                    asset_id=asset.asset_id,
                    asset_type=asset.asset_type,
                    url=asset.url,
                    local_path=asset.local_path,
                    caption=asset.caption,
                    alt_text=asset.alt_text,
                    placement=placement,
                    relevance_reason=reason,
                    table_markdown=asset.table_markdown,
                )
            )

        return result

    def get_max_visuals_for_section(self, section_type: SectionType) -> int:
        """Get the default maximum visuals for a section type."""
        return self.DEFAULT_MAX_VISUALS.get(section_type, 2)
