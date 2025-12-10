"""Refinement engine orchestrator for iterative pitch improvement."""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.llm import (
    LLMProvider,
    LLMResponse,
    ProviderType,
    create_llm_provider,
    parse_json,
)
from src.models.pitch import (
    Pitch,
    PitchSection,
    PitchTone,
    SectionType,
)

from .history import RefinementHistory
from .models import (
    LengthDirection,
    RefinementConfig,
    RefinementContext,
    RefinementRequest,
    RefinementResult,
    RefinementType,
    SectionChange,
)
from .prompts import (
    CLASSIFICATION_PROMPT,
    get_refinement_prompt,
)

logger = logging.getLogger(__name__)


class RefinementEngine:
    """
    Orchestrates pitch refinement through LLM-powered transformations.

    Supports:
    - Tone adjustments
    - Section-specific refinements
    - Length modifications
    - Audience adaptation
    - Custom free-form refinements
    - Undo/redo with persistent history

    Usage:
        config = RefinementConfig()
        async with RefinementEngine(config) as engine:
            result = await engine.refine(
                pitch,
                RefinementRequest(instruction="make it more technical")
            )
            print(result.changes_summary)

        # With history tracking
        history = RefinementHistory.create_for_pitch(pitch)
        result = await engine.refine(pitch, request, history=history)
        history.save()
    """

    def __init__(self, config: Optional[RefinementConfig] = None):
        self.config = config or RefinementConfig()
        self._client: Optional[LLMProvider] = None

    async def __aenter__(self) -> "RefinementEngine":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Initialize resources."""
        self._client = create_llm_provider(
            provider=self.config.provider,
            default_model=self.config.model,
            timeout_seconds=self.config.timeout_seconds,
            max_retries=3,
            requests_per_minute=self.config.requests_per_minute,
            track_costs=True,
            log_requests=self.config.log_requests,
            log_responses=self.config.log_responses,
        )
        await self._client.start()
        logger.info(
            f"RefinementEngine initialized with {self.config.provider.value} provider"
        )

    async def stop(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.stop()
            self._client = None
        logger.info("RefinementEngine stopped")

    def _ensure_client(self) -> LLMProvider:
        """Ensure LLM client is available."""
        if self._client is None:
            raise RuntimeError(
                "Engine not started. Use 'async with RefinementEngine()' or call start()."
            )
        return self._client

    async def refine(
        self,
        pitch: Pitch,
        request: RefinementRequest,
        history: Optional[RefinementHistory] = None,
    ) -> RefinementResult:
        """
        Refine a pitch based on the request.

        Args:
            pitch: The pitch to refine
            request: The refinement request
            history: Optional history for context and tracking

        Returns:
            RefinementResult with the refined pitch
        """
        start_time = time.time()
        client = self._ensure_client()

        # Assign request ID if not set
        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        # Auto-classify refinement type if set to CUSTOM and we can detect it
        if request.refinement_type == RefinementType.CUSTOM:
            classified = await self._classify_instruction(request.instruction)
            if classified:
                request = self._merge_classification(request, classified)
                logger.debug(f"Auto-classified as: {request.refinement_type.value}")

        # Build context
        history_context = history.get_context_for_llm() if history else ""
        context = RefinementContext(
            pitch=pitch,
            request=request,
            config=self.config,
            history_context=history_context,
        )

        # Route to appropriate refinement method
        if request.refinement_type == RefinementType.SECTION and request.target_section:
            result = await self._refine_section(context)
        elif request.refinement_type == RefinementType.TONE:
            result = await self._refine_tone(context)
        elif request.refinement_type == RefinementType.LENGTH:
            result = await self._refine_length(context)
        elif request.refinement_type == RefinementType.AUDIENCE:
            result = await self._refine_audience(context)
        else:
            result = await self._refine_custom(context)

        # Calculate latency
        result.latency_ms = (time.time() - start_time) * 1000
        result.instruction = request.instruction
        result.refinement_type = request.refinement_type

        # Add to history if provided
        if history and result.success:
            history.add(request, result)
            if self.config.auto_save_history and history._file_path:
                history.save()

        logger.info(
            f"Refinement complete: {len(result.changes_summary)} changes, "
            f"{result.tokens_used} tokens, ${result.cost_usd:.4f}"
        )

        return result

    async def refine_section(
        self,
        pitch: Pitch,
        section_type: SectionType,
        instruction: str,
        history: Optional[RefinementHistory] = None,
    ) -> RefinementResult:
        """
        Convenience method to refine a specific section.

        Args:
            pitch: The pitch to refine
            section_type: The section to refine
            instruction: Natural language instruction
            history: Optional history for tracking

        Returns:
            RefinementResult with the refined pitch
        """
        request = RefinementRequest(
            instruction=instruction,
            refinement_type=RefinementType.SECTION,
            target_section=section_type,
        )
        return await self.refine(pitch, request, history)

    async def change_tone(
        self,
        pitch: Pitch,
        target_tone: PitchTone,
        instruction: Optional[str] = None,
        history: Optional[RefinementHistory] = None,
    ) -> RefinementResult:
        """
        Convenience method to change pitch tone.

        Args:
            pitch: The pitch to refine
            target_tone: The target tone
            instruction: Optional additional instruction
            history: Optional history for tracking

        Returns:
            RefinementResult with the refined pitch
        """
        request = RefinementRequest(
            instruction=instruction or f"Change the tone to {target_tone.value}",
            refinement_type=RefinementType.TONE,
            target_tone=target_tone,
        )
        return await self.refine(pitch, request, history)

    async def _classify_instruction(
        self, instruction: str
    ) -> Optional[dict[str, Any]]:
        """Classify a refinement instruction using LLM."""
        client = self._ensure_client()

        prompt = CLASSIFICATION_PROMPT.format(instruction=instruction)

        try:
            response = await client.complete(
                prompt,
                max_tokens=256,
                temperature=0.3,
            )
            result = parse_json(response.content, strict=False)
            if isinstance(result, dict) and result.get("confidence", 0) > 0.5:
                return result
        except Exception as e:
            logger.debug(f"Instruction classification failed: {e}")

        return None

    def _merge_classification(
        self, request: RefinementRequest, classified: dict[str, Any]
    ) -> RefinementRequest:
        """Merge classification results into the request."""
        # Create a copy of the request with updated fields
        updates = {}

        if classified.get("refinement_type"):
            try:
                updates["refinement_type"] = RefinementType(
                    classified["refinement_type"].lower()
                )
            except ValueError:
                pass

        if classified.get("target_section"):
            try:
                updates["target_section"] = SectionType(
                    classified["target_section"].lower()
                )
            except ValueError:
                pass

        if classified.get("target_tone"):
            try:
                updates["target_tone"] = PitchTone(classified["target_tone"].lower())
            except ValueError:
                pass

        if classified.get("length_direction"):
            try:
                updates["length_direction"] = LengthDirection(
                    classified["length_direction"].lower()
                )
            except ValueError:
                pass

        if classified.get("target_audience"):
            updates["target_audience"] = classified["target_audience"]

        if updates:
            return request.model_copy(update=updates)

        return request

    async def _refine_section(self, context: RefinementContext) -> RefinementResult:
        """Refine a specific section."""
        client = self._ensure_client()
        pitch = context.pitch
        request = context.request
        section_type = request.target_section

        # Get the current section
        current_section = pitch.get_section(section_type)
        if not current_section:
            return RefinementResult(
                success=False,
                error=f"Section '{section_type.value}' not found in pitch",
                original_pitch=pitch,
                refined_pitch=pitch,
            )

        # Get prompt template
        prompt_template = get_refinement_prompt(RefinementType.SECTION)

        # Build prompt
        pitch_json = json.dumps(
            {"sections": [s.model_dump() for s in pitch.sections]},
            indent=2,
            default=str,
        )
        section_json = json.dumps(current_section.model_dump(), indent=2, default=str)

        prompt = prompt_template.user_prompt_template.format(
            product_name=pitch.product_name,
            pitch_json=pitch_json,
            section_json=section_json,
            instruction=request.instruction,
            target_section=section_type.value,
            constraints="\n".join(request.constraints) if request.constraints else "None",
        )

        try:
            response = await client.complete(
                prompt,
                system=prompt_template.system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            result_data = parse_json(response.content, strict=False)
            if not isinstance(result_data, dict):
                raise ValueError("Invalid response format")

            # Build refined section
            refined_section = PitchSection(
                section_type=section_type,
                title=result_data.get("title", current_section.title),
                content=result_data.get("content", current_section.content),
                key_points=result_data.get("key_points", current_section.key_points),
                talking_points=result_data.get(
                    "talking_points", current_section.talking_points
                ),
                visual_suggestions=result_data.get(
                    "visual_suggestions", current_section.visual_suggestions
                ),
                visual_assets=current_section.visual_assets,  # Preserve visuals
                order=current_section.order,
            )

            # Build refined pitch
            refined_pitch = self._update_pitch_section(pitch, refined_section)

            # Build changes
            changes_summary = result_data.get("changes_summary", [])
            if not changes_summary:
                changes_summary = [f"Refined {section_type.value} section"]

            section_changes = self._detect_section_changes(
                current_section, refined_section
            )

            return RefinementResult(
                success=True,
                original_pitch=pitch,
                refined_pitch=refined_pitch,
                changes_summary=changes_summary,
                section_changes=section_changes,
                refinement_rationale=result_data.get("rationale", ""),
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error(f"Section refinement failed: {e}")
            return RefinementResult(
                success=False,
                error=str(e),
                original_pitch=pitch,
                refined_pitch=pitch,
            )

    async def _refine_tone(self, context: RefinementContext) -> RefinementResult:
        """Refine the overall tone."""
        client = self._ensure_client()
        pitch = context.pitch
        request = context.request

        prompt_template = get_refinement_prompt(RefinementType.TONE)

        pitch_json = self._pitch_to_json(pitch)

        prompt = prompt_template.user_prompt_template.format(
            product_name=pitch.product_name,
            pitch_json=pitch_json,
            instruction=request.instruction,
            target_tone=request.target_tone.value if request.target_tone else "as instructed",
            current_tone=pitch.config.tone.value,
            constraints="\n".join(request.constraints) if request.constraints else "None",
            preserve_sections=", ".join(
                s.value for s in request.preserve_sections
            ) if request.preserve_sections else "None",
        )

        try:
            response = await client.complete(
                prompt,
                system=prompt_template.system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            result_data = parse_json(response.content, strict=False)
            if not isinstance(result_data, dict):
                raise ValueError("Invalid response format")

            # Build refined pitch
            refined_pitch = self._apply_full_refinement(pitch, result_data, request)

            # Update tone in config if specified
            if request.target_tone:
                refined_pitch.config.tone = request.target_tone

            changes_summary = result_data.get("changes_summary", [])
            if not changes_summary:
                tone_name = request.target_tone.value if request.target_tone else "new"
                changes_summary = [f"Changed tone to {tone_name}"]

            return RefinementResult(
                success=True,
                original_pitch=pitch,
                refined_pitch=refined_pitch,
                changes_summary=changes_summary,
                refinement_rationale=result_data.get("rationale", ""),
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error(f"Tone refinement failed: {e}")
            return RefinementResult(
                success=False,
                error=str(e),
                original_pitch=pitch,
                refined_pitch=pitch,
            )

    async def _refine_length(self, context: RefinementContext) -> RefinementResult:
        """Refine the content length."""
        client = self._ensure_client()
        pitch = context.pitch
        request = context.request

        prompt_template = get_refinement_prompt(RefinementType.LENGTH)

        direction = request.length_direction or LengthDirection.MAINTAIN
        if direction == LengthDirection.EXPAND:
            direction_instruction = "Expand the content with more detail, examples, and supporting points."
        elif direction == LengthDirection.CONDENSE:
            direction_instruction = "Condense the content by removing redundancy and tightening language."
        else:
            direction_instruction = "Adjust the length as specified in the instruction."

        pitch_json = self._pitch_to_json(pitch)

        prompt = prompt_template.user_prompt_template.format(
            product_name=pitch.product_name,
            pitch_json=pitch_json,
            instruction=request.instruction,
            length_direction=direction.value,
            current_word_count=pitch.word_count(),
            direction_specific_instruction=direction_instruction,
            constraints="\n".join(request.constraints) if request.constraints else "None",
            preserve_sections=", ".join(
                s.value for s in request.preserve_sections
            ) if request.preserve_sections else "None",
        )

        try:
            response = await client.complete(
                prompt,
                system=prompt_template.system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            result_data = parse_json(response.content, strict=False)
            if not isinstance(result_data, dict):
                raise ValueError("Invalid response format")

            refined_pitch = self._apply_full_refinement(pitch, result_data, request)

            changes_summary = result_data.get("changes_summary", [])
            old_count = pitch.word_count()
            new_count = refined_pitch.word_count()
            if not changes_summary:
                changes_summary = [
                    f"Adjusted length from {old_count} to {new_count} words"
                ]

            return RefinementResult(
                success=True,
                original_pitch=pitch,
                refined_pitch=refined_pitch,
                changes_summary=changes_summary,
                refinement_rationale=result_data.get("rationale", ""),
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error(f"Length refinement failed: {e}")
            return RefinementResult(
                success=False,
                error=str(e),
                original_pitch=pitch,
                refined_pitch=pitch,
            )

    async def _refine_audience(self, context: RefinementContext) -> RefinementResult:
        """Refine for a different audience."""
        client = self._ensure_client()
        pitch = context.pitch
        request = context.request

        prompt_template = get_refinement_prompt(RefinementType.AUDIENCE)

        pitch_json = self._pitch_to_json(pitch)

        prompt = prompt_template.user_prompt_template.format(
            product_name=pitch.product_name,
            pitch_json=pitch_json,
            instruction=request.instruction,
            target_audience=request.target_audience or "as specified",
            current_audience=pitch.config.target_audience or "general",
            constraints="\n".join(request.constraints) if request.constraints else "None",
        )

        try:
            response = await client.complete(
                prompt,
                system=prompt_template.system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            result_data = parse_json(response.content, strict=False)
            if not isinstance(result_data, dict):
                raise ValueError("Invalid response format")

            refined_pitch = self._apply_full_refinement(pitch, result_data, request)

            # Update audience in config
            if request.target_audience:
                refined_pitch.config.target_audience = request.target_audience

            # Update objections if provided
            if result_data.get("common_objections"):
                refined_pitch.common_objections = result_data["common_objections"]

            changes_summary = result_data.get("changes_summary", [])
            if not changes_summary:
                changes_summary = [
                    f"Adapted pitch for {request.target_audience or 'new audience'}"
                ]

            return RefinementResult(
                success=True,
                original_pitch=pitch,
                refined_pitch=refined_pitch,
                changes_summary=changes_summary,
                refinement_rationale=result_data.get("rationale", ""),
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error(f"Audience refinement failed: {e}")
            return RefinementResult(
                success=False,
                error=str(e),
                original_pitch=pitch,
                refined_pitch=pitch,
            )

    async def _refine_custom(self, context: RefinementContext) -> RefinementResult:
        """Apply a custom refinement."""
        client = self._ensure_client()
        pitch = context.pitch
        request = context.request

        prompt_template = get_refinement_prompt(RefinementType.CUSTOM)

        pitch_json = self._pitch_to_json(pitch)

        prompt = prompt_template.user_prompt_template.format(
            product_name=pitch.product_name,
            pitch_json=pitch_json,
            instruction=request.instruction,
            history_context=context.history_context or "No previous refinements.",
            constraints="\n".join(request.constraints) if request.constraints else "None",
            preserve_sections=", ".join(
                s.value for s in request.preserve_sections
            ) if request.preserve_sections else "None",
        )

        try:
            response = await client.complete(
                prompt,
                system=prompt_template.system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            result_data = parse_json(response.content, strict=False)
            if not isinstance(result_data, dict):
                raise ValueError("Invalid response format")

            refined_pitch = self._apply_full_refinement(pitch, result_data, request)

            changes_summary = result_data.get("changes_summary", [])
            if not changes_summary:
                changes_summary = ["Applied custom refinement"]

            return RefinementResult(
                success=True,
                original_pitch=pitch,
                refined_pitch=refined_pitch,
                changes_summary=changes_summary,
                refinement_rationale=result_data.get("rationale", ""),
                tokens_used=response.usage.total_tokens,
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error(f"Custom refinement failed: {e}")
            return RefinementResult(
                success=False,
                error=str(e),
                original_pitch=pitch,
                refined_pitch=pitch,
            )

    def _pitch_to_json(self, pitch: Pitch) -> str:
        """Convert pitch to JSON for prompts."""
        data = {
            "title": pitch.title,
            "subtitle": pitch.subtitle,
            "executive_summary": pitch.executive_summary,
            "sections": [s.model_dump() for s in pitch.sections],
            "elevator_pitch": pitch.elevator_pitch,
            "one_liner": pitch.one_liner,
            "key_messages": pitch.key_messages,
        }
        return json.dumps(data, indent=2, default=str)

    def _update_pitch_section(
        self, pitch: Pitch, refined_section: PitchSection
    ) -> Pitch:
        """Update a single section in the pitch."""
        refined_pitch = pitch.model_copy(deep=True)

        # Find and replace the section
        for i, section in enumerate(refined_pitch.sections):
            if section.section_type == refined_section.section_type:
                refined_pitch.sections[i] = refined_section
                break
        else:
            # Section not found, append it
            refined_pitch.sections.append(refined_section)

        return refined_pitch

    def _apply_full_refinement(
        self,
        pitch: Pitch,
        result_data: dict[str, Any],
        request: RefinementRequest,
    ) -> Pitch:
        """Apply a full pitch refinement from LLM response."""
        refined_pitch = pitch.model_copy(deep=True)

        # Update sections
        if result_data.get("refined_sections"):
            section_map = {s.section_type: s for s in refined_pitch.sections}

            for section_data in result_data["refined_sections"]:
                try:
                    section_type = SectionType(section_data.get("section_type", ""))

                    # Skip preserved sections
                    if section_type in request.preserve_sections:
                        continue

                    # Get existing section or create new
                    existing = section_map.get(section_type)

                    new_section = PitchSection(
                        section_type=section_type,
                        title=section_data.get(
                            "title", existing.title if existing else ""
                        ),
                        content=section_data.get(
                            "content", existing.content if existing else ""
                        ),
                        key_points=section_data.get(
                            "key_points", existing.key_points if existing else []
                        ),
                        talking_points=section_data.get(
                            "talking_points",
                            existing.talking_points if existing else [],
                        ),
                        visual_suggestions=section_data.get(
                            "visual_suggestions",
                            existing.visual_suggestions if existing else [],
                        ),
                        visual_assets=existing.visual_assets if existing else [],
                        order=existing.order if existing else 0,
                    )

                    section_map[section_type] = new_section
                except (ValueError, KeyError) as e:
                    logger.warning(f"Invalid section in response: {e}")
                    continue

            # Rebuild sections list preserving order
            refined_pitch.sections = sorted(
                section_map.values(), key=lambda s: s.order
            )

        # Update other fields
        if result_data.get("elevator_pitch"):
            refined_pitch.elevator_pitch = result_data["elevator_pitch"]

        if result_data.get("one_liner"):
            refined_pitch.one_liner = result_data["one_liner"]

        if result_data.get("key_messages"):
            refined_pitch.key_messages = result_data["key_messages"]

        return refined_pitch

    def _detect_section_changes(
        self, old_section: PitchSection, new_section: PitchSection
    ) -> list[SectionChange]:
        """Detect changes between sections."""
        changes = []

        if old_section.title != new_section.title:
            changes.append(
                SectionChange(
                    section_type=old_section.section_type,
                    field_changed="title",
                    change_description="Title updated",
                    original_value=old_section.title,
                    new_value=new_section.title,
                )
            )

        if old_section.content != new_section.content:
            changes.append(
                SectionChange(
                    section_type=old_section.section_type,
                    field_changed="content",
                    change_description="Content modified",
                )
            )

        if old_section.key_points != new_section.key_points:
            changes.append(
                SectionChange(
                    section_type=old_section.section_type,
                    field_changed="key_points",
                    change_description=f"Key points updated ({len(old_section.key_points)} -> {len(new_section.key_points)})",
                )
            )

        return changes
