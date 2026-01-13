"""
Stage 2: LLM Processing ,  Abstract Base Class
==============================================
This module defines the interface that every LLM backend must implement.
LLM processing is the second stage of the D&D audio-to-video pipeline:

    Transcript  ->  [Stage 2: LLM]  ->  Storyboard  ->  ProductionScript

The stage performs three sequential operations:
    1. **map_speakers**            ,  Identify real character names from generic labels.
    2. **generate_storyboard**     ,  Segment the transcript into cinematic scenes.
    3. **generate_production_script** ,  Expand each scene with detailed director notes.

All LLM backends (Claude, GPT-4, Gemini, local Llama, etc.) inherit from
``BaseLLMProcessor`` and return the same standardized Pydantic schemas, so the
rest of the pipeline is LLM-agnostic.

Shared utility methods on this base class handle common parsing tasks that every
backend needs (JSON extraction from LLM text, fallback speaker maps, schema builders).
Prompt builders live in ``prompts.py``; parsing/schema helpers live in ``parsing.py``.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from src.shared.schemas import Transcript, Storyboard, ProductionScript, Scene

from src.stages.stage2_llm.prompts import (
	build_speaker_mapping_prompt,
	build_speaker_visualization_prompt,
	build_storyboard_prompt,
	build_production_script_prompt,
	build_scene_relevance_prompt,
	build_scene_shots_prompt,
)
from src.stages.stage2_llm.parsing import (
	extract_json_from_text,
	identity_speaker_map,
	build_storyboard_from_data,
	build_production_script_from_data,
	get_scene_transcript,
)


class BaseLLMProcessor(ABC):
	"""
    Abstract Base Class for Stage 2 LLM processing backends.

    Each concrete subclass wraps a specific LLM API or local model and
    implements the three core transformation methods.  The pipeline only
    interacts with this interface.

    Responsibilities of a concrete processor:
        - Initialize an API client or local model handle in ``__init__``.
        - Implement ``map_speakers`` to infer character names from dialogue context.
        - Implement ``generate_storyboard`` to divide the transcript into scenes with
          visual prompts suitable for video generation AI.
        - Implement ``generate_production_script`` to enrich each scene with
          camera directions, character actions, and a final cinematic prompt.
    """

	@abstractmethod
	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the LLM client with credentials and model configuration.

        Implementations should:
            - Extract API key from ``config`` or fall back to environment variables.
            - Instantiate the SDK client (e.g. ``anthropic.Anthropic()``, ``OpenAI()``).
            - Set the model name to use for all subsequent calls.

        Args:
            config: A flat dictionary of configuration values.  Keys vary by backend.

        Raises:
            ValueError: If a required API key is absent.
        """
		pass

	@abstractmethod
	def map_speakers(self, transcript: Transcript, character_sheet_context: str = "") -> dict:
		"""
        Use an LLM to map generic speaker labels to real character names.

        Args:
            transcript: The Stage 1 output containing utterances with
                generic speaker labels.
            character_sheet_context: Optional free-text description of the campaign
                characters (names, races, classes).

        Returns:
            A dict mapping generic labels to "Real Name - Character Name - Class" strings.
            Falls back to an identity map (passthrough) on LLM errors.
        """
		pass

	@abstractmethod
	def generate_storyboard(self, transcript: Transcript, speaker_map: dict) -> Storyboard:
		"""
        Convert a transcript into a structured storyboard of cinematic scenes.

        Args:
            transcript:  The Stage 1 Transcript with utterances.
            speaker_map: The dict returned by ``map_speakers``.

        Returns:
            A ``Storyboard`` Pydantic object containing a list of ``Scene`` objects.
        """
		pass

	@abstractmethod
	def generate_speaker_visualizations(self, speaker_map: dict) -> dict:
		"""
        Use an LLM to generate a visual/character description for each speaker.

        Args:
            speaker_map: The dict returned by ``map_speakers``.

        Returns:
            A dict mapping each speaker label to a description string.
            Falls back to ``{speaker: entry}`` (passthrough) on LLM errors.
        """
		pass

	@abstractmethod
	def generate_production_script(self, storyboard: Storyboard, transcript: Transcript) -> ProductionScript:
		"""
        Expand a storyboard into a detailed production script with director notes.

        Args:
            storyboard:  The ``Storyboard`` returned by ``generate_storyboard``.
            transcript:  The original Transcript (used for additional context).

        Returns:
            A ``ProductionScript`` Pydantic object with enriched scene objects.
        """
		pass

	@abstractmethod
	def review_scene_relevance(self, storyboard: Storyboard) -> Storyboard:
		"""
        Use an LLM to mark each scene as in-game (relevant) or out-of-game (irrelevant).

        Args:
            storyboard: The ``Storyboard`` returned by ``generate_storyboard``.

        Returns:
            A ``Storyboard`` with ``is_relevant`` and ``relevance_reason`` set on
            every scene.  Falls back to marking all scenes relevant on LLM error.
        """
		pass

	@abstractmethod
	def generate_scene_shots(self, storyboard: Storyboard, transcript: Transcript) -> ProductionScript:
		"""
        Expand a storyboard into a production script where each scene is broken
        into sequential 3-10 second shots.

        Args:
            storyboard:  The (relevance-filtered) ``Storyboard`` from Stage 2.
            transcript:  The original Transcript for additional context.

        Returns:
            A ``ProductionScript`` where every ``ProductionScene`` has its ``shots``
            list populated.  Returns an empty ``ProductionScript`` if no tool call is found.
        """
		pass

	# --------------------------------------------------------------------------
	# Static wrapper methods - delegate to prompts.py and parsing.py so that
	# all concrete implementations calling self._build_*() or self._extract_*()
	# continue to work without modification.
	# --------------------------------------------------------------------------

	def _build_speaker_mapping_prompt(
		transcript_text: str,
		character_sheet_context: str = "",
		distinct_speakers: list = None,
	) -> str:
		"""Delegates to internal builder for speaker mapping prompt."""

		return build_speaker_mapping_prompt(transcript_text, character_sheet_context, distinct_speakers)

	@staticmethod
	def _build_speaker_visualization_prompt(speaker_map: dict) -> str:
		"""Delegates to internal builder for speaker visualization prompt."""

		return build_speaker_visualization_prompt(speaker_map)

	@staticmethod
	def _build_storyboard_prompt(
		transcript_text: str,
		speaker_map: dict,
		fn_keyword: str = "tool",
	) -> str:
		"""Delegates to internal builder for storyboard prompt."""

		return build_storyboard_prompt(transcript_text, speaker_map, fn_keyword)

	@staticmethod
	def _build_production_script_prompt(
		storyboard_json: str,
		transcript_text: str,
		fn_keyword: str = "tool",
	) -> str:
		"""Delegates to internal builder for production script prompt."""

		return build_production_script_prompt(storyboard_json, transcript_text, fn_keyword)

	@staticmethod
	def _extract_json_from_text(text: str) -> dict:
		"""Delegates to internal helper to extract JSON from LLM text."""

		return extract_json_from_text(text)

	@staticmethod
	def _identity_speaker_map(transcript: Transcript) -> dict:
		"""Delegates to internal helper for fallback speaker identity map."""

		return identity_speaker_map(transcript)

	@staticmethod
	def _build_storyboard_from_data(data: dict) -> Storyboard:
		"""Delegates to internal builder for Storyboard object."""

		return build_storyboard_from_data(data)

	@staticmethod
	def _build_production_script_from_data(data: dict) -> ProductionScript:
		"""Delegates to internal builder for ProductionScript object."""

		return build_production_script_from_data(data)

	@staticmethod
	def _get_scene_transcript(transcript: Transcript, scene: Scene) -> str:
		"""Delegates to internal helper to slice transcript for a scene."""

		return get_scene_transcript(transcript, scene)

	@staticmethod
	def _build_scene_relevance_prompt(scenes_summary: str, fn_keyword: str = "tool") -> str:
		"""Delegates to internal builder for scene relevance review prompt."""

		return build_scene_relevance_prompt(scenes_summary, fn_keyword)

	@staticmethod
	def _build_scene_shots_prompt(
		storyboard_json: str,
		scene_transcripts_json: str,
		fn_keyword: str = "tool",
	) -> str:
		"""Delegates to internal builder for scene shots prompt."""

		return build_scene_shots_prompt(storyboard_json, scene_transcripts_json, fn_keyword)
