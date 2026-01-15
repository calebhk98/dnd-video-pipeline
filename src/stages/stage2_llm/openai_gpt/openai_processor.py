"""
Stage 2 LLM Processing ,  OpenAI GPT-4 Backend
==============================================
Uses the OpenAI GPT-4o model to perform all three LLM processing steps:
speaker mapping, storyboard generation, and production script generation.

Structured output strategy: JSON mode
    Unlike the Claude backend (which uses tool calling), this backend uses
    OpenAI's ``response_format={"type": "json_object"}`` to guarantee that
    the model returns valid JSON.  The JSON structure is guided entirely by
    prompt instructions rather than a schema definition.

    Tradeoff: JSON mode guarantees parseable JSON but not a specific schema ,
    the model might omit fields or use different key names.  The shared
    ``_build_storyboard_from_data`` and ``_build_production_script_from_data``
    helpers (from ``BaseLLMProcessor``) handle this by only picking known keys.

Transcript pre-processing:
    ``generate_storyboard`` maps speaker labels and formats utterances as
    ``"[HH.HHs] Character Name: text"`` before sending to the model.  This
    gives GPT-4 explicit timestamps it can use to set scene start/end times.

Dependencies:
    pip install openai
"""

import os
import json
import logging
from typing import Dict, Any
from openai import OpenAI
from src.stages.stage2_llm.base import BaseLLMProcessor
from src.shared.schemas import Transcript, Storyboard, ProductionScript, SceneShot, ProductionScene
from src.stages.stage2_llm.openai_gpt.openai_schemas import RELEVANCE_TOOL, SCENE_SHOTS_TOOL
from src.stages.stage2_llm.openai_gpt.openai_errors import call_api

logger = logging.getLogger(__name__)


class OpenAIGPTProcessor(BaseLLMProcessor):
	"""
    Concrete implementation of BaseLLMProcessor using OpenAI GPT-4o.

    Uses OpenAI's ``json_object`` response format to ensure valid JSON responses.
    All three pipeline methods use separate chat completion calls with role-specific
    system prompts for clarity.

    Config keys:
        OPENAI_API_KEY ,  OpenAI API key (falls back to OPENAI_API_KEY env var).
        model          ,  Model name (default: ``"gpt-4o"``).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the OpenAI client.

        The ``api_key != "test_key"`` guard allows tests to pass a sentinel value
        to bypass the ValueError while still exercising the client initialization.

        Args:
            config: Configuration dict.  See class docstring for keys.

        Raises:
            ValueError: If the API key is missing and config doesn't contain ``"test_key"``.
        """
		self.api_key = config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
		if not self.api_key and config.get("api_key") != "test_key":
			raise ValueError(
				"OpenAI API key is missing. Provide it in config or OPENAI_API_KEY env var."
			)

		self.model = config.get("model", "gpt-4o")

		# OpenAI client is safe to reuse across calls.
		# Falls back to "test_key" so that mock-patched test suites can instantiate.
		self.client = OpenAI(api_key=self.api_key or "test_key")

	def map_speakers(self, transcript: Transcript, character_sheet_context: str = "") -> dict:
		"""
        Ask GPT-4o to identify speaker identities from transcript dialogue.

        Uses ``json_object`` response format so the response is always valid JSON
        without needing to strip markdown fences.

        Args:
            transcript:              Stage 1 Transcript with generic speaker labels.
            character_sheet_context: Optional campaign character descriptions.

        Returns:
            Dict mapping speaker labels to "Real Name - Character Name - Class" strings.
        """
		prompt = self._build_speaker_mapping_prompt(transcript.full_text, character_sheet_context)

		response = self._call_api(lambda: self.client.chat.completions.create(
			model=self.model,
			messages=[{"role": "user", "content": prompt}],
			# json_object mode guarantees parseable JSON; no need for _extract_json_from_text.
			response_format={"type": "json_object"}
		))

		return json.loads(response.choices[0].message.content)

	def generate_speaker_visualizations(self, speaker_map: dict) -> dict:
		"""
        Ask GPT-4o to generate a visual description for each speaker.

        Args:
            speaker_map: Dict mapping speaker labels to "Real Name - Character Name - Class".

        Returns:
            Dict mapping speaker labels to appearance/role description strings.
            Falls back to passthrough on errors.
        """
		prompt = self._build_speaker_visualization_prompt(speaker_map)

		try:
			response = self._call_api(lambda: self.client.chat.completions.create(
				model=self.model,
				messages=[{"role": "user", "content": prompt}],
				response_format={"type": "json_object"}
			))
			return json.loads(response.choices[0].message.content)
		except Exception as e:
			logger.error(f"Error generating speaker visualizations: {e}")
			return {speaker: entry for speaker, entry in speaker_map.items()}

	def generate_storyboard(self, transcript: Transcript, speaker_map: dict) -> Storyboard:
		"""
        Generate a storyboard from the transcript using GPT-4o in JSON mode.

        Pre-processes the transcript into a human-readable format with timestamps
        and mapped character names before sending to the model.  The model returns
        a ``{"scenes": [...]}`` JSON object which is parsed into a ``Storyboard``.

        Args:
            transcript:  Stage 1 Transcript.
            speaker_map: Dict from ``map_speakers``.

        Returns:
            A ``Storyboard`` with scene objects parsed from the LLM's JSON output.
        """
		# Format transcript with character names and timestamps for better scene detection.
		# E.g. "[12.50s] Magnus - Human Fighter: I strike the orc with my axe."
		mapped_transcript = "\n".join([
			f"[{u.start:.2f}s] {speaker_map.get(u.speaker, u.speaker)}: {u.text}"
			for u in transcript.utterances
		])

		prompt = f"""
        You are a storyboard artist. Convert the following transcript into a storyboard.

        Transcript:
        {mapped_transcript}

        Instructions:
        1. Filter out Out-Of-Character (OOC) talk or table talk.
        2. Divide the dialogue into logical scenes.
        3. For each scene, provide: scene_number, start_time, end_time, location, narrative_summary, and visual_prompt.

        Return a JSON object with a key "scenes" containing an array of scene objects.
        """

		response = self._call_api(lambda: self.client.chat.completions.create(
			model=self.model,
			messages=[{"role": "user", "content": prompt}],
			response_format={"type": "json_object"}
		))

		data = json.loads(response.choices[0].message.content)
		return self._build_storyboard_from_data(data)

	def generate_production_script(self, storyboard: Storyboard, transcript: Transcript) -> ProductionScript:
		"""
        Expand storyboard scenes into detailed production directions using GPT-4o.

        Serializes the storyboard to JSON (via ``model_dump_json()``) and passes it
        to the model alongside the original transcript text for context.  The model
        returns an expanded ``{"scenes": [...]}`` JSON object.

        Args:
            storyboard:  The ``Storyboard`` returned by ``generate_storyboard``.
            transcript:  Original transcript for additional context.

        Returns:
            A ``ProductionScript`` with enriched scene objects.
        """
		prompt = f"""
        You are a film director. Expand the following storyboard into a production script.

        Storyboard:
        {storyboard.model_dump_json()}

        For each scene, add:
        - stage_directions: Detailed camera and environment setup.
        - character_actions: Specific movements and expressions.
        - final_video_prompt: A highly detailed prompt for video generation.

        Return a JSON object with a key "scenes" containing the expanded scene objects.
        """

		response = self._call_api(lambda: self.client.chat.completions.create(
			model=self.model,
			messages=[{"role": "user", "content": prompt}],
			response_format={"type": "json_object"}
		))

		data = json.loads(response.choices[0].message.content)
		return self._build_production_script_from_data(data)

	def review_scene_relevance(self, storyboard: Storyboard) -> Storyboard:
		"""
        Use GPT-4o tool calling to mark each scene as in-game (relevant) or OOC (irrelevant).

        Forces the model to call the ``review_scene_relevance`` function via tool_choice,
        guaranteeing structured output. Falls back to marking all scenes relevant on failure.
        """
		scenes_summary = json.dumps([
			{
				"scene_number": s.scene_number,
				"location": s.location,
				"narrative_summary": s.narrative_summary,
			}
			for s in storyboard.scenes
		], indent=2)

		prompt = self._build_scene_relevance_prompt(scenes_summary)

		try:
			response = self._call_api(lambda: self.client.chat.completions.create(
				model=self.model,
				messages=[{"role": "user", "content": prompt}],
				tools=[RELEVANCE_TOOL],
				tool_choice={"type": "function", "function": {"name": "review_scene_relevance"}},
			))
			message = response.choices[0].message
			if not message.tool_calls:
				raise ValueError("No tool calls found in OpenAI response")

			args = json.loads(message.tool_calls[0].function.arguments)
			relevance_map = {item["scene_number"]: item for item in args.get("scenes", [])}

			updated_scenes = []
			for scene in storyboard.scenes:
				data = relevance_map.get(scene.scene_number)
				is_relevant = data["is_relevant"] if data else True
				reason = data.get("relevance_reason", "Assumed relevant.") if data else "Review skipped."

				updated_scenes.append(scene.model_copy(update={
					"is_relevant": is_relevant,
					"relevance_reason": reason,
				}))
			return Storyboard(scenes=updated_scenes)
		except Exception as e:
			logger.error(f"Error in OpenAI review_scene_relevance: {e}")

		logger.warning("review_scene_relevance: falling back to marking all scenes relevant.")
		fallback_scenes = [
			s.model_copy(update={"is_relevant": True, "relevance_reason": "Relevance review skipped."})
			for s in storyboard.scenes
		]
		return Storyboard(scenes=fallback_scenes)

	def generate_scene_shots(self, storyboard: Storyboard, transcript: Transcript) -> ProductionScript:
		"""
        Use GPT-4o tool calling to expand each scene into production details and shot breakdowns.

        Forces the model to call the ``generate_scene_shots`` function, guaranteeing structured
        output with stage directions, character actions, and a list of sequential shots per scene.
        Returns an empty ProductionScript if the tool call is not found in the response.
        """
		scene_transcripts = {
			scene.scene_number: self._get_scene_transcript(transcript, scene)
			for scene in storyboard.scenes
		}
		prompt = self._build_scene_shots_prompt(
			storyboard.model_dump_json(),
			json.dumps(scene_transcripts, indent=2),
		)

		try:
			response = self._call_api(lambda: self.client.chat.completions.create(
				model=self.model,
				messages=[{"role": "user", "content": prompt}],
				tools=[SCENE_SHOTS_TOOL],
				tool_choice={"type": "function", "function": {"name": "generate_scene_shots"}},
			))
			message = response.choices[0].message
			if not message.tool_calls:
				raise ValueError("No tool calls found in OpenAI response")

			args = json.loads(message.tool_calls[0].function.arguments)
			production_scenes = []
			for s in args.get("scenes", []):
				shots = [SceneShot(**shot) for shot in s.pop("shots", [])]
				production_scenes.append(ProductionScene(**s, shots=shots))
			return ProductionScript(scenes=production_scenes)
		except Exception as e:
			logger.error(f"Error in OpenAI generate_scene_shots: {e}")

		return ProductionScript(scenes=[])

	# --------------------------------------------------------------------------
	# Internal helpers
	# --------------------------------------------------------------------------

	def _call_api(self, fn):
		"""Call an OpenAI API function and translate SDK errors into typed ProviderErrors."""
		return call_api(fn)
