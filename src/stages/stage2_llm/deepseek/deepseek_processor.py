"""
Stage 2 LLM Processing ,  Deepseek Backend
==========================================
Uses the Deepseek API for all three LLM processing steps.  Deepseek exposes an
OpenAI-compatible REST API, so we use the official OpenAI Python SDK pointed at
Deepseek's base URL.

Key design: OpenAI-compatible tool calling
    Deepseek supports OpenAI-format function/tool calling.  ``generate_storyboard``
    and ``generate_production_script`` both use the ``"function"`` tool type with
    forced ``tool_choice`` to guarantee structured output matching our schema.

    This is the same pattern as the Claude backend's tool calling, but using the
    OpenAI SDK's tool format (``{"type": "function", "function": {...}}``) rather
    than Anthropic's ``input_schema`` format.

All three methods include try/except blocks and fall back to empty results rather
than letting exceptions propagate, since Deepseek's free tier may occasionally
have rate limits or short outages.

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
from src.stages.stage2_llm.deepseek.deepseek_schemas import (
	STORYBOARD_TOOL,
	PRODUCTION_SCRIPT_TOOL,
	RELEVANCE_TOOL,
	SCENE_SHOTS_TOOL,
)

logger = logging.getLogger(__name__)


class DeepseekProcessor(BaseLLMProcessor):
	"""
    Concrete implementation of BaseLLMProcessor using the Deepseek API.

    Deepseek provides an OpenAI-compatible endpoint, so the OpenAI Python SDK
    is used with a custom ``base_url``.  Supports tool calling for structured
    JSON output.

    Config keys:
        api_key / DEEPSEEK_API_KEY ,  Deepseek API key.
        model                      ,  Model name (default: ``"deepseek-chat"``).
    """

	DEFAULT_MODEL = "deepseek-chat"
	# Deepseek's API is OpenAI-compatible at this base URL.
	API_BASE_URL = "https://api.deepseek.com"

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Deepseek client via the OpenAI SDK with a custom base URL.

        Falls back to ``"dummy_key"`` to allow instantiation in test environments.

        Args:
            config: Configuration dict.  See class docstring for keys.
        """
		self.api_key = config.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
		if not self.api_key:
			# Permit test instantiation with a placeholder key.
			self.api_key = "dummy_key"

		# Point the OpenAI SDK at Deepseek's compatible endpoint.
		self.client = OpenAI(api_key=self.api_key, base_url=self.API_BASE_URL)
		self.model = config.get("model", self.DEFAULT_MODEL)

	def _chat(self, messages: list, tools: list = None, tool_choice: str = "auto") -> Any:
		"""
        Internal helper for chat completion calls with optional tool use.

        Centralizes the ``client.chat.completions.create`` call so all three
        public methods share the same request pattern.

        Args:
            messages:    OpenAI-format message list (role + content dicts).
            tools:       Optional list of tool definitions in OpenAI format.
            tool_choice: How to select tools: ``"auto"``, ``"none"``, or a specific
                         function selector dict (default: ``"auto"``).

        Returns:
            The raw OpenAI ``ChatCompletion`` response object.
        """
		kwargs = {
			"model": self.model,
			"messages": messages,
		}
		if tools:
			kwargs["tools"] = tools
			kwargs["tool_choice"] = tool_choice
		return self.client.chat.completions.create(**kwargs)

	def map_speakers(self, transcript: Transcript, character_sheet_context: str = "") -> dict:
		"""
        Use Deepseek to infer speaker identities from transcript dialogue.

        Sends a plain text prompt and parses the JSON response using
        ``_extract_json_from_text`` (handles markdown fences if present).
        Falls back to identity map on any error.

        Args:
            transcript:              Stage 1 Transcript.
            character_sheet_context: Optional character descriptions.

        Returns:
            Dict mapping speaker labels to "Real Name - Character Name - Class" strings, or identity map on error.
        """
		prompt = self._build_speaker_mapping_prompt(transcript.full_text, character_sheet_context)
		try:
			response = self._chat([{"role": "user", "content": prompt}])
			content_text = response.choices[0].message.content
			return self._extract_json_from_text(content_text)
		except Exception as e:
			logger.error(f"Error in Deepseek map_speakers: {e}")
			return self._identity_speaker_map(transcript)

	def generate_speaker_visualizations(self, speaker_map: dict) -> dict:
		"""
        Use Deepseek to generate a visual description for each speaker.

        Uses the same plain-text + ``_extract_json_from_text`` pattern as
        ``map_speakers``.  Falls back to passthrough on any error.

        Args:
            speaker_map: Dict mapping speaker labels to "Real Name - Character Name - Class".

        Returns:
            Dict mapping speaker labels to appearance/role description strings.
        """
		prompt = self._build_speaker_visualization_prompt(speaker_map)
		try:
			response = self._chat([{"role": "user", "content": prompt}])
			content_text = response.choices[0].message.content
			return self._extract_json_from_text(content_text)
		except Exception as e:
			logger.error(f"Error in Deepseek generate_speaker_visualizations: {e}")
			return {speaker: entry for speaker, entry in speaker_map.items()}

	def generate_storyboard(self, transcript: Transcript, speaker_map: dict) -> Storyboard:
		"""
        Use Deepseek tool calling to generate a structured storyboard.

        Forces the model to call the ``generate_storyboard`` function via
        ``tool_choice={"type": "function", "function": {"name": "generate_storyboard"}}``,
        guaranteeing the response contains a structured tool call rather than text.

        Args:
            transcript:  Stage 1 Transcript.
            speaker_map: Speaker-to-character name mapping.

        Returns:
            A ``Storyboard`` parsed from the tool call arguments, or empty on error.
        """
		prompt = self._build_storyboard_prompt(transcript.full_text, speaker_map)
		try:
			response = self._chat(
				[{"role": "user", "content": prompt}],
				tools=[STORYBOARD_TOOL],
				tool_choice={"type": "function", "function": {"name": "generate_storyboard"}},
			)
			message = response.choices[0].message
			if message.tool_calls:
				args = json.loads(message.tool_calls[0].function.arguments)
				return self._build_storyboard_from_data(args)
		except Exception as e:
			logger.error(f"Error in Deepseek generate_storyboard: {e}")

		return Storyboard(scenes=[])

	def generate_production_script(self, storyboard: Storyboard, transcript: Transcript) -> ProductionScript:
		"""
        Use Deepseek tool calling to expand storyboard into a production script.

        Forces the model to call the ``generate_production_script`` function with
        the extended schema that adds stage_directions, character_actions, and
        final_video_prompt to each scene.

        Args:
            storyboard:  The ``Storyboard`` to expand.
            transcript:  Original transcript for LLM context.

        Returns:
            A ``ProductionScript`` parsed from the tool call, or empty on error.
        """
		prompt = self._build_production_script_prompt(storyboard.model_dump_json(), transcript.full_text)
		try:
			response = self._chat(
				[{"role": "user", "content": prompt}],
				tools=[PRODUCTION_SCRIPT_TOOL],
				tool_choice={"type": "function", "function": {"name": "generate_production_script"}},
			)
			message = response.choices[0].message
			if message.tool_calls:
				args = json.loads(message.tool_calls[0].function.arguments)
				return self._build_production_script_from_data(args)
		except Exception as e:
			logger.error(f"Error in Deepseek generate_production_script: {e}")

		return ProductionScript(scenes=[])

	def review_scene_relevance(self, storyboard: Storyboard) -> Storyboard:
		"""
        Use Deepseek tool calling to mark each scene as in-game (relevant) or OOC (irrelevant).

        Forces the model to call the ``review_scene_relevance`` function via tool_choice.
        Falls back to marking all scenes relevant on any error.
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
			response = self._chat(
				[{"role": "user", "content": prompt}],
				tools=[RELEVANCE_TOOL],
				tool_choice={"type": "function", "function": {"name": "review_scene_relevance"}},
			)
			message = response.choices[0].message
			if not message.tool_calls:
				raise ValueError("No tool calls found in Deepseek response")

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
			logger.error(f"Error in Deepseek review_scene_relevance: {e}")

		logger.warning("review_scene_relevance: falling back to marking all scenes relevant.")
		fallback_scenes = [
			s.model_copy(update={"is_relevant": True, "relevance_reason": "Relevance review skipped."})
			for s in storyboard.scenes
		]
		return Storyboard(scenes=fallback_scenes)

	def generate_scene_shots(self, storyboard: Storyboard, transcript: Transcript) -> ProductionScript:
		"""
        Use Deepseek tool calling to expand each scene into production details and shot breakdowns.

        Forces the model to call the ``generate_scene_shots`` function. Returns an empty
        ProductionScript if the tool call is not found in the response.
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
			response = self._chat(
				[{"role": "user", "content": prompt}],
				tools=[SCENE_SHOTS_TOOL],
				tool_choice={"type": "function", "function": {"name": "generate_scene_shots"}},
			)
			message = response.choices[0].message
			if not message.tool_calls:
				raise ValueError("No tool calls found in Deepseek response")

			args = json.loads(message.tool_calls[0].function.arguments)
			production_scenes = []
			for s in args.get("scenes", []):
				shots = [SceneShot(**shot) for shot in s.pop("shots", [])]
				production_scenes.append(ProductionScene(**s, shots=shots))
			return ProductionScript(scenes=production_scenes)
		except Exception as e:
			logger.error(f"Error in Deepseek generate_scene_shots: {e}")

		return ProductionScript(scenes=[])
