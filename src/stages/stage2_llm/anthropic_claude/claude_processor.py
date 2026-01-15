"""
Stage 2 LLM Processing ,  Anthropic Claude Backend
==================================================
Uses Anthropic's Claude 3.5 Sonnet model to perform all three LLM processing
steps: speaker mapping, storyboard generation, and production script generation.

Key design: Tool calling for structured output
	Claude's tool-calling API (also called "function calling") allows us to
	define a JSON schema and instruct Claude to *always* return output matching
	that schema via a tool call rather than as free-form text.  This produces
	reliably parseable, schema-valid responses without requiring fragile JSON
	parsing of the model's text output.

	``generate_storyboard`` and ``generate_production_script`` both define a
	single tool and use ``tool_choice="any"`` to force the model to call it.

Speaker mapping uses plain text:
	``map_speakers`` uses a regular text prompt + ``_extract_json_from_text``
	because speaker mapping is simpler and the output is a small flat dict.

Dependencies:
	pip install anthropic python-dotenv
"""

import anthropic
import json
import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from src.stages.stage2_llm.base import BaseLLMProcessor
from src.shared.schemas import Transcript, Storyboard, ProductionScript, Scene, ProductionScene, SceneShot
from src.shared.exceptions import InsufficientCreditsError, InvalidAPIKeyError, ProviderUnavailableError
from src.stages.stage2_llm.anthropic_claude.claude_tool_schemas import (
	STORYBOARD_TOOL,
	PRODUCTION_SCRIPT_TOOL,
	SCENE_RELEVANCE_TOOL,
	SCENE_SHOTS_TOOL,
)

# Load .env file so that ANTHROPIC_API_KEY can be set there during local development.
load_dotenv()

logger = logging.getLogger(__name__)


class ClaudeProcessor(BaseLLMProcessor):
	"""
	Concrete implementation of BaseLLMProcessor using Anthropic Claude 3.5 Sonnet.

	Uses Claude's tool-calling API to guarantee structured JSON output for
	storyboard and production script generation.  Falls back to text parsing
	for speaker mapping (simpler, smaller output).

	Config keys:
		api_key ,  Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
		model   ,  Claude model name (default: ``"claude-3-5-sonnet-latest"``).
	"""

	DEFAULT_MODEL = "claude-3-5-sonnet-latest"

	def __init__(self, config: Dict[str, Any]):
		"""
		Initialize the Anthropic client.

		Uses a ``"dummy_key"`` fallback so that test suites that patch the client
		can import and instantiate this class without needing a real API key.

		Args:
			config: Configuration dict.  See class docstring for keys.
		"""
		self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
		if not self.api_key:
			# Allow instantiation in test environments where a mock client is injected.
			self.api_key = "dummy_key"

		self.client = anthropic.Anthropic(api_key=self.api_key)
		self.model = config.get("model", self.DEFAULT_MODEL)

	def map_speakers(self, transcript: Transcript, character_sheet_context: str = "") -> dict:
		"""
		Ask Claude to infer speaker identities from transcript dialogue context.

		Sends the full transcript text and optional character sheet context in a
		single user message.  Instructs Claude to return *only* a JSON dict
		(no preamble) mapping generic speaker labels to a string containing the
		player's real name, their in-character name, and their character class.

		Falls back to the identity speaker map if JSON parsing fails.

		Args:
			transcript:              Stage 1 Transcript with generic speaker labels.
			character_sheet_context: Optional description of campaign characters.

		Returns:
			Dict mapping ``"Speaker X"`` -> ``"Real Name - Character Name - Class"`` (or identity map on error).
		"""
		prompt = self._build_speaker_mapping_prompt(transcript.full_text, character_sheet_context)

		response = self._call_api(lambda: self.client.messages.create(
			model=self.model,
			max_tokens=1024,
			messages=[
				{"role": "user", "content": prompt}
			]
		))

		try:
			content_text = response.content[0].text
			return self._extract_json_from_text(content_text)
		except (json.JSONDecodeError, IndexError, AttributeError) as e:
			logger.error(f"Error parsing speaker map: {e}")
			# Fall back to passthrough mapping so the pipeline can continue.
			return self._identity_speaker_map(transcript)

	def generate_speaker_visualizations(self, speaker_map: dict) -> dict:
		"""
		Ask Claude to generate a visual description for each speaker.

		Uses the same plain-text + JSON extraction pattern as ``map_speakers``
		since the output is a small flat dict.

		Args:
			speaker_map: Dict mapping speaker labels to "Real Name - Character Name - Class".

		Returns:
			Dict mapping speaker labels to appearance/role description strings.
			Falls back to passthrough (label -> original entry) on parse errors.
		"""
		prompt = self._build_speaker_visualization_prompt(speaker_map)

		response = self._call_api(lambda: self.client.messages.create(
			model=self.model,
			max_tokens=1024,
			messages=[{"role": "user", "content": prompt}]
		))

		try:
			content_text = response.content[0].text
			return self._extract_json_from_text(content_text)
		except (json.JSONDecodeError, IndexError, AttributeError) as e:
			logger.error(f"Error parsing speaker visualizations: {e}")
			return {speaker: entry for speaker, entry in speaker_map.items()}

	def generate_storyboard(self, transcript: Transcript, speaker_map: dict) -> Storyboard:
		"""
		Use Claude tool calling to produce a structured storyboard from the transcript.

		Defines a ``generate_storyboard`` tool with a JSON schema describing the
		``scenes`` array.  Claude is forced to call this tool, ensuring output is
		schema-valid without any post-processing regex or JSON extraction.

		The prompt instructs Claude to:
			1. Filter out OOC table talk (rules discussions, bathroom breaks, etc.).
			2. Identify key narrative moments as separate scenes.
			3. Write a cinematic ``visual_prompt`` for each scene suitable as input
               to a text-to-video model.

		Args:
			transcript:  Stage 1 Transcript.
			speaker_map: Dict mapping generic labels to character names.

		Returns:
			A ``Storyboard`` with one ``Scene`` per narrative beat identified by Claude.
			Returns an empty ``Storyboard`` if no tool call is found in the response.
		"""
		prompt = self._build_storyboard_prompt(transcript.full_text, speaker_map)

		response = self._call_api(lambda: self.client.messages.create(
			model=self.model,
			max_tokens=4096,
			tools=STORYBOARD_TOOL,
			messages=[
				{"role": "user", "content": prompt}
			]
		))

		# Find the tool_use block in the response and extract its input directly.
		# Claude guarantees the input matches the schema we defined above.
		for content in response.content:
			if content.type == "tool_use" and content.name == "generate_storyboard":
				return Storyboard(**content.input)

		# Fallback: return empty storyboard if Claude didn't use the tool.
		return Storyboard(scenes=[])

	def generate_production_script(self, storyboard: Storyboard, transcript: Transcript) -> ProductionScript:
		"""
		Use Claude tool calling to expand storyboard scenes into production directions.

		Defines a ``generate_production_script`` tool that adds three fields to each
		scene: ``stage_directions``, ``character_actions``, and ``final_video_prompt``.

		The ``final_video_prompt`` is the most important output ,  it is the fully
		elaborated prompt sent to the video generation model in Stage 3.

		Args:
			storyboard:  The ``Storyboard`` returned by ``generate_storyboard``.
			transcript:  Original transcript for additional LLM context.

		Returns:
			A ``ProductionScript`` with enriched ``ProductionScene`` objects.
			Returns an empty ``ProductionScript`` if no tool call is found.
		"""
		prompt = self._build_production_script_prompt(storyboard.model_dump_json(), transcript.full_text)

		response = self._call_api(lambda: self.client.messages.create(
			model=self.model,
			max_tokens=4096,
			tools=PRODUCTION_SCRIPT_TOOL,
			messages=[
				{"role": "user", "content": prompt}
			]
		))

		# Extract the structured tool call output and hydrate the Pydantic model.
		for content in response.content:
			if content.type == "tool_use" and content.name == "generate_production_script":
				return ProductionScript(**content.input)

		return ProductionScript(scenes=[])

	def review_scene_relevance(self, storyboard: Storyboard) -> Storyboard:
		"""
		Use Claude tool calling to mark each scene as in-game (relevant) or OOC (irrelevant).

		A scene is NOT relevant when it consists entirely of out-of-character table talk
		(lunch breaks, bathroom breaks, player introductions, rules debates with no gameplay).
		Mixed scenes -- OOC chatter followed by any in-game event -- are marked relevant.

		Returns the same scenes with ``is_relevant`` and ``relevance_reason`` populated.
		Falls back to marking all scenes relevant if the tool call fails.
		"""
		scenes_summary = json.dumps([
			{
				"scene_number": s.scene_number,
				"location": s.location,
				"narrative_summary": s.narrative_summary
			}
			for s in storyboard.scenes
		], indent=2)

		prompt = f"""
		You are reviewing scenes from a D&D session recording to decide which contain
		actual in-game narrative content versus out-of-character (OOC) table talk.

		Mark a scene as NOT relevant (is_relevant: false) ONLY when it is ENTIRELY made
		up of OOC content such as:
          - Players taking a lunch or bathroom break
          - Pre-game or post-game player introductions (players introducing themselves,
			not their characters)
          - Extended rules debates or technical setup with zero in-game events
          - Session housekeeping (scheduling, logistics)

		Mark a scene as relevant (is_relevant: true) if ANY in-game narrative events
		occur within it, even if it starts or ends with OOC chatter.

		Scenes to review:
		{scenes_summary}

		Use the 'review_scene_relevance' tool to return your assessment for every scene.
		"""

		response = self._call_api(lambda: self.client.messages.create(
			model=self.model,
			max_tokens=2048,
			tools=SCENE_RELEVANCE_TOOL,
			tool_choice={"type": "any"},
			messages=[{"role": "user", "content": prompt}]
		))

		for content in response.content:
			if content.type != "tool_use" or content.name != "review_scene_relevance":
				continue

			# Build a lookup from scene_number -> relevance data
			relevance_map = {
				item["scene_number"]: item
				for item in content.input.get("scenes", [])
			}
			updated_scenes = []
			for scene in storyboard.scenes:
				data = relevance_map.get(scene.scene_number)
				if not data:
					# Scene not mentioned by LLM -- default to relevant
					updated_scenes.append(scene.model_copy(update={"is_relevant": True, "relevance_reason": "Not reviewed; assumed relevant."}))
					continue

				updated = scene.model_copy(update={
					"is_relevant": data["is_relevant"],
					"relevance_reason": data.get("relevance_reason", ""),
				})
				updated_scenes.append(updated)
			return Storyboard(scenes=updated_scenes)

		# Fallback: mark all scenes relevant so the pipeline continues
		logger.warning("review_scene_relevance: no tool call found in response; marking all scenes relevant.")
		fallback_scenes = [
			s.model_copy(update={"is_relevant": True, "relevance_reason": "Relevance review skipped."})
			for s in storyboard.scenes
		]
		return Storyboard(scenes=fallback_scenes)

	def generate_scene_shots(self, storyboard: Storyboard, transcript: Transcript) -> ProductionScript:
		"""
		Use Claude tool calling to expand each scene into production details plus a
		breakdown of sequential 3-10 second shots for video generation.

		Each scene receives stage_directions, character_actions, final_video_prompt, and
		an ordered list of SceneShot objects.  Stage 3 renders one video clip per shot.

		Returns an empty ProductionScript if no tool call is found in the response.
		"""
		# Build a per-scene transcript slice so each scene receives only the dialogue
		# that actually occurred during its time range rather than a truncated global dump.
		scene_transcripts = {
			scene.scene_number: self._get_scene_transcript(transcript, scene)
			for scene in storyboard.scenes
		}

		prompt = f"""
		You are a cinematic director expanding a D&D storyboard into a detailed production
		script with shot-by-shot breakdowns for AI video generation.

		For EACH scene:
		1. Write concise stage_directions (camera angles, lighting, mood).
		2. Write character_actions (what each character physically does and expresses).
		3. Write a final_video_prompt combining location, mood, and action for the whole scene.
		4. Break the scene into as many SHOTS as needed to cover it completely.
           Each shot is a discrete visual moment lasting ~5-10 seconds that can be rendered
           as a standalone video clip.  Use the transcript for that scene to identify every
           distinct action, spell cast, movement, or dialogue beat -- each one should be its
           own shot.  A short 2-minute travel scene might need 3-4 shots; a 12-minute combat
           encounter might need 15 or more.  Shots must:
           - Flow sequentially to tell the scene's story from start to finish.
           - Have a self-contained visual_prompt (no "continued from previous" references).
           - Cover distinct actions or beats (don't repeat the same image with different words).

		Example shots for "Road to Phandalin -- party encounters dead horses":
          Shot 1: Wide establishing shot of a dirt road through lush green hills, a wooden
                  wagon pulled by a horse trotting peacefully under a bright afternoon sky.
          Shot 2: Medium shot from the wagon driver's POV -- two dead horses lying across the
                  road ahead, blocking the path, flies buzzing around them.
          Shot 3: Close-up of an armored adventurer's face tightening with suspicion, hand
                  moving toward weapon hilt.
          Shot 4: Wide shot of four adventurers leaping down from the wagon, spreading out
                  in a defensive formation, weapons drawn, scanning the treeline.

		Storyboard: {storyboard.model_dump_json()}

		Scene transcripts (dialogue for each scene's time range):
		{json.dumps(scene_transcripts, indent=2)}

		Use the 'generate_scene_shots' tool to return the complete structured output.
		"""

		response = self._call_api(lambda: self.client.messages.create(
			model=self.model,
			max_tokens=8192,
			tools=SCENE_SHOTS_TOOL,
			tool_choice={"type": "any"},
			messages=[{"role": "user", "content": prompt}]
		))

		for content in response.content:
			if content.type != "tool_use" or content.name != "generate_scene_shots":
				continue

			scenes_data = content.input.get("scenes", [])
			production_scenes = []
			for s in scenes_data:
				shots = [SceneShot(**shot) for shot in s.pop("shots", [])]
				ps = ProductionScene(**s, shots=shots)
				production_scenes.append(ps)
			return ProductionScript(scenes=production_scenes)

		return ProductionScript(scenes=[])

	# --------------------------------------------------------------------------
	# Internal helpers
	# --------------------------------------------------------------------------

	def _call_api(self, fn):
		"""Call an Anthropic API function and translate SDK errors into typed ProviderErrors."""
		try:
			return fn()
		except anthropic.AuthenticationError as e:
			raise InvalidAPIKeyError(
				f"Anthropic API key is invalid or missing: {e}",
				provider_name="Anthropic Claude",
				help_url="https://console.anthropic.com/settings/keys",
			) from e
		except anthropic.BadRequestError as e:
			msg = str(e).lower()
			if "credit balance" in msg or "billing" in msg:
				raise InsufficientCreditsError(
					f"Your Anthropic credit balance is too low: {e}",
					provider_name="Anthropic Claude",
					help_url="https://console.anthropic.com/settings/plans",
				) from e
			raise
		except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
			raise ProviderUnavailableError(
				f"Anthropic service is currently unavailable: {e}",
				provider_name="Anthropic Claude",
				help_url="https://status.anthropic.com",
			) from e

