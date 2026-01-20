"""
Stage 2 LLM Processing ,  Google Gemini Backend
===============================================
Uses the Google Gemini 2.0 Flash model for all three LLM processing steps.

Key design: Gemini proto-based function calling
    Gemini's function calling API is structurally similar to OpenAI's, but uses
    Protocol Buffer (proto) schema objects instead of plain Python dicts.  We
    define tools using ``genai.protos.Tool``, ``genai.protos.FunctionDeclaration``,
    and ``genai.protos.Schema`` objects, which the SDK then serializes to the
    Gemini API format.

    ``tool_config={"function_calling_config": {"mode": "ANY"}}`` forces Gemini
    to always call a function (equivalent to OpenAI's forced ``tool_choice``).

Proto schema types:
    - ``genai.protos.Type.OBJECT``  ,  corresponds to a JSON object / Python dict.
    - ``genai.protos.Type.ARRAY``   ,  corresponds to a JSON array / Python list.
    - ``genai.protos.Type.STRING``  ,  string field.
    - ``genai.protos.Type.INTEGER`` ,  integer field.
    - ``genai.protos.Type.NUMBER``  ,  float/number field.

Dependencies:
    pip install google-generativeai
"""

import os
import logging
from typing import Dict, Any
import google.generativeai as genai
from src.stages.stage2_llm.base import BaseLLMProcessor
from src.shared.schemas import Transcript, Storyboard, ProductionScript, SceneShot, ProductionScene

logger = logging.getLogger(__name__)


class GeminiProcessor(BaseLLMProcessor):
	"""
    Concrete implementation of BaseLLMProcessor using Google Gemini 2.0 Flash.

    Uses Gemini's function calling (via proto schema definitions) to produce
    reliably structured output for storyboard and production script generation.
    Speaker mapping uses a plain text prompt with JSON extraction.

    Config keys:
        api_key / GOOGLE_API_KEY ,  Google AI API key.
        model                    ,  Gemini model name (default: ``"gemini-2.0-flash-exp"``).
    """

	DEFAULT_MODEL = "gemini-2.0-flash-exp"

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Gemini client via the ``genai.configure`` module-level call.

        The ``google-generativeai`` SDK uses a module-level configuration pattern
        (similar to AssemblyAI) where ``genai.configure(api_key=...)`` sets the
        credentials for all subsequent calls.

        Args:
            config: Configuration dict.  See class docstring for keys.
        """
		self.api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
		if not self.api_key:
			# Allow instantiation in test environments.
			self.api_key = "dummy_key"

		# Configure the SDK globally ,  all genai calls in this process will use this key.
		genai.configure(api_key=self.api_key)
		self.model_name = config.get("model", self.DEFAULT_MODEL)

		# Create a base model instance (without tools); tool-equipped variants are
		# created per-call because Gemini requires tools to be attached at model init.
		self.model = genai.GenerativeModel(self.model_name)

	def map_speakers(self, transcript: Transcript, character_sheet_context: str = "") -> dict:
		"""
        Use Gemini to infer speaker identities from transcript dialogue.

        Uses a plain text generation (no function calling) since speaker mapping
        returns a small, flat JSON dict.  Falls back to identity map on any error.

        Args:
            transcript:              Stage 1 Transcript.
            character_sheet_context: Optional character descriptions.

        Returns:
            Dict mapping speaker labels to "Real Name - Character Name - Class" strings, or identity map on error.
        """
		prompt = self._build_speaker_mapping_prompt(transcript.full_text, character_sheet_context)
		try:
			response = self.model.generate_content(prompt)
			# response.text is the model's string output; extract JSON from it.
			return self._extract_json_from_text(response.text)
		except Exception as e:
			logger.error(f"Error in Gemini map_speakers: {e}")
			return self._identity_speaker_map(transcript)

	def generate_speaker_visualizations(self, speaker_map: dict) -> dict:
		"""
        Use Gemini to generate a visual description for each speaker.

        Uses plain text generation (same as ``map_speakers``) since the output
        is a small flat dict.  Falls back to passthrough on any error.

        Args:
            speaker_map: Dict mapping speaker labels to "Real Name - Character Name - Class".

        Returns:
            Dict mapping speaker labels to appearance/role description strings.
        """
		prompt = self._build_speaker_visualization_prompt(speaker_map)
		try:
			response = self.model.generate_content(prompt)
			return self._extract_json_from_text(response.text)
		except Exception as e:
			logger.error(f"Error in Gemini generate_speaker_visualizations: {e}")
			return {speaker: entry for speaker, entry in speaker_map.items()}

	def generate_storyboard(self, transcript: Transcript, speaker_map: dict) -> Storyboard:
		"""
        Use Gemini function calling to generate a structured storyboard.

        Defines a proto-based ``generate_storyboard`` function and creates a
        fresh ``GenerativeModel`` instance with the tool attached.
        ``mode="ANY"`` forces the model to always call a function.

        After the call, we iterate over ``response.candidates[0].content.parts``
        to find the ``function_call`` part matching our function name, then
        convert its ``args`` (a proto ``MapComposite``) to a plain Python dict.

        Args:
            transcript:  Stage 1 Transcript.
            speaker_map: Speaker-to-character name mapping.

        Returns:
            A ``Storyboard`` parsed from the function call args, or empty on error.
        """
		# Build the proto-based tool definition for the storyboard function.
		# genai.protos.Schema describes each field's type; nested OBJECT/ARRAY schemas
		# are composed recursively.
		_STORYBOARD_SCENE_SCHEMA = genai.protos.Schema(
			type=genai.protos.Type.OBJECT,
			properties={
				"scene_number": genai.protos.Schema(type=genai.protos.Type.INTEGER),
				"start_time": genai.protos.Schema(type=genai.protos.Type.NUMBER),
				"end_time": genai.protos.Schema(type=genai.protos.Type.NUMBER),
				"location": genai.protos.Schema(type=genai.protos.Type.STRING),
				"narrative_summary": genai.protos.Schema(type=genai.protos.Type.STRING),
				"visual_prompt": genai.protos.Schema(type=genai.protos.Type.STRING),
			},
			required=[
				"scene_number", "start_time", "end_time",
				"location", "narrative_summary", "visual_prompt"
			],
		)

		_storyboard_params_schema = genai.protos.Schema(
			type=genai.protos.Type.OBJECT,
			properties={
				"scenes": genai.protos.Schema(
					type=genai.protos.Type.ARRAY,
					items=_STORYBOARD_SCENE_SCHEMA
				)
			},
			required=["scenes"],
		)

		storyboard_tool = genai.protos.Tool(
			function_declarations=[
				genai.protos.FunctionDeclaration(
					name="generate_storyboard",
					description="Generate a storyboard from a D&D session transcript.",
					parameters=_storyboard_params_schema,
				)
			]
		)

		prompt = self._build_storyboard_prompt(transcript.full_text, speaker_map, fn_keyword="function")
		try:
			# Attach the tool to a fresh model instance ,  Gemini requires this.
			model_with_tools = genai.GenerativeModel(self.model_name, tools=[storyboard_tool])
			response = model_with_tools.generate_content(
				prompt,
				# mode="ANY" means the model MUST call a function (no free text).
				tool_config={"function_calling_config": {"mode": "ANY"}},
			)

			# Walk response parts to find our function call.
			for part in response.candidates[0].content.parts:
				if part.function_call and part.function_call.name == "generate_storyboard":
					# part.function_call.args is a proto MapComposite; cast to plain dict.
					args = dict(part.function_call.args)
					return self._build_storyboard_from_data(args)

		except Exception as e:
			logger.error(f"Error in Gemini generate_storyboard: {e}")

		return Storyboard(scenes=[])

	def generate_production_script(self, storyboard: Storyboard, transcript: Transcript) -> ProductionScript:
		"""
        Use Gemini function calling to expand storyboard into a production script.

        Same pattern as ``generate_storyboard`` but with the extended schema that
        includes stage_directions, character_actions, and final_video_prompt fields.

        Args:
            storyboard:  The ``Storyboard`` to expand.
            transcript:  Original transcript for context.

        Returns:
            A ``ProductionScript`` parsed from the function call, or empty on error.
        """
		# Extended proto schema adding the three production-specific fields.
		_PRODUCTION_SCRIPT_SCENE_SCHEMA = genai.protos.Schema(
			type=genai.protos.Type.OBJECT,
			properties={
				"scene_number": genai.protos.Schema(type=genai.protos.Type.INTEGER),
				"start_time": genai.protos.Schema(type=genai.protos.Type.NUMBER),
				"end_time": genai.protos.Schema(type=genai.protos.Type.NUMBER),
				"location": genai.protos.Schema(type=genai.protos.Type.STRING),
				"narrative_summary": genai.protos.Schema(type=genai.protos.Type.STRING),
				"visual_prompt": genai.protos.Schema(type=genai.protos.Type.STRING),
				"stage_directions": genai.protos.Schema(type=genai.protos.Type.STRING),
				"character_actions": genai.protos.Schema(type=genai.protos.Type.STRING),
				"final_video_prompt": genai.protos.Schema(type=genai.protos.Type.STRING),
			},
			required=[
				"scene_number", "start_time", "end_time", "location",
				"narrative_summary", "visual_prompt", "stage_directions",
				"character_actions", "final_video_prompt"
			],
		)

		_production_params_schema = genai.protos.Schema(
			type=genai.protos.Type.OBJECT,
			properties={
				"scenes": genai.protos.Schema(
					type=genai.protos.Type.ARRAY,
					items=_PRODUCTION_SCRIPT_SCENE_SCHEMA
				)
			},
			required=["scenes"],
		)

		production_tool = genai.protos.Tool(
			function_declarations=[
				genai.protos.FunctionDeclaration(
					name="generate_production_script",
					description="Expand a storyboard into a detailed production script.",
					parameters=_production_params_schema,
				)
			]
		)

		prompt = self._build_production_script_prompt(storyboard.model_dump_json(), transcript.full_text, fn_keyword="function")
		try:
			model_with_tools = genai.GenerativeModel(self.model_name, tools=[production_tool])
			response = model_with_tools.generate_content(
				prompt,
				tool_config={"function_calling_config": {"mode": "ANY"}},
			)

			for part in response.candidates[0].content.parts:
				if part.function_call and part.function_call.name == "generate_production_script":
					args = dict(part.function_call.args)
					return self._build_production_script_from_data(args)

		except Exception as e:
			logger.error(f"Error in Gemini generate_production_script: {e}")

		return ProductionScript(scenes=[])

	def _apply_relevance(self, storyboard: Storyboard, relevance_map: dict) -> Storyboard:
		"""Updates a storyboard with relevance flags from a mapping dict."""

		updated_scenes = []
		for scene in storyboard.scenes:
			data = relevance_map.get(scene.scene_number)
			if data:
				update_dict = {
					"is_relevant": data["is_relevant"],
					"relevance_reason": data.get("relevance_reason", ""),
				}
			else:
				update_dict = {
					"is_relevant": True,
					"relevance_reason": "Not reviewed; assumed relevant.",
				}
			updated_scenes.append(scene.model_copy(update=update_dict))
		return Storyboard(scenes=updated_scenes)

	def review_scene_relevance(self, storyboard: Storyboard) -> Storyboard:
		"""
        Use Gemini function calling to mark each scene as in-game (relevant) or OOC (irrelevant).

        Defines a proto-based ``review_scene_relevance`` function and forces the model to call
        it with ``mode="ANY"``. Falls back to marking all scenes relevant on any error.
        """
		_RELEVANCE_ITEM_SCHEMA = genai.protos.Schema(
			type=genai.protos.Type.OBJECT,
			properties={
				"scene_number": genai.protos.Schema(type=genai.protos.Type.INTEGER),
				"relevance_reason": genai.protos.Schema(type=genai.protos.Type.STRING),
				"is_relevant": genai.protos.Schema(type=genai.protos.Type.BOOLEAN),
			},
			required=["scene_number", "relevance_reason", "is_relevant"],
		)

		_relevance_params_schema = genai.protos.Schema(
			type=genai.protos.Type.OBJECT,
			properties={
				"scenes": genai.protos.Schema(
					type=genai.protos.Type.ARRAY,
					items=_RELEVANCE_ITEM_SCHEMA
				)
			},
			required=["scenes"],
		)

		relevance_tool = genai.protos.Tool(
			function_declarations=[
				genai.protos.FunctionDeclaration(
					name="review_scene_relevance",
					description="Review each storyboard scene and decide whether it contains in-game narrative content.",
					parameters=_relevance_params_schema,
				)
			]
		)

		import json as _json
		scenes_summary = _json.dumps([
			{
				"scene_number": s.scene_number,
				"location": s.location,
				"narrative_summary": s.narrative_summary,
			}
			for s in storyboard.scenes
		], indent=2)

		prompt = self._build_scene_relevance_prompt(scenes_summary, fn_keyword="function")
		try:
			model_with_tools = genai.GenerativeModel(self.model_name, tools=[relevance_tool])
			response = model_with_tools.generate_content(
				prompt,
				tool_config={"function_calling_config": {"mode": "ANY"}},
			)
			for part in response.candidates[0].content.parts:
				if part.function_call and part.function_call.name == "review_scene_relevance":
					args = dict(part.function_call.args)
					relevance_map = {
						item["scene_number"]: item
						for item in args.get("scenes", [])
					}
					return self._apply_relevance(storyboard, relevance_map)
		except Exception as e:
			logger.error(f"Error in Gemini review_scene_relevance: {e}")
			raise

		logger.warning("Gemini failed to generate a relevance review; passing all scenes.")
		fallback_scenes = [
			s.model_copy(update={"is_relevant": True, "relevance_reason": "Relevance review skipped."})
			for s in storyboard.scenes
		]
		return Storyboard(scenes=fallback_scenes)

	def generate_scene_shots(self, storyboard: Storyboard, transcript: Transcript) -> ProductionScript:
		"""
        Use Gemini function calling to expand each scene into production details and shot breakdowns.

        Defines a proto-based ``generate_scene_shots`` function with nested shots schema.
        Returns an empty ProductionScript if no function call is found in the response.
        """
		import json as _json

		shots_schema = genai.protos.Schema(
			type=genai.protos.Type.ARRAY,
			items=genai.protos.Schema(
				type=genai.protos.Type.OBJECT,
				properties={
					"shot_number": genai.protos.Schema(type=genai.protos.Type.INTEGER),
					"description": genai.protos.Schema(type=genai.protos.Type.STRING),
					"visual_prompt": genai.protos.Schema(type=genai.protos.Type.STRING),
					"duration_hint": genai.protos.Schema(type=genai.protos.Type.INTEGER),
				},
				required=["shot_number", "description", "visual_prompt", "duration_hint"],
			),
		)

		_SCENE_SHOTS_ITEM_SCHEMA = genai.protos.Schema(
			type=genai.protos.Type.OBJECT,
			properties={
				"scene_number": genai.protos.Schema(type=genai.protos.Type.INTEGER),
				"start_time": genai.protos.Schema(type=genai.protos.Type.NUMBER),
				"end_time": genai.protos.Schema(type=genai.protos.Type.NUMBER),
				"location": genai.protos.Schema(type=genai.protos.Type.STRING),
				"narrative_summary": genai.protos.Schema(type=genai.protos.Type.STRING),
				"visual_prompt": genai.protos.Schema(type=genai.protos.Type.STRING),
				"stage_directions": genai.protos.Schema(type=genai.protos.Type.STRING),
				"character_actions": genai.protos.Schema(type=genai.protos.Type.STRING),
				"final_video_prompt": genai.protos.Schema(type=genai.protos.Type.STRING),
				"shots": shots_schema,
			},
			required=[
				"scene_number", "start_time", "end_time", "location",
				"narrative_summary", "visual_prompt",
				"stage_directions", "character_actions", "final_video_prompt",
				"shots",
			],
		)

		_scene_shots_params_schema = genai.protos.Schema(
			type=genai.protos.Type.OBJECT,
			properties={
				"scenes": genai.protos.Schema(
					type=genai.protos.Type.ARRAY,
					items=_SCENE_SHOTS_ITEM_SCHEMA
				)
			},
			required=["scenes"],
		)

		scene_shots_tool = genai.protos.Tool(
			function_declarations=[
				genai.protos.FunctionDeclaration(
					name="generate_scene_shots",
					description="Expand storyboard scenes into production details and break each into short video shots.",
					parameters=_scene_shots_params_schema,
				)
			]
		)

		scene_transcripts = {
			scene.scene_number: self._get_scene_transcript(transcript, scene)
			for scene in storyboard.scenes
		}
		prompt = self._build_scene_shots_prompt(
			storyboard.model_dump_json(),
			_json.dumps(scene_transcripts, indent=2),
			fn_keyword="function",
		)

		try:
			model_with_tools = genai.GenerativeModel(self.model_name, tools=[scene_shots_tool])
			response = model_with_tools.generate_content(
				prompt,
				tool_config={"function_calling_config": {"mode": "ANY"}},
			)
			for part in response.candidates[0].content.parts:
				if part.function_call and part.function_call.name == "generate_scene_shots":
					args = dict(part.function_call.args)
					production_scenes = []
					for s in args.get("scenes", []):
						s = dict(s)
						shots = [SceneShot(**dict(shot)) for shot in s.pop("shots", [])]
						production_scenes.append(ProductionScene(**s, shots=shots))
					return ProductionScript(scenes=production_scenes)
		except Exception as e:
			logger.error(f"Error in Gemini generate_scene_shots: {e}")

		return ProductionScript(scenes=[])
