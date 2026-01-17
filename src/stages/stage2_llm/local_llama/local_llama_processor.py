"""
Stage 2 LLM Processing ,  Local Llama via Ollama Backend
========================================================
Uses a locally-running Llama model served by Ollama for all three LLM processing
steps.  Requires no API keys or cloud access ,  runs entirely on-device.

Why Ollama?
    Ollama provides a simple REST API server that manages local model weights,
    quantization, and GPU/CPU scheduling.  The API is a single HTTP POST endpoint
    at ``/api/chat`` that accepts OpenAI-compatible message arrays and returns
    a JSON response.

Chunking strategy for long transcripts:
    Local models typically have smaller context windows than cloud APIs (e.g. 8K
    tokens for Llama 3.1 vs. 200K for Claude 3.5 Sonnet).  ``generate_storyboard``
    splits the utterance list into chunks of 40 utterances each, processes each
    chunk separately, and merges the resulting scenes with globally unique
    scene numbers.

``generate_production_script`` processes scenes one at a time for the same reason.
Falls back gracefully (logging the error, continuing to the next chunk/scene)
rather than raising, so a single bad LLM response doesn't abort the whole pipeline.

Dependencies:
    ollama (running locally at http://localhost:11434 by default)
    pip install httpx
"""

import os
import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from src.stages.stage2_llm.base import BaseLLMProcessor
from src.shared.schemas import Transcript, Storyboard, ProductionScript, Scene, ProductionScene, SceneShot

logger = logging.getLogger(__name__)


class LocalLlamaProcessor(BaseLLMProcessor):
	"""
    Concrete implementation of BaseLLMProcessor using Llama via Ollama.

    Communicates with an Ollama server over HTTP.  No API key needed.
    Used directly for Llama and as the base class for Mistral, Dolphin,
    Gemma, and Qwen thin subclasses (each just sets a different default model).

    Config keys:
        host    ,  Ollama server URL (default: ``"http://localhost:11434"``).
        model   ,  Ollama model tag (default: ``"llama3.1"``).
                  Must match a model you have already pulled:
                  ``ollama pull llama3.1``
        timeout ,  HTTP request timeout in seconds (default: 120.0).
                  Long transcripts may need a higher value.
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Local Llama processor with Ollama connection settings.

        No network call is made here; the Ollama server is contacted only when
        a pipeline method is called.

        Args:
            config: Configuration dict.  See class docstring for keys.
        """
		self.host = config.get("host", "http://localhost:11434")
		self.model = config.get("model", "llama3.1")
		# Timeout in seconds for the entire HTTP request (connect + read).
		# Large models on CPU can be slow; 120 s is a conservative default.
		self.timeout = config.get("timeout", 120.0)

	def _call_ollama(self, messages: List[Dict[str, str]], format: Optional[str] = None) -> str:
		"""
        Send a chat request to the Ollama server and return the response text.

        Uses the ``/api/chat`` endpoint with ``stream=False`` so the full response
        is returned in a single JSON object rather than as a stream of tokens.

        Args:
            messages: OpenAI-compatible list of ``{"role": ..., "content": ...}`` dicts.
            format:   Optional response format hint.  Pass ``"json"`` to request
                      that Ollama coerces the model's output to valid JSON.
                      Note: this does not guarantee schema correctness, only
                      that the raw string is parseable JSON.

        Returns:
            The model's response content as a plain string.

        Raises:
            httpx.ConnectError:     If Ollama is not running or the host is wrong.
            httpx.TimeoutException: If the model takes longer than ``self.timeout``.
            httpx.HTTPStatusError:  On non-2xx HTTP responses (e.g. model not found).
        """
		url = f"{self.host}/api/chat"
		payload = {
			"model": self.model,
			"messages": messages,
			"stream": False,  # Return the full response in one go, not as a stream.
		}
		if format == "json":
			# Ollama's JSON format mode requests the model to produce valid JSON.
			# This is a best-effort hint ,  the model may still produce malformed JSON.
			payload["format"] = "json"

		try:
			with httpx.Client(timeout=self.timeout) as client:
				response = client.post(url, json=payload)
				response.raise_for_status()
				# Ollama's response structure: {"message": {"role": "...", "content": "..."}}
				return response.json()["message"]["content"]

		except httpx.ConnectError as e:
			logger.error(
				"Cannot connect to Ollama at %s. Ensure Ollama is running and the host is correct. Error: %s",
				self.host, e,
			)
			raise
		except httpx.TimeoutException as e:
			logger.error(
				"Request to Ollama at %s timed out after %.1fs. Consider increasing the timeout. Error: %s",
				self.host, self.timeout, e,
			)
			raise
		except httpx.HTTPStatusError as e:
			logger.error(
				"Ollama returned HTTP %s for model '%s': %s",
				e.response.status_code, self.model, e,
			)
			raise
		except Exception as e:
			logger.error("Unexpected error calling Ollama API at %s: %s", self.host, e)
			raise

	def map_speakers(self, transcript: Transcript, character_sheet_context: str = "") -> dict:
		"""
        Use the local LLM to map generic speaker labels to real name, character name, and class.

        Only sends the first 5000 characters of the transcript to stay within
        the model's context window.  Includes all distinct speaker labels so
        the model can generate a complete mapping even from a partial view.

        Falls back to an identity map (each speaker maps to itself) on any error.

        Args:
            transcript:              Stage 1 Transcript.
            character_sheet_context: Optional character descriptions.

        Returns:
            Dict mapping speaker labels to "Real Name - Character Name - Class" strings, or identity map on error.
        """
		# Pre-compute distinct speakers so the prompt explicitly lists what to map.
		distinct_speakers = sorted(list(set(u.speaker for u in transcript.utterances)))

		prompt = self._build_speaker_mapping_prompt(
			transcript.full_text[:5000], character_sheet_context, distinct_speakers
		)

		messages = [{"role": "user", "content": prompt}]
		try:
			# Request JSON format to improve parse success rate.
			response_text = self._call_ollama(messages, format="json")
			return json.loads(response_text)
		except Exception as e:
			logger.error(f"Error in map_speakers: {e}")
			# Identity map: each speaker stays as their generic label.
			return {s: s for s in distinct_speakers}

	def generate_speaker_visualizations(self, speaker_map: dict) -> dict:
		"""
        Use the local LLM to generate a visual description for each speaker.

        The speaker_map is small enough to send in full even on local models.
        Falls back to passthrough on any error.

        Args:
            speaker_map: Dict mapping speaker labels to "Real Name - Character Name - Class".

        Returns:
            Dict mapping speaker labels to appearance/role description strings.
        """
		prompt = self._build_speaker_visualization_prompt(speaker_map)
		messages = [{"role": "user", "content": prompt}]
		try:
			response_text = self._call_ollama(messages, format="json")
			return json.loads(response_text)
		except Exception as e:
			logger.error(f"Error in generate_speaker_visualizations: {e}")
			return {speaker: entry for speaker, entry in speaker_map.items()}

	def generate_storyboard(self, transcript: Transcript, speaker_map: dict) -> Storyboard:
		"""
        Generate a storyboard by processing the transcript in utterance chunks.

        Local models have smaller context windows, so we split the transcript into
        chunks of ``MAX_UTTERANCES_PER_CHUNK`` utterances and call the LLM once
        per chunk.  Each chunk produces its own scenes, which are merged into a
        single scene list with globally unique scene numbers.

        If a chunk fails (bad JSON, timeout, etc.), it is skipped and a warning
        is logged ,  the pipeline continues with remaining chunks.

        Args:
            transcript:  Stage 1 Transcript.
            speaker_map: Speaker-to-character name mapping.

        Returns:
            A ``Storyboard`` containing scenes from all successfully processed chunks.
        """
		# Maximum utterances per LLM call ,  tuned to fit in ~8K token context windows.
		MAX_UTTERANCES_PER_CHUNK = 40
		all_scenes = []

		utterances = transcript.utterances
		# Split utterances into fixed-size chunks.
		chunks = [
			utterances[i:i + MAX_UTTERANCES_PER_CHUNK]
			for i in range(0, len(utterances), MAX_UTTERANCES_PER_CHUNK)
		]

		for i, chunk in enumerate(chunks):
			# Format this chunk with character names and timestamps.
			mapped_text = "\n".join([
				f"{speaker_map.get(u.speaker, u.speaker)}: {u.text} [{u.start:.1f}-{u.end:.1f}]"
				for u in chunk
			])

			prompt = f"""
            You are a storyboard artist. Convert the following transcript chunk into storyboard scenes.

            Transcript Chunk {i+1}/{len(chunks)}:
            {mapped_text}

            Instructions:
            1. Filter out Out-Of-Character (OOC) talk or non-narrative dialogue.
            2. Divide the dialogue into logical scenes.
            3. For each scene, provide: scene_number, start_time, end_time, location, narrative_summary, and visual_prompt.

            Return a JSON object with a key "scenes" containing an array of scene objects.
            Example: {{"scenes": [{{ "scene_number": 1, "start_time": 0.0, "end_time": 10.5, "location": "Forest", "narrative_summary": "...", "visual_prompt": "..." }}]}}

            ONLY return the JSON object.
            """

			messages = [{"role": "user", "content": prompt}]
			try:
				response_text = self._call_ollama(messages, format="json")
				chunk_data = json.loads(response_text)
				for scene in chunk_data.get("scenes", []):
					# Overwrite scene_number with the global sequence to avoid
					# numbering conflicts across chunks (each chunk starts at 1).
					scene["scene_number"] = len(all_scenes) + 1
					all_scenes.append(Scene(**scene))
			except Exception as e:
				logger.error(f"Error in generate_storyboard chunk {i}: {e}")
				continue  # Skip this chunk and proceed with the next.

		return Storyboard(scenes=all_scenes)

	def generate_production_script(self, storyboard: Storyboard, transcript: Transcript) -> ProductionScript:
		"""
        Expand storyboard scenes into production directions, one scene at a time.

        Processes each scene in a separate LLM call to avoid context overflow.
        Falls back to a minimal ``ProductionScene`` (with ``"N/A"`` directions and
        the original ``visual_prompt`` as the ``final_video_prompt``) if any
        individual scene fails, so the pipeline always produces a complete script.

        Args:
            storyboard:  The ``Storyboard`` to expand.
            transcript:  Original transcript (not used here but kept for interface).

        Returns:
            A ``ProductionScript`` containing one ``ProductionScene`` per input scene.
        """
		all_production_scenes = []

		for scene in storyboard.scenes:
			prompt = f"""
            You are a film director. Expand the following storyboard scene into a production script.

            Scene Context:
            {scene.model_dump_json()}

            Instructions:
            - Expand the visual_prompt into a highly detailed final_video_prompt (cinematic, lighting, camera angles).
            - Add stage_directions: Detailed camera and environment setup.
            - Add character_actions: Specific movements, expressions, and non-verbal cues.

            Return a JSON object matching the ProductionScene schema.
            Example: {{
                "scene_number": {scene.scene_number},
                "start_time": {scene.start_time},
                "end_time": {scene.end_time},
                "location": "{scene.location}",
                "narrative_summary": "...",
                "visual_prompt": "...",
                "stage_directions": "...",
                "character_actions": "...",
                "final_video_prompt": "..."
            }}

            ONLY return the JSON object.
            """

			messages = [{"role": "user", "content": prompt}]
			try:
				response_text = self._call_ollama(messages, format="json")
				scene_data = json.loads(response_text)
				all_production_scenes.append(ProductionScene(**scene_data))
			except Exception as e:
				logger.error(f"Error in generate_production_script scene {scene.scene_number}: {e}")
				# Fallback: promote the storyboard scene to a ProductionScene with
				# placeholder directions so the pipeline can continue.
				all_production_scenes.append(ProductionScene(
					**scene.model_dump(),
					stage_directions="N/A",
					character_actions="N/A",
					final_video_prompt=scene.visual_prompt,
				))

		return ProductionScript(scenes=all_production_scenes)

	def review_scene_relevance(self, storyboard: Storyboard) -> Storyboard:
		"""
        Use the local LLM to mark each scene as in-game (relevant) or OOC (irrelevant).

        Sends all scenes in a single call (output is small) using JSON format mode.
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

		prompt = self._build_scene_relevance_prompt(scenes_summary, fn_keyword="") + (
			'\n\nReturn a JSON object: {"scenes": [{"scene_number": <int>, "is_relevant": <bool>, "relevance_reason": "<str>"}]}\n'
			"ONLY return the JSON object."
		)

		messages = [{"role": "user", "content": prompt}]
		try:
			response_text = self._call_ollama(messages, format="json")
			data = json.loads(response_text)
			relevance_map = {
				item["scene_number"]: item
				for item in data.get("scenes", [])
			}
			updated_scenes = []
			for scene in storyboard.scenes:
				item = relevance_map.get(scene.scene_number)
				if item:
					updated_scenes.append(scene.model_copy(update={
						"is_relevant": item["is_relevant"],
						"relevance_reason": item.get("relevance_reason", ""),
					}))
				else:
					updated_scenes.append(scene.model_copy(update={
						"is_relevant": True,
						"relevance_reason": "Not reviewed; assumed relevant.",
					}))
			return Storyboard(scenes=updated_scenes)
		except Exception as e:
			logger.error(f"Error in review_scene_relevance: {e}")

		logger.warning("review_scene_relevance: falling back to marking all scenes relevant.")
		fallback_scenes = [
			s.model_copy(update={"is_relevant": True, "relevance_reason": "Relevance review skipped."})
			for s in storyboard.scenes
		]
		return Storyboard(scenes=fallback_scenes)

	def generate_scene_shots(self, storyboard: Storyboard, transcript: Transcript) -> ProductionScript:
		"""
        Expand storyboard scenes into production details with shot breakdowns, one scene at a time.

        Processes each scene in a separate LLM call to avoid context overflow.
        Falls back to a minimal ProductionScene with empty shots if any individual scene fails.
        """
		all_production_scenes = []

		for scene in storyboard.scenes:
			scene_transcript = self._get_scene_transcript(transcript, scene)
			prompt = f"""
            You are a cinematic director. Expand the following D&D scene into a production script
            with shot-by-shot breakdowns for video generation.

            Scene:
            {scene.model_dump_json()}

            Scene transcript (dialogue during this scene):
            {scene_transcript}

            Instructions:
            - Write stage_directions: camera angles, lighting, and mood.
            - Write character_actions: what each character does and expresses.
            - Write final_video_prompt: combines location, mood, and action for the whole scene.
            - Break the scene into shots -- each a 3-10 second visual moment. Every distinct
              action beat should be its own shot. Use as many as the scene requires.

            Return a JSON object with all scene fields plus stage_directions, character_actions,
            final_video_prompt, and a shots array.
            Example: {{
                "scene_number": {scene.scene_number},
                "start_time": {scene.start_time},
                "end_time": {scene.end_time},
                "location": "{scene.location}",
                "narrative_summary": "...",
                "visual_prompt": "...",
                "stage_directions": "...",
                "character_actions": "...",
                "final_video_prompt": "...",
                "shots": [
                    {{"shot_number": 1, "description": "...", "visual_prompt": "...", "duration_hint": 5}},
                    {{"shot_number": 2, "description": "...", "visual_prompt": "...", "duration_hint": 5}}
                ]
            }}

            ONLY return the JSON object.
            """

			messages = [{"role": "user", "content": prompt}]
			try:
				response_text = self._call_ollama(messages, format="json")
				scene_data = json.loads(response_text)
				shots = [SceneShot(**shot) for shot in scene_data.pop("shots", [])]
				all_production_scenes.append(ProductionScene(**scene_data, shots=shots))
			except Exception as e:
				logger.error(f"Error in generate_scene_shots scene {scene.scene_number}: {e}")
				all_production_scenes.append(ProductionScene(
					**scene.model_dump(),
					stage_directions="N/A",
					character_actions="N/A",
					final_video_prompt=scene.visual_prompt,
					shots=[],
				))

		return ProductionScript(scenes=all_production_scenes)
