"""
Stage 2 LLM - Parsing & Schema Utilities
==========================================
Shared helpers for JSON extraction and Pydantic schema hydration used by all
LLM backends.
"""

import json
from src.shared.schemas import Transcript, Storyboard, ProductionScript, Scene


def extract_json_from_text(text: str) -> dict:
	"""
    Strip markdown code fences and parse JSON from an LLM text response.

    Handles the three common patterns:
        - Plain JSON string (no fences).
        - ```json ... ``` fenced block.
        - ``` ... ``` fenced block (no language tag).

    Args:
        text: Raw string content from an LLM response.

    Returns:
        A parsed Python dict.

    Raises:
        json.JSONDecodeError: If the cleaned content is not valid JSON.
    """
	content = text.strip()
	if "```json" in content:
		content = content.split("```json")[1].split("```")[0].strip()
	elif "```" in content:
		content = content.split("```")[1].split("```")[0].strip()
	return json.loads(content)


def identity_speaker_map(transcript: Transcript) -> dict:
	"""
    Return a passthrough speaker map as a safe fallback.

    Used when LLM-based speaker mapping fails. Each speaker label maps to
    itself so the pipeline continues with generic labels rather than crashing.

    Args:
        transcript: The transcript to extract distinct speaker labels from.

    Returns:
        A dict where every speaker maps to itself.
    """
	return {u.speaker: u.speaker for u in transcript.utterances}


def build_storyboard_from_data(data: dict) -> Storyboard:
	"""
    Construct a ``Storyboard`` Pydantic object from a parsed scenes dict.

    Args:
        data: A dict with a ``"scenes"`` key containing a list of scene dicts.

    Returns:
        A ``Storyboard`` containing validated ``Scene`` objects.
    """
	return Storyboard(scenes=[Scene(**s) for s in data.get("scenes", [])])


def build_production_script_from_data(data: dict) -> ProductionScript:
	"""
    Construct a ``ProductionScript`` Pydantic object from a parsed scenes dict.

    Args:
        data: A dict with a ``"scenes"`` key containing production scene dicts.

    Returns:
        A ``ProductionScript`` containing validated ``ProductionScene`` objects.
    """
	from src.shared.schemas import ProductionScene
	return ProductionScript(scenes=[ProductionScene(**s) for s in data.get("scenes", [])])


def get_scene_transcript(transcript: Transcript, scene: Scene) -> str:
	"""
    Extract and format transcript utterances that overlap a scene's time range.

    Uses an overlap condition (utterance ends after scene start AND utterance
    starts before scene end) to capture utterances that straddle the boundary.

    Returns a formatted string of timestamped dialogue lines, or empty string
    if no utterances fall within the scene's range.
    """
	utterances = [
		u for u in transcript.utterances
		if u.end > scene.start_time and u.start < scene.end_time
	]
	return "\n".join(
		f"[{u.start:.1f}s] {u.speaker}: {u.text}"
		for u in utterances
	)
