"""Stage 3 Entry Point - Video Generation
==========================================
Reads ``shot_script.json`` (preferred) or ``production_script.json`` from the
session directory, renders one video clip per scene/shot using the configured
video generator, and writes scene_*.mp4 files plus a run report to ``output_dir``.

Character visual descriptions (from speaker_visualization.json) and the
relevant transcript lines for each scene are prepended to each shot's
final_video_prompt to improve visual consistency across generated clips.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Callable, Awaitable, Optional, Dict, List, Any

from src.shared.schemas import ProductionScript, ProductionScene
from src.orchestrator.providers import _get_video_generator
from src.orchestrator.reporting import _print_scene_report, _save_run_report

logger = logging.getLogger(__name__)


def _enrich_scene_prompts(
	scenes: List[ProductionScene],
	speaker_viz: Dict[str, str],
	transcript_utterances: List[Dict[str, Any]],
) -> List[ProductionScene]:
	"""Prepend character descriptions and relevant dialogue to each scene's video prompt.

    Args:
        scenes: List of ProductionScene objects to enrich.
        speaker_viz: Map of speaker/character name -> visual description string.
        transcript_utterances: List of utterance dicts with keys: speaker, text, start, end.

    Returns:
        A new list of ProductionScene objects with enriched final_video_prompt fields.
    """
	if not speaker_viz and not transcript_utterances:
		return scenes

	# Build character description block (used for all scenes)
	char_lines = [
		f"{name}: {desc.strip()}"
		for name, desc in speaker_viz.items()
		if desc and desc.strip()
	]
	char_block = "Characters:\n" + "\n".join(char_lines) if char_lines else ""

	enriched = []
	for scene in scenes:
		original_prompt = scene.final_video_prompt or scene.visual_prompt or ""

		# Find transcript lines that overlap with this scene's time range
		relevant_lines = []
		if transcript_utterances:
			for utt in transcript_utterances:
				utt_start = utt.get("start", utt.get("start_time", 0))
				utt_end   = utt.get("end",   utt.get("end_time",   utt_start))
				# Include utterance if it overlaps with the scene time window
				if utt_end >= scene.start_time and utt_start <= scene.end_time:
					speaker = utt.get("speaker", "")
					text    = utt.get("text", "").strip()
					if text:
						relevant_lines.append(f"{speaker}: {text}" if speaker else text)

		# Cap dialogue to avoid excessively long prompts
		if len(relevant_lines) > 8:
			relevant_lines = relevant_lines[:8]

		dialogue_block = (
			"Relevant dialogue:\n" + "\n".join(relevant_lines)
			if relevant_lines else ""
		)

		parts = [p for p in [char_block, original_prompt, dialogue_block] if p]
		enriched_prompt = "\n\n".join(parts)

		enriched.append(scene.model_copy(update={"final_video_prompt": enriched_prompt}))

	return enriched


async def run_stage3(
	output_dir: str,
	video_name: str,
	progress_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
	read_dir: Optional[str] = None,
	scene_callback: Optional[Callable] = None,
	speaker_viz: Optional[Dict[str, str]] = None,
	transcript_utterances: Optional[List[Dict[str, Any]]] = None,
):
	"""Run Stage 3 (Video Generation) only.

    Reads shot_script.json (preferred) or production_script.json from read_dir
    (falls back to output_dir if not given) and writes scene_*.mp4 clips to output_dir.

    Optional speaker_viz and transcript_utterances are used to enrich each
    scene's video prompt with character descriptions and relevant dialogue.
    """
	out_path = Path(output_dir)
	read_path = Path(read_dir) if read_dir else out_path
	out_path.mkdir(parents=True, exist_ok=True)

	# Prefer the shot-level script; fall back to the scene-level script
	shot_script_path   = read_path / "shot_script.json"
	scene_script_path  = read_path / "production_script.json"

	if shot_script_path.exists():
		script_path = shot_script_path
		logger.info("Stage 3: using shot_script.json")
	elif scene_script_path.exists():
		script_path = scene_script_path
		logger.info("Stage 3: shot_script.json not found, falling back to production_script.json")
	else:
		err = "Neither shot_script.json nor production_script.json found - run Stage 2 first"
		logger.error(err)
		if progress_callback:
			await progress_callback({"status": "error", "stage": "3/4: Video Generation", "detail": err})
		return

	with open(script_path) as f:
		production_script = ProductionScript.model_validate_json(f.read())

	logger.info(f"Stage 3: Video Generation with {video_name} (reading from {read_path}, writing to {out_path})")
	if progress_callback:
		await progress_callback({
			"status": "stage_started",
			"stage": "stage3",
			"timestamp": datetime.datetime.utcnow().isoformat(),
		})
		await progress_callback({"status": "processing", "stage": "3/4: Video Generation", "detail": f"Initializing {video_name} generator"})

	try:
		video_generator = _get_video_generator(video_name)
	except ValueError as e:
		logger.error(f"Failed to load video generator: {e}")
		if progress_callback:
			await progress_callback({"status": "error", "stage": "3/4: Video Generation", "detail": str(e)})
		return

	# Enrich prompts with character descriptions and relevant transcript dialogue
	scenes_to_render = production_script.scenes
	if speaker_viz or transcript_utterances:
		scenes_to_render = _enrich_scene_prompts(
			scenes_to_render,
			speaker_viz or {},
			transcript_utterances or [],
		)

	try:
		video_paths, failures = await video_generator.generate_all_scenes(scenes_to_render, str(out_path), scene_callback=scene_callback)
		logger.info(f"Generated {len(video_paths)} video clips ({len(failures)} failed).")
	except Exception as e:
		logger.error(f"Error during video generation: {e}")
		if progress_callback:
			await progress_callback({"status": "error", "stage": "3/4: Video Generation", "detail": str(e)})
		return

	_print_scene_report(production_script.scenes, failures)
	_save_run_report(out_path, production_script.scenes, failures)

	if progress_callback:
		failed_count = len(failures)
		total_count = len(scenes_to_render)
		await progress_callback({
			"status": "stage_complete",
			"stage": "stage3",
			"detail": f"{total_count - failed_count}/{total_count} scenes generated",
			"timestamp": datetime.datetime.utcnow().isoformat(),
		})
