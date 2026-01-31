"""Stage 2 Entry Point - LLM Processing
========================================
Reads ``transcript.json`` from the session directory, runs speaker mapping,
speaker visualisation, storyboard generation, relevance review, and
scene-shot generation via the configured LLM, and saves the results to
``output_dir``.

Outputs:
  storyboard.json         - All scenes with is_relevant markers (for UI display)
  speaker_visualization.json
  speaker_map.json
  production_script.json  - Relevant scenes with per-scene shot breakdowns
  shot_script.json        - Flattened shot-per-scene list ready for Stage 3
"""

import datetime
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Awaitable, Optional

from src.shared.schemas import Transcript, Storyboard, ProductionScript, ProductionScene
from src.orchestrator.providers import _get_llm_processor

logger = logging.getLogger(__name__)


async def run_stage2(
	output_dir: str,
	llm_name: str,
	speaker_mapping: dict,
	progress_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
	read_dir: Optional[str] = None,
	force_rerun: bool = False,
	ver_count: int = 0,
):
	"""Run Stage 2 (LLM Processing) only.

    Reads transcript.json from read_dir (falls back to output_dir if not given)
    and writes storyboard.json, production_script.json, shot_script.json, and
    speaker_map.json to output_dir.

    If force_rerun is True, existing outputs are backed up to .versions/{ver_count}/
    and regenerated from scratch.
    """
	out_path = Path(output_dir)
	read_path = Path(read_dir) if read_dir else out_path
	out_path.mkdir(parents=True, exist_ok=True)

	_CACHED_FILES = [
		"production_script.json",
		"storyboard.json",
		"speaker_visualization.json",
		"shot_script.json",
		"speaker_map.json",
	]

	# Check if outputs already exist (cache hit)
	cached = (
		(out_path / "production_script.json").exists()
		and (out_path / "storyboard.json").exists()
		and (out_path / "speaker_visualization.json").exists()
		and (out_path / "shot_script.json").exists()
	)

	if cached and not force_rerun:
		logger.info(f"Stage 2: outputs already exist at {out_path}, skipping LLM processing")
		if progress_callback:
			await progress_callback({"status": "stage_complete", "stage": "stage2", "detail": "Using cached storyboard and script"})
		return

	if cached and force_rerun:
		# Back up existing files to .versions/{ver_count}/ before overwriting
		ver_dir = out_path / ".versions" / str(ver_count)
		ver_dir.mkdir(parents=True, exist_ok=True)
		for fname in _CACHED_FILES:
			src = out_path / fname
			if src.exists():
				shutil.copy2(str(src), str(ver_dir / fname))
				logger.info(f"Stage 2: backed up {fname} -> .versions/{ver_count}/{fname}")

	# Load transcript written by Stage 1
	transcript_path = read_path / "transcript.json"
	if not transcript_path.exists():
		err = "transcript.json not found - run Stage 1 first"
		logger.error(err)
		if progress_callback:
			await progress_callback({"status": "error", "stage": "2/4: LLM Processing", "detail": err})
		return

	with open(transcript_path) as f:
		transcript = Transcript.model_validate_json(f.read())

	logger.info(f"Stage 2: LLM Processing with {llm_name} (reading from {read_path}, writing to {out_path})")
	if progress_callback:
		await progress_callback({
			"status": "stage_started",
			"stage": "stage2",
			"timestamp": datetime.datetime.utcnow().isoformat(),
		})
		await progress_callback({"status": "processing", "stage": "2/4: LLM Processing", "detail": "Mapping speakers..."})

	try:
		llm_processor = _get_llm_processor(llm_name)
	except ValueError as e:
		logger.error(f"Failed to load LLM processor: {e}")
		if progress_callback:
			await progress_callback({"status": "error", "stage": "2/4: LLM Processing", "detail": str(e)})
		return

	try:
		# Use the user-provided mapping if supplied, otherwise ask the LLM
		if speaker_mapping:
			resolved_map = speaker_mapping
		else:
			resolved_map = llm_processor.map_speakers(transcript)
		logger.info("Speaker mapping complete.")

		if progress_callback:
			await progress_callback({"status": "speaker_map_ready", "speaker_map": resolved_map})
			await progress_callback({"status": "processing", "stage": "2/4: LLM Processing", "detail": "Generating speaker visualizations..."})

		speaker_viz = llm_processor.generate_speaker_visualizations(resolved_map)
		with open(out_path / "speaker_visualization.json", "w") as f:
			json.dump(speaker_viz, f, indent=2)
		logger.info(f"Speaker visualizations saved to {out_path / 'speaker_visualization.json'}")
		if progress_callback:
			await progress_callback({"status": "speaker_visualization_ready", "speaker_visualization": speaker_viz})
			await progress_callback({"status": "processing", "stage": "2/4: LLM Processing", "detail": "Generating storyboard..."})

		storyboard = llm_processor.generate_storyboard(transcript, resolved_map)
		logger.info("Storyboard generated.")

		if progress_callback:
			await progress_callback({"status": "processing", "stage": "2/4: LLM Processing", "detail": "Reviewing scene relevance..."})

		# Review relevance - marks each scene is_relevant True/False
		storyboard = llm_processor.review_scene_relevance(storyboard)
		# Save with ALL scenes including irrelevant ones (for frontend display with "Not included" badges)
		with open(out_path / "storyboard.json", "w") as f:
			f.write(storyboard.model_dump_json(indent=2))
		logger.info(f"Storyboard (with relevance review) saved to {out_path / 'storyboard.json'}")

		if progress_callback:
			await progress_callback({"status": "processing", "stage": "2/4: LLM Processing", "detail": "Generating production script..."})

		# Only pass relevant scenes to shot generation (skip OOC content)
		relevant_scenes = [s for s in storyboard.scenes if s.is_relevant is not False]
		if not relevant_scenes:
			logger.warning("Stage 2: all scenes marked irrelevant; treating all as relevant for shot generation")
			relevant_scenes = storyboard.scenes
		relevant_storyboard = Storyboard(scenes=relevant_scenes)

		production_script = llm_processor.generate_scene_shots(relevant_storyboard, transcript)
		with open(out_path / "production_script.json", "w") as f:
			f.write(production_script.model_dump_json(indent=2))
		logger.info(f"Production script saved to {out_path / 'production_script.json'}")

		# Flatten shots into a linear sequence of ProductionScene objects, one per shot.
		# Stage 3 reads shot_script.json so it can render one clip per shot.
		flat_scenes: list[ProductionScene] = []
		flat_index = 1
		for scene in production_script.scenes:
			if scene.shots:
				total_hint = sum(shot.duration_hint for shot in scene.shots) or 1
				scene_audio_duration = scene.end_time - scene.start_time
				shot_audio_start = scene.start_time
				for shot in scene.shots:
					fraction = shot.duration_hint / total_hint
					shot_audio_end = shot_audio_start + fraction * scene_audio_duration
					flat_scenes.append(ProductionScene(
						scene_number=flat_index,
						start_time=shot_audio_start,
						end_time=shot_audio_end,
						location=scene.location,
						narrative_summary=f"Scene {scene.scene_number}, Shot {shot.shot_number}: {shot.description}",
						visual_prompt=shot.visual_prompt,
						stage_directions=scene.stage_directions,
						character_actions=scene.character_actions,
						final_video_prompt=shot.visual_prompt,
					))
					shot_audio_start = shot_audio_end
					flat_index += 1
			else:
				# No shots: treat entire scene as one clip
				flat_scenes.append(scene.model_copy(update={"scene_number": flat_index}))
				flat_index += 1

		shot_script = ProductionScript(scenes=flat_scenes)
		with open(out_path / "shot_script.json", "w") as f:
			f.write(shot_script.model_dump_json(indent=2))
		logger.info(f"Shot script ({len(flat_scenes)} shots) saved to {out_path / 'shot_script.json'}")

		# Persist the resolved speaker map so the UI can display it
		with open(out_path / "speaker_map.json", "w") as f:
			json.dump(resolved_map, f, indent=2)

		if progress_callback:
			await progress_callback({
				"status": "stage_complete",
				"stage": "stage2",
				"detail": "LLM processing complete",
				"timestamp": datetime.datetime.utcnow().isoformat(),
			})
	except Exception as e:
		logger.error(f"Error during LLM processing: {e}")
		if progress_callback:
			await progress_callback({"status": "error", "stage": "2/4: LLM Processing", "detail": str(e)})
