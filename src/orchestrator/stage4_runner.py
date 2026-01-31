"""Stage 4 Entry Point - Final Assembly
========================================
Reads ``production_script.json`` and ``scene_*.mp4`` clips from the session
directory, stitches them together, adds captions, overlays the original audio,
and writes the final video files to ``output_dir``.
"""

import datetime
import logging
import os
from pathlib import Path
from typing import Callable, Awaitable, Optional

from src.shared.schemas import ProductionScript

logger = logging.getLogger(__name__)


async def run_stage4(
	audio_path: str,
	output_dir: str,
	progress_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
	read_dir: Optional[str] = None,
	force_rerun: bool = False,
	ver_count: int = 0,
):
	"""Run Stage 4 (Assembly) only.

    Reads production_script.json and scene_*.mp4 files from read_dir
    (falls back to output_dir if not given) and writes final video files to output_dir.

    If force_rerun is True, existing outputs are moved to .s4_versions/{ver_count}/ and
    assembly runs again.
    """
	import glob as _glob

	out_path = Path(output_dir)
	read_path = Path(read_dir) if read_dir else out_path
	out_path.mkdir(parents=True, exist_ok=True)

	_FINAL_FILES = ["final_with_audio.mp4", "final_captioned.mp4"]
	cached = (out_path / "final_with_audio.mp4").exists()

	# Skip if final video already exists
	if cached and not force_rerun:
		logger.info(f"Stage 4: final_with_audio.mp4 already exists at {out_path}, skipping assembly")
		if progress_callback:
			await progress_callback({"status": "completed", "stage": "Complete", "detail": "Video generation successful."})
		return

	if cached and force_rerun:
		# Move existing final video files to .s4_versions/{ver_count}/ (O(1) rename)
		ver_dir = out_path / ".s4_versions" / str(ver_count)
		ver_dir.mkdir(parents=True, exist_ok=True)
		for fname in _FINAL_FILES:
			src = out_path / fname
			if src.exists():
				os.rename(str(src), str(ver_dir / fname))
				logger.info(f"Stage 4: moved {fname} -> .s4_versions/{ver_count}/{fname}")

	script_path = read_path / "production_script.json"
	if not script_path.exists():
		err = "production_script.json not found - run Stage 2 first"
		logger.error(err)
		if progress_callback:
			await progress_callback({"status": "error", "stage": "4/4: Final Assembly", "detail": err})
		return

	with open(script_path) as f:
		production_script = ProductionScript.model_validate_json(f.read())

	# Reconstruct ordered video_paths from scene clip files on disk
	scene_files = sorted(_glob.glob(str(read_path / "scene_*.mp4")))
	if not scene_files:
		err = "No scene_*.mp4 files found - run Stage 3 first"
		logger.error(err)
		if progress_callback:
			await progress_callback({"status": "error", "stage": "4/4: Final Assembly", "detail": err})
		return

	import json as _json
	run_report_path = read_path / "run_report.json"
	failed_scene_numbers = set()
	if run_report_path.exists():
		with open(run_report_path) as _f:
			_report = _json.load(_f)
		failed_scene_numbers = {
			s["scene_number"] for s in _report.get("scenes", [])
			if s.get("status") == "failed"
		}
		if failed_scene_numbers:
			logger.warning(f"Stage 4: skipping {len(failed_scene_numbers)} scenes that failed in Stage 3: {sorted(failed_scene_numbers)}")
	succeeded_scenes = [s for s in production_script.scenes if s.scene_number not in failed_scene_numbers]

	logger.info("Stage 4: Video Assembly")
	if progress_callback:
		await progress_callback({
			"status": "stage_started",
			"stage": "stage4",
			"timestamp": datetime.datetime.utcnow().isoformat(),
		})
		await progress_callback({"status": "processing", "stage": "4/4: Final Assembly", "detail": "Stitching video clips..."})

	from src.stages.stage4_assembly.ffmpeg_stitcher.ffmpeg_assembler import FFmpegAssembler

	try:
		assembler = FFmpegAssembler()

		stitched_path = out_path / "final_stitched.mp4"
		assembler.stitch_videos(scene_files, str(stitched_path))
		logger.info(f"Stitched: {stitched_path}")

		if progress_callback:
			await progress_callback({"status": "processing", "stage": "4/4: Final Assembly", "detail": "Adding captions..."})

		captioned_path = out_path / "final_captioned.mp4"
		assembler.add_captions(str(stitched_path), succeeded_scenes, scene_files, str(captioned_path))
		logger.info(f"Captions added: {captioned_path}")

		if progress_callback:
			await progress_callback({"status": "processing", "stage": "4/4: Final Assembly", "detail": "Overlaying audio..."})

		audio_start = production_script.scenes[0].start_time
		audio_end = production_script.scenes[-1].end_time

		final_with_audio_path = out_path / "final_with_audio.mp4"
		assembler.overlay_audio(
			str(captioned_path), audio_path, str(final_with_audio_path),
			audio_start=audio_start, audio_end=audio_end
		)
		logger.info(f"Pipeline complete! Final video: {final_with_audio_path}")

		if progress_callback:
			await progress_callback({
				"status": "stage_complete",
				"stage": "stage4",
				"detail": "Assembly complete",
				"timestamp": datetime.datetime.utcnow().isoformat(),
			})
			await progress_callback({"status": "completed", "stage": "Complete", "detail": "Video generation successful."})
	except Exception as e:
		logger.error(f"Error during video assembly: {e}")
		if progress_callback:
			await progress_callback({"status": "error", "stage": "4/4: Final Assembly", "detail": str(e)})
