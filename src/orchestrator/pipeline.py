"""
DND Video Generation Pipeline Orchestrator
===========================================

This module is the top-level coordinator for a 4-stage pipeline that transforms
a raw audio recording of a Dungeons & Dragons session into a fully produced video:

    Stage 1 - Transcription:   Raw audio  ->  structured transcript with per-speaker utterances
    Stage 2 - LLM Processing:  Transcript ->  storyboard + production script (scene descriptions)
    Stage 3 - Video Generation: Script    ->  individual scene video clips via an AI video API
    Stage 4 - Assembly:        Clips      ->  stitched, captioned, audio-mixed final MP4

Provider selection at each stage uses a factory pattern: callers pass a short
string name (e.g. "whisperx", "anthropic", "kling") and the appropriate implementation
is imported lazily and instantiated.  This avoids loading heavyweight dependencies
(e.g. local model weights, ffmpeg bindings) unless they are actually needed.
The canonical list of provider IDs lives in config/providers_registry.json.

The main entry point for the pipeline is the async function `run_pipeline()`.
A synchronous CLI wrapper (`main()`) is provided for direct command-line use.

Sub-modules:
    providers.py      - Factory functions for each stage's provider.
    stage_runners.py  - Individual stage entry points (run_stage1 ... run_stage4).
    reporting.py      - Console and JSON run reports.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Callable, Awaitable, Optional

# -- Logging --------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -- Stage base interfaces ------------------------------------------------------
from src.stages.stage1_transcription.base import BaseTranscriber
from src.stages.stage2_llm.base import BaseLLMProcessor
from src.stages.stage3_video.base import BaseVideoGenerator

# -- Shared schemas and utilities -----------------------------------------------
from src.shared.schemas import Transcript, Storyboard, ProductionScript, ProductionScene, Utterance
from src.shared.utils.audio_preprocessor import prepare_audio
from src.shared.exceptions import ProviderError

# -- Sub-module imports ---------------------------------------------------------
from src.orchestrator.providers import (
	_compute_file_hash,
	_get_transcriber,
	_get_llm_processor,
	_get_video_generator,
)
from src.orchestrator.reporting import _print_scene_report, _save_run_report

# Re-export individual stage runners so existing callers (upload.py, tests) continue
# to import them from this module without changes.
from src.orchestrator.stage_runners import (  # noqa: F401
	run_stage1,
	run_stage2,
	run_stage3,
	run_stage4,
)


# ==============================================================================
# Directory Helpers
# ==============================================================================

def setup_directories():
	"""Ensure the base inputs/ and outputs/ directories exist at the project root.

    Resolves the project root by walking 3 levels up from this file:
        orchestrator/ -> src/ -> project_root/

    Returns:
        Tuple[Path, Path]: (inputs_dir, outputs_dir) as absolute Path objects.
    """
	# Walk up 3 levels: this file -> orchestrator/ -> src/ -> project root
	base_dir = Path(__file__).resolve().parent.parent.parent
	inputs_dir = base_dir / "inputs"
	outputs_dir = base_dir / "outputs"
	# exist_ok=True means this is safe to call even if the directories already exist
	inputs_dir.mkdir(exist_ok=True)
	outputs_dir.mkdir(exist_ok=True)
	return inputs_dir, outputs_dir


# ==============================================================================
# Main Pipeline
# ==============================================================================

async def run_pipeline(
	audio_path: str,
	transcriber_name: str,
	llm_name: str,
	video_name: str,
	output_dir: str,
	progress_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
	preprocess_audio: bool = False,
	max_chunk_minutes: float = 90.0,
):
	"""Run the full 4-stage DND Video Generation Pipeline end-to-end.

    Stages run sequentially; any unrecoverable error in one stage causes an
    early return so that subsequent stages are not attempted with incomplete data.
    All intermediate artifacts (transcript, storyboard, production script, scene
    videos) are written to `output_dir` so a failed run can be resumed manually.

    Args:
        audio_path:        Absolute path to the input audio file (MP3, WAV, etc.).
        transcriber_name:  Provider key for Stage 1 ,  one of "assembly", "deepgram",
                           "rev", "whisper".
        llm_name:          Provider key for Stage 2 ,  one of "openai", "claude", "local".
        video_name:        Provider key for Stage 3 ,  one of "luma", "replicate", "minimax".
        output_dir:        Directory where all output files will be written.  Created
                           automatically if it does not exist.
        progress_callback: Optional async callable that receives a status dict of the
                           form {"status": str, "stage": str, "detail": str}.
        preprocess_audio:  When True, the audio file is split into chunks of at most
                           `max_chunk_minutes` before transcription.
        max_chunk_minutes: Maximum chunk duration in minutes when preprocessing is
                           enabled.  Defaults to 90.0.

    Returns:
        None.  Side effects: writes output files to `output_dir` and logs progress
        through the module logger.
    """
	out_path = Path(output_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	# --------------------------------------------------------------------------
	# Stage 1: Transcription
	# --------------------------------------------------------------------------
	logger.info(f"Starting Stage 1: Transcription with {transcriber_name}")
	if progress_callback:
		await progress_callback({"status": "processing", "stage": "1/4: Transcription", "detail": f"Parsing audio via {transcriber_name}"})

	try:
		transcriber = _get_transcriber(transcriber_name)
	except ValueError as e:
		logger.error(f"Failed to load transcriber: {e}")
		return

	try:
		if preprocess_audio:
			logger.info(f"Preprocessing audio (max chunk: {max_chunk_minutes} min)...")
			chunks_dir = str(out_path / "audio_chunks")
			manifest = prepare_audio(audio_path, chunks_dir, max_duration_minutes=max_chunk_minutes)
			logger.info(f"Audio split into {len(manifest)} chunk(s).")

			chunk_transcripts = []
			for chunk in manifest:
				ct = transcriber.transcribe(chunk["filepath"])
				offset_s = chunk["global_start_ms"] / 1000.0
				adjusted_utterances = [
					Utterance(
						speaker=u.speaker,
						text=u.text,
						start=u.start + offset_s,
						end=u.end + offset_s,
					)
					for u in ct.utterances
				]
				chunk_transcripts.append((ct, adjusted_utterances, chunk["global_end_ms"] / 1000.0))

			all_utterances = [u for _, ulist, _ in chunk_transcripts for u in ulist]
			full_text = "\n".join(ct.full_text for ct, _, _ in chunk_transcripts)
			audio_duration = chunk_transcripts[-1][2] if chunk_transcripts else 0.0
			transcript = Transcript(
				audio_duration=audio_duration,
				status="completed",
				utterances=all_utterances,
				full_text=full_text,
			)
		else:
			transcript = transcriber.transcribe(audio_path)

		with open(out_path / "transcript.json", "w") as f:
			f.write(transcript.model_dump_json(indent=2))
		logger.info(f"Transcription complete. Saved to {out_path / 'transcript.json'}")
	except ProviderError as e:
		logger.error(f"Provider error during transcription: {e}")
		if progress_callback:
			await progress_callback({
				"status": "error",
				"stage": "1/4: Transcription",
				"detail": str(e),
				"error_type": e.error_type,
				"provider": e.provider_name,
				"help_url": e.help_url,
			})
		return
	except Exception as e:
		logger.error(f"Error during transcription: {e}")
		if progress_callback:
			await progress_callback({"status": "error", "stage": "1/4: Transcription", "detail": str(e), "error_type": "", "provider": transcriber_name, "help_url": ""})
		return

	# --------------------------------------------------------------------------
	# Stage 2: LLM Processing
	# --------------------------------------------------------------------------
	logger.info(f"Starting Stage 2: LLM Processing with {llm_name}")
	if progress_callback:
		await progress_callback({"status": "processing", "stage": "2/4: LLM Processing", "detail": f"Mapping speakers and generating storyboard with {llm_name}"})

	try:
		llm_processor = _get_llm_processor(llm_name)
	except ValueError as e:
		logger.error(f"Failed to load LLM processor: {e}")
		return

	try:
		speaker_map = llm_processor.map_speakers(transcript)
		logger.info("Speaker mapping complete.")

		storyboard = llm_processor.generate_storyboard(transcript, speaker_map)
		with open(out_path / "storyboard.json", "w") as f:
			f.write(storyboard.model_dump_json(indent=2))
		logger.info(f"Storyboard generated. Saved to {out_path / 'storyboard.json'}")

		storyboard = llm_processor.review_scene_relevance(storyboard)
		total_scenes = len(storyboard.scenes)
		relevant_scenes = [s for s in storyboard.scenes if s.is_relevant is not False]
		filtered_count = total_scenes - len(relevant_scenes)
		if filtered_count:
			logger.info(f"Relevance review: filtered {filtered_count} out-of-game scene(s) from {total_scenes}.")
		storyboard = Storyboard(scenes=relevant_scenes)
		with open(out_path / "storyboard.json", "w") as f:
			f.write(storyboard.model_dump_json(indent=2))
		logger.info("Relevance review complete. Storyboard updated.")

		production_script = llm_processor.generate_scene_shots(storyboard, transcript)
		with open(out_path / "production_script.json", "w") as f:
			f.write(production_script.model_dump_json(indent=2))
		logger.info(f"Production script with shot breakdown generated. Saved to {out_path / 'production_script.json'}")

		# Flatten shots into a linear sequence of ProductionScene objects -- one entry
		# per shot -- so Stage 3 can render them without any changes to its interface.
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
				flat_scenes.append(scene.model_copy(update={"scene_number": flat_index}))
				flat_index += 1

		shot_script = ProductionScript(scenes=flat_scenes)
		with open(out_path / "shot_script.json", "w") as f:
			f.write(shot_script.model_dump_json(indent=2))
		logger.info(f"Shot script ({len(flat_scenes)} shots) saved to {out_path / 'shot_script.json'}")
	except ProviderError as e:
		logger.error(f"Provider error during LLM processing: {e}")
		if progress_callback:
			await progress_callback({
				"status": "error",
				"stage": "2/4: LLM Processing",
				"detail": str(e),
				"error_type": e.error_type,
				"provider": e.provider_name,
				"help_url": e.help_url,
			})
		return
	except Exception as e:
		logger.error(f"Error during LLM processing: {e}")
		if progress_callback:
			await progress_callback({"status": "error", "stage": "2/4: LLM Processing", "detail": str(e), "error_type": "", "provider": llm_name, "help_url": ""})
		return

	# --------------------------------------------------------------------------
	# Stage 3: Video Generation
	# --------------------------------------------------------------------------
	logger.info(f"Starting Stage 3: Video Generation with {video_name}")
	if progress_callback:
		await progress_callback({"status": "processing", "stage": "3/4: Video Generation", "detail": f"Initializing {video_name} generator"})

	try:
		video_generator = _get_video_generator(video_name)
	except ValueError as e:
		logger.error(f"Failed to load video generator: {e}")
		return

	try:
		video_paths, failures = await video_generator.generate_all_scenes(flat_scenes, str(out_path))
		logger.info(f"Generated {len(video_paths)} video clips ({len(failures)} failed).")
	except ProviderError as e:
		logger.error(f"Provider error during video generation: {e}")
		if progress_callback:
			await progress_callback({
				"status": "error",
				"stage": "3/4: Video Generation",
				"detail": str(e),
				"error_type": e.error_type,
				"provider": e.provider_name,
				"help_url": e.help_url,
			})
		return
	except Exception as e:
		logger.error(f"Error during video generation: {e}")
		if progress_callback:
			await progress_callback({"status": "error", "stage": "3/4: Video Generation", "detail": str(e), "error_type": "", "provider": video_name, "help_url": ""})
		return

	failed_scene_numbers = {f["scene_number"] for f in failures}
	succeeded_scenes = [s for s in flat_scenes if s.scene_number not in failed_scene_numbers]

	_print_scene_report(flat_scenes, failures)
	_save_run_report(out_path, flat_scenes, failures)

	if not video_paths:
		logger.error("No video paths generated. Aborting assembly.")
		return

	# --------------------------------------------------------------------------
	# Stage 4: Video Assembly
	# --------------------------------------------------------------------------
	logger.info("Starting Stage 4: Video Assembly")
	if progress_callback:
		await progress_callback({"status": "processing", "stage": "4/4: Final Assembly", "detail": "Stitching videos and mixing audio..."})

	from src.stages.stage4_assembly.ffmpeg_stitcher.ffmpeg_assembler import FFmpegAssembler

	try:
		assembler = FFmpegAssembler()

		stitched_path = out_path / "final_stitched.mp4"
		assembler.stitch_videos(video_paths, str(stitched_path))
		logger.info(f"Stitched videos without audio: {stitched_path}")

		captioned_path = out_path / "final_captioned.mp4"
		assembler.add_captions(str(stitched_path), succeeded_scenes, video_paths, str(captioned_path))
		logger.info(f"Captions added: {captioned_path}")

		audio_segments = [(scene.start_time, scene.end_time) for scene in succeeded_scenes]

		final_with_audio_path = out_path / "final_with_audio.mp4"
		assembler.overlay_audio_segments(
			str(captioned_path),
			audio_path,
			str(final_with_audio_path),
			segments=audio_segments,
		)
		logger.info(f"Pipeline complete! Final video: {final_with_audio_path}")

		if progress_callback:
			await progress_callback({"status": "completed", "stage": "Complete", "detail": "Video generation successful."})

	except Exception as e:
		logger.error(f"Error during video assembly: {e}")
		return


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
	"""CLI entry point: parses arguments and runs the full pipeline for a single audio file.

    Creates a session-specific subdirectory under the configured output directory
    using a timestamp suffix (e.g. `outputs/session_2024_01_15_T143022/`) so that
    repeated runs do not overwrite each other's artifacts.
    """
	parser = argparse.ArgumentParser(description="DND-Video-Pipeline Orchestrator")
	parser.add_argument("--audio", type=str, required=True, help="Path to the input raw audio file")
	# Transcription provider IDs (see config/providers_registry.json for the full registry source)
	TRANSCRIBER_CHOICES = [
		"assemblyai", "deepgram", "revai",
		"google_cloud", "amazon_transcribe",
		"whisper", "whisperx", "nemo",
	]

	# LLM provider IDs (api-based and local via Ollama)
	LLM_CHOICES = [
		"openai", "anthropic", "gemini", "deepseek",
		"llama", "qwen", "gemma", "mistral", "dolphin",
	]

	# Video generation provider IDs
	VIDEO_CHOICES = [
		"luma", "kling", "runway", "pika",
		"minimax", "hunyuan", "ltx", "cogvideox", "mochi",
		"runware", "replicate",
	]

	parser.add_argument("--transcriber", type=str, default="assemblyai", choices=TRANSCRIBER_CHOICES, help="Model choice for transcription")
	parser.add_argument("--llm", type=str, default="openai", choices=LLM_CHOICES, help="Model choice for LLM processing")
	parser.add_argument("--video", type=str, default="luma", choices=VIDEO_CHOICES, help="Model choice for video generation")
	parser.add_argument("--output_dir", type=str, default="outputs", help="Base directory to save all artifacts and final video")
	parser.add_argument("--preprocess_audio", action="store_true", help="Chunk long audio files before transcription")
	parser.add_argument("--max_chunk_minutes", type=float, default=90.0, help="Max chunk length in minutes when --preprocess_audio is set")

	args = parser.parse_args()

	audio_path = os.path.abspath(args.audio)

	_, base_outputs_dir = setup_directories()

	session_timestamp = datetime.datetime.now().strftime("session_%Y_%m_%d_T%H%M%S")

	if os.path.isabs(args.output_dir):
		session_out_dir = Path(args.output_dir) / session_timestamp
	else:
		session_out_dir = base_outputs_dir / session_timestamp

	session_out_dir.mkdir(parents=True, exist_ok=True)
	logger.info(f"Created session output directory: {session_out_dir}")

	asyncio.run(run_pipeline(
		audio_path=audio_path,
		transcriber_name=args.transcriber,
		llm_name=args.llm,
		video_name=args.video,
		output_dir=str(session_out_dir),
		preprocess_audio=args.preprocess_audio,
		max_chunk_minutes=args.max_chunk_minutes,
	))


if __name__ == "__main__":
	main()
