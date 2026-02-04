"""
run_comparisons.py
==================
Step 1 of the two-step evaluation workflow:
  1. run_comparisons.py  -> executes pipeline permutations, writes results  <- (this file)
  2. generate_report.py  -> reads results, writes FINAL_REPORT.md

Responsibility
--------------
Orchestrate the execution of multiple end-to-end pipeline "permutations".
A permutation is one specific combination of four model choices:
  - Stage 1: Transcriber      (audio -> transcript)
  - Stage 2: LLM              (transcript -> storyboard + production script)
  - Stage 3: Video Generator  (production script scenes -> video clips)
  - Stage 4: Assembler        (video clips + original audio -> final video)

Each permutation is run sequentially (not in parallel) to avoid simultaneous
API rate-limit exhaustion and to keep timing measurements independent.

Outputs (written to the `results/` directory at the project root):
  - evaluation_metrics.json    machine-readable; consumed by generate_report.py
  - evaluation_summary.csv     human-readable; useful for quick review in spreadsheets

Usage
-----
    python -m src.evaluation.run_comparisons --audio path/to/audio.mp3
"""

import asyncio
import time
import json
import csv
import traceback
import os
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Imports grouped by pipeline stage for readability.
# Adding a new provider means: (1) add one import here, (2) add one entry to
# MODEL_REGISTRY below.  No other file needs to change.
# Note: these are real production classes.  Unit tests replace them with mocks
# rather than importing them directly, so tests don't require live API keys.
# ---------------------------------------------------------------------------

# Transcribers (Stage 1)
from src.stages.stage1_transcription.assembly_ai.assembly_ai_transcriber import AssemblyAITranscriber
# LLMs (Stage 2)
from src.stages.stage2_llm.openai_gpt.openai_processor import OpenAIGPTProcessor
# Video generators (Stage 3)
from src.stages.stage3_video.luma_dream_machine.luma_video_generator import LumaVideoGenerator
# Assemblers (Stage 4)
from src.stages.stage4_assembly.ffmpeg_stitcher.ffmpeg_assembler import FFmpegAssembler

# ---------------------------------------------------------------------------
# MODEL_REGISTRY
# ---------------------------------------------------------------------------
# The registry pattern decouples human-readable string keys (used in permutation
# config dicts) from actual Python class objects.  This lets permutations be
# expressed as plain data (lists of dicts) rather than imperative code.
#
# Adding a new provider:
#   1. Add an import above (e.g. `from src.stages... import WhisperTranscriber`)
#   2. Add an entry here:   "whisper": WhisperTranscriber
#   3. Reference the key in PERMUTATIONS inside main()
#
# In tests, the registry values are replaced with mock classes so no real API
# calls are made during unit testing.
MODEL_REGISTRY = {
	"assembly": AssemblyAITranscriber,
	"openai": OpenAIGPTProcessor,
	"luma": LumaVideoGenerator,
	"ffmpeg": FFmpegAssembler
}

@dataclass
class RunMetrics:
	"""Timing and status metrics collected for a single pipeline permutation run.

    Each field maps directly to one pipeline stage or overall run metadata.
    The dataclass is used with `asdict()` for JSON/CSV serialization, so field
    names must match the keys that generate_report.py expects in the JSON file.
    """
	permutation_name: str

	# Individual stage wall-clock durations in seconds.
	# Stored separately (not just total_time) so generate_report.py can identify
	# which stage is the bottleneck across different model combinations.
	stage1_time: float = 0.0  # Transcription
	stage2_time: float = 0.0  # LLM processing (speaker mapping + storyboard + script)
	stage3_time: float = 0.0  # Video generation (all scenes, async)
	stage4_time: float = 0.0  # Video assembly (stitch + audio overlay)
	total_time: float = 0.0   # Wall time from first line of run_permutation to last

	# "Pending" is the default ,  if a run is interrupted before completion,
	# any unstarted entries in the output JSON will show "Pending" rather than
	# misleadingly showing "Success" or "Failed".
	status: str = "Pending"

	# Full Python traceback string for debugging failed runs.
	# Note: generate_report.py reads the key "error_message" from the JSON output,
	# but this field is named "error_traceback".  When serialized via asdict(),
	# the key in JSON will be "error_traceback".  Ensure generate_report.py is
	# updated if the key name changes, or add a separate error_message field.
	error_traceback: str = ""

	# Reserved for future use: cost per run could be computed from API usage
	# metadata (tokens processed, video seconds generated, etc.).
	estimated_cost: float = 0.0


async def run_permutation(config: dict, audio_path: str, base_output_dir: str) -> RunMetrics:
	"""Execute a single pipeline permutation end-to-end and return its timing metrics.

    Each stage's duration is timed independently so that bottlenecks can be
    identified across permutations (e.g. "Stage 3 with Luma took 3x longer than
    with Replicate").

    Args:
        config:          A dict with keys: "name", "transcriber", "llm", "video",
                         "assembler".  Values for model keys must exist in MODEL_REGISTRY.
        audio_path:      Absolute or relative path to the source audio file (.mp3/.wav).
        base_output_dir: Parent directory under which a per-run subdirectory is created.

    Returns:
        RunMetrics instance with all timing fields populated and status set to
        "Success" or "Failed".  total_time is always set, even for failed runs.
    """
	name = config["name"]
	metrics = RunMetrics(permutation_name=name)
	# Record total start time before any work begins so total_time captures
	# directory creation and model instantiation overhead as well.
	start_time_total = time.time()

	# Create a unique output directory for this run's artifacts (video clips,
	# stitched video, final video).  Spaces are replaced with underscores to keep
	# filesystem paths shell-friendly and avoid quoting issues in downstream tools.
	run_output_dir = os.path.join(base_output_dir, f"run_{name.replace(' ', '_').lower()}")
	os.makedirs(run_output_dir, exist_ok=True)

	try:
		# ------------------------------------------------------------------
		# Model instantiation ,  wrapped in its own inner try/except so that a
		# misconfigured registry key (KeyError) produces a different, more
		# informative error message than a model crashing during actual inference.
		# ------------------------------------------------------------------
		try:
			transcriber_class = MODEL_REGISTRY[config["transcriber"]]
			llm_class = MODEL_REGISTRY[config["llm"]]
			video_class = MODEL_REGISTRY[config["video"]]
			assembler_class = MODEL_REGISTRY[config["assembler"]]

			transcriber = transcriber_class()
			llm = llm_class()
			video_generator = video_class()
			assembler = assembler_class()
		except KeyError as e:
			raise KeyError(f"Configuration specifies a model not found in registry: {e}")

		# ------------------------------------------------------------------
		# Stage 1: Transcription
		# Input:  audio_path (str) ,  path to source audio file
		# Output: transcript (object) ,  provider-specific transcript object
		# ------------------------------------------------------------------
		stage_start = time.time()
		transcript = transcriber.transcribe(audio_path)
		metrics.stage1_time = time.time() - stage_start

		# ------------------------------------------------------------------
		# Stage 2: LLM Processing
		# Input:  transcript from Stage 1
		# Output: prod_script ,  production script with scene-level detail
		#
		# Three LLM calls in sequence:
		#   1. map_speakers	,  identify and label each speaker in the transcript
		#   2. generate_storyboard ,  turn the labelled transcript into a visual story
		#   3. generate_production_script ,  add camera directions, timing, etc.
		# All three calls are timed together as a single stage because they form a
		# logical unit; breaking them apart would make the metrics harder to interpret.
		# ------------------------------------------------------------------
		stage_start = time.time()
		speaker_map = llm.map_speakers(transcript)
		storyboard = llm.generate_storyboard(transcript, speaker_map)
		prod_script = llm.generate_production_script(storyboard, transcript)
		metrics.stage2_time = time.time() - stage_start

		# ------------------------------------------------------------------
		# Stage 3: Video Generation (async)
		# Input:  prod_script.scenes ,  list of scene descriptors
		# Output: video_paths ,  list of local file paths to generated video clips
		#
		# This stage uses `await` because video generation APIs work asynchronously:
		# the client submits a job and polls for completion.  Making this stage truly
		# async allows future refactoring to run multiple permutations concurrently
		# (e.g. using asyncio.gather) without blocking on each video generation job.
		# This is typically the longest-running stage (often several minutes).
		# ------------------------------------------------------------------
		stage_start = time.time()
		video_paths = await video_generator.generate_all_scenes(prod_script.scenes, run_output_dir)
		metrics.stage3_time = time.time() - stage_start

		# ------------------------------------------------------------------
		# Stage 4: Video Assembly
		# Input:  video_paths (list[str]) ,  individual scene clips from Stage 3
		# Output: final_path ,  single MP4 with all scenes concatenated and
		#                       original audio overlaid
		#
		# Assembly is intentionally two steps:
		#   a. stitch_videos  ,  concatenates clips into one silent video
		#   b. overlay_audio  ,  mixes the original audio track onto the stitched video
		# Keeping them separate means the stitched video can be reused if only the
		# audio source changes (e.g. re-running with a noise-cancelled audio file).
		# ------------------------------------------------------------------
		stage_start = time.time()
		stitched_path = os.path.join(run_output_dir, "stitched_no_audio.mp4")
		final_path = os.path.join(run_output_dir, "final_video.mp4")

		assembler.stitch_videos(video_paths, stitched_path)
		assembler.overlay_audio(stitched_path, audio_path, final_path)
		metrics.stage4_time = time.time() - stage_start

		metrics.status = "Success"

	except Exception as e:
		# Catch all exceptions so a single failing permutation does not abort
		# the entire evaluation run.  The full traceback is stored for post-hoc
		# debugging; the status is set to "Failed" so generate_report.py can
		# include this run in the Error Log section.
		metrics.status = "Failed"
		metrics.error_traceback = traceback.format_exc()
		# Print to console immediately so the operator sees failures in real time
		# rather than only discovering them when reviewing the final report.
		print(f"Run {name} failed: {e}")

	# total_time is set OUTSIDE the try/except block so it is always populated,
	# even for failed runs.  This lets the report show how long was spent before
	# the failure occurred ,  useful for cost estimation and timeout planning.
	metrics.total_time = time.time() - start_time_total
	return metrics


async def process_permutations(permutations: List[dict], audio_path: str, results_dir: str):
	"""Run all permutations sequentially, collect metrics, and save results as JSON and CSV.

    Runs are executed one at a time (sequential, not concurrent) for two reasons:
      1. API rate limits: firing all permutations simultaneously would likely trigger
         rate limiting on third-party services (LLMs, video generators).
      2. Measurement independence: concurrent runs would share CPU/network resources,
         making per-run timing less meaningful for head-to-head comparisons.

    Args:
        permutations: List of config dicts (see run_permutation for the dict schema).
        audio_path:   Path to the source audio file shared across all permutations.
        results_dir:  Directory where output artifacts and metrics files are written.
    """
	os.makedirs(results_dir, exist_ok=True)

	all_metrics = []

	for config in permutations:
		print(f"Starting permutation: {config['name']}")
		metrics = await run_permutation(config, audio_path, results_dir)
		all_metrics.append(metrics)
		print(f"Finished {config['name']} with status: {metrics.status}")

	# ------------------------------------------------------------------
	# Persist results in two formats.
	# `asdict()` converts a RunMetrics dataclass instance to a plain dict,
	# which is required for both json.dump and csv.DictWriter.
	# ------------------------------------------------------------------

	# JSON ,  structured format consumed programmatically by generate_report.py.
	# indent=4 makes the file human-readable when inspected directly.
	json_path = os.path.join(results_dir, "evaluation_metrics.json")
	with open(json_path, "w") as f:
		json.dump([asdict(m) for m in all_metrics], f, indent=4)

	# CSV ,  flat tabular format; easy to open in Excel/Google Sheets for a quick
	# overview or to feed into a data analysis notebook.
	csv_path = os.path.join(results_dir, "evaluation_summary.csv")
	with open(csv_path, "w", newline='') as f:
		if all_metrics:
			# Derive column names from the first RunMetrics instance so the CSV
			# header always stays in sync with the dataclass definition.
			writer = csv.DictWriter(f, fieldnames=asdict(all_metrics[0]).keys())
			writer.writeheader()
			for m in all_metrics:
				writer.writerow(asdict(m))

	print(f"Saved evaluation metrics to {results_dir}")


def main():
	"""CLI entry point: parses arguments and runs the comparison evaluation harness."""
	parser = argparse.ArgumentParser(description="Run comparisons evaluation harness")
	# --audio is required: the harness must have a real audio file to feed into the
	# transcription stage.  There is no sensible default ,  a missing audio file
	# would cause every permutation to fail with an unhelpful FileNotFoundError.
	parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
	args = parser.parse_args()

	# PERMUTATIONS is a declarative list of configuration dicts.
	# Each dict is purely data: the harness (process_permutations -> run_permutation)
	# handles all the imperative logic.  To add a new permutation, append a dict here
	# with valid MODEL_REGISTRY keys for "transcriber", "llm", "video", "assembler".
	PERMUTATIONS = [
		{
			"name": "Run 1 Deep Claude Luma",
			"transcriber": "assembly",  # Maps to AssemblyAITranscriber via MODEL_REGISTRY
			"llm": "openai",			# Maps to OpenAIGPTProcessor via MODEL_REGISTRY
			"video": "luma",			# Maps to LumaVideoGenerator via MODEL_REGISTRY
			"assembler": "ffmpeg"       # Maps to FFmpegAssembler via MODEL_REGISTRY
		}
		# Example of how to add a second permutation when more providers are available:
		# {
		#     "name": "Run 2 AssemblyAI OpenAI Replicate",
		#     "transcriber": "assemblyai",
		#     "llm": "openai",
		#     "video": "replicate",
		#     "assembler": "ffmpeg"
		# }
	]

	# asyncio.run() is the standard top-level entry point for an async coroutine.
	# It creates a new event loop, runs process_permutations to completion, and
	# then closes the loop.  This is necessary because run_permutation uses `await`
	# for Stage 3 video generation.
	asyncio.run(process_permutations(PERMUTATIONS, args.audio, "results"))

if __name__ == "__main__":
	main()
