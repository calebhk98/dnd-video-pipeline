"""
Upload Background Task Helpers
================================
Async background tasks that execute each pipeline stage for a job.
These are scheduled via FastAPI's BackgroundTasks and run independently
of the HTTP response lifecycle.
"""

import os
import json
import logging
import datetime

import dotenv

from core import (
	jobs_db, manager, ENV_PATH, OUTPUTS_DIR, save_job_meta,
	DEFAULT_TRANSCRIBER, DEFAULT_LLM, DEFAULT_VIDEO_GEN
)
from provider_dirs import (
	get_stage1_dir, get_stage2_dir, get_stage34_dir, is_legacy_session,
	resolve_stage1_dir, resolve_stage2_dir, resolve_stage34_dir
)

logger = logging.getLogger(__name__)

_STAGE_NUM = {"stage1": 1, "stage2": 2, "stage3": 3, "stage4": 4}

# Maps detail-text substrings to (sub_phase_key, key_to_stop_first)
# The order matters: first match wins.
_SUB_PHASE_DETAIL_TRANSITIONS = [
	("Converting",             "1-convert",    None),
	("Uploading",              "1-upload",     "1-convert"),
	("Transcribing",           "1-transcribe", "1-upload"),
	("speaker visualizations", "2-viz",        "2-speakers"),
	("storyboard",             "2-storyboard", "2-viz"),
	("Reviewing scene",        "2-relevance",  "2-storyboard"),
	("production script",      "2-production", "2-relevance"),
	("Stitching",              "4-stitch",     None),
	("Adding captions",        "4-captions",   "4-stitch"),
	("Overlaying audio",       "4-audio",      "4-captions"),
]

# Sub-phases that start when their parent stage starts (before any detail text)
_STAGE_START_SUB_PHASES = {
	"stage2": "2-speakers",
	"stage3": "3-scenes",
}

# Sub-phases that end when their parent stage completes
_STAGE_COMPLETE_SUB_PHASES = {
	"stage1": "1-transcribe",
	"stage2": "2-production",
	"stage3": "3-scenes",
	"stage4": "4-audio",
}


async def _progress_callback_factory(job_id: str):
	"""Return an async progress callback that broadcasts to WebSocket subscribers."""
	async def progress_callback(update: dict):
		"""Internal callback that updates the job DB and broadcasts via WebSocket."""
		now_iso = datetime.datetime.utcnow().isoformat()

		await manager.broadcast_to_job(job_id, update)
		jobs_db[job_id]["last_update"] = update

		status = update.get("status")
		stage  = update.get("stage", "")
		detail = update.get("detail", "")

		# ---- Phase-level timer recording --------------------------------
		stage_num = _STAGE_NUM.get(stage)
		if stage_num and status in ("stage_started", "stage_complete"):
			timings = jobs_db[job_id].setdefault("stage_timings", {})
			entry = timings.setdefault(stage_num, {})
			ts = update.get("timestamp", now_iso)
			if status == "stage_started":
				entry["started_at"] = ts
			else:
				entry["ended_at"] = ts

		# ---- Sub-phase timer recording ----------------------------------
		sub_timings = jobs_db[job_id].setdefault("sub_stage_timings", {})

		def _start_sub(key: str):
			sub_timings.setdefault(key, {})["started_at"] = now_iso

		def _stop_sub(key: str):
			if key and key in sub_timings:
				sub_timings[key]["ended_at"] = now_iso

		if status == "stage_started" and stage in _STAGE_START_SUB_PHASES:
			_start_sub(_STAGE_START_SUB_PHASES[stage])

		elif status == "stage_complete" and stage in _STAGE_COMPLETE_SUB_PHASES:
			_stop_sub(_STAGE_COMPLETE_SUB_PHASES[stage])

		elif status == "processing" and detail:
			for substr, start_key, stop_key in _SUB_PHASE_DETAIL_TRANSITIONS:
				if substr in detail:
					if stop_key:
						_stop_sub(stop_key)
					# Only start if not already started (avoid duplicate entries)
					if start_key not in sub_timings or "started_at" not in sub_timings[start_key]:
						_start_sub(start_key)
					break

		# ---- Job status --------------------------------------------------
		if status == "completed":
			jobs_db[job_id]["status"] = "completed"
		elif status == "error":
			jobs_db[job_id]["status"] = "error"
		save_job_meta(job_id)
	return progress_callback


async def _run_stage(job_id: str, stage_label: str, run_fn, *args, **kwargs):
	"""Generic wrapper to fetch job, handle errors, and attach progress callbacks."""
	job = jobs_db.get(job_id)
	if not job:
		return
	dotenv.load_dotenv(ENV_PATH)
	progress_callback = await _progress_callback_factory(job_id)
	try:
		await run_fn(progress_callback=progress_callback, *args, **kwargs)
	except Exception as exc:
		logger.error("%s error for job %s: %s", stage_label, job_id, exc)
		jobs_db[job_id]["status"] = "error"
		await manager.broadcast_to_job(job_id, {"status": "error", "stage": stage_label, "detail": str(exc)})


async def _run_stage1(job_id: str, force_rerun: bool = False):
	"""Execute Stage 1 (transcription + optional speaker suggestions) for a job."""
	from src.orchestrator.pipeline import run_stage1
	job = jobs_db.get(job_id)
	if not job:
		return

	out_dir = resolve_stage1_dir(job)
	ver_count = job.get("stage1_ver_count", 0)

	await _run_stage(
		job_id, "1/4: Transcription", run_stage1,
		audio_path=job["audio_path"],
		transcriber_name=job.get("transcriber", DEFAULT_TRANSCRIBER),
		llm_name=job.get("llm", DEFAULT_LLM),
		output_dir=out_dir,
		force_rerun=force_rerun,
		ver_count=ver_count,
	)

	if force_rerun and job_id in jobs_db:
		jobs_db[job_id]["stage1_ver_count"] = ver_count + 1
		jobs_db[job_id]["stage1_cur_ver"] = -1
		save_job_meta(job_id)


async def _run_stage2(job_id: str, force_rerun: bool = False):
	"""Execute Stage 2 (LLM processing) for a job."""
	from src.orchestrator.pipeline import run_stage2
	job = jobs_db.get(job_id)
	if not job:
		return

	out_dir = resolve_stage2_dir(job)
	r_dir = job["session_dir"] if is_legacy_session(job.get("session_dir", "")) else resolve_stage1_dir(job)
	ver_count = job.get("stage2_ver_count", 0)

	await _run_stage(
		job_id, "2/4: LLM Processing", run_stage2,
		output_dir=out_dir,
		llm_name=job.get("llm", DEFAULT_LLM),
		speaker_mapping=job.get("speaker_mapping", {}),
		read_dir=r_dir,
		force_rerun=force_rerun,
		ver_count=ver_count,
	)

	if force_rerun and job_id in jobs_db:
		jobs_db[job_id]["stage2_ver_count"] = ver_count + 1
		jobs_db[job_id]["stage2_cur_ver"] = -1
		save_job_meta(job_id)


async def _run_stage3(job_id: str, backup_first: bool = False):
	"""Execute Stage 3 (video generation) for a job."""
	import glob as _glob
	from src.orchestrator.pipeline import run_stage3
	job = jobs_db.get(job_id)
	if not job:
		return

	if backup_first:
		s34_dir = resolve_stage34_dir(job)
		ver_count = job.get("stage3_ver_count", 0)
		ver_dir = os.path.join(s34_dir, ".s3_versions", str(ver_count))
		os.makedirs(ver_dir, exist_ok=True)
		for f in _glob.glob(os.path.join(s34_dir, "scene_*.mp4")):
			os.rename(f, os.path.join(ver_dir, os.path.basename(f)))
		logger.info("Stage 3: backed up scene videos to .s3_versions/%d", ver_count)

	async def _scene_ready_callback(scene, video_path):
		"""Broadcast a scene_ready message as each individual clip finishes."""
		rel = os.path.relpath(video_path, OUTPUTS_DIR).replace(os.sep, '/')
		await manager.broadcast_to_job(job_id, {
			"status": "scene_ready",
			"scene_number": scene.scene_number,
			"video_url": f"/outputs/{rel}",
		})

	out_dir = resolve_stage34_dir(job)
	r_dir = job["session_dir"] if is_legacy_session(job.get("session_dir", "")) else resolve_stage2_dir(job)
	video_gen = job.get("video_gen", DEFAULT_VIDEO_GEN)

	# Load character descriptions from Stage 2 output
	speaker_viz: dict = {}
	s2_dir = resolve_stage2_dir(job)
	viz_path = os.path.join(s2_dir, "speaker_visualization.json")
	if os.path.exists(viz_path):
		try:
			with open(viz_path) as fh:
				speaker_viz = json.load(fh)
		except Exception as exc:
			logger.warning("Could not load speaker_visualization.json: %s", exc)

	# Load transcript from Stage 1 output
	transcript_utterances: list = []
	s1_dir = resolve_stage1_dir(job)
	transcript_path = os.path.join(s1_dir, "transcript.json")
	if os.path.exists(transcript_path):
		try:
			with open(transcript_path) as fh:
				t_data = json.load(fh)
				transcript_utterances = t_data.get("utterances", [])
		except Exception as exc:
			logger.warning("Could not load transcript.json for stage 3 enrichment: %s", exc)

	await _run_stage(
		job_id, "3/4: Video Generation", run_stage3,
		output_dir=out_dir,
		video_name=video_gen,
		read_dir=r_dir,
		scene_callback=_scene_ready_callback,
		speaker_viz=speaker_viz,
		transcript_utterances=transcript_utterances,
	)

	if backup_first and job_id in jobs_db:
		ver_count = jobs_db[job_id].get("stage3_ver_count", 0)
		jobs_db[job_id]["stage3_ver_count"] = ver_count + 1
		jobs_db[job_id]["stage3_cur_ver"] = -1
		save_job_meta(job_id)


async def _rerun_scene(job_id: str, scene_number: int):
	"""Re-generate a single failed scene clip for a job."""
	from src.orchestrator.stage3_runner import _enrich_scene_prompts
	from src.orchestrator.providers import _get_video_generator
	from src.shared.schemas import ProductionScript

	job = jobs_db.get(job_id)
	if not job:
		return

	out_dir = resolve_stage34_dir(job)
	r_dir = job["session_dir"] if is_legacy_session(job.get("session_dir", "")) else resolve_stage2_dir(job)
	video_gen = job.get("video_gen", DEFAULT_VIDEO_GEN)

	# Load the script (prefer shot_script.json)
	shot_script_path = os.path.join(r_dir, "shot_script.json")
	scene_script_path = os.path.join(r_dir, "production_script.json")
	script_path = shot_script_path if os.path.exists(shot_script_path) else scene_script_path
	if not os.path.exists(script_path):
		logger.error("rerun_scene: no script found for job %s", job_id)
		return

	with open(script_path) as fh:
		production_script = ProductionScript.model_validate_json(fh.read())

	scene = next((s for s in production_script.scenes if s.scene_number == scene_number), None)
	if scene is None:
		logger.error("rerun_scene: scene %d not found in script for job %s", scene_number, job_id)
		return

	# Load character descriptions and transcript for prompt enrichment
	speaker_viz: dict = {}
	s2_dir = resolve_stage2_dir(job)
	viz_path = os.path.join(s2_dir, "speaker_visualization.json")
	if os.path.exists(viz_path):
		try:
			with open(viz_path) as fh:
				speaker_viz = json.load(fh)
		except Exception as exc:
			logger.warning("rerun_scene: could not load speaker_visualization.json: %s", exc)

	transcript_utterances: list = []
	s1_dir = resolve_stage1_dir(job)
	transcript_path = os.path.join(s1_dir, "transcript.json")
	if os.path.exists(transcript_path):
		try:
			with open(transcript_path) as fh:
				t_data = json.load(fh)
				transcript_utterances = t_data.get("utterances", [])
		except Exception as exc:
			logger.warning("rerun_scene: could not load transcript.json: %s", exc)

	# Enrich the single scene's prompt
	if speaker_viz or transcript_utterances:
		enriched = _enrich_scene_prompts([scene], speaker_viz, transcript_utterances)
		scene = enriched[0]

	try:
		video_generator = _get_video_generator(video_gen)
	except ValueError as exc:
		logger.error("rerun_scene: could not load video generator: %s", exc)
		return

	try:
		video_path = await video_generator.generate_scene(scene, out_dir)
	except Exception as exc:
		logger.error("rerun_scene: generation failed for scene %d: %s", scene_number, exc)
		await manager.broadcast_to_job(job_id, {
			"status": "scene_failed",
			"scene_number": scene_number,
			"error": str(exc),
		})
		return

	# Broadcast scene_ready so the frontend injects the video
	rel = os.path.relpath(video_path, OUTPUTS_DIR).replace(os.sep, '/')
	await manager.broadcast_to_job(job_id, {
		"status": "scene_ready",
		"scene_number": scene_number,
		"video_url": f"/outputs/{rel}",
	})

	# Update run_report.json: mark this scene as succeeded
	report_path = os.path.join(out_dir, "run_report.json")
	if os.path.exists(report_path):
		try:
			with open(report_path) as fh:
				report = json.load(fh)
			for rec in report.get("scenes", []):
				if rec.get("scene_number") == scene_number:
					rec["status"] = "success"
					rec.pop("error", None)
					break
			# Recount summary
			total = len(report["scenes"])
			succeeded = sum(1 for r in report["scenes"] if r.get("status") == "success")
			report["summary"]["succeeded"] = succeeded
			report["summary"]["failed"] = total - succeeded
			with open(report_path, "w") as fh:
				json.dump(report, fh, indent=2)
		except Exception as exc:
			logger.warning("rerun_scene: could not update run_report.json: %s", exc)


async def _run_stage4(job_id: str, force_rerun: bool = False):
	"""Execute Stage 4 (assembly) for a job."""
	from src.orchestrator.pipeline import run_stage4
	job = jobs_db.get(job_id)
	if not job:
		return

	out_dir = resolve_stage34_dir(job)
	r_dir = out_dir  # Stage 4 reads and writes in the same dir
	ver_count = job.get("stage4_ver_count", 0)

	await _run_stage(
		job_id, "4/4: Final Assembly", run_stage4,
		audio_path=job["audio_path"],
		output_dir=out_dir,
		read_dir=r_dir,
		force_rerun=force_rerun,
		ver_count=ver_count,
	)

	if force_rerun and job_id in jobs_db:
		jobs_db[job_id]["stage4_ver_count"] = ver_count + 1
		jobs_db[job_id]["stage4_cur_ver"] = -1
		save_job_meta(job_id)
