"""
Upload Route Handlers
======================
FastAPI route definitions for the audio upload and pipeline workflow.

All routes here form the core pipeline workflow:
  POST   /api/upload                  -> accept audio file, create a job (Stage 1 NOT auto-started)
  PATCH  /api/job/{job_id}            -> update AI provider selections or session name
  POST   /api/start_stage1/{job_id}   -> explicitly start Stage 1 (transcription)
  GET    /api/transcript/{job_id}     -> return transcript once stage 1 is done
  POST   /api/map_speakers/{job_id}   -> save speaker names, start Stage 2
  POST   /api/speaker_visualization/{job_id} -> save user-edited character descriptions
  GET    /api/stage2_results/{job_id} -> return speaker map + storyboard after Stage 2
  GET    /api/job_status/{job_id}     -> return job status and stage timings
  POST   /api/run_stage3/{job_id}     -> start Stage 3 (video generation)
  GET    /api/stage3_results/{job_id} -> return per-scene video paths after Stage 3
  POST   /api/run_stage4/{job_id}     -> start Stage 4 (assembly)
  WS     /api/ws/progress/{job_id}    -> stream real-time progress to the browser
  GET    /api/videos/{job_id}         -> return finished video URL(s)
"""

import re
import os
import json
import uuid
import glob
import datetime
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks

from core import (
	jobs_db, manager,
	SUPPORTED_EXTENSIONS, MAX_UPLOAD_BYTES,
	INPUTS_DIR, OUTPUTS_DIR,
	save_job_meta,
	DEFAULT_TRANSCRIBER, DEFAULT_LLM, DEFAULT_VIDEO_GEN
)
from provider_dirs import (
	resolve_stage1_dir, resolve_stage2_dir, resolve_stage34_dir
)
from routers.upload_tasks import _run_stage1, _run_stage2, _run_stage3, _run_stage4, _rerun_scene
import shutil

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["pipeline"])


def require_job(job_id: str) -> dict:
	"""Fetch a job from the DB or raise a 404."""
	if job_id not in jobs_db:
		raise HTTPException(status_code=404, detail="Job not found")
	return jobs_db[job_id]


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------
@router.post("/upload")
async def upload_audio(
	file: UploadFile = File(...),
	num_speakers: int = Form(0),
	transcriber: str = Form(DEFAULT_TRANSCRIBER),
	llm: str = Form(DEFAULT_LLM),
	video_gen: str = Form(DEFAULT_VIDEO_GEN),
	auto_run: bool = Form(False),
):
	"""Accept an audio file upload and persist it to disk.

    Stage 1 is NOT started automatically ,  the client must call
    POST /api/start_stage1/{job_id} when ready.  This gives the user a
    chance to review the uploaded file or change AI model selections before
    committing to processing.

    Returns a ``job_id`` that the client uses for all subsequent requests.
    """
	try:
		if not file.filename.endswith(SUPPORTED_EXTENSIONS):
			raise HTTPException(status_code=400, detail="Unsupported file format.")

		if not (0 <= num_speakers <= 50):
			raise HTTPException(status_code=400, detail="num_speakers must be between 0 and 50.")

		job_id = str(uuid.uuid4())

		contents = await file.read()
		if len(contents) > MAX_UPLOAD_BYTES:
			raise HTTPException(
				status_code=413,
				detail=f"File too large. Maximum is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
			)

		os.makedirs(INPUTS_DIR, exist_ok=True)
		safe_filename = re.sub(r'[^\w\-.]', '_', os.path.basename(file.filename))
		audio_path = os.path.join(INPUTS_DIR, f"{job_id}_{safe_filename}")
		with open(audio_path, "wb") as f:
			f.write(contents)

		# Create the session output directory so stages can write there
		timestamp = datetime.datetime.now().strftime("session_%Y_%m_%d_T%H%M%S")
		session_out_dir = os.path.join(OUTPUTS_DIR, timestamp)
		os.makedirs(session_out_dir, exist_ok=True)

		jobs_db[job_id] = {
			"status": "uploaded",
			"num_speakers": num_speakers,
			"filename": file.filename,
			"audio_path": audio_path,
			"transcriber": transcriber,
			"llm": llm,
			"video_gen": video_gen,
			"session_dir": session_out_dir,
			"auto_run": auto_run,
		}
		save_job_meta(job_id)
		logger.info("Accepted upload for %s -> job %s (auto_run=%s)", file.filename, job_id, auto_run)

		return {"job_id": job_id, "filename": file.filename, "auto_run": auto_run}

	except HTTPException:
		raise
	except Exception as exc:
		logger.error("Upload error: %s", exc)
		raise HTTPException(status_code=500, detail="Internal server error during upload")


# ---------------------------------------------------------------------------
# PATCH /api/job/{job_id}
# ---------------------------------------------------------------------------
@router.patch("/job/{job_id}")
async def update_job_providers(
	job_id: str,
	transcriber: str = Form(None),
	llm: str = Form(None),
	video_gen: str = Form(None),
	name: str = Form(None),
):
	"""Update AI provider selections or display name for a job.

    Called by the frontend when the user changes an AI model dropdown on the
    pipeline screen, or when the user edits the session name.  Only fields
    that are provided are updated.
    """
	job = require_job(job_id)
	if transcriber is not None:
		job["transcriber"] = transcriber
	if llm is not None:
		job["llm"] = llm
	if video_gen is not None:
		job["video_gen"] = video_gen
	if name is not None:
		job["name"] = name
		save_job_meta(job_id)
	return {"status": "updated"}


# ---------------------------------------------------------------------------
# POST /api/start_stage1/{job_id}
# ---------------------------------------------------------------------------
@router.post("/start_stage1/{job_id}")
async def start_stage1(job_id: str, background_tasks: BackgroundTasks):
	"""Explicitly kick off Stage 1 (transcription) for an uploaded job.

    The client calls this after confirming the file and AI model selections,
    giving the user a chance to correct mistakes before processing begins.
    """
	job = require_job(job_id)
	job["status"] = "processing"
	save_job_meta(job_id)
	background_tasks.add_task(_run_stage1, job_id)
	return {"status": "started"}


# ---------------------------------------------------------------------------
# GET /api/transcript/{job_id}
# ---------------------------------------------------------------------------
@router.get("/transcript/{job_id}")
async def get_transcript(job_id: str):
	"""Return the transcript and detected speaker list for a job.

    The transcript is written to disk by Stage 1; this endpoint reads that
    file so the frontend can render the speaker-mapping UI.
    """
	job = require_job(job_id)
	session_dir = job.get("session_dir")

	if session_dir:
		stage1_dir = resolve_stage1_dir(job)
		transcript_path = os.path.join(stage1_dir, "transcript.json")

		if os.path.exists(transcript_path):
			with open(transcript_path) as f:
				try:
					data = json.load(f)
				except (json.JSONDecodeError, ValueError):
					return {"job_id": job_id, "transcript": [], "speakers_detected": [], "converted_audio_url": None}
			utterances = data.get("utterances", [])
			speakers = list({u["speaker"] for u in utterances})

			# Resolve converted audio URL: prefer the shared hash-based cache
			converted_audio_url = None
			hash_ref = os.path.join(session_dir, "input_audio.hash")
			if os.path.exists(hash_ref):
				with open(hash_ref) as fh:
					file_hash = fh.read().strip()
				converted_audio_url = f"/outputs/wav_cache/{file_hash}.wav"
			elif os.path.exists(os.path.join(session_dir, "input_converted.wav")):
				session_name = os.path.basename(session_dir)
				converted_audio_url = f"/outputs/{session_name}/input_converted.wav"

			return {
				"job_id": job_id,
				"transcript": [{"speaker": u["speaker"], "text": u["text"]} for u in utterances],
				"speakers_detected": speakers,
				"converted_audio_url": converted_audio_url,
			}

	return {"job_id": job_id, "transcript": [], "speakers_detected": [], "converted_audio_url": None}


# ---------------------------------------------------------------------------
# POST /api/map_speakers/{job_id}  - stores mapping, starts Stage 2
# ---------------------------------------------------------------------------
@router.post("/map_speakers/{job_id}")
async def map_speakers(job_id: str, mapping: dict, background_tasks: BackgroundTasks):
	"""Store the user-provided speaker->character name mapping and start Stage 2 (LLM).

    The pipeline runs as a FastAPI BackgroundTask so the HTTP response returns
    immediately while processing continues in the background.
    """
	job = require_job(job_id)

	job["speaker_mapping"] = mapping
	job["status"] = "processing"
	save_job_meta(job_id)
	logger.info("Speaker mapping for job %s: %s - starting Stage 2", job_id, mapping)

	background_tasks.add_task(_run_stage2, job_id)
	return {"status": "success", "mapped": mapping}


# ---------------------------------------------------------------------------
# POST /api/speaker_visualization/{job_id}  - save user-edited descriptions
# ---------------------------------------------------------------------------
@router.post("/speaker_visualization/{job_id}")
async def save_speaker_visualization(job_id: str, visualization: dict):
	"""Save user-edited speaker visualization descriptions to disk.

    The frontend calls this when the user edits a character description and
    clicks Save.  The updated dict is written to ``speaker_visualization.json``
    in the Stage 2 output directory.
    """
	job = require_job(job_id)
	
	s2_dir = resolve_stage2_dir(job)
	os.makedirs(s2_dir, exist_ok=True)
	viz_path = os.path.join(s2_dir, "speaker_visualization.json")
	with open(viz_path, "w") as f:
		json.dump(visualization, f, indent=2)

	job["speaker_visualization"] = visualization
	save_job_meta(job_id)
	logger.info("Speaker visualization saved for job %s", job_id)
	return {"status": "saved"}


# ---------------------------------------------------------------------------
# GET /api/stage2_results/{job_id}
# ---------------------------------------------------------------------------
@router.get("/stage2_results/{job_id}")
async def get_stage2_results(job_id: str):
	"""Return speaker map + storyboard scenes after Stage 2 completes."""
	job = require_job(job_id)
	result = {"job_id": job_id, "speaker_map": {}, "scenes": []}
	s2_dir = resolve_stage2_dir(job)

	speaker_map_path = os.path.join(s2_dir, "speaker_map.json")
	if os.path.exists(speaker_map_path):
		with open(speaker_map_path) as f:
			result["speaker_map"] = json.load(f)

	speaker_viz_path = os.path.join(s2_dir, "speaker_visualization.json")
	if os.path.exists(speaker_viz_path):
		with open(speaker_viz_path) as f:
			result["speaker_visualization"] = json.load(f)

	storyboard_path = os.path.join(s2_dir, "storyboard.json")
	if os.path.exists(storyboard_path):
		with open(storyboard_path) as f:
			data = json.load(f)
		scenes = data.get("scenes", [])

		# Merge shots from production_script.json if available
		shots_by_scene = {}
		production_script_path = os.path.join(s2_dir, "production_script.json")
		if os.path.exists(production_script_path):
			with open(production_script_path) as f:
				ps_data = json.load(f)
			for ps in ps_data.get("scenes", []):
				shots_by_scene[ps.get("scene_number")] = ps.get("shots", [])

		result["scenes"] = [
			{
				"scene_number":     s.get("scene_number"),
				"location":         s.get("location", ""),
				"summary":          s.get("narrative_summary", ""),
				"start_time":       s.get("start_time"),
				"end_time":         s.get("end_time"),
				"prompt":           s.get("visual_prompt", ""),
				"is_relevant":      s.get("is_relevant"),
				"relevance_reason": s.get("relevance_reason", ""),
				"shots":			shots_by_scene.get(s.get("scene_number"), []),
			}
			for s in scenes
		]

	# Version navigation info
	ver_count = job.get("stage2_ver_count", 0)
	cur_ver   = job.get("stage2_cur_ver", -1)
	result["ver_count"]  = ver_count
	result["cur_ver"]    = cur_ver
	result["has_prev"]   = (cur_ver > 0) or (cur_ver == -1 and ver_count > 0)
	result["has_next"]   = (cur_ver != -1)
	# Legacy compat: keep has_previous_version
	result["has_previous_version"] = result["has_prev"]

	return result


# ---------------------------------------------------------------------------
# GET /api/job_status/{job_id}
# ---------------------------------------------------------------------------
@router.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
	"""Return the current status, stage timings, and version info for a job.

    Used by the frontend to restore elapsed-time displays when switching back
    to a session that was already running or has completed stages.
    """
	job = require_job(job_id)
	return {
		"job_id": job_id,
		"status": job.get("status"),
		"stage_timings": job.get("stage_timings", {}),
		"sub_stage_timings": job.get("sub_stage_timings", {}),
		# Version navigation state for all stages
		"stage1_ver_count": job.get("stage1_ver_count", 0),
		"stage1_cur_ver":   job.get("stage1_cur_ver", -1),
		"stage2_ver_count": job.get("stage2_ver_count", 0),
		"stage2_cur_ver":   job.get("stage2_cur_ver", -1),
		"stage3_ver_count": job.get("stage3_ver_count", 0),
		"stage3_cur_ver":   job.get("stage3_cur_ver", -1),
		"stage4_ver_count": job.get("stage4_ver_count", 0),
		"stage4_cur_ver":   job.get("stage4_cur_ver", -1),
	}


# ---------------------------------------------------------------------------
# POST /api/run_stage3/{job_id}
# ---------------------------------------------------------------------------
@router.post("/run_stage3/{job_id}")
async def run_stage3_endpoint(job_id: str, background_tasks: BackgroundTasks):
	"""Start Stage 3 (video generation) for a job."""
	job = require_job(job_id)

	job["status"] = "processing"
	save_job_meta(job_id)
	background_tasks.add_task(_run_stage3, job_id)
	return {"status": "started"}


# ---------------------------------------------------------------------------
# POST /api/rerun_scene/{job_id}/{scene_number}
# ---------------------------------------------------------------------------
@router.post("/rerun_scene/{job_id}/{scene_number}")
async def rerun_scene_endpoint(job_id: str, scene_number: int, background_tasks: BackgroundTasks):
	"""Re-generate a single failed scene clip."""
	require_job(job_id)
	background_tasks.add_task(_rerun_scene, job_id, scene_number)
	return {"status": "started", "scene_number": scene_number}


# ---------------------------------------------------------------------------
# GET /api/stage3_results/{job_id}
# ---------------------------------------------------------------------------
@router.get("/stage3_results/{job_id}")
async def get_stage3_results(job_id: str):
	"""Return per-scene video URLs after Stage 3 completes."""
	job = require_job(job_id)
	search_dir = resolve_stage34_dir(job)
	scene_files = sorted(glob.glob(os.path.join(search_dir, "scene_*.mp4")))

	scenes = []
	for path in scene_files:
		basename = os.path.basename(path)
		rel = os.path.relpath(path, OUTPUTS_DIR)
		# Extract scene number from filename like "scene_001.mp4"
		try:
			num = int(basename.replace("scene_", "").replace(".mp4", ""))
		except ValueError:
			num = None
		scenes.append({"scene_number": num, "video_url": f"/outputs/{rel}"})

	return {"job_id": job_id, "scenes": scenes}


# ---------------------------------------------------------------------------
# DELETE /api/job/{job_id}  - delete an active session
# ---------------------------------------------------------------------------
@router.delete("/job/{job_id}")
async def delete_active_job(job_id: str):
	"""Delete an active session: remove it from the in-memory job store and
	optionally delete its output directory from disk.
	"""
	if job_id not in jobs_db:
		raise HTTPException(status_code=404, detail="Job not found")
	job = jobs_db[job_id]
	session_dir = job.get("session_dir")

	# Remove from in-memory store
	del jobs_db[job_id]

	# Delete session directory if it exists
	if session_dir and os.path.isdir(session_dir):
		try:
			shutil.rmtree(session_dir)
		except Exception as exc:
			logger.warning("Could not delete session dir %s: %s", session_dir, exc)

	# Delete the uploaded audio file
	audio_path = job.get("audio_path")
	if audio_path and os.path.isfile(audio_path):
		try:
			os.remove(audio_path)
		except Exception as exc:
			logger.warning("Could not delete audio file %s: %s", audio_path, exc)

	logger.info("Deleted active job %s", job_id)
	return {"status": "deleted", "job_id": job_id}


# ---------------------------------------------------------------------------
# POST /api/regenerate_stage2/{job_id}  - force-rerun Stage 2
# ---------------------------------------------------------------------------
@router.post("/regenerate_stage2/{job_id}")
async def regenerate_stage2(job_id: str, background_tasks: BackgroundTasks):
	"""Force Stage 2 to re-run even if outputs already exist.

    Backs up existing outputs to .versions/{N}/ before overwriting,
    so the user can navigate between versions.
    """
	job = require_job(job_id)
	job["status"] = "processing"
	save_job_meta(job_id)
	background_tasks.add_task(_run_stage2, job_id, True)  # force_rerun=True
	return {"status": "started", "force": True}


# ---------------------------------------------------------------------------
# POST /api/navigate_stage/{job_id}/{stage_num}  - navigate versions
# ---------------------------------------------------------------------------
@router.post("/navigate_stage/{job_id}/{stage_num}")
async def navigate_stage(job_id: str, stage_num: int, direction: str = "prev"):
	"""Navigate to the previous or next saved version of a stage's outputs.

    For JSON stages (1, 2): copies files from .versions/{N}/ into the canonical
    stage dir so downstream stages always read the displayed version.
    For video stages (3, 4): renames files between canonical dir and
    .s3_versions/ or .s4_versions/ (O(1) moves, no file duplication).

    Returns updated version navigation info plus a 'stage_data' key with
    stage-specific result data for re-rendering.
    """
	job = require_job(job_id)

	ver_count_key = f"stage{stage_num}_ver_count"
	cur_ver_key   = f"stage{stage_num}_cur_ver"
	ver_count = job.get(ver_count_key, 0)
	cur_ver   = job.get(cur_ver_key, -1)

	# Compute target version
	if direction == "prev":
		if cur_ver == -1 and ver_count > 0:
			new_cur_ver = ver_count - 1
		elif cur_ver > 0:
			new_cur_ver = cur_ver - 1
		else:
			raise HTTPException(status_code=400, detail="No previous version available")
	else:  # "next"
		if cur_ver == -1:
			raise HTTPException(status_code=400, detail="Already at the latest version")
		elif cur_ver == ver_count - 1:
			new_cur_ver = -1
		else:
			new_cur_ver = cur_ver + 1

	# --- File manipulation per stage ---
	if stage_num == 1:
		stage_dir = resolve_stage1_dir(job)
		_S1_FILES = ["transcript.json", "speaker_map_suggestions.json"]
		if new_cur_ver == -1:
			# Navigating back to latest: need to save current displayed files back
			# and load the "live" state. But we use copy so the live files are always there.
			# Actually for stage 1 (copy semantics): just load from versions if not -1,
			# or do nothing if going to -1 (live = canonical = already there from last run).
			# However, navigating left overwrites canonical with version data.
			# Going right back to -1 means we need to restore the "latest run" state.
			# Since we always COPY (not move), canonical dir still has the latest run files
			# AS LONG AS navigation always copies INTO canonical (overwriting).
			# So going back to -1: we need to restore the original latest files.
			# We save them to a special "latest" backup before first left-navigation.
			latest_dir = os.path.join(stage_dir, ".versions", "latest")
			if os.path.isdir(latest_dir):
				for fname in _S1_FILES:
					src = os.path.join(latest_dir, fname)
					if os.path.exists(src):
						shutil.copy2(src, os.path.join(stage_dir, fname))
		else:
			# Before first left-navigation, save canonical as "latest"
			if cur_ver == -1:
				latest_dir = os.path.join(stage_dir, ".versions", "latest")
				os.makedirs(latest_dir, exist_ok=True)
				for fname in _S1_FILES:
					src = os.path.join(stage_dir, fname)
					if os.path.exists(src):
						shutil.copy2(src, os.path.join(latest_dir, fname))
			# Copy target version files to canonical
			ver_dir = os.path.join(stage_dir, ".versions", str(new_cur_ver))
			for fname in _S1_FILES:
				src = os.path.join(ver_dir, fname)
				if os.path.exists(src):
					shutil.copy2(src, os.path.join(stage_dir, fname))

	elif stage_num == 2:
		stage_dir = resolve_stage2_dir(job)
		_S2_FILES = [
			"production_script.json", "storyboard.json",
			"speaker_visualization.json", "shot_script.json", "speaker_map.json",
		]
		if new_cur_ver == -1:
			latest_dir = os.path.join(stage_dir, ".versions", "latest")
			if os.path.isdir(latest_dir):
				for fname in _S2_FILES:
					src = os.path.join(latest_dir, fname)
					if os.path.exists(src):
						shutil.copy2(src, os.path.join(stage_dir, fname))
		else:
			if cur_ver == -1:
				latest_dir = os.path.join(stage_dir, ".versions", "latest")
				os.makedirs(latest_dir, exist_ok=True)
				for fname in _S2_FILES:
					src = os.path.join(stage_dir, fname)
					if os.path.exists(src):
						shutil.copy2(src, os.path.join(latest_dir, fname))
			ver_dir = os.path.join(stage_dir, ".versions", str(new_cur_ver))
			for fname in _S2_FILES:
				src = os.path.join(ver_dir, fname)
				if os.path.exists(src):
					shutil.copy2(src, os.path.join(stage_dir, fname))

	elif stage_num == 3:
		stage_dir = resolve_stage34_dir(job)
		versions_dir = os.path.join(stage_dir, ".s3_versions")
		os.makedirs(versions_dir, exist_ok=True)
		# Save current scene_*.mp4 files to a temp swap slot, load target
		swap_dir = os.path.join(versions_dir, "swap_tmp")
		os.makedirs(swap_dir, exist_ok=True)
		# Move canonical scene files to swap_tmp
		for f in glob.glob(os.path.join(stage_dir, "scene_*.mp4")):
			os.rename(f, os.path.join(swap_dir, os.path.basename(f)))
		# Move target version files to canonical
		src_dir = os.path.join(versions_dir, "latest") if new_cur_ver == -1 else os.path.join(versions_dir, str(new_cur_ver))
		if os.path.isdir(src_dir):
			for f in glob.glob(os.path.join(src_dir, "scene_*.mp4")):
				os.rename(f, os.path.join(stage_dir, os.path.basename(f)))
		# Save swap_tmp as the old cur_ver's archive
		old_slot_label = "latest" if cur_ver == -1 else str(cur_ver)
		old_dir = os.path.join(versions_dir, old_slot_label)
		if swap_dir != old_dir:
			if os.path.isdir(old_dir):
				shutil.rmtree(old_dir)
			os.rename(swap_dir, old_dir)
		else:
			pass  # already in right place

	elif stage_num == 4:
		stage_dir = resolve_stage34_dir(job)
		_S4_FILES = ["final_with_audio.mp4", "final_captioned.mp4"]
		versions_dir = os.path.join(stage_dir, ".s4_versions")
		os.makedirs(versions_dir, exist_ok=True)
		swap_dir = os.path.join(versions_dir, "swap_tmp")
		os.makedirs(swap_dir, exist_ok=True)
		for fname in _S4_FILES:
			src = os.path.join(stage_dir, fname)
			if os.path.exists(src):
				os.rename(src, os.path.join(swap_dir, fname))
		src_dir = os.path.join(versions_dir, "latest") if new_cur_ver == -1 else os.path.join(versions_dir, str(new_cur_ver))
		if os.path.isdir(src_dir):
			for fname in _S4_FILES:
				src = os.path.join(src_dir, fname)
				if os.path.exists(src):
					os.rename(src, os.path.join(stage_dir, fname))
		old_slot_label = "latest" if cur_ver == -1 else str(cur_ver)
		old_dir = os.path.join(versions_dir, old_slot_label)
		if swap_dir != old_dir:
			if os.path.isdir(old_dir):
				shutil.rmtree(old_dir)
			os.rename(swap_dir, old_dir)

	# Update metadata
	job[cur_ver_key] = new_cur_ver
	save_job_meta(job_id)

	has_prev = (new_cur_ver > 0) or (new_cur_ver == -1 and ver_count > 0)
	has_next = (new_cur_ver != -1)

	return {
		"status": "navigated",
		"stage_num": stage_num,
		"ver_count": ver_count,
		"cur_ver": new_cur_ver,
		"has_prev": has_prev,
		"has_next": has_next,
	}


# ---------------------------------------------------------------------------
# POST /api/rerun_stage1/{job_id}  - force re-run Stage 1
# ---------------------------------------------------------------------------
@router.post("/rerun_stage1/{job_id}")
async def rerun_stage1(job_id: str, background_tasks: BackgroundTasks):
	"""Force Stage 1 (transcription) to re-run even if transcript already exists.

    Backs up existing transcript to .versions/{N}/ before overwriting.
    """
	job = require_job(job_id)
	job["status"] = "processing"
	save_job_meta(job_id)
	background_tasks.add_task(_run_stage1, job_id, True)  # force_rerun=True
	return {"status": "started", "force": True}


# ---------------------------------------------------------------------------
# POST /api/rerun_stage3/{job_id}  - re-run Stage 3
# ---------------------------------------------------------------------------
@router.post("/rerun_stage3/{job_id}")
async def rerun_stage3(job_id: str, background_tasks: BackgroundTasks):
	"""Re-run Stage 3 (video generation), backing up existing scene videos first."""
	job = require_job(job_id)
	job["status"] = "processing"
	save_job_meta(job_id)
	background_tasks.add_task(_run_stage3, job_id, True)  # backup_first=True
	return {"status": "started"}


# ---------------------------------------------------------------------------
# POST /api/rerun_stage4/{job_id}  - force re-run Stage 4
# ---------------------------------------------------------------------------
@router.post("/rerun_stage4/{job_id}")
async def rerun_stage4(job_id: str, background_tasks: BackgroundTasks):
	"""Force Stage 4 (assembly) to re-run even if final video already exists.

    Moves existing final video files to .versions/{N}/ before overwriting.
    """
	job = require_job(job_id)
	job["status"] = "processing"
	save_job_meta(job_id)
	background_tasks.add_task(_run_stage4, job_id, True)  # force_rerun=True
	return {"status": "started", "force": True}


# ---------------------------------------------------------------------------
# POST /api/run_stage4/{job_id}
# ---------------------------------------------------------------------------
@router.post("/run_stage4/{job_id}")
async def run_stage4_endpoint(job_id: str, background_tasks: BackgroundTasks):
	"""Start Stage 4 (final assembly) for a job."""
	job = require_job(job_id)

	job["status"] = "processing"
	save_job_meta(job_id)
	background_tasks.add_task(_run_stage4, job_id)
	return {"status": "started"}


# ---------------------------------------------------------------------------
# WebSocket /api/ws/progress/{job_id}
# ---------------------------------------------------------------------------
@router.websocket("/ws/progress/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
	"""Stream real-time pipeline progress updates to the browser via WebSocket."""
	await manager.connect(websocket, job_id)
	try:
		while True:
			await websocket.receive_text()
	except WebSocketDisconnect:
		manager.disconnect(websocket, job_id)


# ---------------------------------------------------------------------------
# GET /api/videos/{job_id}
# ---------------------------------------------------------------------------
@router.get("/videos/{job_id}")
async def get_videos(job_id: str):
	"""Return the generated video URL(s) for a completed job, or its status."""
	job = require_job(job_id)
	status = job.get("status")

	if status == "completed":
		stage34_dir = resolve_stage34_dir(job)
		# Also search session root as a safety fallback for legacy/partial runs
		session_dir = job.get("session_dir", "")
		search_dirs = [stage34_dir, session_dir]
		for candidate in ("final_with_audio.mp4", "final_captioned.mp4", "final_stitched.mp4"):
			for search_dir in search_dirs:
				path = os.path.join(search_dir, candidate)
				if os.path.exists(path):
					rel = os.path.relpath(path, OUTPUTS_DIR)
					return {"status": "completed", "videos": [f"/outputs/{rel}"]}
		return {"status": "completed", "videos": []}

	if status == "error":
		detail = job.get("last_update", {}).get("detail", "Unknown error")
		return {"status": "error", "detail": detail}

	return {"status": job.get("status", "processing")}
