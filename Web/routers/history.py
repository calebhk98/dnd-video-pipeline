"""
routers/history.py - Session history and resume endpoints.

Scans the outputs directory for past pipeline sessions and returns a
sorted list with metadata (transcript available, video available, resume stage).
The frontend sidebar uses this to let users revisit or resume previous runs.

Supports both legacy (flat-directory) sessions and new provider-namespaced sessions.
"""

import os
import json
import uuid
import shutil
import glob as glob_module
import logging

from fastapi import APIRouter, HTTPException

from core import OUTPUTS_DIR, jobs_db, save_job_meta, JOB_META_FILENAME, DEFAULT_TRANSCRIBER, DEFAULT_LLM, DEFAULT_VIDEO_GEN
from provider_dirs import (
	is_legacy_session, get_stage1_dir, get_stage2_dir, get_stage34_dir,
	detect_all_run_keys, parse_key,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["history"])

# Candidate video filenames produced by stage 4, in priority order
VIDEO_CANDIDATES = ("final_with_audio.mp4", "final_captioned.mp4", "final_stitched.mp4")


def _detect_resume_stage_legacy(session_dir: str) -> dict:
	"""Inspect artifact files in a flat (legacy) session_dir and return resume guidance."""
	def exists(*parts):
		"""Checks if a file exists within the session directory."""

		return os.path.exists(os.path.join(session_dir, *parts))

	# Stage 4 complete
	for candidate in VIDEO_CANDIDATES:
		if exists(candidate):
			return {
				"resume_from": None,
				"completed_stages": [1, 2, 3, 4],
				"detail": "Pipeline fully complete",
			}

	# Stage 3 complete (check for partial completion)
	scene_files = glob_module.glob(os.path.join(session_dir, "scene_*.mp4"))
	if scene_files:
		script_path = os.path.join(session_dir, "production_script.json")
		if os.path.exists(script_path):
			try:
				with open(script_path) as fh:
					script_data = json.load(fh)
				expected = len(script_data.get("scenes", []))
				actual = len(scene_files)
				if expected > 0 and actual < expected:
					return {
						"resume_from": 3,
						"completed_stages": [1, 2],
						"detail": f"Stage 3 partially complete ({actual}/{expected} scenes); re-run to finish",
					}
			except Exception:
				pass
		return {
			"resume_from": 4,
			"completed_stages": [1, 2, 3],
			"detail": "Scene videos exist; ready to run Stage 4 (Assembly)",
		}

	# Stage 2 complete
	if exists("production_script.json") and exists("storyboard.json"):
		return {
			"resume_from": 3,
			"completed_stages": [1, 2],
			"detail": "Storyboard and script exist; ready to run Stage 3 (Video Generation)",
		}

	# Stage 1 complete
	if exists("transcript.json"):
		return {
			"resume_from": 2,
			"completed_stages": [1],
			"detail": "Transcript exists; ready for speaker mapping and Stage 2 (LLM)",
		}

	# Nothing done yet
	return {
		"resume_from": 1,
		"completed_stages": [],
		"detail": "No pipeline artifacts found; start from Stage 1 (Transcription)",
	}


def _detect_resume_stage(session_dir: str, providers: dict = None) -> dict:
	"""Inspect artifact files and return resume guidance for the given provider combination.

    For legacy (flat-directory) sessions, falls through to _detect_resume_stage_legacy.
    For new sessions, checks the provider-namespaced subdirectories.

    providers: dict with keys "transcriber", "llm", "video_gen"
    """
	if is_legacy_session(session_dir) or not providers:
		return _detect_resume_stage_legacy(session_dir)

	transcriber = providers.get("transcriber", "assembly")
	llm = providers.get("llm", "claude")
	video_gen = providers.get("video_gen", "luma")

	s1_dir = get_stage1_dir(session_dir, transcriber)
	s2_dir = get_stage2_dir(session_dir, transcriber, llm)
	s34_dir = get_stage34_dir(session_dir, transcriber, llm, video_gen)

	def exists(base, *parts):
		"""Checks if a file exists within a specific provider-namespaced directory."""

		return os.path.exists(os.path.join(base, *parts))

	# Stage 4 complete
	for candidate in VIDEO_CANDIDATES:
		if exists(s34_dir, candidate):
			return {
				"resume_from": None,
				"completed_stages": [1, 2, 3, 4],
				"detail": "Pipeline fully complete",
			}

	# Stage 3 complete (check for partial completion)
	scene_files = glob_module.glob(os.path.join(s34_dir, "scene_*.mp4"))
	if scene_files:
		script_path = os.path.join(s2_dir, "production_script.json")
		if os.path.exists(script_path):
			try:
				with open(script_path) as fh:
					script_data = json.load(fh)
				expected = len(script_data.get("scenes", []))
				actual = len(scene_files)
				if expected > 0 and actual < expected:
					return {
						"resume_from": 3,
						"completed_stages": [1, 2],
						"detail": f"Stage 3 partially complete ({actual}/{expected} scenes); re-run to finish",
					}
			except Exception:
				pass
		return {
			"resume_from": 4,
			"completed_stages": [1, 2, 3],
			"detail": "Scene videos exist; ready to run Stage 4 (Assembly)",
		}

	# Stage 2 complete
	if exists(s2_dir, "production_script.json") and exists(s2_dir, "storyboard.json"):
		return {
			"resume_from": 3,
			"completed_stages": [1, 2],
			"detail": "Storyboard and script exist; ready to run Stage 3 (Video Generation)",
		}

	# Stage 1 complete
	if exists(s1_dir, "transcript.json"):
		return {
			"resume_from": 2,
			"completed_stages": [1],
			"detail": "Transcript exists; ready for speaker mapping and Stage 2 (LLM)",
		}

	# Nothing done yet for this provider combination
	return {
		"resume_from": 1,
		"completed_stages": [],
		"detail": "No pipeline artifacts found; start from Stage 1 (Transcription)",
	}


def _summarize_runs(session_dir: str) -> list:
	"""Return a list of completed Stage-3 run summaries for display in the history sidebar."""
	runs_dir = os.path.join(session_dir, "runs")
	if not os.path.isdir(runs_dir):
		return []
	result = []
	for key in detect_all_run_keys(session_dir):
		parsed = parse_key(key)
		if parsed["stage"] == 3:  # Only Stage 3+4 combos produce final videos
			s34_dir = os.path.join(runs_dir, key)
			video_url = None
			for candidate in VIDEO_CANDIDATES:
				if os.path.exists(os.path.join(s34_dir, candidate)):
					# Build a relative URL for the video
					rel_session = os.path.basename(session_dir)
					video_url = f"/outputs/{rel_session}/runs/{key}/{candidate}"
					break
			result.append({
				"key": key,
				"transcriber": parsed["transcriber"],
				"llm": parsed["llm"],
				"video_gen": parsed["video_gen"],
				"has_video": video_url is not None,
				"video_url": video_url,
			})
	return result


def _find_any_transcript(session_dir: str) -> tuple[bool, str | None]:
	"""
	Search for any transcript.json in any provider-namespaced run directory.
	Returns (found, url).
	"""
	for key in detect_all_run_keys(session_dir):
		if parse_key(key)["stage"] != 1:
			continue

		candidate_path = os.path.join(session_dir, "runs", key, "transcript.json")
		if os.path.exists(candidate_path):
			rel_session = os.path.basename(session_dir)
			return True, f"/outputs/{rel_session}/runs/{key}/transcript.json"

	return False, None


@router.get("/history")
async def get_history():
	"""Scan the outputs directory and return a list of past sessions.

    Each session entry contains:
    - ``id``: directory name (also used as a sortable timestamp key)
    - ``name``: human-readable label derived from the directory name
    - ``has_transcript``: whether a transcript is present (any provider)
    - ``video_url``: URL of the first finished video found, or None
    - ``transcript_url``: URL of transcript.json, or None
    - ``resume_stage``: next stage to run (None = complete)
    - ``completed_stages``: list of stages with existing artifacts
    - ``runs``: list of completed Stage-3 provider combinations (new sessions only)
    - ``active_providers``: the last-used provider selections
    """
	if not os.path.exists(OUTPUTS_DIR):
		return {"sessions": []}

	sessions = []
	for item in os.listdir(OUTPUTS_DIR):
		item_path = os.path.join(OUTPUTS_DIR, item)
		if not os.path.isdir(item_path):
			continue  # skip stray files

		# Load saved metadata (if any) to get active providers and runs index
		meta = {}
		meta_path = os.path.join(item_path, JOB_META_FILENAME)
		if os.path.exists(meta_path):
			try:
				with open(meta_path) as fh:
					meta = json.load(fh)
			except Exception:
				pass

		active_providers = meta.get("active_providers", {
			"transcriber": meta.get("transcriber", DEFAULT_TRANSCRIBER),
			"llm": meta.get("llm", DEFAULT_LLM),
			"video_gen": meta.get("video_gen", DEFAULT_VIDEO_GEN),
		})

		# Determine where to look for transcript and video
		if is_legacy_session(item_path):
			has_transcript = os.path.exists(os.path.join(item_path, "transcript.json"))
			video_url = None
			for candidate in VIDEO_CANDIDATES:
				if os.path.exists(os.path.join(item_path, candidate)):
					video_url = f"/outputs/{item}/{candidate}"
					break
			transcript_url = f"/outputs/{item}/transcript.json" if has_transcript else None
			runs_list = []
		else:
			# Check the active Stage 1 dir for transcript
			s1_dir = get_stage1_dir(item_path, active_providers.get("transcriber", DEFAULT_TRANSCRIBER))
			has_transcript = os.path.exists(os.path.join(s1_dir, "transcript.json"))

			if has_transcript:
				rel_s1 = os.path.relpath(s1_dir, OUTPUTS_DIR)
				transcript_url = f"/outputs/{rel_s1}/transcript.json"
			else:
				# Also accept any transcript in any provider subdir
				has_transcript, transcript_url = _find_any_transcript(item_path)

			# Use active providers to find the primary video
			video_url = None
			runs_list = _summarize_runs(item_path)
			# First check the active Stage 3/4 dir
			s34_dir = get_stage34_dir(
				item_path,
				active_providers.get("transcriber", DEFAULT_TRANSCRIBER),
				active_providers.get("llm", DEFAULT_LLM),
				active_providers.get("video_gen", DEFAULT_VIDEO_GEN),
			)
			for candidate in VIDEO_CANDIDATES:
				if os.path.exists(os.path.join(s34_dir, candidate)):
					rel = os.path.relpath(os.path.join(s34_dir, candidate), OUTPUTS_DIR)
					video_url = f"/outputs/{rel}"
					break
			# If not found in active dir, use first run that has a video
			if video_url is None and runs_list:
				for run in runs_list:
					if run["video_url"]:
						video_url = run["video_url"]
						break

		# Parse the directory timestamp into a sortable ISO string and a display string.
		# Directory name format: session_YYYY_MM_DD_THHMMSS
		created_at_iso = None
		try:
			ts_part = item.replace("session_", "")  # "YYYY_MM_DD_THHMMSS"
			date_part, time_part = ts_part.rsplit("_T", 1)   # "YYYY_MM_DD", "HHMMSS"
			date_clean = date_part.replace("_", "-")  # "YYYY-MM-DD"
			time_clean = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
			created_at_iso = f"{date_clean}T{time_clean}"
		except Exception:
			created_at_iso = None

		# Prefer the user-provided name saved in job_meta.json; fall back to filename, then timestamp.
		display_name = meta.get("name") or ""
		if not display_name:
			filename = meta.get("filename") or ""
			if filename:
				display_name = os.path.splitext(filename)[0]
			elif created_at_iso:
				display_name = created_at_iso.replace("T", " ")
			else:
				display_name = item.replace("session_", "").replace("_", " ")

		# Include sessions that have at least one artifact OR that have been uploaded
		# but not yet transcribed (so they survive a page refresh before Stage 1 starts).
		is_uploaded_only = (
			not has_transcript and not video_url
			and meta.get("status") == "uploaded"
			and meta.get("audio_path")
		)
		if has_transcript or video_url or is_uploaded_only:
			resume_info = _detect_resume_stage(item_path, active_providers)
			sessions.append({
				"id": item,
				"name": display_name,
				"created_at": created_at_iso,
				"has_transcript": has_transcript,
				"video_url": video_url,
				"transcript_url": transcript_url,
				"resume_stage": resume_info["resume_from"],
				"completed_stages": resume_info["completed_stages"],
				"runs": runs_list,
				"active_providers": active_providers,
			})

	# Most recent first
	sessions.sort(key=lambda x: x["id"], reverse=True)
	return {"sessions": sessions}


@router.post("/resume/{session_id}")
async def resume_session(
	session_id: str,
	transcriber: str = None,
	llm: str = None,
	video_gen: str = None,
):
	"""Load a past session into jobs_db and return resume instructions.

    Accepts optional provider override query parameters so the frontend can
    request a specific provider combination when resuming.  The resume stage
    reflects what still needs to be done for that exact combination.

    Args:
        session_id: The directory name under outputs/ (e.g. "session_2026_03_16_T120000").
        transcriber: Optional transcription provider override.
        llm:         Optional LLM provider override.
        video_gen:   Optional video generator override.

    Returns a job_id that the client can use for all subsequent pipeline API calls,
    along with which stage to trigger next.
    """
	session_dir = os.path.join(OUTPUTS_DIR, session_id)
	if not os.path.isdir(session_dir):
		raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

	meta_path = os.path.join(session_dir, JOB_META_FILENAME)

	# Check if any live jobs_db entry already points to this session
	existing_job_id = None
	for jid, jdata in jobs_db.items():
		if jdata.get("session_dir") == session_dir:
			existing_job_id = jid
			break

	if existing_job_id:
		job_id = existing_job_id
		job = jobs_db[job_id]
	elif os.path.exists(meta_path):
		with open(meta_path) as fh:
			meta = json.load(fh)
		job_id = meta.get("job_id")
		if not job_id:
			raise HTTPException(status_code=422, detail="job_meta.json is missing job_id field")
		if job_id not in jobs_db:
			# Reconstruct flat job dict from saved metadata
			active = meta.get("active_providers", {})
			job = {
				"status": meta.get("status", "unknown"),
				"session_dir": session_dir,
				"filename": meta.get("filename"),
				"name": meta.get("name", ""),
				"audio_path": meta.get("audio_path"),
				"num_speakers": meta.get("num_speakers", 0),
				"transcriber": active.get("transcriber") or meta.get("transcriber", DEFAULT_TRANSCRIBER),
				"llm": active.get("llm") or meta.get("llm", DEFAULT_LLM),
				"video_gen": active.get("video_gen") or meta.get("video_gen", DEFAULT_VIDEO_GEN),
				"auto_run": meta.get("auto_run", False),
				"speaker_mapping": meta.get("speaker_mapping", {}),
				"stage_timings": meta.get("stage_timings", {}),
				"sub_stage_timings": meta.get("sub_stage_timings", {}),
			}
			if job["status"] == "processing":
				job["status"] = "interrupted"
			jobs_db[job_id] = job
		else:
			job = jobs_db[job_id]
		logger.info("Loaded session %s into jobs_db as job %s", session_id, job_id)
	else:
		# Old session without metadata: create a minimal stub so stages can be triggered
		job_id = str(uuid.uuid4())
		job = {
			"status": "unknown",
			"session_dir": session_dir,
			"filename": None,
			"audio_path": None,
			"num_speakers": 0,
			"transcriber": DEFAULT_TRANSCRIBER,
			"llm": DEFAULT_LLM,
			"video_gen": DEFAULT_VIDEO_GEN,
			"auto_run": False,
			"speaker_mapping": {},
		}
		jobs_db[job_id] = job
		save_job_meta(job_id)
		logger.info("Created stub entry for session %s as job %s (no job_meta.json)", session_id, job_id)

	# Apply provider overrides from query parameters
	if transcriber is not None:
		jobs_db[job_id]["transcriber"] = transcriber
	if llm is not None:
		jobs_db[job_id]["llm"] = llm
	if video_gen is not None:
		jobs_db[job_id]["video_gen"] = video_gen

	providers = {
		"transcriber": jobs_db[job_id].get("transcriber", DEFAULT_TRANSCRIBER),
		"llm": jobs_db[job_id].get("llm", DEFAULT_LLM),
		"video_gen": jobs_db[job_id].get("video_gen", DEFAULT_VIDEO_GEN),
	}

	resume_info = _detect_resume_stage(session_dir, providers)

	# Sync status if artifacts indicate completion
	if resume_info["resume_from"] is None and jobs_db[job_id].get("status") != "completed":
		jobs_db[job_id]["status"] = "completed"

	save_job_meta(job_id)

	return {
		"job_id": job_id,
		"session_id": session_id,
		"status": jobs_db[job_id]["status"],
		"resume_from": resume_info["resume_from"],
		"completed_stages": resume_info["completed_stages"],
		"detail": resume_info["detail"],
		"has_speaker_mapping": bool(jobs_db[job_id].get("speaker_mapping")),
		"active_providers": providers,
	}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
	"""Delete a past session directory and remove it from the in-memory job store.

    The session directory (and all its contents) under outputs/ is removed from
    disk.  Any matching live entry in jobs_db is also purged.  This is
    irreversible -- the client should confirm with the user before calling.
    """
	session_dir = os.path.join(OUTPUTS_DIR, session_id)
	if not os.path.isdir(session_dir):
		raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

	# Remove from in-memory job store
	to_remove = [jid for jid, jdata in jobs_db.items() if jdata.get("session_dir") == session_dir]
	for jid in to_remove:
		del jobs_db[jid]

	# Delete the session directory from disk
	try:
		shutil.rmtree(session_dir)
	except Exception as exc:
		logger.error("Failed to delete session directory %s: %s", session_dir, exc)
		raise HTTPException(status_code=500, detail=f"Failed to delete session: {exc}")

	logger.info("Deleted session %s", session_id)
	return {"status": "deleted", "session_id": session_id}
