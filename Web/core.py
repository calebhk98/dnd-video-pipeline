"""
core.py - Shared state and utilities for the DND Video Pipeline API.

This module is imported by all router modules so they share the same
in-memory job database and WebSocket connection manager without needing
to pass objects around or use global variables spread across files.
"""

import os
import json
import datetime
import logging
from typing import Dict, List, Any
from fastapi import WebSocket

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants (resolved relative to this file so they work regardless of
# the current working directory when uvicorn is started)
# ---------------------------------------------------------------------------
# Web/ directory (where app.py lives)
WEB_DIR = os.path.dirname(__file__)
# Project root (one level above Web/)
ROOT_DIR = os.path.dirname(WEB_DIR)

FRONTEND_DIR = os.path.join(WEB_DIR, "frontend")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
ENV_PATH = os.path.join(ROOT_DIR, ".env")
EXAMPLE_PATH = os.path.join(CONFIG_DIR, ".env.example")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
INPUTS_DIR = os.path.join(ROOT_DIR, "inputs")

JOB_META_FILENAME = "job_meta.json"

SUPPORTED_EXTENSIONS = ('.mp3', '.wav', '.m4a')

# Maximum upload size - configurable via environment variable (default 500 MB)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024

DEFAULT_TRANSCRIBER = os.getenv("DEFAULT_TRANSCRIBER", "assemblyai")
DEFAULT_LLM         = os.getenv("DEFAULT_LLM", "anthropic")
DEFAULT_VIDEO_GEN   = os.getenv("DEFAULT_VIDEO_GEN", "luma")

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
# Maps job_id (str UUID) -> metadata dict.  Backed by per-session job_meta.json
# files so state survives server restarts (see save_job_meta / load_jobs_from_disk).
jobs_db: Dict[str, Dict[str, Any]] = {}


def save_job_meta(job_id: str) -> None:
	"""Persist jobs_db[job_id] to {session_dir}/job_meta.json.

    Uses a session-level format that tracks all provider combinations attempted
    in a ``runs`` dict.  Reads any existing file first to preserve previously
    recorded run entries before merging in the current job's state.

    Embeds the job_id so load_jobs_from_disk can recover the UUID on restart.
    Errors are logged as warnings so a write failure never crashes the pipeline.
    """
	job = jobs_db.get(job_id)
	if not job:
		return
	session_dir = job.get("session_dir")
	if not session_dir:
		return
	meta_path = os.path.join(session_dir, JOB_META_FILENAME)

	# Load existing metadata so we can merge rather than overwrite the runs index
	existing = {}
	if os.path.exists(meta_path):
		try:
			with open(meta_path) as fh:
				existing = json.load(fh)
		except Exception:
			pass

	# Build or update the per-provider-combination run entries
	from provider_dirs import stage1_key, stage2_key, stage3_key
	transcriber = job.get("transcriber", "")
	llm = job.get("llm", "")
	video_gen = job.get("video_gen", "")

	runs = existing.get("runs", {})
	run_entry = {
		"job_id": job_id,
		"status": job.get("status"),
		"started_at": job.get("started_at"),
	}
	if transcriber:
		runs[stage1_key(transcriber)] = run_entry
	if transcriber and llm:
		runs[stage2_key(transcriber, llm)] = run_entry
	if transcriber and llm and video_gen:
		runs[stage3_key(transcriber, llm, video_gen)] = run_entry

	try:
		payload = {
			"job_id": job_id,
			"session_dir": session_dir,
			"filename": job.get("filename"),
			"name": job.get("name", ""),
			"audio_path": job.get("audio_path"),
			"num_speakers": job.get("num_speakers"),
			"status": job.get("status"),
			"transcriber": transcriber,
			"llm": llm,
			"video_gen": video_gen,
			"auto_run": job.get("auto_run", False),
			"speaker_mapping": job.get("speaker_mapping", {}),
			"stage_timings": job.get("stage_timings", {}),
			"sub_stage_timings": job.get("sub_stage_timings", {}),
			"active_providers": {
				"transcriber": transcriber,
				"llm": llm,
				"video_gen": video_gen,
			},
			"runs": runs,
			"saved_at": datetime.datetime.utcnow().isoformat(),
		}
		with open(meta_path, "w") as fh:
			json.dump(payload, fh, indent=2)
	except Exception as exc:
		logger.warning("Could not save job meta for %s: %s", job_id, exc)


def load_jobs_from_disk() -> None:
	"""Scan OUTPUTS_DIR and restore persisted job metadata into jobs_db.

    Called once at application startup. Jobs that are already in jobs_db
    (e.g. from a hot-reload) are skipped. Any job that was mid-run when
    the server stopped is marked 'interrupted' instead of 'processing'.
    """
	if not os.path.exists(OUTPUTS_DIR):
		return
	loaded = 0
	for item in os.listdir(OUTPUTS_DIR):
		item_path = os.path.join(OUTPUTS_DIR, item)
		if not os.path.isdir(item_path):
			continue
		meta_path = os.path.join(item_path, JOB_META_FILENAME)
		if not os.path.exists(meta_path):
			continue
		try:
			with open(meta_path) as fh:
				meta = json.load(fh)
			job_id = meta.get("job_id")
			if not job_id:
				logger.warning("job_meta.json in %s has no job_id, skipping", item)
				continue
			if job_id in jobs_db:
				continue
			# Reconstruct the flat job dict from the new-format meta.
			# active_providers holds the last-used provider selections; fall back
			# to top-level keys for backward compatibility with old format files.
			active = meta.get("active_providers", {})
			job = {
				"status": meta.get("status", "unknown"),
				"session_dir": item_path,
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
			loaded += 1
		except Exception as exc:
			logger.warning("Failed to load job_meta.json from %s: %s", item, exc)
	logger.info("Restored %d job(s) from disk.", loaded)


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
	"""Manages active WebSocket connections grouped by job ID.

    Multiple browser tabs can subscribe to the same job's progress feed;
    all of them receive every broadcast message.
    """

	def __init__(self):
		"""Initializes an empty map of active WebSocket connections."""

		# job_id -> list of currently-connected WebSocket objects
		self.active_connections: Dict[str, List[WebSocket]] = {}

	async def connect(self, websocket: WebSocket, job_id: str):
		"""Accept a new WebSocket handshake and register it for the given job."""
		await websocket.accept()
		self.active_connections.setdefault(job_id, []).append(websocket)
		logger.info("WebSocket connected for job %s", job_id)

	def disconnect(self, websocket: WebSocket, job_id: str):
		"""Remove a closed WebSocket connection; clean up empty job entries."""
		connections = self.active_connections.get(job_id, [])
		if websocket in connections:
			connections.remove(websocket)
		# Delete the key entirely when no clients are left
		if not connections and job_id in self.active_connections:
			del self.active_connections[job_id]
		logger.info("WebSocket disconnected for job %s", job_id)

	async def broadcast_to_job(self, job_id: str, message: dict):
		"""Send a JSON message to every subscriber for a particular job."""
		for connection in self.active_connections.get(job_id, []):
			try:
				await connection.send_json(message)
			except (RuntimeError, ConnectionResetError) as exc:
				logger.error("Error sending to websocket: %s", exc)


# Module-level singleton - import this instance from every router module
manager = ConnectionManager()
