"""
provider_dirs.py - Utilities for provider-namespaced output directories.

Each pipeline stage's output is namespaced by the combination of providers used
to produce it.  This allows multiple provider combinations to coexist within a
single session directory without overwriting each other.

Directory layout:
  outputs/{session}/runs/{transcriber}/               <- Stage 1 artifacts
  outputs/{session}/runs/{transcriber}__{llm}/        <- Stage 2 artifacts
  outputs/{session}/runs/{transcriber}__{llm}__{video_gen}/  <- Stage 3+4 artifacts

The double-underscore separator avoids ambiguity with provider names that may
contain a single hyphen or underscore.
"""

import os

from core import DEFAULT_TRANSCRIBER, DEFAULT_LLM, DEFAULT_VIDEO_GEN

SEP = "__"


def stage1_key(transcriber: str) -> str:
	"""Return the run-directory key for a Stage 1 (transcription) run."""
	return transcriber


def stage2_key(transcriber: str, llm: str) -> str:
	"""Return the run-directory key for a Stage 2 (LLM) run."""
	return f"{transcriber}{SEP}{llm}"


def stage3_key(transcriber: str, llm: str, video_gen: str) -> str:
	"""Return the run-directory key for a Stage 3+4 (video + assembly) run."""
	return f"{transcriber}{SEP}{llm}{SEP}{video_gen}"


def get_run_dir(session_dir: str, key: str) -> str:
	"""Return the absolute path for a named run directory (does not create it)."""
	return os.path.join(session_dir, "runs", key)


def get_stage1_dir(session_dir: str, transcriber: str) -> str:
	"""Return the Stage 1 output directory for the given transcriber."""
	return get_run_dir(session_dir, stage1_key(transcriber))


def get_stage2_dir(session_dir: str, transcriber: str, llm: str) -> str:
	"""Return the Stage 2 output directory for the given transcriber + LLM pair."""
	return get_run_dir(session_dir, stage2_key(transcriber, llm))


def get_stage34_dir(session_dir: str, transcriber: str, llm: str, video_gen: str) -> str:
	"""Return the Stage 3+4 output directory for all three providers."""
	return get_run_dir(session_dir, stage3_key(transcriber, llm, video_gen))


def resolve_stage1_dir(job: dict) -> str:
	"""Return the Stage 1 output dir for a job, respecting legacy sessions."""
	session_dir = job.get("session_dir", "")
	if not session_dir or is_legacy_session(session_dir):
		return session_dir
	return get_stage1_dir(session_dir, job.get("transcriber", DEFAULT_TRANSCRIBER))


def resolve_stage2_dir(job: dict) -> str:
	"""Return the Stage 2 output dir for a job, respecting legacy sessions."""
	session_dir = job.get("session_dir", "")
	if not session_dir or is_legacy_session(session_dir):
		return session_dir
	return get_stage2_dir(
		session_dir, 
		job.get("transcriber", DEFAULT_TRANSCRIBER), 
		job.get("llm", DEFAULT_LLM)
	)


def resolve_stage34_dir(job: dict) -> str:
	"""Return the Stage 3+4 output dir for a job, respecting legacy sessions."""
	session_dir = job.get("session_dir", "")
	if not session_dir or is_legacy_session(session_dir):
		return session_dir
	return get_stage34_dir(
		session_dir,
		job.get("transcriber", DEFAULT_TRANSCRIBER),
		job.get("llm", DEFAULT_LLM),
		job.get("video_gen", DEFAULT_VIDEO_GEN)
	)


def is_legacy_session(session_dir: str) -> bool:
	"""Return True if this session uses the old flat structure (no runs/ subdir)."""
	return not os.path.isdir(os.path.join(session_dir, "runs"))


def detect_all_run_keys(session_dir: str) -> list:
	"""Scan the runs/ subdirectory and return a list of all run-key directory names."""
	runs_dir = os.path.join(session_dir, "runs")
	if not os.path.isdir(runs_dir):
		return []
	return [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]


def parse_key(key: str) -> dict:
	"""Split a run key back into its constituent provider names and stage number.

    Examples:
        parse_key("deepgram")               -> {transcriber: "deepgram", llm: None, video_gen: None, stage: 1}
        parse_key("deepgram__claude")       -> {transcriber: "deepgram", llm: "claude", video_gen: None, stage: 2}
        parse_key("deepgram__claude__luma") -> {transcriber: "deepgram", llm: "claude", video_gen: "luma", stage: 3}
    """
	parts = key.split(SEP)
	return {
		"transcriber": parts[0] if len(parts) > 0 else None,
		"llm": parts[1] if len(parts) > 1 else None,
		"video_gen": parts[2] if len(parts) > 2 else None,
		"stage": len(parts),
	}
