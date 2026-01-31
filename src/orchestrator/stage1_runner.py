"""Stage 1 Entry Point - Transcription
======================================
Runs audio conversion and transcription, then optionally asks an LLM for
speaker-name suggestions.  Saves ``transcript.json`` (and optionally
``speaker_map_suggestions.json``) to ``output_dir``.
"""

import asyncio
import datetime
import json
import logging
import shutil
from pathlib import Path
from typing import Callable, Awaitable, Optional

from src.shared.schemas import Transcript, Utterance
from src.shared.utils.audio_preprocessor import prepare_audio
from src.orchestrator.providers import (
	_compute_file_hash,
	_get_transcriber,
	_get_llm_processor,
)

logger = logging.getLogger(__name__)


async def run_stage1(
	audio_path: str,
	transcriber_name: str,
	output_dir: str,
	progress_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
	preprocess_audio: bool = False,
	max_chunk_minutes: float = 90.0,
	llm_name: Optional[str] = None,
	force_rerun: bool = False,
	ver_count: int = 0,
):
	"""Run Stage 1 (Transcription) only and save transcript.json to output_dir.

    If llm_name is supplied, also calls map_speakers() after transcription to
    produce speaker name suggestions, saved to speaker_map_suggestions.json and
    broadcast via a ``speaker_suggestions`` WebSocket event.

    If force_rerun is True, existing outputs are backed up to .versions/{ver_count}/
    and regenerated from scratch.
    """
	out_path = Path(output_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	_CACHED_FILES = ["transcript.json", "speaker_map_suggestions.json"]

	cached = (out_path / "transcript.json").exists()

	# Skip if transcript already exists for this exact provider combination
	if cached and not force_rerun:
		logger.info(f"Stage 1: transcript.json already exists at {out_path}, skipping transcription")
		if progress_callback:
			await progress_callback({"status": "stage_complete", "stage": "stage1", "detail": "Using cached transcript"})
		# Still emit speaker suggestions from cached file if LLM requested
		if llm_name:
			suggestions_path = out_path / "speaker_map_suggestions.json"
			if suggestions_path.exists():
				with open(suggestions_path) as f:
					suggestions = json.load(f)
				if progress_callback:
					await progress_callback({"status": "speaker_suggestions", "suggestions": suggestions})
			else:
				if progress_callback:
					await progress_callback({"status": "speaker_suggestions", "suggestions": {}})
		return

	if cached and force_rerun:
		# Back up existing files to .versions/{ver_count}/ before overwriting
		ver_dir = out_path / ".versions" / str(ver_count)
		ver_dir.mkdir(parents=True, exist_ok=True)
		for fname in _CACHED_FILES:
			src = out_path / fname
			if src.exists():
				shutil.copy2(str(src), str(ver_dir / fname))
				logger.info(f"Stage 1: backed up {fname} -> .versions/{ver_count}/{fname}")

	logger.info(f"Stage 1: Transcription with {transcriber_name}")

	if progress_callback:
		await progress_callback({
			"status": "stage_started",
			"stage": "stage1",
			"timestamp": datetime.datetime.utcnow().isoformat(),
		})

	# -- Step 1a: Audio format conversion --------------------------------------
	# If the uploaded file is not already a WAV, convert it to 16 kHz mono WAV
	# so all transcribers receive a consistent format.
	#
	# Converted files are stored in a shared cache keyed by the SHA-256 hash of
	# the source file: outputs/wav_cache/{hash}.wav.  This lets any session that
	# uploads the same audio file reuse the cached WAV instead of reconverting.
	# The hash is written to {session_dir}/input_audio.hash so the API can
	# reconstruct the cache URL when serving the audio player.
	audio_to_transcribe = audio_path
	file_ext = Path(audio_path).suffix.lower()
	if file_ext not in ('.wav',):
		wav_cache_dir = out_path.parent / "wav_cache"
		wav_cache_dir.mkdir(parents=True, exist_ok=True)

		file_hash = _compute_file_hash(audio_path)
		cached_wav = wav_cache_dir / f"{file_hash}.wav"
		# Persist the hash so the transcript API can build the audio player URL
		(out_path / "input_audio.hash").write_text(file_hash)

		if cached_wav.exists():
			audio_to_transcribe = str(cached_wav)
			logger.info(f"Reusing cached WAV ({file_hash[:12]}...): {cached_wav}")
			if progress_callback:
				await progress_callback({
					"status": "wav_ready",
					"stage": "1/4: Transcription",
					"detail": "Reusing previously converted WAV file...",
					"wav_url": f"/outputs/wav_cache/{file_hash}.wav",
				})
		else:
			if progress_callback:
				await progress_callback({
					"status": "processing",
					"stage": "1/4: Transcription",
					"detail": f"Converting {file_ext.lstrip('.')} -> WAV...",
				})
			try:
				from pydub import AudioSegment
				def _do_convert():
					"""Performs the actual audio conversion using Pydub."""

					seg = AudioSegment.from_file(audio_path)
					seg = seg.set_frame_rate(16000).set_channels(1)
					seg.export(str(cached_wav), format="wav")
				await asyncio.to_thread(_do_convert)
				audio_to_transcribe = str(cached_wav)
				logger.info(f"Converted {file_ext} -> WAV: {cached_wav}")
				if progress_callback:
					await progress_callback({
						"status": "wav_ready",
						"stage": "1/4: Transcription",
						"detail": "Audio converted -- transcription starting...",
						"wav_url": f"/outputs/wav_cache/{file_hash}.wav",
					})
			except Exception as conv_err:
				# Conversion is best-effort; fall back to the original file
				logger.warning(f"Audio conversion failed (will try original file): {conv_err}")
				audio_to_transcribe = audio_path

	if progress_callback:
		await progress_callback({
			"status": "processing",
			"stage": "1/4: Transcription",
			"detail": f"Uploading audio to {transcriber_name}...",
		})

	try:
		transcriber = _get_transcriber(transcriber_name)
	except ValueError as e:
		logger.error(f"Failed to load transcriber: {e}")
		if progress_callback:
			await progress_callback({"status": "error", "stage": "1/4: Transcription", "detail": str(e)})
		return

	# Build a thread-safe upload progress callback that posts WS updates from
	# the worker thread back to the asyncio event loop without blocking it.
	_event_loop = asyncio.get_event_loop()

	def _make_upload_cb():
		"""Factory for creating a debounced upload progress callback."""

		_last_pct = [-1]

		def _upload_cb(pct: int):
			"""Internal callback to handle percentage updates and broadcast via WS."""

			if pct == _last_pct[0]:
				return  # debounce identical values
			_last_pct[0] = pct
			if pct < 100:
				detail = f"Uploading to {transcriber_name}... {pct}%"
			else:
				detail = f"Transcribing audio via {transcriber_name}..."
			status_msg = {
				"status": "processing",
				"stage": "1/4: Transcription",
				"detail": detail,
				"percent": pct,
			}
			try:
				asyncio.run_coroutine_threadsafe(progress_callback(status_msg), _event_loop)
			except Exception:
				pass

		return _upload_cb

	upload_cb = _make_upload_cb() if progress_callback else None

	try:
		if preprocess_audio:
			chunks_dir = str(out_path / "audio_chunks")
			manifest = prepare_audio(audio_to_transcribe, chunks_dir, max_duration_minutes=max_chunk_minutes)
			chunk_transcripts = []
			for chunk in manifest:
				ct = await asyncio.to_thread(transcriber.transcribe, chunk["filepath"], upload_cb)
				offset_s = chunk["global_start_ms"] / 1000.0
				adjusted = [
					Utterance(speaker=u.speaker, text=u.text, start=u.start + offset_s, end=u.end + offset_s)
					for u in ct.utterances
				]
				chunk_transcripts.append((ct, adjusted, chunk["global_end_ms"] / 1000.0))
			all_utterances = [u for _, ulist, _ in chunk_transcripts for u in ulist]
			full_text = "\n".join(ct.full_text for ct, _, _ in chunk_transcripts)
			audio_duration = chunk_transcripts[-1][2] if chunk_transcripts else 0.0
			transcript = Transcript(
				audio_duration=audio_duration, status="completed",
				utterances=all_utterances, full_text=full_text
			)
		else:
			transcript = await asyncio.to_thread(transcriber.transcribe, audio_to_transcribe, upload_cb)

		with open(out_path / "transcript.json", "w") as f:
			f.write(transcript.model_dump_json(indent=2))
		logger.info(f"Transcription complete. Saved to {out_path / 'transcript.json'}")

		if progress_callback:
			await progress_callback({
				"status": "stage_complete",
				"stage": "stage1",
				"detail": "Transcription complete",
				"timestamp": datetime.datetime.utcnow().isoformat(),
			})

	except Exception as e:
		logger.error(f"Error during transcription: {e}")
		if progress_callback:
			await progress_callback({"status": "error", "stage": "1/4: Transcription", "detail": str(e)})
		return

	# -- Step 1b: LLM speaker-name suggestions (optional, non-fatal) -----------
	# Ask the LLM to infer human-readable names from the raw speaker IDs so the
	# UI can pre-fill the speaker-mapping form.  This runs after transcription
	# completes so the transcript is visible in the UI without waiting.
	if llm_name:
		if progress_callback:
			await progress_callback({
				"status": "processing",
				"stage": "1/4: Transcription",
				"detail": "Getting speaker name suggestions from LLM...",
			})
		try:
			llm_proc = _get_llm_processor(llm_name)
			suggestions = llm_proc.map_speakers(transcript)
			with open(out_path / "speaker_map_suggestions.json", "w") as f:
				json.dump(suggestions, f, indent=2)
			logger.info(f"Speaker suggestions: {suggestions}")
			if progress_callback:
				await progress_callback({
					"status": "speaker_suggestions",
					"suggestions": suggestions,
				})
		except Exception as sugg_err:
			logger.warning(f"Speaker suggestions failed (non-fatal): {sugg_err}")
			# Broadcast empty suggestions so the frontend unblocks the form
			if progress_callback:
				await progress_callback({"status": "speaker_suggestions", "suggestions": {}})
