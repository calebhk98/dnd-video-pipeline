"""
Stage 1 Transcription ,  AssemblyAI Backend
===========================================
Uses the AssemblyAI cloud service for automatic speech recognition (ASR) with
built-in speaker diarization.

Dependencies:
    pip install assemblyai requests
"""

import os
import logging
import requests
import assemblyai as aai
from typing import Dict, Any, Callable, Optional
from src.stages.stage1_transcription.base import BaseTranscriber
from src.shared.schemas import Transcript, Utterance
from src.shared.exceptions import (
	InsufficientCreditsError,
	InvalidAPIKeyError,
	ProviderUnavailableError
)

logger = logging.getLogger(__name__)


class AssemblyAITranscriber(BaseTranscriber):
	"""
    Concrete implementation of BaseTranscriber using the AssemblyAI API.

    Config keys:
        ASSEMBLYAI_API_KEY: AssemblyAI API key.
    """

	def __init__(self, config: Dict[str, Any]):
		"""Initialize the AssemblyAI transcriber."""
		self.api_key = config.get("ASSEMBLYAI_API_KEY") or os.getenv("ASSEMBLYAI_API_KEY")
		if not self.api_key:
			raise ValueError("AssemblyAI API key is missing.")

		aai.settings.api_key = self.api_key
		self.transcriber = aai.Transcriber()

	def _report_progress(self, progress_cb, uploaded, total):
		"""Helper to report progress without deep nesting."""
		if not progress_cb or total <= 0:
			return
		try:
			progress_cb(min(99, int(uploaded / total * 100)))
		except Exception:
			pass

	def _upload_file(self, filepath: str, progress_cb: Optional[Callable[[int], None]] = None) -> str:
		"""Upload file to AssemblyAI with progress reporting."""
		file_size = os.path.getsize(filepath)
		uploaded = 0
		CHUNK = 1024 * 1024

		def body_gen():
			"""Generator that reads the file in chunks and report progress during upload."""

			nonlocal uploaded
			with open(filepath, "rb") as f_in:
				while True:
					chunk = f_in.read(CHUNK)
					if not chunk:
						break
					uploaded += len(chunk)
					self._report_progress(progress_cb, uploaded, file_size)
					yield chunk

		resp = requests.post(
			"https://api.assemblyai.com/v2/upload",
			headers={"authorization": self.api_key},
			data=body_gen(),
			stream=True,
		)
		resp.raise_for_status()
		if progress_cb:
			try:
				progress_cb(100)
			except Exception:
				pass
		return resp.json()["upload_url"]

	def transcribe(self, audio_filepath: str, upload_progress_cb=None) -> Transcript:
		"""Transcribe an audio file with speaker diarization."""
		if not os.path.exists(audio_filepath):
			raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

		config = aai.TranscriptionConfig(speaker_labels=True)
		try:
			upload_url = self._upload_file(audio_filepath, upload_progress_cb)
			transcript_group = self.transcriber.transcribe(upload_url, config=config)
			if transcript_group.status == aai.TranscriptStatus.error:
				error_msg = transcript_group.error or "Unknown AssemblyAI error"
				raise Exception(f"AssemblyAI transcription failed: {error_msg}")
			return self._map_response_to_transcript(transcript_group)
		except Exception as e:
			logger.error(f"Error during AssemblyAI transcription: {e}")
			raise

	def _map_response_to_transcript(self, aai_transcript: aai.Transcript) -> Transcript:
		"""Convert AssemblyAI response to shared schema."""
		utterances = []
		# AssemblyAI returns utterance objects directly if speaker_labels=True.
		if aai_transcript.utterances:
			for utt in aai_transcript.utterances:
				utterances.append(Utterance(
					speaker=f"Speaker {utt.speaker}",
					text=utt.text,
					start=utt.start / 1000.0,
					end=utt.end / 1000.0,
				))
		return Transcript(
			audio_duration=aai_transcript.audio_duration or 0.0,
			status="completed",
			utterances=utterances,
			full_text=aai_transcript.text,
		)
