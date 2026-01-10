"""
Stage 1 Transcription ,  Rev.ai Backend
=======================================
Uses the Rev.ai cloud API for human-quality audio transcription with
speaker diarization.

Rev.ai workflow:
    1. Submit the local audio file as an async transcription job.
    2. Poll ``get_job_details`` every ``polling_interval`` seconds until the
       job status becomes ``"completed"`` or ``"failed"``.
    3. Retrieve the transcript JSON via ``get_transcript_json``.
    4. Map the ``monologues`` array (each monologue is one speaker's turn) to
       the shared ``Utterance`` list.

Rev.ai JSON structure:
    The transcript JSON contains a top-level ``"monologues"`` array.  Each
    monologue has:
        speaker   ,  integer speaker ID
        elements  ,  list of dicts, each either {"type": "text", "value": "word",
                    "ts": 1.23, "end_ts": 1.45} or {"type": "punct", "value": ","}

Dependencies:
    pip install rev-ai
"""

import os
import time
import logging
from typing import Dict, Any, List
from rev_ai import apiclient
from src.stages.stage1_transcription.base import BaseTranscriber
from src.shared.schemas import Transcript, Utterance

logger = logging.getLogger(__name__)


class RevAiTranscriber(BaseTranscriber):
	"""
    Concrete implementation of BaseTranscriber using the Rev.ai async API.

    Submits an audio file to Rev.ai, polls for job completion, and maps
    the monologue-based JSON response to the shared ``Transcript`` schema.

    Config keys:
        api_key          ,  Rev.ai API key (**required**).
        polling_interval ,  Seconds between job status checks (default: 5).
        timeout          ,  Maximum seconds to wait for job completion (default: 600).
        language         ,  BCP-47 language code for transcription (default: ``"en"``).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Rev.ai API client.

        Args:
            config: Configuration dictionary.  See class docstring for keys.

        Raises:
            ValueError: If ``api_key`` is absent from ``config``.
        """
		self.api_key = config.get("api_key")
		if not self.api_key:
			raise ValueError("Rev AI API key is required in config.")

		# RevAiAPIClient handles HTTP auth and endpoint routing.
		self.client = apiclient.RevAiAPIClient(self.api_key)

		# How many seconds to wait between polling for job completion.
		self.polling_interval = config.get("polling_interval", 5)

		# Maximum seconds to wait for job completion before raising TimeoutError.
		self.timeout = config.get("timeout", 600)

		# BCP-47 language code; "en" covers US/UK English.
		self.language = config.get("language", "en")

	def transcribe(self, audio_filepath: str, upload_progress_cb=None) -> Transcript:
		"""
        Submit an audio file to Rev.ai, wait for completion, and return a Transcript.

        This method blocks until the Rev.ai job finishes (or raises on failure).
        Job processing time varies with audio length ,  typically 15-50% of audio
        duration for standard async jobs.

        Args:
            audio_filepath: Path to the local audio file to transcribe.

        Returns:
            A ``Transcript`` with speaker-labeled utterances sorted by start time.

        Raises:
            FileNotFoundError: Implicitly if the file doesn't exist (Rev.ai will error).
            Exception: If the Rev.ai job fails or an API error occurs.
        """
		# Step 1: Upload the file and start the transcription job asynchronously.
		# submit_job_local_file reads the file and POSTs it to Rev.ai.
		job = self.client.submit_job_local_file(audio_filepath, language=self.language)
		job_id = job.id

		# Step 2: Poll until the job reaches a terminal state (transcribed/completed or failed).
		# Rev AI SDK JobStatus enum defines four values: in_progress, transcribed, completed, failed.
		# "transcribed" is the terminal state for machine transcription (the common case).
		# "completed" also signals success (e.g. after human review). Both mean the transcript is ready.
		start_time = time.monotonic()
		while True:
			elapsed = time.monotonic() - start_time
			if elapsed >= self.timeout:
				raise TimeoutError(
					f"Rev AI job {job_id} timed out after {self.timeout}s"
				)

			try:
				job_details = self.client.get_job_details(job_id)
			except Exception as poll_err:
				logger.warning(
					"Rev AI poll error for job %s (elapsed %.1fs): %s retrying in %ss",
					job_id, elapsed, poll_err, self.polling_interval,
				)
				time.sleep(self.polling_interval)
				continue

			# Normalise to plain string so enum comparison is never ambiguous.
			status_obj = job_details.status
			status_str = status_obj.value if hasattr(status_obj, "value") else str(status_obj)

			logger.info("Rev AI job %s status: %s (elapsed: %.1fs)", job_id, status_str, elapsed)

			if status_str in ("completed", "transcribed"):
				break  # Job finished successfully; proceed to retrieve transcript.
			elif status_str == "failed":
				logger.error(
					"Rev AI transcription job %s failed: %s",
					job_id,
					job_details.failure_detail,
				)
				raise Exception(
					f"Rev AI transcription job {job_id} failed: {job_details.failure_detail}"
				)

			# Job is still in_progress or queued wait before checking again.
			time.sleep(self.polling_interval)

		# Step 3: Download the transcript in Rev.ai's JSON format (monologue-based).
		transcript_json = self.client.get_transcript_json(job_id)

		# Step 4: Convert Rev.ai's monologue format to our standard Utterance list.
		utterances = self._map_to_utterances(transcript_json)

		# Use the API-reported duration if available; otherwise fall back to the
		# end time of the last utterance (less accurate but always present).
		duration = getattr(job_details, "duration_seconds", 0.0)
		if duration == 0.0 and utterances:
			duration = utterances[-1].end

		full_text = " ".join([u.text for u in utterances])

		return Transcript(
			audio_duration=float(duration),
			status="completed",
			utterances=utterances,
			full_text=full_text,
		)

	def _map_to_utterances(self, rev_json: Dict[str, Any]) -> List[Utterance]:
		"""
        Convert Rev.ai's monologue JSON format to a list of Utterance objects.

        Rev.ai organizes transcripts as *monologues* ,  contiguous segments where
        a single speaker is active.  Each monologue's ``elements`` list mixes
        text tokens (with timestamps) and punctuation tokens (without timestamps).

        Processing per monologue:
            - Collect ``"text"`` elements to build the utterance text string.
            - Collect ``"punct"`` elements and append them directly to text (no space).
            - Track the first ``ts`` (start) and maximum ``end_ts`` (end).

        Args:
            rev_json: Parsed JSON dict from ``get_transcript_json``.

        Returns:
            A list of ``Utterance`` objects sorted by start time.
        """
		standard_utterances = []

		for monologue in rev_json.get("monologues", []):
			speaker = str(monologue.get("speaker", "unknown"))
			elements = monologue.get("elements", [])

			text_parts = []
			start_time = None  # Will be set from the first text element with a timestamp.
			end_time = 0.0

			for elem in elements:
				val = elem.get("value", "")
				ts = elem.get("ts")       # Start timestamp (float seconds), only on text.
				end_ts = elem.get("end_ts")  # End timestamp (float seconds), only on text.

				if elem.get("type") == "text":
					# Add a leading space before subsequent words for readability.
					if text_parts and val:
						text_parts.append(" " + val)
					else:
						text_parts.append(val)
				elif elem.get("type") == "punct":
					# Punctuation attaches directly to the preceding word (no space).
					text_parts.append(val)

				# Record the start time from the first element that has a timestamp.
				if ts is not None and start_time is None:
					start_time = ts
				# Track the maximum end_ts to get the true end of this monologue.
				if end_ts is not None:
					end_time = max(end_time, end_ts)

			# Default start to 0.0 if no element had a timestamp (e.g. silent monologue).
			if start_time is None:
				start_time = 0.0

			combined_text = "".join(text_parts).strip()

			# Only add non-empty utterances to avoid empty turns in the transcript.
			if combined_text:
				standard_utterances.append(Utterance(
					speaker=speaker,
					text=combined_text,
					start=float(start_time),
					end=float(end_time),
				))

		# Sort by start time to ensure chronological ordering.
		# Rev.ai monologues are usually ordered but sorting is a safety measure.
		standard_utterances.sort(key=lambda x: x.start)

		return standard_utterances
