"""
Stage 1 Transcription ,  Deepgram Backend
=========================================
Uses the Deepgram Nova-2 model for pre-recorded audio transcription with
speaker diarization (``diarize=True``) and smart formatting.

Deepgram's API accepts the raw audio bytes directly (no cloud storage required)
and returns the result synchronously via a single HTTP call.

Key features used:
    - ``model="nova-2"``   ,  Deepgram's most accurate pre-recorded model.
    - ``smart_format=True`` ,  Adds punctuation, capitalization, and formatting.
    - ``diarize=True``      ,  Segments the transcript by speaker.

Output parsing strategy:
    When ``smart_format=True`` is combined with ``diarize=True``, Deepgram
    groups words into ``paragraphs`` objects with per-paragraph speaker IDs.
    We prefer this higher-level representation because it gives natural utterance
    boundaries.  If paragraphs are unavailable (e.g. older SDK versions), we fall
    back to grouping consecutive words that share the same speaker ID.

Dependencies:
    pip install deepgram-sdk
"""

from typing import Dict, Any, List
import logging
import os
import httpx
from deepgram import (
	DeepgramClient,
	PrerecordedOptions,
	FileSource,
)
from src.stages.stage1_transcription.base import BaseTranscriber
from src.shared.schemas import Transcript, Utterance

logger = logging.getLogger(__name__)


class DeepgramTranscriber(BaseTranscriber):
	"""
    Concrete implementation of BaseTranscriber using Deepgram Nova-2.

    Submits the audio file as raw bytes to Deepgram's pre-recorded transcription
    endpoint, then parses the response into the shared ``Transcript`` schema.

    Config keys:
        DEEPGRAM_API_KEY ,  Deepgram API key (**required**; no env var fallback).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Deepgram client with the provided API key.

        Args:
            config: Configuration dictionary.  Must contain ``DEEPGRAM_API_KEY``.

        Raises:
            ValueError: If ``DEEPGRAM_API_KEY`` is absent from ``config``.
        """
		self.api_key = config.get("DEEPGRAM_API_KEY")
		if not self.api_key:
			raise ValueError("DEEPGRAM_API_KEY is required in config")

		# DeepgramClient wraps the HTTP session; it is safe to reuse across calls.
		self.client = DeepgramClient(self.api_key)

	def transcribe(self, audio_filepath: str, upload_progress_cb=None) -> Transcript:
		"""
        Transcribe an audio file using Deepgram Nova-2 with speaker diarization.

        Reads the entire audio file into memory and sends it as a byte buffer to
        Deepgram's pre-recorded v1 endpoint.  The call blocks until the response
        is available (typically a few seconds for short files, longer for hours
        of audio).

        Args:
            audio_filepath: Path to the local audio file.

        Returns:
            A ``Transcript`` with diarized utterances.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            Exception: On Deepgram API errors (auth, rate limit, etc.).
        """
		if not os.path.exists(audio_filepath):
			raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

		# Read the entire audio file into memory as bytes.
		# Deepgram accepts a ``FileSource`` dict with a "buffer" key.
		with open(audio_filepath, "rb") as file:
			buffer_data = file.read()

		# Wrap in the FileSource dict structure expected by the SDK.
		payload: FileSource = {
			"buffer": buffer_data,
		}

		# Configure transcription options:
		#   nova-2     ,  Deepgram's best accuracy model for pre-recorded audio.
		#   smart_format ,  Auto-punctuation, capitalization, and number formatting.
		#   diarize      ,  Segment transcript by speaker (required for utterances).
		options = PrerecordedOptions(
			model="nova-2",
			smart_format=True,
			diarize=True,
		)

		# Deepgram's SDK offers no incremental upload-progress hooks; signal 100% now
		# (file is fully buffered) so the frontend transitions to "Transcribing..." status.
		if upload_progress_cb:
			try:
				upload_progress_cb(100)
			except Exception:
				pass

		try:
			# Call the pre-recorded endpoint (v1 API via .v("1")).
			# This is synchronous ,  it blocks until Deepgram returns the full response.
			# Use a long timeout (10 min) so large audio files have time to upload.
			response = self.client.listen.prerecorded.v("1").transcribe_file(
				payload, options, timeout=httpx.Timeout(600.0, connect=10.0)
			)
		except Exception as e:
			# Log differentiated error messages for common HTTP error codes.
			# Note: Deepgram SDK may raise generic exceptions depending on version.
			error_msg = str(e).lower()
			if "429" in error_msg or "rate limit" in error_msg:
				logger.warning(f"Deepgram Rate Limit Hit: {e}")
			elif "401" in error_msg or "unauthorized" in error_msg:
				logger.error(f"Deepgram Authentication Error: {e}")
			else:
				logger.error(f"Error during Deepgram transcription: {e}")
			raise
		return self._map_response_to_transcript(response)

	def _flush_utterance(self, utterances, text_list, speaker, start, end):
		"""Helper to avoid deep nesting when creating Utterance objects."""
		if speaker is not None:
			utterances.append(Utterance(
				speaker=f"Speaker {speaker}",
				text=" ".join(text_list),
				start=start,
				end=end,
			))

	def _map_response_to_transcript(self, response: Any) -> Transcript:
		"""
        Convert the Deepgram API response to the shared Transcript Pydantic schema.

        Deepgram response structure (simplified):
            response.metadata.duration          ,  total audio length in seconds
            response.results.channels[0]
                .alternatives[0]
                    .transcript                 ,  full concatenated text
                    .paragraphs.paragraphs[]    ,  paragraph objects (preferred)
                        .speaker                ,  numeric speaker ID
                        .start / .end           ,  paragraph timestamps
                        .sentences[].words[]    ,  individual words
                    .words[]                    ,  flat word list (fallback)
                        .word / .start / .end / .speaker

        Args:
            response: The ``PrerecordedResponse`` object returned by the SDK.

        Returns:
            A populated ``Transcript``.
        """
		metadata = response.metadata
		results = response.results

		audio_duration = metadata.duration

		utterances: List[Utterance] = []
		full_text = ""

		# Deepgram may return multiple channels; we only need the first alternative.
		if not results.channels or not results.channels[0].alternatives:
			return Transcript(audio_duration=audio_duration, status="completed", utterances=[], full_text="")

		alternative = results.channels[0].alternatives[0]
		full_text = alternative.transcript

		# Prefer paragraph-level grouping when smart_format produces it.
		if hasattr(alternative, 'paragraphs') and alternative.paragraphs:
			for para in alternative.paragraphs.paragraphs:
				speaker = str(para.speaker)
				if hasattr(para, "words") and para.words:
					text = " ".join(w.word for w in para.words)
				elif para.sentences:
					text = " ".join(s.text for s in para.sentences)
				else:
					text = ""
				utterances.append(Utterance(speaker=f"Speaker {speaker}", text=text, start=para.start, end=para.end))
			return Transcript(audio_duration=audio_duration, status="completed", utterances=utterances, full_text=full_text)

		# Fallback: walk the flat word list and group consecutive words.
		current_speaker = None
		current_text = []
		current_start = 0.0
		current_end = 0.0

		for word in alternative.words:
			speaker = str(getattr(word, 'speaker', '0'))
			if speaker != current_speaker:
				self._flush_utterance(utterances, current_text, current_speaker, current_start, current_end)
				current_speaker = speaker
				current_text = [word.word]
				current_start = word.start
			else:
				current_text.append(word.word)
			current_end = word.end

		self._flush_utterance(utterances, current_text, current_speaker, current_start, current_end)

		return Transcript(
			audio_duration=audio_duration,
			status="completed",
			utterances=utterances,
			full_text=full_text,
		)
