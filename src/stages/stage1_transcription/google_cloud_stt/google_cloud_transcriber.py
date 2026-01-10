"""
Stage 1 Transcription ,  Google Cloud Speech-to-Text Backend
============================================================
Uses the Google Cloud Speech-to-Text API (Chirp model family) for audio
transcription with speaker diarization.

Why ``long_running_recognize``:
    Google's ``recognize`` endpoint only handles audio up to 1 minute.
    D&D sessions are typically hours long, so we always use
    ``long_running_recognize``, which returns a long-polling Operation that
    we wait on via ``operation.result(timeout=600)``.

Speaker diarization:
    Google STT attaches speaker tags to the *words* in the *last* result's
    alternative.  We walk those words, grouping consecutive words from the
    same ``speaker_tag`` integer into utterances.

Authentication options:
    1. Explicit ``api_key`` in config / ``GOOGLE_CLOUD_API_KEY`` env var , 
       passed via ``ClientOptions``.
    2. Application Default Credentials (ADC) ,  when no key is provided, the
       client uses ``GOOGLE_APPLICATION_CREDENTIALS`` or the gcloud login.

Dependencies:
    pip install google-cloud-speech
"""

import os
import logging
import time
from typing import Dict, Any
from google.cloud import speech
from src.stages.stage1_transcription.base import BaseTranscriber
from src.shared.schemas import Transcript, Utterance

logger = logging.getLogger(__name__)


class GoogleCloudTranscriber(BaseTranscriber):
	"""
    Concrete implementation of BaseTranscriber using Google Cloud Speech-to-Text.

    Uses the Chirp 2 model by default, which is Google's latest long-form
    transcription model.  Speaker diarization is enabled via
    ``SpeakerDiarizationConfig``.

    Config keys:
        api_key      ,  Google Cloud API key (optional; falls back to
                        GOOGLE_CLOUD_API_KEY env var, then ADC).
        model        ,  Speech recognition model name (default: ``"chirp_2"``).
        max_speakers ,  Maximum number of distinct speakers (default: 6).
    """

	# Default to Chirp 2, Google's most accurate long-form model.
	DEFAULT_MODEL = "chirp_2"

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Google Cloud Speech client with diarization settings.

        Args:
            config: Configuration dictionary.  See class docstring for keys.
        """
		self.model = config.get("model", self.DEFAULT_MODEL)
		self.max_speakers = config.get("max_speakers", 6)

		# Resolve authentication: prefer explicit API key, then fall back to ADC.
		api_key = config.get("api_key") or os.getenv("GOOGLE_CLOUD_API_KEY")
		if api_key:
			# When an API key is available, pass it through ClientOptions so
			# the SDK includes it in every request header.
			from google.api_core.client_options import ClientOptions
			self.client = speech.SpeechClient(
				client_options=ClientOptions(api_key=api_key)
			)
		else:
			# Fall back to Application Default Credentials (gcloud auth / service account).
			self.client = speech.SpeechClient()

	def transcribe(self, audio_filepath: str, upload_progress_cb=None) -> Transcript:
		"""
        Transcribe an audio file with speaker diarization using Google Cloud STT.

        Reads the entire audio into memory and sends it as inline content via
        ``long_running_recognize``, which handles files of arbitrary length.
        We wait up to 600 seconds (10 minutes) for the operation to complete.

        Args:
            audio_filepath: Path to the audio file (MP3 assumed; 44100 Hz).

        Returns:
            A ``Transcript`` with speaker-diarized utterances.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            Exception: On Google Cloud API errors or timeout.
        """
		if not os.path.exists(audio_filepath):
			raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

		# Load the entire audio file into memory as bytes for inline content.
		# For very large files, consider using a GCS URI instead to avoid memory pressure.
		with open(audio_filepath, "rb") as f:
			audio_content = f.read()

		audio = speech.RecognitionAudio(content=audio_content)

		# Configure diarization: identify up to max_speakers distinct voices.
		# Providing a reasonable max_speaker_count helps the model tune its clustering.
		diarization_config = speech.SpeakerDiarizationConfig(
			enable_speaker_diarization=True,
			max_speaker_count=self.max_speakers,
		)

		# Recognition configuration:
		#   encoding		,  MP3 is the expected format from the pipeline.
		#   sample_rate_hertz ,  Standard CD quality; adjust if source differs.
		#   language_code   ,  D&D campaigns are assumed to be in English (US).
		#   model           ,  Chirp 2 for best long-form accuracy.
		#   diarization_config ,  Enables speaker identification.
		#   enable_automatic_punctuation ,  Improves readability of transcript text.
		config = speech.RecognitionConfig(
			encoding=speech.RecognitionConfig.AudioEncoding.MP3,
			sample_rate_hertz=44100,
			language_code="en-US",
			model=self.model,
			diarization_config=diarization_config,
			enable_automatic_punctuation=True,
		)

		try:
			# long_running_recognize returns an Operation; .result() blocks until done.
			# timeout=600 raises google.api_core.exceptions.DeadlineExceeded if exceeded.
			operation = self.client.long_running_recognize(config=config, audio=audio)
			response = operation.result(timeout=600)
			return self._map_response_to_transcript(response)
		except Exception as e:
			logger.error(f"Error during Google Cloud STT transcription: {str(e)}")
			raise

	def _flush_utterance(self, utterances, words, speaker, start, end):
		"""Helper to avoid deep nesting when creating Utterance objects."""
		if words and speaker is not None:
			utterances.append(Utterance(
				speaker=f"Speaker {speaker}",
				text=" ".join(words),
				start=start,
				end=end,
			))

	def _map_response_to_transcript(self, response) -> Transcript:
		"""
        Parse the Google Cloud STT response into the shared Transcript schema.

        Google STT quirk ,  diarization data placement:
            Speaker tags are only available on the *words* array of the *last*
            result's first alternative.  Earlier results may have words without
            speaker_tag set.  We iterate over all results/alternatives to collect
            full_text, but use word-level speaker_tag for diarization.

        Grouping logic:
            Walk words sequentially.  When ``speaker_tag`` changes, flush the
            accumulated words as a new Utterance and start a fresh run.

        Args:
            response: The ``LongRunningRecognizeResponse`` from the Google SDK.

        Returns:
            A ``Transcript`` with ordered, speaker-labeled utterances.
        """
		utterances = []
		full_text_parts = []

		for result in response.results:
			alternative = result.alternatives[0]
			full_text_parts.append(alternative.transcript)

			# Only results that have word-level info with speaker_tag are usable for diarization.
			if not alternative.words:
				continue

			current_speaker = None
			current_words = []
			current_start = None
			current_end = None

			for word_info in alternative.words:
				# speaker_tag is an integer (1-based); default to 1 if absent.
				speaker_tag = getattr(word_info, "speaker_tag", 1)
				start_sec = word_info.start_time.total_seconds()
				end_sec = word_info.end_time.total_seconds()

				if speaker_tag != current_speaker:
					self._flush_utterance(
						utterances, current_words, current_speaker,
						current_start, current_end
					)
					# Reset
					current_speaker = speaker_tag
					current_words = [word_info.word]
					current_start = start_sec
					current_end = end_sec
				else:
					current_words.append(word_info.word)
					current_end = end_sec

			# Flush the last accumulated utterance for this result.
			self._flush_utterance(
				utterances, current_words, current_speaker,
				current_start, current_end
			)

		# Derive total duration from the end time of the last utterance.
		audio_duration = utterances[-1].end if utterances else 0.0

		return Transcript(
			audio_duration=audio_duration,
			status="completed",
			utterances=utterances,
			full_text=" ".join(full_text_parts),
		)
