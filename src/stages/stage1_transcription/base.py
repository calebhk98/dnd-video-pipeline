"""
Stage 1: Transcription ,  Abstract Base Class
=============================================
This module defines the interface that every transcription backend must implement.
Transcription is the first stage of the D&D audio-to-video pipeline:

    Audio file  ->  [Stage 1: Transcribe]  ->  Transcript (with speaker diarization)

All concrete transcribers (Amazon, AssemblyAI, Deepgram, WhisperX, etc.) inherit
from BaseTranscriber and return the same standardized ``Transcript`` Pydantic schema,
which is then consumed by Stage 2 (LLM processing).

Adding a new transcription backend:
    1. Create a subdirectory under ``stage1_transcription/``.
    2. Subclass ``BaseTranscriber`` and implement ``__init__`` and ``transcribe``.
    3. Return a ``Transcript`` object from ``transcribe()``.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from src.shared.schemas import Transcript


class BaseTranscriber(ABC):
	"""
    Abstract Base Class for Stage 1 Transcription backends.

    Each subclass wraps a specific transcription service or local model and
    normalizes its output into the shared ``Transcript`` schema.  The pipeline
    only interacts with this interface, so backends are interchangeable.

    Responsibilities of a concrete transcriber:
        - Accept an audio file path and transcription-related config values.
        - Call the underlying API or local model.
        - Perform (or delegate) speaker diarization to identify *who* said *what*.
        - Return a ``Transcript`` containing a list of ``Utterance`` objects with
          speaker labels, timestamps, and text.
    """

	@abstractmethod
	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the transcriber with API credentials or model configuration.

        This method is responsible for:
            - Extracting required keys from ``config`` or falling back to
              environment variables (e.g. ``os.getenv("API_KEY")``).
            - Instantiating the API client or loading the local model into memory.
            - Raising ``ValueError`` for missing required configuration.

        Args:
            config: A flat dictionary of configuration values.  Keys vary by
                    implementation ,  see each subclass docstring for details.

        Raises:
            ValueError: If a required configuration key is absent.
        """
		pass

	@abstractmethod
	def transcribe(self, audio_filepath: str, upload_progress_cb=None) -> Transcript:
		"""
        Transcribe an audio file and perform speaker diarization.

        This is the primary public method that the pipeline calls.  Implementations
        must handle:
            - Validating that the file exists before attempting upload or loading.
            - Invoking the transcription service/model (possibly async with polling).
            - Parsing the service-specific response format into standard types.
            - Converting speaker labels to the "Speaker N" convention used throughout
              the pipeline (e.g. ``spk_0`` -> ``"Speaker 0"``).
            - Converting timestamps to floating-point *seconds* (some APIs return
              milliseconds and require ``/ 1000.0``).

        Args:
            audio_filepath: Absolute or relative path to the audio file on disk.
                Common formats: mp3, wav, m4a, ogg.
            upload_progress_cb: Optional synchronous callable(percent: int) called
                during the upload phase with 0-100 progress values.
                Implementations that do not support progress reporting
                can safely ignore this argument.

        Returns:
            A ``Transcript`` Pydantic object containing:
                - ``utterances``: Ordered list of ``Utterance`` objects, each with
                  speaker, text, start, and end (in seconds).
                - ``full_text``: Concatenated transcript text (no speaker labels).
                - ``audio_duration``: Total length of the audio in seconds.
                - ``status``: Always ``"completed"`` on success.

        Raises:
            FileNotFoundError: If ``audio_filepath`` does not exist.
            RuntimeError: If the transcription service reports a failure.
            TimeoutError: If the service does not respond within the allowed window.
        """
		pass
