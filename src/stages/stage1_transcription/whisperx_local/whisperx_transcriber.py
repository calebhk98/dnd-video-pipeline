"""
Stage 1 Transcription ,  WhisperX Local Backend (whisperx_local)
================================================================
Full WhisperX pipeline with configurable speaker count bounds and lazy import.

This implementation is structurally similar to ``whisper_local`` but adds:
    - Optional ``min_speakers`` / ``max_speakers`` hints to the diarizer,
      which can improve accuracy when the number of D&D players is known.
    - Lazy ``import whisperx`` inside methods so the module can be imported
      even if whisperx is not installed (fails at use time, not import time).
    - Explicit speaker label normalization: ``"SPEAKER_00"`` -> ``"Speaker 0"``.

Three-stage pipeline:
    1. **Transcribe** ,  Whisper produces rough text segments with chunk timestamps.
    2. **Align**      ,  Phoneme-level alignment refines word timestamps.
    3. **Diarize**    ,  PyAnnote identifies speaker turns (needs HF token).

Dependencies:
    pip install whisperx torch
"""

import os
import logging
from typing import Dict, Any
from src.stages.stage1_transcription.base import BaseTranscriber
from src.shared.schemas import Transcript, Utterance

logger = logging.getLogger(__name__)


class WhisperXTranscriber(BaseTranscriber):
	"""
    Concrete implementation of BaseTranscriber using the full WhisperX pipeline.

    WhisperX extends OpenAI Whisper with:
        - Faster batched inference via faster-whisper (CTranslate2 backend).
        - Word-level forced alignment using wav2vec2 phoneme models.
        - Speaker diarization via PyAnnote.audio.

    Config keys / env vars:
        model        ,  Whisper model size: ``"tiny"``, ``"base"``, ``"small"``,
                        ``"medium"``, ``"large-v2"`` (default: ``"large-v2"``).
        device       ,  ``"cpu"`` or ``"cuda"`` (default: ``"cuda"`` if available, else ``"cpu"``).
        compute_type ,  ``"int8"``, ``"float16"``, or ``"float32"``
                        (default: ``"float16"`` on CUDA, ``"int8"`` on CPU).
        batch_size   ,  Batch size for Whisper inference (default: 16).
        hf_token / HUGGING_FACE_TOKEN ,  HuggingFace token for PyAnnote models.
        min_speakers ,  Minimum expected speakers for diarizer hint (optional).
        max_speakers ,  Maximum expected speakers for diarizer hint (optional).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Load the WhisperX model at initialization time.

        whisperx is imported lazily here to avoid hard dependency errors when
        the package is not installed.  The ``# noqa: F401`` suppresses the
        "imported but unused" lint warning since we verify the import works.

        Args:
            config: Configuration dictionary.  See class docstring for keys.
        """
		try:
			import torch
		except ImportError:
			torch = None

		import whisperx  # noqa: F401  ,  verifies whisperx is installed.

		self.model_name = config.get("model", "large-v2")
		# Auto-detect GPU; fall back to CPU if CUDA is not available.
		self.device = config.get("device", "cuda" if torch and torch.cuda.is_available() else "cpu")
		# float16 is fast and accurate on GPU; int8 is quantized for CPU efficiency.
		self.compute_type = config.get("compute_type", "float16" if self.device == "cuda" else "int8")
		self.batch_size = config.get("batch_size", 16)

		# HF token is required to download PyAnnote's gated diarization model.
		self.hf_token = config.get("hf_token") or os.getenv("HUGGING_FACE_TOKEN")

		# Optional speaker count hints; None means auto-detect.
		self.min_speakers = config.get("min_speakers", None)
		self.max_speakers = config.get("max_speakers", None)

		# Load the Whisper model once at init to avoid reloading on every call.
		self.model = whisperx.load_model(
			self.model_name,
			device=self.device,
			compute_type=self.compute_type,
		)

	def transcribe(self, audio_filepath: str, upload_progress_cb=None) -> Transcript:
		"""
        Run the three-stage WhisperX pipeline and return a Transcript.

        Stage 1 ,  Whisper transcription:
            Produces text segments with start/end timestamps (chunk-level precision).

        Stage 2 ,  Alignment:
            Refines timestamps to word level using a language-specific wav2vec2 model.
            ``return_char_alignments=False`` skips character-level data to save memory.

        Stage 3 ,  Diarization (conditional on HF token):
            If ``hf_token`` is set, uses PyAnnote to identify speaker turns.
            ``min_speakers`` / ``max_speakers`` constrain the clustering if known.
            ``assign_word_speakers`` attaches speaker IDs to each Whisper segment.
            If no token is available, segments keep their default ``"SPEAKER_00"`` label.

        Args:
            audio_filepath: Path to the local audio file.

        Returns:
            A ``Transcript`` with per-segment speaker labels, text, and timestamps.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            Exception: On model inference or API errors.
        """
		import whisperx

		if not os.path.exists(audio_filepath):
			raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

		try:
			# Stage 1: Transcribe ,  load audio at 16 kHz mono and run Whisper batched.
			audio = whisperx.load_audio(audio_filepath)
			result = self.model.transcribe(audio, batch_size=self.batch_size)
			language = result.get("language", "en")

			# Stage 2: Align ,  improve timestamp precision to the word level.
			align_model, metadata = whisperx.load_align_model(
				language_code=language, device=self.device
			)
			result = whisperx.align(
				result["segments"],
				align_model,
				metadata,
				audio,
				self.device,
				return_char_alignments=False,
			)

			# Stage 3: Diarize ,  identify who is speaking when (requires HF token).
			if self.hf_token:
				diarize_model = whisperx.DiarizationPipeline(
					use_auth_token=self.hf_token, device=self.device
				)
				# Pass optional speaker count hints to improve diarization accuracy.
				# Omit keys whose values are None so the diarizer uses auto-detection.
				diarize_kwargs = {}
				if self.min_speakers:
					diarize_kwargs["min_speakers"] = self.min_speakers
				if self.max_speakers:
					diarize_kwargs["max_speakers"] = self.max_speakers

				diarize_segments = diarize_model(audio, **diarize_kwargs)
				# Merge diarization output with transcription segments.
				result = whisperx.assign_word_speakers(diarize_segments, result)

			return self._map_result_to_transcript(result)

		except Exception as e:
			logger.error(f"Error during WhisperX transcription: {str(e)}")
			raise

	def _map_result_to_transcript(self, result: dict) -> Transcript:
		"""
        Map WhisperX output segments to the shared Transcript Pydantic schema.

        WhisperX segment keys:
            text    ,  transcribed text for this segment (may have leading whitespace).
            start   ,  segment start time in seconds (float).
            end     ,  segment end time in seconds (float).
            speaker ,  speaker label string (e.g. ``"SPEAKER_00"``); absent if diarization
                       was skipped, in which case we default to ``"SPEAKER_00"``.

        Speaker label normalization:
            WhisperX uses the format ``"SPEAKER_NN"`` (zero-padded two digits).
            We strip the prefix and leading zeros so ``"SPEAKER_00"`` becomes
            ``"Speaker 0"`` and ``"SPEAKER_01"`` becomes ``"Speaker 1"``.

        Empty segments (blank ``text``) are skipped to avoid noise in the output.

        Args:
            result: The dict returned by WhisperX after all three stages.

        Returns:
            A ``Transcript`` with normalized speaker labels and second-precision timestamps.
        """
		utterances = []
		full_text_parts = []
		segments = result.get("segments", [])

		for segment in segments:
			text = segment.get("text", "").strip()
			start = float(segment.get("start", 0.0))
			end = float(segment.get("end", 0.0))

			# Default speaker label when diarization was not run.
			speaker_label = segment.get("speaker", "SPEAKER_00")

			# Normalize "SPEAKER_00" -> "Speaker 0", "SPEAKER_01" -> "Speaker 1", etc.
			# strip("SPEAKER_") then lstrip("0") handles zero-padding; "or '0'"
			# ensures the label doesn't become empty for "SPEAKER_00".
			speaker_num = speaker_label.replace("SPEAKER_", "").lstrip("0") or "0"
			speaker = f"Speaker {speaker_num}"

			if text:
				full_text_parts.append(text)
				utterances.append(Utterance(
					speaker=speaker,
					text=text,
					start=start,
					end=end,
				))

		# Derive total audio duration from the last segment's end time.
		audio_duration = utterances[-1].end if utterances else 0.0

		return Transcript(
			audio_duration=audio_duration,
			status="completed",
			utterances=utterances,
			full_text=" ".join(full_text_parts),
		)
