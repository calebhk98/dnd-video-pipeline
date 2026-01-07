"""
Stage 1 Transcription ,  WhisperX Local Backend (whisper_local)
===============================================================
Uses WhisperX ,  an enhanced version of OpenAI's Whisper ,  for fully local,
offline transcription with word-level alignment and speaker diarization.

Three-stage pipeline:
    1. **Transcribe** ,  WhisperX runs Whisper to convert audio to text segments.
    2. **Align**      ,  WhisperX aligns the text to the audio for accurate word
                        timestamps using a phoneme-level alignment model.
    3. **Diarize**    ,  WhisperX's DiarizationPipeline (PyAnnote under the hood)
                        identifies speaker turns and assigns them to segments.

HuggingFace token requirement:
    PyAnnote's diarization models are gated on HuggingFace ,  you must:
        1. Accept the model license at https://huggingface.co/pyannote/speaker-diarization
        2. Generate a HF access token and supply it via ``hf_token`` config key
           or the ``HF_TOKEN`` environment variable.

This module differs from ``whisperx_local`` in that it uses ``whisperx`` as the
top-level package throughout (no separate ``load_align_model`` guard), and it
always runs all three stages.

Dependencies:
    pip install whisperx
    pip install torch  (GPU support optional but strongly recommended)
"""

import os
import logging
from typing import Dict, Any
try:
	import whisperx
	import torch
except ImportError:
	# Allow import to succeed in environments without whisperx/torch installed.
	# Instantiation will fail at model loading time if these are missing.
	whisperx = None
	torch = None
from src.stages.stage1_transcription.base import BaseTranscriber
from src.shared.schemas import Transcript, Utterance

logger = logging.getLogger(__name__)


class WhisperTranscriber(BaseTranscriber):
	"""
    Local transcription using WhisperX: Whisper + word alignment + PyAnnote diarization.

    Runs entirely on-device ,  no API calls are made during transcription.  The
    HuggingFace models are downloaded on first use and cached locally.

    Config keys:
        model_size   ,  Whisper model size: ``"tiny"``, ``"base"``, ``"small"``,
                        ``"medium"``, ``"large-v2"`` (default: ``"base"``).
        device       ,  ``"cuda"`` or ``"cpu"`` (default: CUDA if available, else CPU).
        compute_type ,  Quantization: ``"float16"`` (GPU) or ``"int8"`` (CPU)
                        (default: ``"float16"`` on CUDA, ``"int8"`` on CPU).
        hf_token     ,  HuggingFace token for PyAnnote diarization models.
                        Falls back to ``HF_TOKEN`` environment variable.
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Load the WhisperX model into memory.

        Model loading is done eagerly at init time so that the first call to
        ``transcribe()`` does not include model load latency.  GPU memory is
        allocated here; ensure sufficient VRAM (>=4 GB for large-v2 on float16).

        Args:
            config: Configuration dictionary.  See class docstring for keys.
        """
		self.model_size = config.get("model_size", "base")

		# Choose device based on config, falling back to CUDA if available.
		self.device = config.get(
			"device",
			"cuda" if torch and torch.cuda.is_available() else "cpu"
		)

		# float16 is fast and accurate on GPU; int8 is quantized for CPU efficiency.
		self.compute_type = config.get(
			"compute_type",
			"float16" if self.device == "cuda" else "int8"
		)

		# HF token is required for PyAnnote diarization models gated on HuggingFace.
		self.hf_token = config.get("hf_token") or os.getenv("HF_TOKEN")
		if not self.hf_token:
			logger.warning(
				"HF_TOKEN is not set. Diarization models from Hugging Face that require "
				"authentication will fail. Set hf_token in config or the HF_TOKEN environment variable."
			)

		# Load the Whisper model ,  this downloads weights if not cached.
		# Takes 5-60 seconds depending on model size and hardware.
		self.model = whisperx.load_model(self.model_size, self.device, compute_type=self.compute_type)

	def transcribe(self, audio_filepath: str) -> Transcript:
		"""
        Run the full WhisperX pipeline: transcribe -> align -> diarize.

        Step 1 ,  Transcription:
            WhisperX batches audio chunks through Whisper for fast inference.
            ``batch_size=16`` balances throughput and VRAM usage.

        Step 2 ,  Alignment:
            Loads a language-specific phoneme model to align transcribed text
            to the audio waveform, producing accurate word-level timestamps.

        Step 3 ,  Diarization:
            PyAnnote's speaker diarization pipeline identifies speaker changes.
            ``assign_word_speakers`` maps speaker IDs to each Whisper segment.

        Args:
            audio_filepath: Path to the local audio file.

        Returns:
            A ``Transcript`` with per-segment speaker labels and timestamps.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
		if not os.path.exists(audio_filepath):
			raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

		# Step 1: Transcribe audio to text segments with rough timestamps.
		# whisperx.load_audio reads and resamples to 16 kHz mono (Whisper's format).
		audio = whisperx.load_audio(audio_filepath)
		result = self.model.transcribe(audio, batch_size=16)

		# Step 2: Align word-level timestamps for more precise segment boundaries.
		# The alignment model is language-specific and downloaded on first use.
		model_a, metadata = whisperx.load_align_model(
			language_code=result["language"],
			device=self.device,
		)
		result = whisperx.align(
			result["segments"],
			model_a,
			metadata,
			audio,
			self.device,
			return_char_alignments=False,  # Word-level alignment is sufficient.
		)

		# Step 3: Diarize ,  identify who is speaking in each time window.
		# DiarizationPipeline wraps PyAnnote; it requires an HF token to download
		# the speaker-diarization model on first use.
		diarize_model = whisperx.DiarizationPipeline(
			use_auth_token=self.hf_token,
			device=self.device,
		)
		diarize_segments = diarize_model(audio)

		# Assign speaker labels from diarization output to Whisper segments.
		result = whisperx.assign_word_speakers(diarize_segments, result)

		# Step 4: Map WhisperX segment output to the shared Transcript schema.
		utterances = []
		for segment in result["segments"]:
			utterances.append(Utterance(
				speaker=segment.get("speaker", "Unknown"),
				text=segment.get("text", "").strip(),
				start=segment.get("start", 0.0),
				end=segment.get("end", 0.0),
			))

		full_text = " ".join([u.text for u in utterances])

		# Derive audio duration from the end timestamp of the final segment.
		duration = utterances[-1].end if utterances else 0.0

		return Transcript(
			audio_duration=duration,
			status="completed",
			utterances=utterances,
			full_text=full_text,
		)
