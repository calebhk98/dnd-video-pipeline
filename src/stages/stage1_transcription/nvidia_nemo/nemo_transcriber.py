"""
Stage 1 Transcription ,  NVIDIA NeMo (Local) Backend
====================================================
Uses NVIDIA NeMo's Parakeet TDT model for *local, offline* audio transcription.
Optionally runs NeMo's multi-scale speaker diarization pipeline (MSDD) to
identify who said what.

Why local NeMo?
    - No API keys or cloud costs required.
    - GPU-accelerated via CUDA for production throughput.
    - Parakeet TDT 1.1B is one of the most accurate open-source English ASR models.

Diarization approach:
    NeMo's ``ClusteringDiarizer`` uses three components:
        1. Voice Activity Detection (VAD) ,  ``vad_multilingual_marblenet`` model.
        2. Speaker Embeddings ,  ``titanet_large`` model extracts speaker vectors.
        3. Clustering + MSDD ,  Groups embeddings into speaker clusters.
    The diarizer requires a JSON *manifest* file describing the audio.  It outputs
    an RTTM (Rich Transcription Time Mark) file with speaker-labeled segments.

Known limitation:
    NeMo's diarization does NOT align transcribed words to diarized segments.
    The ``_parse_rttm`` method distributes the full transcript text evenly across
    segments as a best-effort approximation.  For precise word-level alignment,
    use the WhisperX backend instead.

Dependencies:
    pip install nemo_toolkit[asr]
"""

import os
import json
import logging
import tempfile
from typing import Dict, Any, List
try:
	import torch
except ImportError:
	torch = None
from src.stages.stage1_transcription.base import BaseTranscriber
from src.shared.schemas import Transcript, Utterance

logger = logging.getLogger(__name__)


class NvidiaNemoTranscriber(BaseTranscriber):
	"""
    Concrete implementation of BaseTranscriber using NVIDIA NeMo Parakeet ASR.

    Transcribes audio using the Parakeet TDT model (a CTC/TDT hybrid) and
    optionally diarizes using NeMo's ``ClusteringDiarizer``.  Runs fully locally
    ,  no internet connection required after model download.

    Config keys:
        model_name         ,  NeMo model identifier (default: ``nvidia/parakeet-tdt-1.1b``).
        enable_diarization ,  Whether to run speaker diarization (default: True).
        device             ,  ``"cpu"`` or ``"cuda"`` (default: ``"cuda"`` if available, else ``"cpu"``).
    """

	DEFAULT_ASR_MODEL = "nvidia/parakeet-tdt-1.1b"
	DEFAULT_DIARIZATION = True

	def __init__(self, config: Dict[str, Any]):
		"""
        Load the NeMo ASR model into memory (and optionally onto GPU).

        Model loading can take 10-60 seconds on first run as NeMo downloads
        weights.  Subsequent runs use a local cache.

        Args:
            config: Configuration dictionary.  See class docstring for keys.
        """
		# Import lazily so that projects not using NeMo don't pay the import cost.
		import nemo.collections.asr as nemo_asr  # noqa: F401

		self.model_name = config.get("model_name", self.DEFAULT_ASR_MODEL)
		self.enable_diarization = config.get("enable_diarization", self.DEFAULT_DIARIZATION)
		# Auto-detect GPU; fall back to CPU if CUDA is not available.
		self.device = config.get("device", "cuda" if torch and torch.cuda.is_available() else "cpu")

		# Load the pre-trained ASR model from HuggingFace/NGC.
		# from_pretrained downloads the model on first call, then caches it.
		self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
			model_name=self.model_name
		)

		# Move model to GPU if requested for faster inference.
		if self.device == "cuda":
			self.asr_model = self.asr_model.cuda()

		# Set to eval mode to disable dropout and batch norm in training mode.
		self.asr_model.eval()

	def transcribe(self, audio_filepath: str) -> Transcript:
		"""
        Transcribe an audio file using NeMo Parakeet ASR with optional diarization.

        When diarization is disabled, returns a single-speaker Utterance containing
        the full transcript text with placeholder timestamps (0.0, 0.0) because
        Parakeet's CTC output does not produce segment-level timestamps natively.

        Args:
            audio_filepath: Path to the local audio file.

        Returns:
            A ``Transcript`` with speaker-labeled utterances.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            Exception: On NeMo model inference errors.
        """
		if not os.path.exists(audio_filepath):
			raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

		try:
			# NeMo's transcribe() accepts a *list* of file paths and returns a list
			# of transcription strings (one per file).
			transcriptions = self.asr_model.transcribe([audio_filepath])
			full_text = transcriptions[0] if transcriptions else ""

			if self.enable_diarization:
				# Run the full diarization pipeline and map RTTM output to utterances.
				utterances = self._run_diarization(audio_filepath, full_text)
			else:
				# No diarization ,  wrap entire transcript in a single unknown-speaker utterance.
				# Timestamps are (0.0, 0.0) because NeMo doesn't provide segment timings here.
				utterances = [Utterance(
					speaker="Speaker 0",
					text=full_text,
					start=0.0,
					end=0.0,
				)]

			return Transcript(
				audio_duration=utterances[-1].end if utterances else 0.0,
				status="completed",
				utterances=utterances,
				full_text=full_text,
			)

		except Exception as e:
			logger.error(f"Error during NeMo transcription: {str(e)}")
			raise

	def _run_diarization(self, audio_filepath: str, full_text: str) -> List[Utterance]:
		"""
        Run NeMo's ClusteringDiarizer pipeline and convert its RTTM output to Utterances.

        The diarizer requires:
            1. A JSON manifest file listing the audio path and metadata.
            2. An output directory for intermediate files and RTTM output.
        We use a temporary directory so all intermediate files are cleaned up
        automatically after this method returns.

        Config values baked in here are reasonable defaults for D&D recordings
        (mostly English speech, variable number of speakers, moderate background noise).

        Args:
            audio_filepath: Absolute path to the audio file.
            full_text:      Already-transcribed text to distribute across segments.

        Returns:
            A list of ``Utterance`` objects with speaker labels and timestamps.
            Falls back to a single-speaker utterance if diarization fails.
        """
		try:
			from nemo.collections.asr.models import ClusteringDiarizer
			from omegaconf import OmegaConf

			with tempfile.TemporaryDirectory() as tmpdir:
				# NeMo requires a manifest JSON file describing each audio file.
				# Fields follow the NeMo diarization manifest format specification.
				manifest_path = os.path.join(tmpdir, "manifest.json")
				with open(manifest_path, "w") as f:
					json.dump({
						"audio_filepath": os.path.abspath(audio_filepath),
						"offset": 0,           # Start from the beginning.
						"duration": None,      # None = process entire file.
						"label": "infer",      # "infer" means auto-detect speakers.
						"text": "-",           # Required field; unused here.
						"num_speakers": None,  # None = auto-detect speaker count.
						"rttm_filepath": None, # No ground-truth RTTM provided.
						"uem_filepath": None,  # No UEM (evaluation map) provided.
					}, f)
					f.write("\n")  # NeMo manifest files must end with a newline.

				# Build the diarizer configuration using OmegaConf structured config.
				# These VAD and embedding parameters are calibrated for typical speech:
				#   window_length_in_sec / shift_length_in_sec ,  sliding window settings
				#   onset/offset thresholds ,  speech detection sensitivity
				vad_params = {
					"window_length_in_sec": 0.15,
					"shift_length_in_sec": 0.01,
					"smoothing": "median",   # Smooth VAD predictions.
					"overlap": 0.875,
					"onset": 0.4,			# Threshold to mark speech start.
					"offset": 0.7,           # Threshold to mark speech end.
					"pad_onset": 0.05,
					"pad_offset": 0.0,
					"min_duration_on": 0.1,  # Ignore very short speech bursts.
					"min_duration_off": 0.4, # Minimum silence between segments.
					"filter_speech_first": True,
				}

				embed_params = {
					# Multi-scale windows: longer windows capture slower voice
					# patterns; shorter windows handle rapid speaker changes.
					"window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
					"shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
					"multiscale_weights": [1, 1, 1, 1, 1],
					"save_embeddings": False,  # Don't persist embeddings to disk.
				}

				clustering_params = {
					"oracle_num_speakers": False,  # Auto-detect speaker count.
					"max_num_speakers": 8,         # Upper bound for clustering.
				}

				diarizer_config = {
					"manifest_filepath": manifest_path,
					"out_dir": tmpdir,
					"oracle_vad": False,
					"vad": {"model_path": "vad_multilingual_marblenet", "parameters": vad_params},
					"speaker_embeddings": {"model_path": "titanet_large", "parameters": embed_params},
					"clustering": {"parameters": clustering_params},
					# MSDD (Multi-Scale Diarization Decoder) refines cluster boundaries.
					"msdd_model": {"model_path": "diar_msdd_telephony", "parameters": {}},
				}

				cfg = OmegaConf.structured({"diarizer": diarizer_config})

				diarizer = ClusteringDiarizer(cfg=cfg)
				diarizer.diarize()

				# Locate the RTTM output file produced by the diarizer.
				rttm_dir = os.path.join(tmpdir, "pred_rttms")
				rttm_files = [f for f in os.listdir(rttm_dir) if f.endswith(".rttm")]
				if not rttm_files:
					raise RuntimeError("No RTTM output found from NeMo diarizer.")

				rttm_path = os.path.join(rttm_dir, rttm_files[0])
				return self._parse_rttm(rttm_path, full_text)

		except Exception as e:
			# Diarization is optional ,  fall back gracefully to a single speaker.
			logger.warning(f"NeMo diarization failed, returning single-speaker transcript: {e}")
			return [Utterance(speaker="Speaker 0", text=full_text, start=0.0, end=0.0)]

	def _parse_rttm(self, rttm_path: str, full_text: str) -> List[Utterance]:
		"""
        Parse an RTTM file and distribute transcript text evenly across segments.

        RTTM format (space-separated fields):
            SPEAKER <file_id> <channel> <start_sec> <duration_sec> <NA> <NA>
                    <speaker_label> <NA> <NA>

        Because NeMo does not align transcribed words to diarized segments, we
        divide the full transcript's words equally across all segments as the best
        available approximation.  This means word counts per speaker may be inaccurate
        but the speaker turns and timestamps are correct.

        Args:
            rttm_path:  Path to the RTTM file output by NeMo's diarizer.
            full_text:  The full transcribed text to distribute across segments.

        Returns:
            A chronologically sorted list of ``Utterance`` objects.
        """
		utterances = []
		with open(rttm_path, "r") as f:
			for line in f:
				parts = line.strip().split()
				if not parts or parts[0] != "SPEAKER":
					continue  # Skip non-SPEAKER lines (e.g. comments or other types).

				start = float(parts[3])
				duration = float(parts[4])
				speaker_label = parts[7]  # e.g. "speaker_0", "speaker_1"

				# Strip the "speaker_" prefix and leading zeros to get a clean number.
				speaker_num = speaker_label.split("_")[-1].lstrip("0") or "0"

				utterances.append(Utterance(
					speaker=f"Speaker {speaker_num}",
					text="[segment]",  # Placeholder; replaced below with real words.
					start=start,
					end=start + duration,
				))

		# Sort segments chronologically (RTTM output is usually sorted, but not guaranteed).
		utterances.sort(key=lambda u: u.start)

		if utterances:
			# Distribute the transcript words evenly across segments.
			# This is a rough approximation ,  actual word-to-segment alignment would
			# require forced alignment (see WhisperX backend for word-level alignment).
			words = full_text.split()
			words_per_segment = max(1, len(words) // len(utterances))
			for i, utt in enumerate(utterances):
				segment_words = words[i * words_per_segment:(i + 1) * words_per_segment]
				utterances[i] = Utterance(
					speaker=utt.speaker,
					text=" ".join(segment_words),
					start=utt.start,
					end=utt.end,
				)

		return utterances
