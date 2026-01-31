"""
audio_preprocessor.py ,  utilities for loading, converting, and chunking audio files.

This module prepares raw audio for downstream services (e.g. speech-to-text APIs)
that impose constraints on file format, sample rate, channel count, or clip length.

Public API
----------
prepare_audio(filepath, output_dir, force_wav, max_duration_minutes) -> list[dict]
	Main entry point.  Loads an audio file, optionally converts it to 16 kHz mono
	WAV, splits it into time-bounded chunks if it exceeds `max_duration_minutes`,
	and writes the results to `output_dir`.  Returns a manifest list that maps each
	output file back to its global timestamp range in the original recording.

Private helper
--------------
_chunk_by_silence(audio, max_duration_ms, silence_thresh) -> list[tuple[int,int]]
	Greedy algorithm that groups speech segments into chunks that respect a maximum
	duration limit, preferring to cut at natural silence boundaries.
"""

import logging
import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

logger = logging.getLogger(__name__)


def _chunk_by_silence(audio: AudioSegment, max_duration_ms: int, silence_thresh: float) -> list[tuple[int, int]]:
	"""
	Compute (start_ms, end_ms) chunk boundaries for an AudioSegment.

	The algorithm tries to split the audio at natural silence boundaries so that
	each resulting chunk is no longer than `max_duration_ms` milliseconds.  If a
	single continuous speech run is longer than `max_duration_ms` (e.g. a speaker
	who never pauses), the run is hard-split at the exact limit as a fallback.

	This is a *private* helper (indicated by the leading underscore) and should
	only be called from `prepare_audio`.

	Args:
		audio: The full AudioSegment to be chunked.
		max_duration_ms: Maximum allowed length of any single output chunk, in
			milliseconds.  Chunks will not exceed this length.
		silence_thresh: dBFS level below which audio is considered silent.
			Typically computed as `audio.dBFS - 14` by the caller.

	Returns:
		A list of (start_ms, end_ms) tuples in chronological order, where each
		tuple defines the inclusive start and exclusive end of one chunk within
		the original audio timeline.  All values are in milliseconds.

	Algorithm overview:
		1. Detect non-silent ranges using a 1-second minimum silence window.
		2. Greedily merge consecutive non-silent ranges into a single chunk as
		long as the merged span does not exceed max_duration_ms.
		3. When adding the next range would exceed the limit, seal the current
		chunk and start a new one.
		4. After grouping, perform a second pass to hard-split any chunk that
		still exceeds max_duration_ms (handles continuous speech with no
		detectable silences).
	"""
	# Use a 1-second minimum silence window.  Shorter windows would cause the
	# detector to split on brief natural pauses within sentences; 1 second is
	# a reliable indicator of an intentional pause or speaker change.
	min_silence_ms = 1000
	ranges = detect_nonsilent(audio, min_silence_len=min_silence_ms, silence_thresh=silence_thresh)

	if not ranges:
		# No non-silent parts detected ,  the audio may be entirely silent, or
		# the silence threshold may be too aggressive.  Treat the whole file as
		# one range so we can still produce a (possibly silent) output chunk.
		ranges = [[0, len(audio)]]

	# --- Greedy grouping pass ---
	# Walk through the non-silent ranges and accumulate them into a chunk.
	# When the next range would push the current chunk over max_duration_ms,
	# seal the existing chunk and start a fresh one from the current range.
	chunks = []
	current_start = ranges[0][0]  # Start of the chunk being accumulated
	current_end = ranges[0][1]	# End of the last range added to this chunk

	for r_start, r_end in ranges[1:]:
		# Check whether extending the current chunk to include this range would
		# exceed the maximum duration.  We measure from current_start (not
		# current_end) to capture any silence gap between ranges as well.
		if (r_end - current_start) > max_duration_ms:
			# Seal current chunk ,  record it using the end of the *last* range
			# we added (current_end), NOT r_end, so we don't include the silence
			# gap or the new range in the sealed chunk.
			chunks.append((current_start, current_end))
			# Start a new chunk from the beginning of the current range.
			current_start = r_start
			current_end = r_end
		else:
			# The new range fits within the limit ,  extend the current chunk.
			# We update current_end to include this range (and any silence gap
			# between the previous range and this one is absorbed into the chunk,
			# which is intentional to avoid very short inter-range fragments).
			current_end = r_end

	# Don't forget to seal the final accumulated chunk.
	chunks.append((current_start, current_end))

	# --- Hard-split pass ---
	# The greedy pass guarantees that no chunk *spans* more than max_duration_ms
	# worth of timeline from start to end, but it's still possible for a single
	# non-silent range to be longer than max_duration_ms (e.g. an uninterrupted
	# 10-minute monologue with no detectable 1-second silences).  This pass
	# enforces a hard upper bound on every chunk regardless.
	final_chunks = []
	for c_start, c_end in chunks:
		# Repeatedly slice off max_duration_ms segments from the front of the
		# chunk until what remains is within the limit.
		while (c_end - c_start) > max_duration_ms:
			final_chunks.append((c_start, c_start + max_duration_ms))
			c_start += max_duration_ms
		# Append the remainder (which is <= max_duration_ms).
		# Guard against zero-length trailing segments that can arise if a chunk
		# was an exact multiple of max_duration_ms.
		if c_end > c_start:
			final_chunks.append((c_start, c_end))

	return final_chunks


def prepare_audio(filepath: str, output_dir: str, force_wav: bool = False, max_duration_minutes: float = 0.0) -> list[dict]:
	"""
	Load, optionally convert, and chunk an audio file for downstream processing.

	This is the main public function in this module.  It:
	1. Validates that the input file and output directory exist (creating the
		output directory if needed).
	2. Loads the audio with pydub (supports MP3, WAV, FLAC, OGG, M4A, etc.).
	3. Optionally converts the audio to 16 kHz mono WAV format ,  the standard
		input format accepted by most cloud speech-to-text APIs.
	4. If `max_duration_minutes` is set and the audio exceeds that length,
		splits the audio into silence-aware chunks using `_chunk_by_silence`.
	5. Exports each chunk (or the single processed file) to `output_dir`.
	6. Returns a manifest describing every output file and its position in the
		original recording timeline.

	Args:
		filepath: Absolute or relative path to the source audio file.
		output_dir: Directory where processed/chunked audio files will be written.
			Created automatically if it does not already exist.
		force_wav: If True, convert the audio to 16 kHz, single-channel (mono)
			WAV before writing.  This is required by many speech-to-text APIs
			(e.g. AWS Transcribe, Google Speech-to-Text) that only accept WAV
			with a specific sample rate.  If False, the output format matches
			the source file's extension.
		max_duration_minutes: Maximum duration of any single output file, in
			minutes.  If the source audio is longer than this value, it will be
			split into multiple chunks.  A value of 0.0 (the default) disables
			chunking ,  the whole file is written as a single output.

	Returns:
		A manifest: a list of dicts, one per output file, each with the keys:
			{
				"filepath":        str   ,  absolute path to the output audio file,
				"global_start_ms": int   ,  start of this chunk in the *original*
					audio timeline, in milliseconds,
				"global_end_ms":   int   ,  end of this chunk in the original
					audio timeline, in milliseconds.
			}
		The global_start_ms / global_end_ms values allow callers to map
		transcription results (which are relative to the chunk) back to the
		correct timestamps in the original recording.

	Raises:
		FileNotFoundError: If `filepath` does not exist on disk.
		Exception: Propagates any exception raised by pydub during audio loading
			or export (e.g. missing codec, corrupt file, disk full).
	"""
	if not os.path.exists(filepath):
		raise FileNotFoundError(f"Audio file not found: {filepath}")
	if not os.path.exists(output_dir):
		# Create the output directory (and any missing parents) if it doesn't
		# exist yet.  exist_ok=True prevents a race condition if two workers
		# try to create the directory simultaneously.
		os.makedirs(output_dir, exist_ok=True)

	try:
		# pydub automatically detects the audio format from the file extension
		# and uses the appropriate decoder (ffmpeg/avconv under the hood).
		audio = AudioSegment.from_file(filepath)
	except Exception as e:
		logger.error(f"Error loading audio file {filepath}: {e}")
		raise e

	# pydub expresses duration as the number of samples, accessible via len().
	# The result is in milliseconds for AudioSegment objects.
	duration_ms = len(audio)

	if force_wav:
		# Resample to 16 000 Hz and downmix to mono ,  both are requirements of
		# most cloud speech-to-text APIs.  16 kHz captures the full frequency
		# range of human speech (max ~8 kHz) while keeping file sizes small.
		audio = audio.set_frame_rate(16000).set_channels(1)
		ext = ".wav"
		export_format = "wav"
	else:
		# Preserve the original format so callers that don't need WAV (e.g.
		# those that just need chunks, not format conversion) don't incur the
		# cost of re-encoding.
		_, ext = os.path.splitext(filepath)
		ext = ext.lower()
		# pydub's export() expects a format string without the leading dot.
		export_format = ext.replace(".", "") if ext else "wav"
		if not export_format or export_format == "":
			# Fallback for files with no extension ,  default to WAV which is
			# universally supported.
			export_format = "wav"
			ext = ".wav"

	# Convert the float minutes limit to milliseconds for comparison with pydub
	# durations (which are always in ms).
	max_duration_ms = int(max_duration_minutes * 60 * 1000)

	# Strip the directory and extension from the source filename to use as the
	# base name for output files (e.g. "interview.mp3" -> "interview").
	filename = os.path.basename(filepath)
	name, _ = os.path.splitext(filename)

	# The manifest accumulates one entry per output file.
	manifest = []

	if max_duration_ms > 0 and duration_ms > max_duration_ms:
		# The audio exceeds the maximum chunk duration ,  we need to split it.

		# Compute a sensible silence detection threshold relative to the overall
		# loudness of the file.  audio.dBFS is the RMS loudness of the whole
		# file; subtracting 14 dB gives a threshold that reliably catches pauses
		# between spoken words without triggering on quiet-but-present speech.
		# This heuristic works well for typical speech recordings and podcasts.
		thresh = audio.dBFS - 14  # standard sensible threshold
		chunk_bounds = _chunk_by_silence(audio, max_duration_ms, thresh)

		for i, (start_ms, end_ms) in enumerate(chunk_bounds):
			# Slice the AudioSegment to extract this chunk.
			# pydub slice syntax is [start_ms:end_ms] (milliseconds).
			chunk_audio = audio[start_ms:end_ms]

			# Zero-pad the chunk index (e.g. 001, 002, ...) so that output files
			# sort correctly in lexicographic order ,  important when filenames
			# are later sorted to reconstruct the original sequence.
			out_filepath = os.path.join(output_dir, f"{name}_chunk_{i:03d}{ext}")

			try:
				chunk_audio.export(out_filepath, format=export_format)
			except Exception as e:
				logger.error(f"Error exporting chunk {i}: {e}")
				raise e

			# Record the output file path and its position in the original
			# recording so callers can convert chunk-relative timestamps back
			# to global timestamps (global_ts = chunk_ts + global_start_ms).
			manifest.append({
				"filepath": out_filepath,
				"global_start_ms": start_ms,
				"global_end_ms": end_ms
			})
	else:
		# No chunking required ,  either max_duration_minutes was not set, or the
		# audio is already within the allowed duration.  Write a single output
		# file with a "_processed" suffix to indicate it has been through this
		# pipeline step (even if no conversion or chunking occurred).
		out_filepath = os.path.join(output_dir, f"{name}_processed{ext}")
		try:
			audio.export(out_filepath, format=export_format)
		except Exception as e:
			logger.error(f"Error exporting audio to {out_filepath}: {e}")
			raise e

		# The single chunk spans the entire original recording.
		manifest.append({
			"filepath": out_filepath,
			"global_start_ms": 0,
			"global_end_ms": duration_ms
		})

	return manifest
