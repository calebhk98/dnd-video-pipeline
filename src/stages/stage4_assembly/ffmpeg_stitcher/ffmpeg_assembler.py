"""
Stage 4 Assembly ,  FFmpeg Backend
==================================
Concrete assembler implementation using FFmpeg for video stitching, audio
overlay, and subtitle embedding.

FFmpeg binary resolution:
    The module first tries to locate FFmpeg via the ``imageio_ffmpeg`` package,
    which ships a bundled binary for cross-platform compatibility.  If
    ``imageio_ffmpeg`` is not installed, it falls back to the system FFmpeg
    binary (assumes ``ffmpeg`` is on PATH).

Two stitching modes:

1. **No transitions** (``add_transitions=False``):
    Uses FFmpeg's ``concat`` filter with per-clip scale+pad stages to normalize
    all clips to 1920x1080 before concatenating.  This is a hard cut (no blend).
    Faster to encode; good for quick review renders.

    Filter pattern (2 clips):
        [0:v]scale=...,pad=...[v0]; [1:v]scale=...,pad=...[v1];
        [v0][v1]concat=n=2:v=1:a=0[outv]

2. **With transitions** (``add_transitions=True``):
    Uses FFmpeg's ``xfade`` filter for 1-second crossfade dissolves between clips.
    The ``offset`` parameter for each xfade is computed from the cumulative clip
    durations minus the transition overlap.

    Filter pattern (3 clips):
        [0:v]...[v0]; [1:v]...[v1]; [2:v]...[v2];
        [v0][v1]xfade=transition=fade:duration=1:offset=D0[xf1];
        [xf1][v2]xfade=transition=fade:duration=1:offset=D0+D1-1[outv]

    where D0, D1 are the durations of clips 0 and 1.

Audio overlay:
    ``overlay_audio`` replaces the video's silent track with the original D&D
    session audio.  Optional ``audio_start`` / ``audio_end`` slice the audio to
    the section covered by the generated scenes.

Subtitle embedding (``add_captions``):
    Generates an SRT file from each scene's ``narrative_summary`` and embeds it
    as a ``mov_text`` soft subtitle track (toggleable in most video players).

Dependencies:
    pip install imageio[ffmpeg]  (optional but recommended for bundled FFmpeg)
    FFmpeg must be installed system-wide if imageio_ffmpeg is not available.
"""

import subprocess
from typing import List, Optional, TYPE_CHECKING
from src.stages.stage4_assembly.base import BaseAssembler
from src.stages.stage4_assembly.ffmpeg_stitcher.ffmpeg_filters import (
	get_duration,
	stitch_no_transitions,
	stitch_with_transitions,
)
from src.stages.stage4_assembly.ffmpeg_stitcher.ffmpeg_audio import (
	overlay_audio as _overlay_audio,
	overlay_audio_segments as _overlay_audio_segments,
)
from src.stages.stage4_assembly.ffmpeg_stitcher.ffmpeg_captions import add_captions as _add_captions

if TYPE_CHECKING:
	# Avoid circular import at runtime; only needed for type hints.
	from src.shared.schemas import ProductionScene

# Attempt to use the imageio-bundled FFmpeg binary for portability.
# If imageio_ffmpeg is not installed, fall back to the system 'ffmpeg' command.
try:
	import imageio_ffmpeg
	FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
	FFMPEG_EXE = "ffmpeg"


class FFmpegAssembler(BaseAssembler):
	"""
    Concrete assembler using FFmpeg for video stitching, audio overlay, and captions.

    All FFmpeg calls use ``subprocess.run(..., check=True)`` so any non-zero
    exit code raises ``CalledProcessError`` immediately.  stdout/stderr are
    suppressed to keep pipeline logs clean (errors still propagate as exceptions).

    Target output resolution: 1920x1080 (Full HD, 16:9).
    """

	def __init__(self):
		"""
        Initialize the FFmpegAssembler and verify that FFmpeg is accessible.

        Runs ``ffmpeg -version`` at initialization to fail fast if FFmpeg is
        missing, rather than discovering this mid-pipeline when generating videos.

        Raises:
            RuntimeError: If the FFmpeg executable is not found or not executable.
        """
		super().__init__()
		try:
			# Quick smoke-test: FFmpeg exits 0 if installed and executable.
			subprocess.run(
				[FFMPEG_EXE, "-version"],
				stdout=subprocess.DEVNULL,
				stderr=subprocess.DEVNULL,
				check=True,
			)
		except Exception:
			raise RuntimeError(f"FFmpeg executable not found at {FFMPEG_EXE}")

	def stitch_videos(
		self,
		video_paths: List[str],
		final_output_path: str,
		add_transitions: bool = True,
	) -> str:
		"""
        Stitch scene clips into a single 1920x1080 output video.

        Delegates to ``stitch_with_transitions`` or ``stitch_no_transitions``
        depending on ``add_transitions``.  Both modes normalize all clips to
        1920x1080 before concatenating.

        Args:
            video_paths:       Ordered list of scene MP4 file paths.
            final_output_path: Output path for the stitched video.
            add_transitions:   If True, apply 1-second crossfade dissolves.

        Returns:
            ``final_output_path`` after the FFmpeg command completes.

        Raises:
            ValueError: If ``video_paths`` is empty.
        """
		if not video_paths:
			raise ValueError("video_paths cannot be empty")

		# Standard Full HD target ,  all clips are scaled/padded to this resolution.
		target_w, target_h = 1920, 1080

		if not add_transitions:
			return stitch_no_transitions(FFMPEG_EXE, video_paths, final_output_path, target_w, target_h)
		else:
			return stitch_with_transitions(FFMPEG_EXE, video_paths, final_output_path, target_w, target_h)

	def overlay_audio(
		self,
		video_path: str,
		original_audio_path: str,
		synced_output_path: str,
		audio_start: float = 0.0,
		audio_end: Optional[float] = None,
	) -> str:
		"""
        Overlay the original D&D session audio onto the (mute) stitched video.

        Args:
            video_path:           Path to the stitched mute video.
            original_audio_path:  Path to the original session audio file.
            synced_output_path:   Path for the output video with audio.
            audio_start:          Seconds into the audio to start from (default: 0.0).
            audio_end:            Seconds into the audio to stop at (default: until end).

        Returns:
            ``synced_output_path`` after the FFmpeg command completes.
        """
		return _overlay_audio(
			FFMPEG_EXE, video_path, original_audio_path, synced_output_path,
			audio_start=audio_start, audio_end=audio_end,
		)

	def overlay_audio_segments(
		self,
		video_path: str,
		original_audio_path: str,
		synced_output_path: str,
		segments: list,
	) -> str:
		"""
        Overlay per-shot audio segments onto the (mute) stitched video.

        Args:
            video_path:           Path to the stitched mute video.
            original_audio_path:  Path to the original session audio file.
            synced_output_path:   Path for the output video with audio.
            segments:             List of (start_sec, end_sec) pairs -- one per shot.

        Returns:
            ``synced_output_path`` after the FFmpeg command completes.
        """
		return _overlay_audio_segments(
			FFMPEG_EXE, video_path, original_audio_path, synced_output_path, segments,
		)

	def add_captions(
		self,
		stitched_path: str,
		scenes: List["ProductionScene"],
		video_paths: List[str],
		output_path: str,
	) -> str:
		"""
        Embed narrative summaries as a soft subtitle track in the stitched video.

        Args:
            stitched_path: Path to the already-stitched (transitions applied) video.
            scenes:        Ordered list of ``ProductionScene`` objects (one per clip).
            video_paths:   Paths to the individual scene clips (for duration measurement).
            output_path:   Path to write the captioned output video.

        Returns:
            ``output_path`` after the FFmpeg command completes.

        Raises:
            ValueError: If ``scenes``/``video_paths`` are empty or have different lengths.
        """
		return _add_captions(
			FFMPEG_EXE, stitched_path, scenes, video_paths, output_path,
			get_duration_fn=lambda path: get_duration(FFMPEG_EXE, path),
		)

	# --------------------------------------------------------------------------
	# Private helpers kept for backward compatibility with any tests that
	# patch these methods directly on the instance.
	# --------------------------------------------------------------------------

	def _get_duration(self, path: str) -> float:
		"""Legacy wrapper for duration measurement (for testing)."""

		return get_duration(FFMPEG_EXE, path)

	def _stitch_no_transitions(self, video_paths, final_output_path, w, h):
		"""Legacy wrapper for no-transition stitching."""

		return stitch_no_transitions(FFMPEG_EXE, video_paths, final_output_path, w, h)

	def _stitch_with_transitions(self, video_paths, final_output_path, w, h):
		"""Legacy wrapper for transition-enabled stitching."""

		return stitch_with_transitions(FFMPEG_EXE, video_paths, final_output_path, w, h)
