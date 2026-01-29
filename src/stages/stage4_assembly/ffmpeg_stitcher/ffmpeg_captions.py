"""
FFmpeg Caption/Subtitle Functions
===================================
Functions for generating SRT subtitle content and embedding it into a video
as a soft ``mov_text`` track via FFmpeg.
"""

import os
import subprocess
import tempfile
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
	from src.shared.schemas import ProductionScene


def seconds_to_srt_timestamp(seconds: float) -> str:
	"""
    Convert a duration in seconds to SRT timestamp format ``HH:MM:SS,mmm``.

    SRT uses comma as the decimal separator for milliseconds (not a period),
    e.g. ``00:01:23,456`` for 83.456 seconds.

    Args:
        seconds: Time in seconds (float).

    Returns:
        String in ``HH:MM:SS,mmm`` format.
    """
	h = int(seconds // 3600)
	m = int((seconds % 3600) // 60)
	s = int(seconds % 60)
	ms = int((seconds % 1) * 1000)
	return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(
	scenes: List["ProductionScene"],
	durations: List[float],
	caption_starts: List[float],
) -> str:
	"""
    Build an SRT subtitle string from scene narrative summaries and timing.

    Each SRT block covers one scene and displays the scene's
    ``narrative_summary`` text for the full duration of that clip in the
    final stitched video.

    SRT block format:
        <index>
        <HH:MM:SS,mmm> --> <HH:MM:SS,mmm>
        <text>

    Args:
        scenes:         Ordered list of ``ProductionScene`` objects.
        durations:      Duration in seconds of each corresponding video clip.
        caption_starts: Start time in the final video for each scene's caption.

    Returns:
        A complete SRT string ready to write to a ``.srt`` file.
    """
	blocks = []
	for i, scene in enumerate(scenes):
		t_start = caption_starts[i]
		t_end = t_start + durations[i]
		start_ts = seconds_to_srt_timestamp(t_start)
		end_ts = seconds_to_srt_timestamp(t_end)
		# SRT blocks are 1-indexed.
		blocks.append(f"{i + 1}\n{start_ts} --> {end_ts}\n{scene.narrative_summary}\n")
	return "\n".join(blocks)


def add_captions(
	ffmpeg_exe: str,
	stitched_path: str,
	scenes: List["ProductionScene"],
	video_paths: List[str],
	output_path: str,
	get_duration_fn,
) -> str:
	"""
    Embed narrative summaries as a soft subtitle track in the stitched video.

    The subtitles are added as a ``mov_text`` track (the standard subtitle
    codec for MP4/MOV containers), which most video players can toggle on/off.
    Subtitles are NOT burned into the video pixels.

    Timing logic:
        With crossfade transitions, each scene in the final video starts
        ``transition_duration`` seconds earlier than a simple concatenation.
        Caption start times are computed accordingly:

            caption_starts[0] = 0.0
            caption_starts[i] = caption_starts[i-1] + durations[i-1] - transition_duration

    Args:
        ffmpeg_exe:      Path to the ffmpeg executable.
        stitched_path:   Path to the already-stitched (transitions applied) video.
        scenes:          Ordered list of ``ProductionScene`` objects (one per clip).
        video_paths:     Paths to the individual scene clips (for duration measurement).
        output_path:     Path to write the captioned output video.
        get_duration_fn: Callable(path) -> float for measuring clip duration.

    Returns:
        ``output_path`` after the FFmpeg command completes.

    Raises:
        ValueError: If ``scenes``/``video_paths`` are empty or have different lengths.
    """
	if not scenes or not video_paths:
		raise ValueError("scenes and video_paths must be non-empty")
	if len(scenes) != len(video_paths):
		raise ValueError("scenes and video_paths must have the same length")

	# Measure each clip's duration for accurate caption timing.
	durations = [get_duration_fn(vp) for vp in video_paths]
	# Use the same clamped transition duration as stitch_with_transitions.
	transition_duration = min(1.0, min(durations))

	# Compute caption start time for each scene in the final stitched timeline.
	caption_starts = [0.0]
	for i in range(1, len(durations)):
		caption_starts.append(
			caption_starts[i - 1] + durations[i - 1] - transition_duration
		)

	srt_content = generate_srt(scenes, durations, caption_starts)

	# Write the SRT to a temporary file; FFmpeg reads subtitles from disk.
	with tempfile.NamedTemporaryFile(
		mode="w", suffix=".srt", delete=False, encoding="utf-8"
	) as srt_file:
		srt_file.write(srt_content)
		srt_path = srt_file.name

	try:
		cmd = [
			ffmpeg_exe, "-y",
			"-i", stitched_path,  # Input: the already-stitched video.
			"-i", srt_path,		# Input: the generated SRT file.
			"-map", "0:v",         # Keep video stream from the stitched video.
			"-map", "1:s",         # Add subtitle stream from the SRT file.
			"-c:v", "copy",		# Copy video without re-encoding.
			"-c:s", "mov_text",	# Encode subtitles as mov_text (MP4-compatible).
			output_path,
		]
		subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	finally:
		# Always clean up the temporary SRT file, even on failure.
		os.unlink(srt_path)

	return output_path
