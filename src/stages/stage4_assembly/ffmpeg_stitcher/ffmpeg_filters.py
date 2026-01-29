"""
FFmpeg Filter Functions - Video Stitching
==========================================
Low-level FFmpeg command builders for concatenating video clips with or without
crossfade transitions.  All functions accept ``ffmpeg_exe`` as the first argument
so the caller controls which binary is used.
"""

import subprocess
from typing import List


def get_duration(ffmpeg_exe: str, path: str) -> float:
	"""
    Return the duration in seconds of a video file by parsing FFmpeg output.

    FFmpeg prints duration to stderr in the format:
        ``Duration: HH:MM:SS.sss, start: ...``

    Args:
        ffmpeg_exe: Path to the ffmpeg executable.
        path:       Path to the video file.

    Returns:
        Duration in seconds as a float.  Returns 0.0 if not found.
    """
	cmd = [ffmpeg_exe, "-i", path]
	# FFmpeg always prints info to stderr (even for -i); capture it.
	res = subprocess.run(cmd, capture_output=True, text=True)
	for line in res.stderr.split('\n'):
		if "Duration:" in line:
			# Format: "  Duration: 00:00:05.04, start: ..."
			time_str = line.split("Duration:")[1].split(",")[0].strip()
			h, m, s = time_str.split(":")
			return float(h) * 3600 + float(m) * 60 + float(s)
	return 0.0


def stitch_no_transitions(
	ffmpeg_exe: str,
	video_paths: List[str],
	final_output_path: str,
	w: int,
	h: int,
) -> str:
	"""
    Concatenate video clips with hard cuts using FFmpeg's ``concat`` filter.

    Clips are individually scaled and padded to ``wxh`` with letterboxing
    (``force_original_aspect_ratio=decrease`` + ``pad``), then concatenated.
    This handles clips of any input resolution without distortion.

    FFmpeg filter graph (example for 2 clips):
        [0:v]scale=1920:1080:force_original_aspect_ratio=decrease,
              pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v0];
        [1:v]scale=...setsar=1[v1];
        [v0][v1]concat=n=2:v=1:a=0[outv]

    Args:
        ffmpeg_exe:        Path to the ffmpeg executable.
        video_paths:       Ordered list of input clip paths.
        final_output_path: Path to write the concatenated output.
        w, h:              Target width and height in pixels.

    Returns:
        ``final_output_path``.
    """
	filter_complex = []
	inputs = []

	for i, vp in enumerate(video_paths):
		inputs.extend(["-i", vp])
		# Scale to fit within wxh (letterbox), then pad to exact wxh.
		# setsar=1 ensures the sample aspect ratio is 1:1 for compatibility.
		scale_filter = (
			f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
			f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
		)
		filter_complex.append(scale_filter)

	# Concatenate all normalized streams into [outv].
	# n= is the number of input segments, v=1:a=0 means one video, no audio.
	concat_inputs = "".join([f"[v{i}]" for i in range(len(video_paths))])
	concat_filter = f"{concat_inputs}concat=n={len(video_paths)}:v=1:a=0[outv]"
	filter_complex.append(concat_filter)

	cmd = [ffmpeg_exe, "-y"] + inputs + [
		"-filter_complex", "".join(filter_complex),
		"-map", "[outv]",
		"-c:v", "libx264",	# H.264 encoding for broad compatibility.
		"-preset", "fast",	# Fast encoding preset (slight quality tradeoff).
		final_output_path,
	]

	subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	return final_output_path


def stitch_with_transitions(
	ffmpeg_exe: str,
	video_paths: List[str],
	final_output_path: str,
	w: int,
	h: int,
) -> str:
	"""
    Concatenate clips with 1-second crossfade dissolves using FFmpeg's ``xfade``.

    ``xfade`` requires an ``offset`` parameter in *absolute* seconds -- the
    start time of the transition in the combined timeline.  We compute these
    offsets from the actual clip durations.

    Offset calculation:
        offset[0] = duration[0] - transition_duration
        offset[1] = offset[0] + duration[1] - transition_duration
        ...

    The ``transition_duration`` is clamped to the shortest clip duration to
    avoid negative offsets (which would crash FFmpeg).

    For a single clip, falls back to ``stitch_no_transitions`` since xfade
    requires at least two inputs.

    Args:
        ffmpeg_exe:        Path to the ffmpeg executable.
        video_paths:       Ordered list of input clip paths.
        final_output_path: Path to write the output video.
        w, h:              Target width and height in pixels.

    Returns:
        ``final_output_path``.
    """
	# Single clip: xfade needs at least two inputs; just copy/normalize it.
	if len(video_paths) == 1:
		return stitch_no_transitions(ffmpeg_exe, video_paths, final_output_path, w, h)

	durations = [get_duration(ffmpeg_exe, vp) for vp in video_paths]
	# Clamp transition to the shortest clip to prevent negative xfade offsets.
	transition_duration = min(1.0, min(durations))

	filter_complex = []
	inputs = []

	# First, normalize all clips to the target resolution.
	for i, vp in enumerate(video_paths):
		inputs.extend(["-i", vp])
		scale_filter = (
			f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
			f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
		)
		filter_complex.append(scale_filter)

	# Compute the xfade offset for the first transition.
	# Offset = start time of the crossfade in the output timeline.
	current_offset = durations[0] - transition_duration

	# Chain xfade filters: each takes (last_out, v_i) and produces (xf_i).
	# The final transition outputs [outv].
	last_out = "[v0]"
	for i in range(1, len(video_paths)):
		next_in = f"[v{i}]"
		# Last xfade outputs [outv]; intermediates output [xf1], [xf2], ...
		out_name = f"[xf{i}]" if i < len(video_paths) - 1 else "[outv]"

		fade_filter = (
			f"{last_out}{next_in}xfade=transition=fade:"
			f"duration={transition_duration}:"
			f"offset={current_offset}{out_name};"
		)
		filter_complex.append(fade_filter)

		last_out = out_name
		# Advance the offset by this clip's contribution (minus overlap).
		current_offset += (durations[i] - transition_duration)

	cmd = [ffmpeg_exe, "-y"] + inputs + [
		"-filter_complex", "".join(filter_complex),
		"-map", "[outv]",
		"-c:v", "libx264",
		"-preset", "fast",
		final_output_path,
	]

	subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	return final_output_path
