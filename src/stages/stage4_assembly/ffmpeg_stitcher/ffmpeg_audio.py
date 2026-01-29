"""
FFmpeg Audio Overlay Functions
================================
Functions for mixing the original session audio with a (mute) stitched video.
Supports both a single contiguous audio slice and per-shot audio segments.
"""

import subprocess
from typing import Optional


def overlay_audio(
	ffmpeg_exe: str,
	video_path: str,
	original_audio_path: str,
	synced_output_path: str,
	audio_start: float = 0.0,
	audio_end: Optional[float] = None,
) -> str:
	"""
    Overlay the original D&D session audio onto the (mute) stitched video.

    Stream mapping:
        - ``-map 0:v``  ,  Video stream from the stitched clip (copied, not re-encoded).
        - ``-map 1:a``  ,  Audio stream from the session recording (re-encoded to AAC).
        - ``-map 0:s?`` ,  Any subtitle streams from the video (optional; ``?`` ignores
                          missing streams).

    Optional audio trimming:
        ``audio_start`` and ``audio_end`` let callers trim the audio to the
        section covered by the generated scenes, avoiding large amounts of silence.

    Args:
        ffmpeg_exe:           Path to the ffmpeg executable.
        video_path:           Path to the stitched mute video.
        original_audio_path:  Path to the original session audio file.
        synced_output_path:   Path for the output video with audio.
        audio_start:          Seconds into the audio to start from (default: 0.0).
        audio_end:            Seconds into the audio to stop at (default: until end).

    Returns:
        ``synced_output_path`` after the FFmpeg command completes.
    """
	cmd = [ffmpeg_exe, "-y", "-i", video_path]

	# Insert optional seek (-ss) and duration (-t) flags before the audio input.
	# These must appear *before* -i original_audio_path to apply to that input.
	if audio_start > 0:
		cmd.extend(["-ss", f"{audio_start:.3f}"])
	if audio_end is not None and audio_end > audio_start:
		cmd.extend(["-t", f"{audio_end - audio_start:.3f}"])

	cmd.extend([
		"-i", original_audio_path,
		"-map", "0:v",	# Video from input 0 (stitched video).
		"-map", "1:a",	# Audio from input 1 (session recording).
		"-map", "0:s?",   # Subtitles from input 0, if present (? = optional).
		"-c:v", "copy",   # Copy video stream without re-encoding.
		"-c:a", "aac",	# Re-encode audio to AAC for broad player compatibility.
		"-c:s", "copy",   # Copy subtitle stream without re-encoding.
		"-shortest",      # Truncate to the shorter of video or audio.
		synced_output_path,
	])

	subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	return synced_output_path


def overlay_audio_segments(
	ffmpeg_exe: str,
	video_path: str,
	original_audio_path: str,
	synced_output_path: str,
	segments: list,
) -> str:
	"""
    Overlay per-shot audio segments onto the (mute) stitched video.

    For each shot in `segments`, the corresponding slice of `original_audio_path`
    is extracted and the slices are concatenated in order to produce a single
    audio track that is then mixed with the video.

    Args:
        ffmpeg_exe:           Path to the ffmpeg executable.
        video_path:           Path to the stitched mute video.
        original_audio_path:  Path to the original session audio file.
        synced_output_path:   Path for the output video with audio.
        segments:             List of (start_sec, end_sec) pairs -- one per shot.

    Returns:
        ``synced_output_path`` after the FFmpeg command completes.
    """
	if len(segments) == 1:
		# Single segment: delegate to the simpler overlay_audio() path.
		return overlay_audio(
			ffmpeg_exe,
			video_path,
			original_audio_path,
			synced_output_path,
			audio_start=segments[0][0],
			audio_end=segments[0][1],
		)

	# Build a filter_complex that:
	#   1. Trims each segment from the audio input with atrim.
	#   2. Resets each segment's timestamps to 0 with asetpts=PTS-STARTPTS.
	#   3. Concatenates all segments into one continuous audio stream.
	filter_parts: list = []
	concat_inputs = ""
	for i, (start, end) in enumerate(segments):
		filter_parts.append(
			f"[1:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]"
		)
		concat_inputs += f"[a{i}]"

	n = len(segments)
	filter_parts.append(f"{concat_inputs}concat=n={n}:v=0:a=1[aout]")
	filter_complex = ";".join(filter_parts)

	cmd = [
		ffmpeg_exe, "-y",
		"-i", video_path,
		"-i", original_audio_path,
		"-filter_complex", filter_complex,
		"-map", "0:v",
		"-map", "[aout]",
		"-map", "0:s?",
		"-c:v", "copy",
		"-c:a", "aac",
		"-c:s", "copy",
		"-shortest",
		synced_output_path,
	]

	subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	return synced_output_path
