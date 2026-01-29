"""
Integration tests for the FFmpeg assembler.
Verifies video stitching, transition effects, captioning, and audio overlay.
"""
import pytest

import os
import subprocess
import shutil
import tempfile
from pathlib import Path

# Try to import imageio_ffmpeg to get a guaranteed ffmpeg binary
try:
	import imageio_ffmpeg
	FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
	FFMPEG_EXE = "ffmpeg"

from src.stages.stage4_assembly.ffmpeg_stitcher.ffmpeg_assembler import FFmpegAssembler

@pytest.fixture(scope="module")
def workspace():
	"""Creates a temporary workspace directory for test media files."""

	# Create a temporary directory for the tests
	test_dir = tempfile.mkdtemp(prefix="test_ffmpeg_")
	yield test_dir
	# Clean up
	shutil.rmtree(test_dir)

def generate_mock_video(filepath: str, duration: int = 5, color: str = "red", resolution: str = "1280x720"):
	"""Helper to generate a mock video using ffmpeg."""
	cmd = [
		FFMPEG_EXE,
		"-f", "lavfi",
		"-i", f"color=c={color}:s={resolution}:d={duration}",
		"-c:v", "libx264",
		"-preset", "ultrafast",
		"-y", filepath
	]
	subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def generate_mock_audio(filepath: str, duration: int = 10):
	"""Helper to generate mock audio (silence or beep) using ffmpeg."""
	cmd = [
		FFMPEG_EXE,
		"-f", "lavfi",
		"-i", f"aevalsrc=0:d={duration}",
		"-c:a", "libmp3lame",
		"-y", filepath
	]
	subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def get_media_duration(filepath: str) -> float:
	"""Helper to get media duration using ffmpeg instead of ffprobe."""
	cmd = [
		FFMPEG_EXE,
		"-i", filepath
	]
	# ffmpeg will error because no output file is provided, but it still prints duration to stderr
	result = subprocess.run(cmd, capture_output=True, text=True)
	# Look for: Duration: 00:00:05.00, start: 0.000000, bitrate: N/A
	for line in result.stderr.split('\n'):
		if "Duration:" in line:
			time_str = line.split("Duration:")[1].split(",")[0].strip()
			h, m, s = time_str.split(":")
			return float(h) * 3600 + float(m) * 60 + float(s)
	return 0.0

@pytest.fixture(scope="module")
def mock_assets(workspace):
	"""Generates mock video and audio assets for integration testing."""

	v1_path = os.path.join(workspace, "vid1.mp4")
	v2_path = os.path.join(workspace, "vid2.mp4")
	v3_path = os.path.join(workspace, "vid3.mp4")
	audio_path = os.path.join(workspace, "audio.mp3")

	# We'll make them different resolutions to test scaling
	generate_mock_video(v1_path, duration=5, color="red", resolution="1280x720")
	generate_mock_video(v2_path, duration=5, color="blue", resolution="1920x1080")
	generate_mock_video(v3_path, duration=5, color="green", resolution="1280x720")
	
	generate_mock_audio(audio_path, duration=15)

	return {
		"videos": [v1_path, v2_path, v3_path],
		"audio": audio_path
	}

def test_stitch_videos_no_transitions(mock_assets, workspace):
	"""Verify video stitching without any transition effects."""

	assembler = FFmpegAssembler()
	output_path = os.path.join(workspace, "stitched_no_trans.mp4")
	
	try:
		res = assembler.stitch_videos(mock_assets["videos"], output_path, add_transitions=False)
		assert res == output_path
		assert os.path.exists(output_path)
		
		# Total duration should be exactly 5 + 5 + 5 = 15 seconds
		duration = get_media_duration(output_path)
		assert 14.5 <= duration <= 15.5
	except Exception as e:
		pytest.fail(f"Stitching without transitions failed: {e}")

def test_stitch_videos_with_transitions(mock_assets, workspace):
	"""Verify video stitching with crossfade transition effects."""

	assembler = FFmpegAssembler()
	output_path = os.path.join(workspace, "stitched_trans.mp4")
	
	try:
		res = assembler.stitch_videos(mock_assets["videos"], output_path, add_transitions=True)
		assert res == output_path
		assert os.path.exists(output_path)
		
		# With 1s crossfades (default behavior usually), total duration should be
		# 5 + 5 + 5 - (1 * 2) = 13 seconds (approx)
		duration = get_media_duration(output_path)
		assert 12.0 <= duration <= 14.0
	except Exception as e:
		pytest.fail(f"Stitching with transitions failed: {e}")

def test_add_captions(mock_assets, workspace):
	"""Verify that subtitles are correctly embedded into the video."""

	assembler = FFmpegAssembler()

	# Stitch videos to get a base for captioning
	base_video = os.path.join(workspace, "base_for_captions.mp4")
	assembler.stitch_videos(mock_assets["videos"], base_video, add_transitions=True)

	# Create minimal mock scenes with narrative_summary
	class MockScene:
		"""Minimal mock class for scene data with narrative summary."""

		def __init__(self, summary):
			"""Initializes a mock scene with a summary."""

			self.narrative_summary = summary

	mock_scenes = [
		MockScene("The adventurers enter the dungeon."),
		MockScene("A dragon blocks the path."),
		MockScene("Victory! The dragon is slain."),
	]

	captioned_output = os.path.join(workspace, "captioned.mp4")

	try:
		res = assembler.add_captions(base_video, mock_scenes, mock_assets["videos"], captioned_output)
		assert res == captioned_output
		assert os.path.exists(captioned_output)

		# Verify a subtitle stream is present in the output
		cmd = [FFMPEG_EXE, "-i", captioned_output]
		result = subprocess.run(cmd, capture_output=True, text=True)
		has_subtitle = any("Subtitle:" in line for line in result.stderr.split('\n'))
		assert has_subtitle, "Expected a subtitle stream in the captioned output"
	except Exception as e:
		pytest.fail(f"add_captions failed: {e}")


def test_overlay_audio(mock_assets, workspace):
	"""Verify that an audio track is correctly synced and overlaid on the video."""

	assembler = FFmpegAssembler()
	
	# First stitch some videos to have a base
	base_video = os.path.join(workspace, "base_for_audio.mp4")
	assembler.stitch_videos(mock_assets["videos"], base_video, add_transitions=False)
	
	synced_output = os.path.join(workspace, "final_synced.mp4")
	
	try:
		res = assembler.overlay_audio(base_video, mock_assets["audio"], synced_output)
		assert res == synced_output
		assert os.path.exists(synced_output)
		
		# Check if the output has an audio stream using ffmpeg
		cmd = [
			FFMPEG_EXE,
			"-i", synced_output
		]
		result = subprocess.run(cmd, capture_output=True, text=True)
		has_audio = False
		for line in result.stderr.split('\n'):
			if "Stream" in line and "Audio:" in line:
				has_audio = True
				break
		assert has_audio
	except Exception as e:
		pytest.fail(f"Overlaying audio failed: {e}")
