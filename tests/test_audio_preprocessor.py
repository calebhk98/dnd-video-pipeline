"""
Tests for the audio preprocessor utility.
Covers audio format conversion, chunking, and error handling.
"""
import os

import tempfile
import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from src.shared.utils.audio_preprocessor import prepare_audio

def create_dummy_audio(duration_ms=1000, frame_rate=44100):
	"""Creates an in-memory silent AudioSegment for testing."""

	return AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate)

def test_prepare_audio_file_not_found():
	"""Test that a non-existent file raises an exception."""
	with pytest.raises(FileNotFoundError):
		prepare_audio("nonexistent_file.mp3", "dummy_dir")

def test_prepare_audio_no_chunking_no_conversion():
	"""Test standard case with no chunking and no conversion."""
	with tempfile.TemporaryDirectory() as temp_dir:
		input_path = os.path.join(temp_dir, "input.wav")
		output_dir = os.path.join(temp_dir, "output")
		
		audio = create_dummy_audio(duration_ms=2000)
		audio.export(input_path, format="wav")
		
		manifest = prepare_audio(input_path, output_dir, force_wav=False, max_duration_minutes=0.0)
		
		assert len(manifest) == 1
		assert "filepath" in manifest[0]
		assert manifest[0]["global_start_ms"] == 0
		assert manifest[0]["global_end_ms"] == 2000
		assert os.path.exists(manifest[0]["filepath"])

def test_prepare_audio_force_wav():
	"""Test format conversion to 16kHz WAV."""
	with tempfile.TemporaryDirectory() as temp_dir:
		# Changing extension to .wav to bypass ffmpeg MP3 export if ffmpeg is missing
		input_path = os.path.join(temp_dir, "input.wav")
		output_dir = os.path.join(temp_dir, "output")
		
		audio = create_dummy_audio(duration_ms=2000)
		audio.export(input_path, format="wav")
		
		manifest = prepare_audio(input_path, output_dir, force_wav=True, max_duration_minutes=0.0)
		
		assert len(manifest) == 1
		out_filepath = manifest[0]["filepath"]
		assert out_filepath.endswith(".wav")
		
		# Verify format changes
		out_audio = AudioSegment.from_file(out_filepath)
		assert out_audio.frame_rate == 16000
		assert out_audio.channels == 1

def test_prepare_audio_chunking():
	"""Test logic for chunking audio when it exceeds max duration."""
	with tempfile.TemporaryDirectory() as temp_dir:
		input_path = os.path.join(temp_dir, "input.wav")
		output_dir = os.path.join(temp_dir, "output")
		
		# We need something that pydub split_on_silence will actually split.
		# Let's generate a "sine wave" for sound and leave silence in between.
		sound = Sine(440).to_audio_segment(duration=3000)
		silence = AudioSegment.silent(duration=2000)
		
		# 3s sound + 2s silence + 3s sound = 8s total
		combined = sound + silence + sound
		combined.export(input_path, format="wav")
		
		# Max duration 5 seconds (5/60 minutes)
		manifest = prepare_audio(input_path, output_dir, force_wav=False, max_duration_minutes=5/60)
		
		assert len(manifest) == 2
		
		# Check that global offsets are continuous (ish) and cover the non-silent bounds
		assert manifest[0]["global_start_ms"] == 0
		assert manifest[1]["global_start_ms"] > 0
		assert manifest[1]["global_start_ms"] >= 3000
		assert os.path.exists(manifest[0]["filepath"])
		assert os.path.exists(manifest[1]["filepath"])
