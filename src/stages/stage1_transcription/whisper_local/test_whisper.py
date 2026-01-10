"""
Tests for the local WhisperX transcriber.
Verifies initialization and missing file error handling.
"""
import sys

import os
import unittest
import json
from pydantic import ValidationError
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
	import whisperx
	HAS_WHISPERX = True
except ImportError:
	HAS_WHISPERX = False

@unittest.skipIf(not HAS_WHISPERX, "whisperx is not installed")
class TestWhisperTranscriber(unittest.TestCase):
	"""Test suite for the local WhisperX transcriber."""

	@patch('whisperx.load_model')
	def setUp(self, mock_load_model):
		"""Sets up configuration for WhisperX testing."""

		self.config = {
			"model_size": "tiny", # Smallest for testing
			"device": "cpu",
			"compute_type": "int8"
		}

	def test_import_and_init(self):
		"""Test that the class can be imported and initialized."""
		try:
			from whisper_transcriber import WhisperTranscriber
			transcriber = WhisperTranscriber(self.config)
			self.assertIsNotNone(transcriber)
		except ImportError as e:
			self.fail(f"Import failed: {e}. Ensure whisperx and torch are installed.")
		except Exception as e:
			self.fail(f"Initialization failed: {e}")

	@patch('whisperx.load_model')
	def test_transcribe_missing_file(self, mock_load):
		"""Test that transcribe raises FileNotFoundError for missing files."""
		from whisper_transcriber import WhisperTranscriber
		transcriber = WhisperTranscriber(self.config)
		with self.assertRaises(FileNotFoundError):
			transcriber.transcribe("non_existent_file.wav")

if __name__ == "__main__":
	unittest.main()
