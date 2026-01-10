"""
Tests for the AssemblyAI transcription stage.
Verifies API integration, status handling, and response mapping.
"""
import sys

import os
import unittest
from unittest.mock import MagicMock, patch

# Add src to sys.path to allow imports from shared and stages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from stages.stage1_transcription.assembly_ai.assembly_ai_transcriber import AssemblyAITranscriber
from shared.schemas import Transcript, Utterance
import assemblyai as aai

class TestAssemblyAITranscriber(unittest.TestCase):
	"""Test suite for the AssemblyAITranscriber component."""

	def setUp(self):
		"""Sets up a mocked AssemblyAI transcriber for testing."""

		self.config = {"ASSEMBLY_AI_API_KEY": "test_key"}
		# Mock settings and Transcriber globally for setup
		with patch('assemblyai.settings'), patch('assemblyai.Transcriber'):
			self.transcriber = AssemblyAITranscriber(self.config)

	def test_class_exists(self):
		"""Test that the AssemblyAITranscriber class can be imported."""
		self.assertIsNotNone(AssemblyAITranscriber, "AssemblyAITranscriber class should exist")

	def test_transcribe_exists(self):
		"""Test that the transcribe method is implemented."""
		self.assertTrue(hasattr(self.transcriber, 'transcribe'))

	def test_missing_api_key(self):
		"""Test that missing API key raises ValueError."""
		with self.assertRaises(ValueError):
			AssemblyAITranscriber({})

	def test_transcribe_success(self):
		"""Test successful transcription and mapping."""
		# Mock AssemblyAI response
		mock_aai_transcript = MagicMock()
		mock_aai_transcript.status = aai.TranscriptStatus.completed
		mock_aai_transcript.audio_duration = 30.0
		mock_aai_transcript.text = "Hello from AssemblyAI"
		
		# Mock utterances
		mock_utt = MagicMock()
		mock_utt.speaker = "A"
		mock_utt.text = "Hello"
		mock_utt.start = 1000  # 1s
		mock_utt.end = 2000	# 2s
		
		mock_aai_transcript.utterances = [mock_utt]
		self.transcriber.transcriber.transcribe.return_value = mock_aai_transcript
		
		with patch('os.path.exists', return_value=True):
			# We must use the transcriber instance that was initialized with the mock settings
			transcript = self.transcriber.transcribe("dummy.wav")
			
		self.assertEqual(transcript.audio_duration, 30.0)
		self.assertEqual(transcript.status, "completed")
		self.assertEqual(transcript.full_text, "Hello from AssemblyAI")
		self.assertEqual(len(transcript.utterances), 1)
		self.assertEqual(transcript.utterances[0].speaker, "Speaker A")
		self.assertEqual(transcript.utterances[0].start, 1.0)
		self.assertEqual(transcript.utterances[0].end, 2.0)

	def test_transcribe_api_error(self):
		"""Test handling of AssemblyAI API errors."""
		mock_aai_transcript = MagicMock()
		mock_aai_transcript.status = aai.TranscriptStatus.error
		mock_aai_transcript.error = "Invalid API Key"
		self.transcriber.transcriber.transcribe.return_value = mock_aai_transcript
		
		with patch('os.path.exists', return_value=True):
			with self.assertRaises(Exception) as cm:
				self.transcriber.transcribe("dummy.wav")
			self.assertIn("Invalid API Key", str(cm.exception))

	def test_file_not_found(self):
		"""Test handling of missing audio file."""
		with self.assertRaises(FileNotFoundError):
			self.transcriber.transcribe("non_existent.wav")

def run_demo():
	"""Run a simple demonstration using mock API responses."""
	print("--- AssemblyAI Transcriber Demo (Mocked) ---")
	config = {"ASSEMBLY_AI_API_KEY": "dummy_key_for_demo"}
	
	# Create a dummy audio file
	dummy_wav = "dummy_assembly.wav"
	with open(dummy_wav, "wb") as f:
		f.write(b"RIFF" + b"\x00" * 32)
	
	try:
		with patch('assemblyai.Transcriber') as mock_transcriber_class:
			mock_transcriber_instance = mock_transcriber_class.return_value
			
			mock_aai_transcript = MagicMock()
			mock_aai_transcript.status = aai.TranscriptStatus.completed
			mock_aai_transcript.audio_duration = 10.5
			mock_aai_transcript.text = "This is a mocked AssemblyAI transcript."
			
			mock_utt1 = MagicMock(speaker="0", text="This is a test", start=0, end=2000)
			mock_utt2 = MagicMock(speaker="1", text="I hear you loud and clear", start=2500, end=5000)
			
			mock_aai_transcript.utterances = [mock_utt1, mock_utt2]
			mock_transcriber_instance.transcribe.return_value = mock_aai_transcript
			
			transcriber = AssemblyAITranscriber(config)
			transcript = transcriber.transcribe(dummy_wav)
			
			print("Transcript Validated JSON:")
			print(transcript.model_dump_json(indent=2))
	finally:
		if os.path.exists(dummy_wav):
			os.remove(dummy_wav)

if __name__ == '__main__':
	if len(sys.argv) > 1 and sys.argv[1] == '--demo':
		run_demo()
	else:
		unittest.main()
