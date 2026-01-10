"""
Tests for the Rev.ai transcription stage.
Covers successful transcription flows and error handling for failed jobs.
"""
import unittest

from unittest.mock import MagicMock, patch
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.stages.stage1_transcription.rev_ai.rev_ai_transcriber import RevAiTranscriber
from src.shared.schemas import Transcript, Utterance

class TestRevAiTranscriber(unittest.TestCase):
	"""Test suite for the RevAiTranscriber component."""

	def setUp(self):
		"""Sets up a RevAiTranscriber instance for testing."""

		self.config = {"api_key": "test_key"}
		self.transcriber = RevAiTranscriber(self.config)

	def test_initialization(self):
		"""Test that the transcriber initializes correctly with config."""
		self.assertEqual(self.transcriber.api_key, "test_key")

	def test_transcribe_success(self):
		"""Test successful transcription flow."""
		# Mock the client and job
		mock_client = MagicMock()
		self.transcriber.client = mock_client
		
		mock_job = MagicMock()
		mock_job.id = "job_123"
		mock_job.status = "transcribed"
		mock_client.submit_job_local_file.return_value = mock_job
		mock_client.get_job_details.return_value = mock_job
		
		# Mock the transcript response
		mock_transcript_json = {
			"monologue_count": 1,
			"monologues": [{
				"speaker": 0,
				"elements": [
					{"type": "text", "value": "Hello", "ts": 0.0, "end_ts": 1.0, "confidence": 0.99},
					{"type": "text", "value": "world", "ts": 1.1, "end_ts": 2.0, "confidence": 0.99}
				]
			}]
		}
		mock_client.get_transcript_json.return_value = mock_transcript_json
		
		# Add metadata mock for duration (Rev AI doesn't directly provide it in JSON transcript usually)
		# We might need to estimate it or get it from job details if available
		mock_job.duration_seconds = 2.0
		
		transcript = self.transcriber.transcribe("fake_audio.mp3")
		
		self.assertIsInstance(transcript, Transcript)
		self.assertEqual(len(transcript.utterances), 1)
		self.assertEqual(transcript.utterances[0].text, "Hello world")
		self.assertEqual(transcript.utterances[0].speaker, "0")
		self.assertEqual(transcript.audio_duration, 2.0)

	def test_transcribe_polls_until_transcribed(self):
		"""Test that the polling loop continues through 'in_progress' and breaks on 'transcribed'."""
		mock_client = MagicMock()
		self.transcriber.client = mock_client

		mock_job = MagicMock()
		mock_job.id = "job_poll_789"
		mock_client.submit_job_local_file.return_value = mock_job

		# First poll returns in_progress; second returns transcribed.
		mock_in_progress = MagicMock()
		mock_in_progress.status = "in_progress"

		mock_done = MagicMock()
		mock_done.status = "transcribed"
		mock_done.duration_seconds = 1.5

		mock_client.get_job_details.side_effect = [mock_in_progress, mock_done]

		mock_transcript_json = {
			"monologues": [{
				"speaker": 1,
				"elements": [
					{"type": "text", "value": "Test", "ts": 0.0, "end_ts": 1.5, "confidence": 0.95}
				]
			}]
		}
		mock_client.get_transcript_json.return_value = mock_transcript_json

		transcript = self.transcriber.transcribe("poll_audio.mp3")

		# get_job_details should have been called exactly twice.
		self.assertEqual(mock_client.get_job_details.call_count, 2)
		self.assertIsInstance(transcript, Transcript)
		self.assertEqual(len(transcript.utterances), 1)
		self.assertEqual(transcript.utterances[0].text, "Test")
		self.assertEqual(transcript.utterances[0].speaker, "1")

	def test_transcribe_failure(self):
		"""Test handling of job failure."""
		mock_client = MagicMock()
		self.transcriber.client = mock_client

		mock_job = MagicMock()
		mock_job.id = "job_fail_456"
		mock_client.submit_job_local_file.return_value = mock_job

		mock_failed_details = MagicMock()
		mock_failed_details.status = "failed"
		mock_failed_details.failure_detail = "Audio file could not be decoded."
		mock_client.get_job_details.return_value = mock_failed_details

		with self.assertRaises(Exception) as ctx:
			self.transcriber.transcribe("bad_audio.mp3")

		self.assertIn("job_fail_456", str(ctx.exception))
		self.assertIn("Audio file could not be decoded.", str(ctx.exception))

if __name__ == "__main__":
	unittest.main()
