"""
Unit tests for the Local Llama LLM processor.
Uses mocks to verify speaker mapping, storyboard generation, and production script creation.
"""
import pytest

import json
from unittest.mock import MagicMock, patch
from src.shared.schemas import Transcript, Utterance, Storyboard, ProductionScript, Scene, ProductionScene
from src.stages.stage2_llm.local_llama.local_llama_processor import LocalLlamaProcessor

@pytest.fixture
def mock_transcript():
	"""Provides a sample Transcript object for testing."""

	return Transcript(
		audio_duration=10.0,
		status="completed",
		utterances=[
			Utterance(speaker="Speaker A", text="Hello world", start=0.0, end=2.0),
			Utterance(speaker="Speaker B", text="Hi there", start=2.5, end=4.0)
		],
		full_text="Hello world. Hi there."
	)

@pytest.fixture
def processor():
	"""Provides a LocalLlamaProcessor instance configured for testing."""

	config = {"host": "http://localhost:11434", "model": "llama3.1"}
	return LocalLlamaProcessor(config)

def test_initialization(processor):
	"""Verify that the processor initializes with the correct host and model."""

	assert processor.host == "http://localhost:11434"
	assert processor.model == "llama3.1"

@patch("httpx.Client.post")
def test_map_speakers(mock_post, processor, mock_transcript):
	"""Test mapping transcript speakers to character names using the LLM."""

	# Mock Ollama response
	mock_response = MagicMock()
	mock_response.status_code = 200
	mock_response.json.return_value = {
		"message": {
			"content": '{"Speaker A": "Magnus", "Speaker B": "Killian"}'
		}
	}
	mock_post.return_value = mock_response

	speaker_map = processor.map_speakers(mock_transcript)
	
	assert isinstance(speaker_map, dict)
	assert speaker_map["Speaker A"] == "Magnus"
	assert speaker_map["Speaker B"] == "Killian"

@patch("httpx.Client.post")
def test_generate_storyboard(mock_post, processor, mock_transcript):
	"""Test generating a storyboard from a transcript and speaker map."""

	mock_response = MagicMock()
	mock_response.status_code = 200
	mock_response.json.return_value = {
		"message": {
			"content": '{"scenes": [{"scene_number": 1, "start_time": 0.0, "end_time": 4.0, "location": "Forest", "narrative_summary": "Greeting", "visual_prompt": "Two people meet"}]}'
		}
	}
	mock_post.return_value = mock_response

	speaker_map = {"Speaker A": "Magnus", "Speaker B": "Killian"}
	storyboard = processor.generate_storyboard(mock_transcript, speaker_map)
	
	assert isinstance(storyboard, Storyboard)
	assert len(storyboard.scenes) == 1
	assert storyboard.scenes[0].location == "Forest"

@patch("httpx.Client.post")
def test_generate_production_script(mock_post, processor, mock_transcript):
	"""Test generating a production script with stage directions and character actions."""

	storyboard = Storyboard(scenes=[
		Scene(
			scene_number=1,
			start_time=0.0,
			end_time=4.0,
			location="Forest",
			narrative_summary="Greeting",
			visual_prompt="Two people meet"
		)
	])
	
	mock_response = MagicMock()
	mock_response.status_code = 200
	mock_response.json.return_value = {
		"message": {
			"content": '{"scene_number": 1, "start_time": 0.0, "end_time": 4.0, "location": "Forest", "narrative_summary": "Greeting", "visual_prompt": "Two people meet", "stage_directions": "Birds chirping", "character_actions": "Walking", "final_video_prompt": "Cinematic"}'
		}
	}
	mock_post.return_value = mock_response

	script = processor.generate_production_script(storyboard, mock_transcript)
	
	assert isinstance(script, ProductionScript)
	assert len(script.scenes) == 1
	assert script.scenes[0].stage_directions == "Birds chirping"
