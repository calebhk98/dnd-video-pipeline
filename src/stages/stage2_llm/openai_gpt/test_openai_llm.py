"""
Unit tests for the OpenAI GPT LLM processor.
Verifies the integration with OpenAI's API for speaker mapping and storyboard/script generation.
"""
import pytest

from unittest.mock import MagicMock, patch
from src.shared.schemas import Transcript, Utterance, Storyboard, ProductionScript
from src.stages.stage2_llm.openai_gpt.openai_processor import OpenAIGPTProcessor

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
	"""Provides an OpenAIGPTProcessor instance configured for testing."""

	config = {"api_key": "test_key", "model": "gpt-4o"}
	with patch("src.stages.stage2_llm.openai_gpt.openai_processor.OpenAI"):
		return OpenAIGPTProcessor(config)

def test_initialization(processor):
	"""Verify that the processor initializes with the correct model and client."""

	assert processor.model == "gpt-4o"
	assert processor.client is not None

def test_map_speakers(processor, mock_transcript):
	"""Test mapping transcript speakers to character names using OpenAI."""

	# Mocking the OpenAI client response
	mock_response = MagicMock()
	mock_response.choices = [
		MagicMock(message=MagicMock(content="{\"Speaker A\": \"Hero\", \"Speaker B\": \"Sidekick\"}"))
	]
	processor.client.chat.completions.create.return_value = mock_response

	speaker_map = processor.map_speakers(mock_transcript)
	
	assert isinstance(speaker_map, dict)
	assert speaker_map["Speaker A"] == "Hero"
	assert speaker_map["Speaker B"] == "Sidekick"

def test_generate_storyboard(processor, mock_transcript):
	"""Test generating a storyboard from a transcript using OpenAI."""

	# Mocking the OpenAI client response for JSON format
	mock_response = MagicMock()
	mock_response.choices = [
		MagicMock(message=MagicMock(content='{"scenes": [{"scene_number": 1, "start_time": 0.0, "end_time": 4.0, "location": "Forest", "narrative_summary": "Greeting", "visual_prompt": "Two people meet in a forest"}]}'))
	]
	processor.client.chat.completions.create.return_value = mock_response

	speaker_map = {"Speaker A": "Hero", "Speaker B": "Sidekick"}
	storyboard = processor.generate_storyboard(mock_transcript, speaker_map)
	
	assert isinstance(storyboard, Storyboard)
	assert len(storyboard.scenes) == 1
	assert storyboard.scenes[0].location == "Forest"

def test_generate_production_script(processor, mock_transcript):
	"""Test generating a production script with stage directions using OpenAI."""

	storyboard = Storyboard(scenes=[
		{
			"scene_number": 1,
			"start_time": 0.0,
			"end_time": 4.0,
			"location": "Forest",
			"narrative_summary": "Greeting",
			"visual_prompt": "Two people meet in a forest"
		}
	])
	
	# Mocking the OpenAI client response
	mock_response = MagicMock()
	mock_response.choices = [
		MagicMock(message=MagicMock(content='{"scenes": [{"scene_number": 1, "start_time": 0.0, "end_time": 4.0, "location": "Forest", "narrative_summary": "Greeting", "visual_prompt": "Two people meet in a forest", "stage_directions": "Birds chirping", "character_actions": "Walking towards each other", "final_video_prompt": "Cinematic shot of meeting"}]}'))
	]
	processor.client.chat.completions.create.return_value = mock_response

	script = processor.generate_production_script(storyboard, mock_transcript)
	
	assert isinstance(script, ProductionScript)
	assert len(script.scenes) == 1
	assert script.scenes[0].stage_directions == "Birds chirping"
