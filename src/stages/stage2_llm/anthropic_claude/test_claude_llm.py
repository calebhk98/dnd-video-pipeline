"""
Unit tests for the Anthropic Claude LLM processor.
Comprehensive tests covering speaker mapping, storyboard/script generation, and edge cases.
"""
import pytest

import json
from unittest.mock import MagicMock, patch
from src.shared.schemas import Transcript, Utterance, Storyboard, ProductionScript, Scene, ProductionScene, SceneShot
from src.stages.stage2_llm.anthropic_claude.claude_processor import ClaudeProcessor

@pytest.fixture
def mock_config():
	"""Provides a sample configuration dictionary for testing."""

	return {"api_key": "test_key"}

@pytest.fixture
def sample_transcript():
	"""Provides a sample Transcript object for testing."""

	return Transcript(
		audio_duration=10.0,
		status="completed",
		utterances=[
			Utterance(speaker="Speaker A", text="Hello everyone, I'm Magnus.", start=0.0, end=2.0),
			Utterance(speaker="Speaker B", text="Hi Magnus, I'm Taako.", start=2.5, end=4.5)
		],
		full_text="Speaker A: Hello everyone, I'm Magnus. Speaker B: Hi Magnus, I'm Taako."
	)

def test_map_speakers(mock_config, sample_transcript):
	"""Test mapping generic speaker labels to character names using Claude."""

	with patch('anthropic.Anthropic') as MockAnthropic:
		mock_client = MockAnthropic.return_value
		mock_client.messages.create.return_value.content = [
			MagicMock(text='{"Speaker A": "Magnus", "Speaker B": "Taako"}')
		]
		
		processor = ClaudeProcessor(mock_config)
		speaker_map = processor.map_speakers(sample_transcript)
		
		assert speaker_map == {"Speaker A": "Magnus", "Speaker B": "Taako"}

def test_generate_storyboard(mock_config, sample_transcript):
	"""Test generating a high-level storyboard from a transcript."""

	speaker_map = {"Speaker A": "Magnus", "Speaker B": "Taako"}
	
	with patch('anthropic.Anthropic') as MockAnthropic:
		mock_client = MockAnthropic.return_value
		
		# Mock tool call response for Storyboard
		mock_tool_use = MagicMock()
		mock_tool_use.type = "tool_use"
		mock_tool_use.name = "generate_storyboard"
		mock_tool_use.input = {
			"scenes": [
				{
					"scene_number": 1,
					"start_time": 0.0,
					"end_time": 4.5,
					"location": "Tavern",
					"narrative_summary": "Introduction",
					"visual_prompt": "Tavern scene"
				}
			]
		}
		
		mock_response = MagicMock()
		mock_response.content = [mock_tool_use]
		mock_client.messages.create.return_value = mock_response
		
		processor = ClaudeProcessor(mock_config)
		storyboard = processor.generate_storyboard(sample_transcript, speaker_map)
		
		assert isinstance(storyboard, Storyboard)
		assert len(storyboard.scenes) == 1

def test_generate_production_script(mock_config, sample_transcript):
	"""Test generating a production script with visual prompts and stage directions."""

	storyboard = Storyboard(scenes=[
		Scene(
			scene_number=1,
			start_time=0.0,
			end_time=4.5,
			location="Tavern",
			narrative_summary="Introduction",
			visual_prompt="Tavern scene"
		)
	])
	
	with patch('anthropic.Anthropic') as MockAnthropic:
		mock_client = MockAnthropic.return_value
		
		mock_tool_use = MagicMock()
		mock_tool_use.type = "tool_use"
		mock_tool_use.name = "generate_production_script"
		mock_tool_use.input = {
			"scenes": [
				{
					"scene_number": 1,
					"start_time": 0.0,
					"end_time": 4.5,
					"location": "Tavern",
					"narrative_summary": "Introduction",
					"visual_prompt": "Tavern scene",
					"stage_directions": "Magnus waves his hand.",
					"character_actions": "Taako tips his hat.",
					"final_video_prompt": "Cinematic shot of Magnus and Taako in a tavern."
				}
			]
		}
		
		mock_response = MagicMock()
		mock_response.content = [mock_tool_use]
		mock_client.messages.create.return_value = mock_response
		
def test_map_speakers_invalid_json(mock_config, sample_transcript):
	"""Test fallback behavior when the LLM returns invalid JSON for speaker mapping."""

	with patch('anthropic.Anthropic') as MockAnthropic:
		mock_client = MockAnthropic.return_value
		# Mock invalid JSON response
		mock_client.messages.create.return_value.content = [
			MagicMock(text='Invalid JSON')
		]
		
		processor = ClaudeProcessor(mock_config)
		speaker_map = processor.map_speakers(sample_transcript)
		
		# Should fallback to basic mapping
		assert speaker_map["Speaker A"] == "Speaker A"

def test_generate_storyboard_empty_transcript(mock_config):
	"""Test storyboard generation with an empty transcript."""

	empty_transcript = Transcript(audio_duration=0.0, status="completed", utterances=[], full_text="")
	speaker_map = {}
	
	with patch('anthropic.Anthropic') as MockAnthropic:
		mock_client = MockAnthropic.return_value
		mock_client.messages.create.return_value.content = [] # No tool use
		
		processor = ClaudeProcessor(mock_config)
		storyboard = processor.generate_storyboard(empty_transcript, speaker_map)
		
		assert isinstance(storyboard, Storyboard)
		assert len(storyboard.scenes) == 0

def test_generate_production_script_fallback(mock_config, sample_transcript):
	"""Test production script generation fallback when no tool calls are returned."""

	storyboard = Storyboard(scenes=[])
	
	with patch('anthropic.Anthropic') as MockAnthropic:
		mock_client = MockAnthropic.return_value
		mock_client.messages.create.return_value.content = [] # No tool use
		
		processor = ClaudeProcessor(mock_config)
		production_script = processor.generate_production_script(storyboard, sample_transcript)
		
		assert isinstance(production_script, ProductionScript)
		assert len(production_script.scenes) == 0

def test_map_speakers_markdown_json(mock_config, sample_transcript):
	"""Test parsing of speaker mapping JSON when wrapped in markdown code blocks."""

	with patch('anthropic.Anthropic') as MockAnthropic:
		mock_client = MockAnthropic.return_value
		# Mock JSON wrapped in markdown
		mock_client.messages.create.return_value.content = [
			MagicMock(text='```json\n{"Speaker A": "Magnus"}\n```')
		]

		processor = ClaudeProcessor(mock_config)
		speaker_map = processor.map_speakers(sample_transcript)

		assert speaker_map == {"Speaker A": "Magnus"}


@pytest.fixture
def sample_storyboard():
	"""Provides a sample Storyboard object with multiple scenes."""

	return Storyboard(scenes=[
		Scene(
			scene_number=1,
			start_time=0.0,
			end_time=60.0,
			location="Tavern",
			narrative_summary="The party gathers at the tavern and meets their quest giver.",
			visual_prompt="Fantasy tavern interior, warm firelight, adventurers gathered around a table."
		),
		Scene(
			scene_number=2,
			start_time=60.0,
			end_time=120.0,
			location="Out of game",
			narrative_summary="Players take a short bathroom break and discuss pizza orders.",
			visual_prompt=""
		),
	])


def test_review_scene_relevance(mock_config, sample_storyboard):
	"""Test reviewing scenes for relevance (filtering out-of-character moments)."""

	with patch('anthropic.Anthropic') as MockAnthropic:
		mock_client = MockAnthropic.return_value

		mock_tool_use = MagicMock()
		mock_tool_use.type = "tool_use"
		mock_tool_use.name = "review_scene_relevance"
		# relevance_reason comes before is_relevant -- matching the tool schema field order
		# so the model reasons before committing to the boolean.
		mock_tool_use.input = {
			"scenes": [
				{"scene_number": 1, "relevance_reason": "Contains in-game tavern events.", "is_relevant": True},
				{"scene_number": 2, "relevance_reason": "Purely out-of-character bathroom break.", "is_relevant": False},
			]
		}

		mock_response = MagicMock()
		mock_response.content = [mock_tool_use]
		mock_client.messages.create.return_value = mock_response

		processor = ClaudeProcessor(mock_config)
		reviewed = processor.review_scene_relevance(sample_storyboard)

		assert isinstance(reviewed, Storyboard)
		assert len(reviewed.scenes) == 2
		assert reviewed.scenes[0].is_relevant is True
		assert reviewed.scenes[0].relevance_reason == "Contains in-game tavern events."
		assert reviewed.scenes[1].is_relevant is False
		assert reviewed.scenes[1].relevance_reason == "Purely out-of-character bathroom break."


def test_review_scene_relevance_fallback_on_no_tool_call(mock_config, sample_storyboard):
	"""When the LLM returns no tool call, all scenes should default to relevant."""
	with patch('anthropic.Anthropic') as MockAnthropic:
		mock_client = MockAnthropic.return_value
		mock_response = MagicMock()
		mock_response.content = []  # No tool use block
		mock_client.messages.create.return_value = mock_response

		processor = ClaudeProcessor(mock_config)
		reviewed = processor.review_scene_relevance(sample_storyboard)

		assert all(s.is_relevant is True for s in reviewed.scenes)


def test_generate_scene_shots(mock_config, sample_storyboard):
	"""Test generating individual shots for each scene in a storyboard."""

	storyboard = Storyboard(scenes=[
		Scene(
			scene_number=1,
			start_time=0.0,
			end_time=60.0,
			location="Road to Phandalin",
			narrative_summary="Party travels and encounters dead horses; prepares for ambush.",
			visual_prompt="Fantasy road, wagon, adventurers."
		)
	])

	with patch('anthropic.Anthropic') as MockAnthropic:
		mock_client = MockAnthropic.return_value

		mock_tool_use = MagicMock()
		mock_tool_use.type = "tool_use"
		mock_tool_use.name = "generate_scene_shots"
		mock_tool_use.input = {
			"scenes": [{
				"scene_number": 1, "start_time": 0.0, "end_time": 60.0,
				"location": "Road to Phandalin",
				"narrative_summary": "Party travels and encounters dead horses.",
				"visual_prompt": "Fantasy road, wagon, adventurers.",
				"stage_directions": "Wide establishing shot.",
				"character_actions": "Party rides in wagon.",
				"final_video_prompt": "Fantasy road through green hills.",
				"shots": [
					{"shot_number": 1, "description": "Wagon rolls.", "visual_prompt": "Wide shot.", "duration_hint": 5},
					{"shot_number": 2, "description": "Dead horses.", "visual_prompt": "POV shot.", "duration_hint": 5},
					{"shot_number": 3, "description": "Party wary.", "visual_prompt": "Close-up.", "duration_hint": 5}
				]
			}]
		}

		mock_response = MagicMock()
		mock_response.content = [mock_tool_use]
		mock_client.messages.create.return_value = mock_response

		processor = ClaudeProcessor(mock_config)
		production_script = processor.generate_scene_shots(storyboard, sample_transcript)

		assert isinstance(production_script, ProductionScript)
		assert len(production_script.scenes) == 1

		scene = production_script.scenes[0]
		assert isinstance(scene, ProductionScene)
		assert scene.stage_directions == "Wide establishing shot transitioning to medium close-ups."
		assert len(scene.shots) == 3

		shot = scene.shots[1]
		assert isinstance(shot, SceneShot)
		assert shot.shot_number == 2
		assert shot.duration_hint == 5
		assert "dead horses" in shot.visual_prompt


def test_generate_scene_shots_empty_storyboard(mock_config, sample_transcript):
	"""Test shot generation with an empty storyboard."""

	storyboard = Storyboard(scenes=[])

	with patch('anthropic.Anthropic') as MockAnthropic:
		mock_client = MockAnthropic.return_value
		mock_response = MagicMock()
		mock_response.content = []  # No tool use
		mock_client.messages.create.return_value = mock_response

		processor = ClaudeProcessor(mock_config)
		production_script = processor.generate_scene_shots(storyboard, sample_transcript)

		assert isinstance(production_script, ProductionScript)
		assert len(production_script.scenes) == 0


def test_generate_scene_shots_no_shot_limit(mock_config, sample_transcript):
	"""A long combat scene should be allowed to produce more than 8 shots."""
	shots_data = [
		{"shot_number": i, "description": f"Combat beat {i}.", "visual_prompt": f"Action shot {i}.", "duration_hint": 5}
		for i in range(1, 16)  # 15 shots -- well above the old 3-8 limit
	]
	storyboard = Storyboard(scenes=[
		Scene(
			scene_number=5,
			start_time=1198.0,
			end_time=1928.0,
			location="Forest path",
			narrative_summary="Goblins attack the party in a prolonged battle.",
			visual_prompt="Goblin ambush on a forest path."
		)
	])

	with patch('anthropic.Anthropic') as MockAnthropic:
		mock_client = MockAnthropic.return_value
		mock_tool_use = MagicMock()
		mock_tool_use.type = "tool_use"
		mock_tool_use.name = "generate_scene_shots"
		mock_tool_use.input = {
			"scenes": [{
				"scene_number": 5,
				"start_time": 1198.0,
				"end_time": 1928.0,
				"location": "Forest path",
				"narrative_summary": "Goblins attack the party in a prolonged battle.",
				"visual_prompt": "Goblin ambush on a forest path.",
				"stage_directions": "Dynamic handheld camera, chaotic battle energy.",
				"character_actions": "Warriors swing swords; wizards cast spells; goblins swarm.",
				"final_video_prompt": "Intense goblin ambush on a dark forest path, magic and steel.",
				"shots": shots_data,
			}]
		}
		mock_response = MagicMock()
		mock_response.content = [mock_tool_use]
		mock_client.messages.create.return_value = mock_response

		processor = ClaudeProcessor(mock_config)
		production_script = processor.generate_scene_shots(storyboard, sample_transcript)

		scene = production_script.scenes[0]
		assert len(scene.shots) == 15, "Long combat scenes should not be limited to 8 shots"


def test_get_scene_transcript_filters_by_time(mock_config):
	"""_get_scene_transcript should return only utterances within the scene's time range."""
	transcript = Transcript(
		audio_duration=20.0,
		status="completed",
		utterances=[
			Utterance(speaker="DM",     text="Before scene.",   start=0.0,  end=2.0),
			Utterance(speaker="Magnus", text="In scene early.", start=5.0,  end=8.0),
			Utterance(speaker="Taako",  text="In scene late.",  start=9.0,  end=12.0),
			Utterance(speaker="DM",     text="After scene.",	start=15.0, end=18.0),
		],
		full_text=""
	)
	scene = Scene(
		scene_number=1, start_time=4.0, end_time=13.0,
		location="Road", narrative_summary="Travel.", visual_prompt=""
	)

	processor = ClaudeProcessor({"api_key": "test_key"})
	result = processor._get_scene_transcript(transcript, scene)

	assert "In scene early." in result
	assert "In scene late." in result
	assert "Before scene." not in result
	assert "After scene." not in result
