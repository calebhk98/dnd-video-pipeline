"""
Tests for Luma Dream Machine video generation.
Covers parallel job submission, polling for status, and video downloading.
"""
import pytest

import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock

from src.stages.stage3_video.luma_dream_machine.luma_video_generator import LumaVideoGenerator
from src.shared.schemas import ProductionScene



@pytest.fixture
def mock_config():
	"""Provides sample configuration for the Luma generator."""

	return {"api_key": "test_luma_key", "model": "ray-2"}

@pytest.fixture
def sample_scenes():
	"""Provides a list of sample ProductionScene objects for testing."""

	scene1 = ProductionScene(
		scene_number=1,
		start_time=0.0,
		end_time=5.0,
		location="Forest",
		narrative_summary="Walking in forest",
		visual_prompt="A man walking in a dark forest",
		stage_directions="Walks slowly",
		character_actions="Looking around",
		final_video_prompt="A man walking in a dark cinematic forest, 4k"
	)
	scene2 = ProductionScene(
		scene_number=2,
		start_time=5.0,
		end_time=10.0,
		location="Castle",
		narrative_summary="Arrives at castle",
		visual_prompt="A large castle in the distance",
		stage_directions="Stops walking",
		character_actions="Stares at castle",
		final_video_prompt="A dark spooky castle under a full moon, 4k"
	)
	return [scene1, scene2]

@pytest.mark.asyncio
@patch('src.stages.stage3_video.luma_dream_machine.luma_video_generator.httpx.AsyncClient')
@patch('src.stages.stage3_video.luma_dream_machine.luma_video_generator.asyncio.sleep', new_callable=AsyncMock)
async def test_generate_all_scenes_parallel(mock_sleep, mock_client_class, mock_config, sample_scenes, tmp_path):
	"""Test parallel generation of multiple scenes with polling and download logic."""

	generator = LumaVideoGenerator(mock_config)
	
	# We need to mock the async methods
	# httpx.AsyncClient is used as a context manager
	mock_client = AsyncMock()
	mock_client_class.return_value.__aenter__.return_value = mock_client
	
	# Mocking POST response
	mock_post_response = MagicMock()
	mock_post_response.json.side_effect = [{"id": "gen_1"}, {"id": "gen_2"}]
	mock_client.post.return_value = mock_post_response
	
	# Mocking GET response for polling and then for downloading
	mock_get_poll_response1 = MagicMock()
	mock_get_poll_response1.json.return_value = {"state": "completed", "assets": {"video": "http://example.com/vid1.mp4"}}
	
	mock_get_poll_response2 = MagicMock()
	mock_get_poll_response2.json.return_value = {"state": "completed", "assets": {"video": "http://example.com/vid2.mp4"}}
	
	mock_get_download_response = MagicMock()
	mock_get_download_response.content = b"fake_video_content"
	
	# A side effect function to return different responses based on url
	def get_side_effect(url, headers=None, **kwargs):
		"""Mock side effect for httpx GET requests, returning responses based on URL patterns."""

		if "gen_1" in url:
			return mock_get_poll_response1
		elif "gen_2" in url:
			return mock_get_poll_response2
		elif "vid1" in url or "vid2" in url:
			return mock_get_download_response
		return MagicMock()
		
	mock_client.get.side_effect = get_side_effect
	
	output_dir = str(tmp_path)
	
	# Run the parallel generation
	result_paths, failures = await generator.generate_all_scenes(sample_scenes, output_dir)

	# Verifying the basic logic
	assert len(result_paths) == 2
	assert len(failures) == 0
	assert "scene_1.mp4" in result_paths[0]
	assert "scene_2.mp4" in result_paths[1]

	# Check that it actually output the files
	assert os.path.exists(result_paths[0])
	assert os.path.exists(result_paths[1])
	
	# Check that post was called twice
	assert mock_client.post.call_count == 2
	
	# Check that get was called for polling (2) + downloading (2) = 4
	assert mock_client.get.call_count == 4

@pytest.mark.asyncio
async def test_api_key_initialization():
	"""Test API key loading from config and environment variables."""

	generator = LumaVideoGenerator({"api_key": "test"})
	assert generator.api_key == "test"
	
	# Test fallback to env
	os.environ["LUMA_API_KEY"] = "env_test"
	generator2 = LumaVideoGenerator({})
	assert generator2.api_key == "env_test"
	
	# Test error on missing key
	del os.environ["LUMA_API_KEY"]
	with pytest.raises(ValueError, match="Luma API key is required either in config or environment variable LUMA_API_KEY."):
		LumaVideoGenerator({})
