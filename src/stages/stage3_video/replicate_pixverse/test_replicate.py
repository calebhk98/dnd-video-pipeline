"""
Tests for Replicate Pixverse video generation.
Verifies parallel scene generation and file downloading.
"""
import sys

from unittest.mock import MagicMock
sys.modules['replicate'] = MagicMock()

import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from src.stages.stage3_video.replicate_pixverse.replicate_video_generator import ReplicateVideoGenerator
from src.shared.schemas import ProductionScene

@pytest.fixture
def mock_scenes():
	"""Provides a list of sample ProductionScene objects for testing."""

	return [
		ProductionScene(
			scene_number=1,
			start_time=0.0,
			end_time=5.0,
			location="Space",
			narrative_summary="A spaceship flying",
			visual_prompt="A spaceship flying through space",
			stage_directions="Fast movement",
			character_actions="None",
			final_video_prompt="A spaceship flying through space, fast movement"
		),
		ProductionScene(
			scene_number=2,
			start_time=5.0,
			end_time=10.0,
			location="Mars",
			narrative_summary="Landing on Mars",
			visual_prompt="Spaceship landing on red planet",
			stage_directions="Slow descent",
			character_actions="None",
			final_video_prompt="Spaceship landing on red planet, slow descent"
		),
		ProductionScene(
			scene_number=3,
			start_time=10.0,
			end_time=15.0,
			location="Mars Surface",
			narrative_summary="Astronaut steps out",
			visual_prompt="Astronaut on Mars",
			stage_directions="Dramatic low angle",
			character_actions="Steps onto dust",
			final_video_prompt="Astronaut on Mars, dramatic low angle, steps onto dust"
		)
	]

@pytest.mark.asyncio
async def test_replicate_video_generator_parallelism(mock_scenes, tmp_path):
	"""Test that video generation calls are made in parallel and results are saved."""

	# Mock replicate client
	with patch('src.stages.stage3_video.replicate_pixverse.replicate_video_generator.replicate.Client') as mock_replicate_client:
		mock_instance = MagicMock()
		mock_replicate_client.return_value = mock_instance
		
		# We also need to mock aiohttp to avoid real downloads
		with patch('src.stages.stage3_video.replicate_pixverse.replicate_video_generator.aiohttp.ClientSession') as mock_session_class:
			mock_session = MagicMock()
			mock_session_class.return_value.__aenter__.return_value = mock_session
			
			mock_response = AsyncMock()
			mock_session.get.return_value.__aenter__.return_value = mock_response
			
			mock_response.read.return_value = b'fake_video_data'
			mock_response.raise_for_status = MagicMock()
			
			config = {"replicate_api_token": "fake_token", "model_version": "pixverse/pixverse-v4"}
			generator = ReplicateVideoGenerator(config)
			
			# Mock the replicate async_run to return a fake URL list
			# It needs to return a URL or list of URLs
			mock_instance.async_run = AsyncMock(return_value=["http://fake.url/video.mp4"])
			
			output_dir = str(tmp_path)
			
			# Run the parallel generation
			results = await generator.generate_all_scenes(mock_scenes, output_dir)
			
			# Verify we got 3 valid paths back
			assert len(results) == 3
			assert all(os.path.exists(path) for path in results)
			
			# Verify async_run was called exactly 3 times
			assert mock_instance.async_run.call_count == 3
