"""
Integration tests for the Hailuo Video Generator (Minimax).
Verifies that the generator can successfully produce an mp4 file from a ProductionScene.
"""
import pytest

import os
import tempfile
from src.stages.stage3_video.minimax_hailuo.hailuo_video_generator import HailuoVideoGenerator
from src.shared.schemas import ProductionScene

@pytest.mark.asyncio
async def test_hailuo_video_generator():
	"""Test the core video generation functionality using the Minimax Hailuo API."""

	if not os.getenv("FAL_KEY"):
		pytest.skip("FAL_KEY environment variable is missing, skipping API test.")
		
	config = {}
	generator = HailuoVideoGenerator(config)
	
	scene = ProductionScene(
		scene_number=1,
		start_time=0.0,
		end_time=6.0,
		location="Beach",
		narrative_summary="A wide shot of a dog running on a beach.",
		visual_prompt="A wide shot of a dog running on a beautiful beach, 4k",
		stage_directions="Pan right and follow the dog",
		character_actions="Dog runs happily",
		final_video_prompt="A wide golden retriever running on a beautiful beach, sunset, 4k, cinematic, slow motion. Professional cinematography"
	)
	
	with tempfile.TemporaryDirectory() as tmpdir:
		filepath = await generator.generate_scene(scene, tmpdir)
		
		assert os.path.exists(filepath), "Video file was not created"
		assert filepath.endswith(".mp4"), "Video file should be an mp4"
		assert os.path.getsize(filepath) > 0, "Video file is empty"
