"""
Stage 3 Video Generation ,  Minimax Hailuo Backend
==================================================
Uses the Minimax Hailuo-02 model via the fal.ai platform for text-to-video
generation.

fal.ai platform:
    fal.ai is an AI inference platform that hosts and serves various video/image
    generation models.  The ``fal_client`` Python SDK provides both sync and async
    interfaces.  This backend uses ``fal_client.submit_async`` to submit the job
    asynchronously and ``handler.get()`` to wait for completion.

Authentication:
    fal_client reads the API key from the ``FAL_KEY`` environment variable
    automatically.  There is no explicit credential injection in this class , 
    if ``FAL_KEY`` is absent, the error is raised lazily by fal_client when the
    first API call is made.

Coordination note:
    Additional video generator integrations are developed in parallel branches.
    Avoid restructuring the class interface or ``__init__`` signature without
    coordinating first.

Dependencies:
    pip install fal-client httpx aiofiles
"""

import os
import asyncio
from typing import Dict, Any
import fal_client
import httpx
import aiofiles

from src.stages.stage3_video.base import BaseVideoGenerator
from src.shared.schemas import ProductionScene


class HailuoVideoGenerator(BaseVideoGenerator):
	"""
    Video generator using Minimax Hailuo-02 via the fal.ai platform.

    Submits a text-to-video request to fal.ai, waits for the result using the
    fal_client async handler, then downloads the generated video.

    Config keys:
        api_key / FAL_KEY: fal.ai API key.  If provided in config, it is written
            to the ``FAL_KEY`` environment variable so fal_client
            can read it automatically.
        max_concurrent: Max simultaneous requests (inherited from base).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Hailuo generator.

        fal_client reads ``FAL_KEY`` from the environment automatically, so no
        explicit client initialization is needed.  If an ``api_key`` is provided
        in config, it can be injected into the environment here.

        Args:
            config: Configuration dict.  See class docstring for keys.
        """
		super().__init__(config)
		self.config = config

		# fal_client automatically picks up FAL_KEY from the environment.
		# If the key is absent when generate_scene is called, fal_client will
		# raise an authentication error at that point.
		if not os.getenv("FAL_KEY") and "api_key" not in config:
			pass  # fal_client handles its own validation when called.

	async def generate_scene(self, scene: ProductionScene, output_dir: str) -> str:
		"""
        Generate a single video scene using Minimax Hailuo-02 on fal.ai.

        Workflow:
            1. Submit the job with ``fal_client.submit_async`` ,  returns a handler.
            2. Await ``handler.get()`` to block until the generation finishes.
            3. Extract the video URL from the result dict.
            4. Download the video using httpx (async) and aiofiles.

        The fal.ai Hailuo-02 standard text-to-video model typically produces
        5-second clips at up to 720p quality.

        Args:
            scene:      ProductionScene with prompt and metadata.
            output_dir: Directory to save the output MP4.

        Returns:
            Path to the saved MP4 file.

        Raises:
            Exception: If the fal.ai submission or download fails.
        """
		prompt = scene.final_video_prompt

		# Submit the text-to-video request to the Hailuo-02 standard endpoint.
		# fal_client.submit_async is an async wrapper around the fal.ai queue API.
		handler = await fal_client.submit_async(
			"fal-ai/minimax/hailuo-02/standard/text-to-video",
			arguments={"prompt": prompt}
		)

		# Wait (async) for the generation to complete.
		# handler.get() returns the full result dict when the job is done.
		result = await handler.get()
		# Extract the video URL from the fal.ai result structure.
		video_url = result['video']['url']

		# Ensure the output directory exists before writing.
		os.makedirs(output_dir, exist_ok=True)
		filename = f"scene_{scene.scene_number}_hailuo.mp4"
		filepath = os.path.join(output_dir, filename)

		# Download the video using httpx + aiofiles for fully async I/O.
		async with httpx.AsyncClient() as client:
			response = await client.get(video_url)
			response.raise_for_status()
			async with aiofiles.open(filepath, 'wb') as f:
				await f.write(response.content)

		return filepath
