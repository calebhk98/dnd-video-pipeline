"""
Stage 3 Video Generation ,  Kling AI Backend
============================================
Uses Kling AI v1.6 via the fal.ai platform for high-quality cinematic video
generation from text prompts.

Kling AI overview:
    Kling AI is a video generation model from Kuaishou Technology known for
    producing high-fidelity cinematic videos with realistic motion and lighting.
    It is particularly well-suited for fantasy and dramatic content, making it
    an excellent match for D&D campaign cinematics.

fal.ai authentication:
    fal_client reads ``FAL_KEY`` from the environment.  If an ``api_key`` is
    provided in config, it is written to ``os.environ["FAL_KEY"]`` so fal_client
    can pick it up automatically for all subsequent calls.

Configurable parameters:
    ``duration`` and ``aspect_ratio`` are surfaced as config options because
    they directly affect generation time, cost, and output format.

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


class KlingVideoGenerator(BaseVideoGenerator):
	"""
    Video generator using Kling AI v1.6 via the fal.ai platform.

    Produces high-quality cinematic video at configurable duration and aspect ratio.

    Config keys:
        api_key / FAL_KEY: fal.ai API key.  If provided in config, it is
            injected into the environment for fal_client.
        duration: Video length in seconds: ``"5"`` or ``"10"``
            (default: ``"5"``).  Longer clips cost more compute.
        aspect_ratio: Output aspect ratio: ``"16:9"``, ``"9:16"``, or ``"1:1"``
            (default: ``"16:9"``).
        max_concurrent: Max simultaneous requests (inherited from base).
    """

	# fal.ai endpoint for Kling AI v1.6 standard text-to-video.
	FAL_ENDPOINT = "fal-ai/kling-video/v1.6/standard/text-to-video"

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Kling AI generator and inject the API key into the environment.

        Args:
            config: Configuration dict.  See class docstring for keys.
        """
		super().__init__(config)

		# If an API key is provided explicitly, write it to the environment
		# so fal_client picks it up via os.environ["FAL_KEY"].
		api_key = config.get("api_key") or os.getenv("FAL_KEY")
		if api_key:
			os.environ["FAL_KEY"] = api_key

		# Duration as a string because fal.ai Kling endpoint expects "5" or "10".
		self.duration = config.get("duration", "5")
		self.aspect_ratio = config.get("aspect_ratio", "16:9")

	async def generate_scene(self, scene: ProductionScene, output_dir: str) -> str:
		"""
        Generate a single video scene using Kling AI v1.6 via fal.ai.

        Workflow:
            1. Submit the job with ``fal_client.submit_async`` ,  non-blocking queue entry.
            2. Await ``handler.get()`` ,  blocks until the generation is complete.
            3. Extract video URL from the result.
            4. Download the video using httpx + aiofiles.

        Args:
            scene:      ProductionScene with prompt and metadata.
            output_dir: Directory to save the output MP4.

        Returns:
            Path to the saved MP4 file.
        """
		prompt = scene.final_video_prompt or scene.visual_prompt

		# Submit the generation request to the Kling AI endpoint on fal.ai.
		handler = await fal_client.submit_async(
			self.FAL_ENDPOINT,
			arguments={
				"prompt": prompt,
				"duration": self.duration,       # "5" or "10" seconds.
				"aspect_ratio": self.aspect_ratio,
			},
		)

		# Wait for the job to finish and retrieve the result.
		result = await handler.get()
		# fal.ai Kling result structure: {"video": {"url": "https://..."}}
		video_url = result["video"]["url"]

		os.makedirs(output_dir, exist_ok=True)
		filepath = os.path.join(output_dir, f"scene_{scene.scene_number}_kling.mp4")

		# Download the generated video using async httpx + aiofiles.
		async with httpx.AsyncClient() as client:
			response = await client.get(video_url)
			response.raise_for_status()
			async with aiofiles.open(filepath, "wb") as f:
				await f.write(response.content)

		return filepath
