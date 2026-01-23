"""
Stage 3 Video Generation ,  Pixverse v4 via Replicate Backend
=============================================================
Uses the Pixverse v4 model via the Replicate API for 720p video generation.

Pixverse is a video generation model that produces cinematic results at 720p
resolution.  This is the original/primary Replicate backend in the pipeline , 
other Replicate-based generators (CogVideoX, Pika, HunyuanVideo, LTX, Mochi)
follow the same pattern established here.

Coordination note:
    Additional video generator integrations are developed in parallel branches.
    Avoid restructuring the class interface, import order, or ``__init__``
    signature without coordinating first, to prevent merge conflicts.

Dependencies:
    pip install replicate aiohttp
"""

import os
import aiohttp
import asyncio
import logging
import replicate
from typing import Dict, Any
from src.stages.stage3_video.base import BaseVideoGenerator
from src.shared.schemas import ProductionScene

logger = logging.getLogger(__name__)


class ReplicateVideoGenerator(BaseVideoGenerator):
	"""
    Video generator using Pixverse v4 via the Replicate API.

    Uses the Replicate Python SDK to run Pixverse v4 with async_run, falling back
    to synchronous execution in a thread if async is unavailable.

    Config keys:
        replicate_api_token: Replicate API token
            (falls back to REPLICATE_API_TOKEN env var).
        model_version: Replicate model path (default: ``"pixverse/replicate-video"``).
        max_concurrent: Max simultaneous API calls (inherited from base).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Replicate API client for Pixverse.

        Args:
            config: Configuration dict.  See class docstring for keys.

        Raises:
            ValueError: If REPLICATE_API_TOKEN is absent from config and env.
        """
		super().__init__(config)
		self.api_token = config.get("replicate_api_token") or os.getenv("REPLICATE_API_TOKEN")
		if not self.api_token:
			raise ValueError("REPLICATE_API_TOKEN is missing from config and environment variables.")

		# Initialize the Replicate client with the API token.
		self.client = replicate.Client(api_token=self.api_token)
		# Model path on Replicate.  Can be pinned to a specific version hash.
		self.model_version = config.get("model_version", "pixverse/pixverse-v4")

	async def _download_video(self, url: str, output_path: str):
		"""Download a video file via HTTP."""
		async with aiohttp.ClientSession() as session:
			async with session.get(str(url)) as resp:
				resp.raise_for_status()
				content = await resp.read()
				with open(output_path, "wb") as f:
					f.write(content)

	async def generate_scene(self, scene: ProductionScene, output_dir: str) -> str:
		"""
        Generate a single video scene using Pixverse v4 via Replicate.

        Attempts async_run first (available in Replicate SDK >= 0.20.0).
        Falls back to synchronous ``run`` wrapped in a thread executor if
        ``async_run`` raises an exception.

        Args:
            scene:      ProductionScene with prompt and metadata.
            output_dir: Directory to save the downloaded MP4.

        Returns:
            Path to the saved MP4 file.

        Raises:
            Exception: If both primary and fallback paths fail.
        """
		# Prefer the detailed production prompt over the basic visual prompt.
		prompt = scene.final_video_prompt
		if not prompt:
			prompt = scene.visual_prompt

		os.makedirs(output_dir, exist_ok=True)
		output_path = os.path.join(output_dir, f"scene_{scene.scene_number}.mp4")

		# Input parameters for the Pixverse v4 model.
		# 720p is the recommended resolution for quality/speed balance.
		input_args = {
			"prompt": prompt,
			"resolution": "720p"  # Pixverse also supports "540p" for faster generation.
		}

		try:
			response_url = await self.client.async_run(self.model_version, input=input_args)
			if isinstance(response_url, list):
				response_url = response_url[0]

			await self._download_video(response_url, output_path)
			return output_path
		except Exception:
			loop = asyncio.get_event_loop()
			run_fn = lambda: self.client.run(self.model_version, input=input_args)
			response_url = await loop.run_in_executor(None, run_fn)
			if isinstance(response_url, list):
				response_url = response_url[0]

			await self._download_video(response_url, output_path)
			return output_path
