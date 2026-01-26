"""
Stage 3 Video Generation ,  Pika Labs Backend
=============================================
Uses the Pika Labs 2.5 model via the Replicate API to generate video from text.

Pika Labs specializes in stylized, high-motion video generation with strong
support for fantasy and cinematic styles ,  a good fit for D&D campaign footage.

Uses the same Replicate async_run / sync fallback pattern as other Replicate
backends (CogVideoX, HunyuanVideo, LTX, Mochi).  See those files for details.

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


class PikaVideoGenerator(BaseVideoGenerator):
	"""
    Video generator using Pika Labs 2.5 via the Replicate API.

    Produces 16:9 aspect ratio videos well-suited for cinematic D&D content.

    Config keys:
        replicate_api_token: Replicate API token
            (falls back to REPLICATE_API_TOKEN env var).
        model_version: Replicate model path (default: ``"pika/pika-v1.0"``).
        max_concurrent: Max simultaneous API calls (inherited from base).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Replicate API client for Pika Labs.

        Args:
            config: Configuration dict.  See class docstring for keys.

        Raises:
            ValueError: If REPLICATE_API_TOKEN is absent from config and env.
        """
		super().__init__(config)
		self.api_token = config.get("replicate_api_token") or os.getenv("REPLICATE_API_TOKEN")
		if not self.api_token:
			raise ValueError("REPLICATE_API_TOKEN is missing from config and environment variables.")
		self.client = replicate.Client(api_token=self.api_token)
		self.model_version = config.get("model_version", "pika/pika-art")

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
        Generate a single video clip using Pika Labs 2.5 via Replicate.

        Tries async_run first; falls back to sync run in a thread executor.
        The resulting video URL is downloaded via aiohttp.

        Args:
            scene:      ProductionScene with prompt and metadata.
            output_dir: Directory to save the downloaded MP4.

        Returns:
            Path to the saved MP4 file.

        Raises:
            Exception: If both async and sync Replicate calls fail.
        """
		prompt = scene.final_video_prompt or scene.visual_prompt
		os.makedirs(output_dir, exist_ok=True)
		output_path = os.path.join(output_dir, f"scene_{scene.scene_number}.mp4")

		input_args = {
			"prompt": prompt,
			"aspect_ratio": "16:9",  # Widescreen format for cinematic output.
		}

		try:
			# Primary async path via Replicate SDK.
			response_url = await self.client.async_run(self.model_version, input=input_args)
			if isinstance(response_url, list):
				response_url = response_url[0]

			await self._download_video(response_url, output_path)
			return output_path
		except Exception:
			# Fallback: wrap synchronous Replicate SDK in a thread executor.
			loop = asyncio.get_event_loop()
			run_fn = lambda: self.client.run(self.model_version, input=input_args)
			response_url = await loop.run_in_executor(None, run_fn)
			if isinstance(response_url, list):
				response_url = response_url[0]

			await self._download_video(response_url, output_path)
			return output_path
