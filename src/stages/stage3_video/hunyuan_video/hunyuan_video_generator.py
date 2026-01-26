"""
Stage 3 Video Generation ,  Tencent HunyuanVideo Backend
========================================================
Uses Tencent's HunyuanVideo 1.5 model via the Replicate API for high-resolution
text-to-video generation.

HunyuanVideo produces 1280x720 (720p) video at up to 129 frames (~5 seconds at
~24 fps), making it one of the highest-quality open-source video models available.
It is accessed through Replicate, which handles GPU scheduling and serves the
model on demand.

Uses the same Replicate async_run / sync fallback pattern as other Replicate
backends.  See cogvideox_generator.py for the detailed pattern description.

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


class HunyuanVideoGenerator(BaseVideoGenerator):
	"""
    Video generator using Tencent HunyuanVideo 1.5 via the Replicate API.

    Produces 1280x720 video at 129 frames (~5.4 seconds).  Good quality for
    complex D&D scenes that benefit from detailed world rendering.

    Config keys:
        replicate_api_token: Replicate API token
            (falls back to REPLICATE_API_TOKEN env var).
        model_version: Replicate model path (default: ``"lucataco/hunyuanvideo:7b"``).
        max_concurrent: Max simultaneous API calls (inherited from base).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Replicate API client for HunyuanVideo.

        Args:
            config: Configuration dict.  See class docstring for keys.

        Raises:
            ValueError: If REPLICATE_API_TOKEN is absent.
        """
		super().__init__(config)
		self.api_token = config.get("replicate_api_token") or os.getenv("REPLICATE_API_TOKEN")
		if not self.api_token:
			raise ValueError("REPLICATE_API_TOKEN is missing from config and environment variables.")
		self.client = replicate.Client(api_token=self.api_token)
		self.model_version = config.get("model_version", "lucataco/hunyuanvideo:7b")

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
        Generate a single video clip using HunyuanVideo 1.5 via Replicate.

        Video parameters:
            - width=1280, height=720 ,  720p HD output.
            - num_frames=129         ,  ~5.4 seconds at 24 fps; near-standard clip length.

        Args:
            scene:      ProductionScene with prompt and metadata.
            output_dir: Directory to save the downloaded MP4.

        Returns:
            Path to the saved MP4 file.
        """
		prompt = scene.final_video_prompt or scene.visual_prompt
		os.makedirs(output_dir, exist_ok=True)
		output_path = os.path.join(output_dir, f"scene_{scene.scene_number}.mp4")

		input_args = {
			"prompt": prompt,
			"width": 1280,      # HD widescreen output.
			"height": 720,
			"num_frames": 129,  # ~5.4 seconds at 24 fps.
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
