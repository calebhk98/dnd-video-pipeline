"""
Stage 3 Video Generation ,  Lightricks LTX-Video Backend
========================================================
Uses Lightricks' LTX-Video model via the Replicate API.

LTX-Video is an open-source real-time text-to-video model from Lightricks that
is optimized for fast inference.  At 121 frames (approximately 5 seconds at 24 fps)
and 1280x720 resolution, it produces cinematic-quality clips quickly compared to
heavier models like HunyuanVideo.

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


class LTXVideoGenerator(BaseVideoGenerator):
	"""
    Video generator using Lightricks LTX-Video via the Replicate API.

    LTX-Video is optimized for speed while maintaining high visual quality.
    Produces 1280x720 video at 121 frames (~5 seconds at 24 fps).

    Config keys:
        replicate_api_token: Replicate API token
            (falls back to REPLICATE_API_TOKEN env var).
        model_version: Replicate model path (default: ``"lightricks/ltx-video:0.9"``).
        max_concurrent: Max simultaneous API calls (inherited from base).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Replicate API client for LTX-Video.

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
		self.model_version = config.get("model_version", "lightricks/ltx-video:0.9")

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
        Generate a single video clip using LTX-Video via Replicate.

        Video parameters:
            - width=1280, height=720 ,  720p HD widescreen.
            - num_frames=121         ,  ~5 seconds at 24 fps.

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
			"width": 1280,      # 720p HD width.
			"height": 720,      # 720p HD height.
			"num_frames": 121,  # ~5 seconds at 24 fps.
		}

		try:
			response_url = await self.client.async_run(self.model_version, input=input_args)
			if isinstance(response_url, list):
				response_url = response_url[0]

			await self._download_video(response_url, output_path)
			return output_path
		except Exception:
			# Fallback to sync execution in a thread executor.
			try:
				loop = asyncio.get_event_loop()
				run_fn = lambda: self.client.run(self.model_version, input=input_args)
				response_url = await loop.run_in_executor(None, run_fn)
				if isinstance(response_url, list):
					response_url = response_url[0]

				await self._download_video(response_url, output_path)
				return output_path
			except Exception as inner_e:
				logger.error(f"Error generating LTX-Video for scene {scene.scene_number}: {inner_e}")
				raise inner_e
