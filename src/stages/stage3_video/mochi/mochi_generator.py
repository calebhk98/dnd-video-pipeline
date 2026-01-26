"""
Stage 3 Video Generation ,  Mochi 1 by Genmo Backend
====================================================
Uses Genmo's Mochi 1 model via the Replicate API for high-fidelity video generation.

Mochi 1 is an open-source video generation model focused on high visual quality
and smooth motion.  It uses a simpler API surface than multi-parameter models
(no explicit resolution or frame count parameters ,  the model uses its defaults).

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


class MochiGenerator(BaseVideoGenerator):
	"""
    Video generator using Mochi 1 by Genmo via the Replicate API.

    Mochi 1 produces high-fidelity video with smooth motion and strong prompt
    adherence.  It accepts only a prompt (no resolution/frame controls exposed
    via Replicate's hosted version).

    Config keys:
        replicate_api_token: Replicate API token
            (falls back to REPLICATE_API_TOKEN env var).
        model_version: Replicate model path (default: ``"genmoai/mochi-1"``).
        max_concurrent: Max simultaneous API calls (inherited from base).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Replicate API client for Mochi 1.

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
		self.model_version = config.get("model_version", "genmoai/mochi-1")

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
        Generate a single video clip using Mochi 1 via Replicate.

        Mochi 1 only requires a ``prompt`` input; resolution and frame count are
        fixed by the model internally.

        Args:
            scene:      ProductionScene with prompt and metadata.
            output_dir: Directory to save the downloaded MP4.

        Returns:
            Path to the saved MP4 file.
        """
		prompt = scene.final_video_prompt or scene.visual_prompt
		os.makedirs(output_dir, exist_ok=True)
		output_path = os.path.join(output_dir, f"scene_{scene.scene_number}.mp4")

		# Mochi 1 on Replicate only accepts a prompt; no resolution/frame overrides.
		input_args = {
			"prompt": prompt,
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
