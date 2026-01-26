"""
Stage 3 Video Generation ,  CogVideoX Backend
=============================================
Uses CogVideoX-5B 1.5 via the Replicate API to generate cinematic video clips.

CogVideoX is an open-source video generation model from ZhipuAI/THUDM that
produces high-quality videos from text prompts.  It is accessed here through
Replicate, which hosts the model and provides a simple async API.

Replicate async_run / sync fallback pattern:
    ``replicate.Client.async_run`` is the preferred async method.  If it raises
    (e.g. older SDK version or transient error), we fall back to the synchronous
    ``client.run`` wrapped in ``run_in_executor`` to avoid blocking the event loop.
    This pattern is repeated across all Replicate-based generators.

Video parameters:
    - ``num_frames=48``   ,  ~2 seconds at 24 fps; sufficient for a scene clip.
    - ``guidance_scale=6.0`` ,  Controls how strictly the model follows the prompt.
      Higher values = more prompt-faithful but potentially less natural motion.

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


class CogVideoXGenerator(BaseVideoGenerator):
	"""
    Video generator using CogVideoX-5B 1.5 via the Replicate API.

    CogVideoX is an open-source video generation model from ZhipuAI / THUDM
    that produces cinematic 720p video from text prompts.

    Config keys:
        replicate_api_token: Replicate API token
            (falls back to REPLICATE_API_TOKEN env var).
        model_version: Replicate model path (default: ``"zsxkib/cogvideox-5b"``).
        max_concurrent: Max simultaneous API calls (inherited from base).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Replicate API client for CogVideoX.

        Args:
            config: Configuration dict.  See class docstring for keys.

        Raises:
            ValueError: If REPLICATE_API_TOKEN is absent from config and env.
        """
		super().__init__(config)  # Initialize semaphore in base class.
		self.api_token = config.get("replicate_api_token") or os.getenv("REPLICATE_API_TOKEN")
		if not self.api_token:
			raise ValueError("REPLICATE_API_TOKEN is missing from config and environment variables.")

		self.client = replicate.Client(api_token=self.api_token)
		# Replicate model path; can be pinned to a specific version hash.
		self.model_version = config.get("model_version", "zsxkib/cogvideox-5b")

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
        Generate a single video clip using CogVideoX-5B via Replicate.

        Tries async_run first for non-blocking execution.  Falls back to
        synchronous run in a thread executor if async_run fails.
        The resulting video URL is downloaded via aiohttp.

        Args:
            scene:      ProductionScene with prompt and scene metadata.
            output_dir: Directory to save the downloaded MP4.

        Returns:
            Path to the saved MP4 file.

        Raises:
            Exception: If both async and sync Replicate calls fail.
        """
		# Prefer the richer final_video_prompt; fall back to basic visual prompt.
		prompt = scene.final_video_prompt or scene.visual_prompt
		os.makedirs(output_dir, exist_ok=True)
		output_path = os.path.join(output_dir, f"scene_{scene.scene_number}.mp4")

		input_args = {
			"prompt": prompt,
			"num_frames": 48,       # ~2 seconds at 24 fps.
			"guidance_scale": 6.0,  # Prompt adherence strength (1.0-20.0 range).
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
