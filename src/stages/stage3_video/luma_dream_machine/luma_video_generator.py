"""
Stage 3 Video Generation ,  Luma Dream Machine Backend
======================================================
Uses the Luma Labs Dream Machine (Ray model) via the official REST API to
generate cinematic video from text prompts.

Luma API flow:
    1. POST to ``/dream-machine/v1/generations`` with prompt and aspect ratio.
    2. API returns a generation ID immediately (async job).
    3. Poll ``GET /dream-machine/v1/generations/{id}`` every 5 seconds.
    4. When ``state == "completed"``, retrieve ``assets.video`` URL.
    5. Download the MP4 from the video URL.

Retry logic:
    The entire ``_attempt`` closure (submit + poll + download) is wrapped in
    ``retry_async`` from the shared utilities module.  Up to 3 attempts are made
    with exponential back-off (base delay: 2 seconds) to handle transient API
    errors (network glitches, brief 5xx responses, etc.).

Polling limits:
    ``MAX_POLL_ATTEMPTS = 120`` at 5-second intervals gives a 10-minute window,
    which is generous for typical Luma generation times (30-120 seconds).

Dependencies:
    pip install httpx aiofiles
"""

import os
import asyncio
import logging
import httpx
from typing import Dict, Any
from pathlib import Path

from src.stages.stage3_video.base import BaseVideoGenerator
from src.shared.schemas import ProductionScene
from src.shared.utils.retry import retry_async

logger = logging.getLogger(__name__)

# Luma Dream Machine REST API endpoint.
LUMA_API_URL = "https://api.lumalabs.ai/dream-machine/v1/generations"
# Seconds between generation status polls.
POLL_INTERVAL_SECONDS = 5
# Default timeout for individual HTTP requests (connect + read).
DEFAULT_TIMEOUT_SECONDS = 30.0
# Maximum number of polls: 120 x 5s = 10 minutes.
MAX_POLL_ATTEMPTS = 120


class LumaVideoGenerator(BaseVideoGenerator):
	"""
    Video generator using Luma Dream Machine (Ray) via the Luma Labs REST API.

    Handles authentication, async generation submission, status polling, and
    video download.  Wraps the full attempt in retry logic for resilience.

    Config keys:
        api_key / LUMA_API_KEY ,  Luma API key (**required**).
        max_concurrent         ,  Max simultaneous requests (inherited from base).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Luma Video API client with authentication headers.

        Args:
            config: Configuration dict.  See class docstring for keys.

        Raises:
            ValueError: If api_key is absent from config and LUMA_API_KEY env var.
        """
		super().__init__(config)
		self.api_key = config.get("api_key") or os.environ.get("LUMA_API_KEY")
		if not self.api_key:
			raise ValueError(
				"Luma API key is required either in config or environment variable LUMA_API_KEY."
			)

		self.model = config.get("model", "ray-2")

		# Authorization headers are constant across all requests.
		self.headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
		}

	async def _poll_for_completion(self, client: httpx.AsyncClient, generation_id: str) -> str:
		"""
        Poll the Luma generation endpoint until the video is complete or fails.

        Luma generations transition through states: ``pending`` -> ``dreaming``
        -> ``completed`` (or ``failed``).  We check every ``POLL_INTERVAL_SECONDS``
        and return the video URL when ``completed``.

        Args:
            client:        Shared ``httpx.AsyncClient``.
            generation_id: The generation ID returned by the initial POST request.

        Returns:
            The CDN URL for the generated video (``assets.video``).

        Raises:
            RuntimeError: If the generation fails or completes with no video URL.
            RuntimeError: If ``MAX_POLL_ATTEMPTS`` are exhausted.
            httpx.HTTPStatusError / httpx.RequestError: On network-level errors.
        """
		url = f"{LUMA_API_URL}/{generation_id}"

		for attempt in range(MAX_POLL_ATTEMPTS):
			try:
				response = await client.get(url, headers=self.headers)
				response.raise_for_status()
				data = response.json()
			except (httpx.HTTPStatusError, httpx.RequestError) as e:
				logger.error(f"Failed to poll status for generation {generation_id}: {e}")
				raise e

			state = data.get("state")

			if state == "completed":
				# ``assets.video`` contains the final download URL.
				video_url = data.get("assets", {}).get("video")
				if not video_url:
					raise RuntimeError(
						f"Video generation {generation_id} completed but no video URL found."
					)
				return video_url

			if state == "failed":
				failure_reason = data.get("failure_reason", "Unknown reason")
				raise RuntimeError(f"Generation {generation_id} failed: {failure_reason}")

			# Generation is still in progress ,  wait before the next poll.
			await asyncio.sleep(POLL_INTERVAL_SECONDS)

		raise RuntimeError(
			f"Generation {generation_id} timed out after "
			f"{MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS}s."
		)

	async def _download_video(self, client: httpx.AsyncClient, video_url: str, output_path: str):
		"""
        Asynchronously download a video from a URL and write it to disk.

        Args:
            client:      Shared ``httpx.AsyncClient``.
            video_url:   Direct URL to the generated MP4 video.
            output_path: Local file path to write the video bytes to.

        Raises:
            httpx.HTTPStatusError: If the download request fails.
            Exception: On any file I/O error.
        """
		try:
			response = await client.get(video_url)
			response.raise_for_status()
			with open(output_path, 'wb') as f:
				f.write(response.content)
		except Exception as e:
			logger.error(f"Error downloading video from {video_url} to {output_path}: {e}")
			raise e

	async def generate_scene(self, scene: ProductionScene, output_dir: str) -> str:
		"""
        Generate a single video scene using Luma Dream Machine.

        The entire submit + poll + download flow is wrapped in ``retry_async``
        (3 attempts, 2-second base delay with exponential back-off).

        Luma generation uses ``aspect_ratio: "16:9"`` to match the 1920x1080
        target output format used by Stage 4 (FFmpeg assembly).

        Args:
            scene:      ProductionScene with prompt and metadata.
            output_dir: Directory to save the output MP4.

        Returns:
            Path to the saved MP4 file.
        """
		# Prefer the richer production prompt over the basic visual prompt.
		prompt = scene.final_video_prompt
		if not prompt:
			prompt = scene.visual_prompt

		payload = {
			"prompt": prompt,
			"model": self.model,
			"aspect_ratio": "16:9",  # Matches the pipeline's 1920x1080 output format.
		}

		Path(output_dir).mkdir(parents=True, exist_ok=True)
		output_filename = f"scene_{scene.scene_number}.mp4"
		output_path = os.path.join(output_dir, output_filename)

		async def _attempt():
			"""Single attempt: submit -> poll -> download.  Retried on failure."""
			async with httpx.AsyncClient(
				timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS)
			) as client:
				try:
					response = await client.post(
						LUMA_API_URL, headers=self.headers, json=payload
					)
					response.raise_for_status()
					data = response.json()
				except Exception as e:
					logger.error(
						f"Failed to start generation for scene {scene.scene_number}: {e}"
					)
					raise e

				generation_id = data.get("id")
				if not generation_id:
					raise RuntimeError(
						f"Failed to obtain generation ID for scene {scene.scene_number}"
					)

				# Wait for the generation to finish and get the video URL.
				video_url = await self._poll_for_completion(client, generation_id)

				# Download the video to the local output path.
				await self._download_video(client, video_url, output_path)

			return output_path

		# Retry the entire attempt up to 3 times with exponential back-off.
		return await retry_async(_attempt, max_attempts=3, base_delay=2.0)
