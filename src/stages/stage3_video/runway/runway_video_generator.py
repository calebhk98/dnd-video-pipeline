"""
Stage 3 Video Generation ,  Runway ML Backend
=============================================
Uses Runway Gen-4 Turbo (or Gen-3 Alpha) via the Runway ML REST API to generate
high-quality cinematic video clips from text prompts.

Runway ML API flow:
    1. POST to ``/image_to_video`` with the text prompt, model, duration, and ratio.
       (Despite the endpoint name, text-only requests are supported.)
    2. The API immediately returns a ``task_id`` ,  generation is asynchronous.
    3. Poll ``GET /tasks/{task_id}`` every ``POLL_INTERVAL_SECONDS`` seconds.
    4. When status is ``"SUCCEEDED"``, retrieve the video URL from ``output[0]``.
    5. Download the MP4 from the video URL using httpx.

Rate limit / timeout considerations:
    - ``POLL_INTERVAL_SECONDS = 10``: Polling too aggressively can trigger rate limits.
    - ``MAX_POLL_ATTEMPTS = 60``: 60 x 10s = 10 minutes total wait time.
    - The httpx client has a 30-second connect/read timeout for the initial POST.
      Polling requests use the same client within the same async context manager.

API version header:
    ``X-Runway-Version: 2024-11-06`` pins the API contract version so responses
    remain stable even when Runway releases breaking changes.

Dependencies:
    pip install httpx aiofiles
"""

import os
import asyncio
import httpx
import aiofiles
from typing import Dict, Any
from pathlib import Path

from src.stages.stage3_video.base import BaseVideoGenerator
from src.shared.schemas import ProductionScene

# Runway ML Gen-4 API base URL.
RUNWAY_API_URL = "https://api.dev.runwayml.com/v1"
# Seconds between task status polls.  10s is a reasonable balance between
# responsiveness and not hammering the API.
POLL_INTERVAL_SECONDS = 10
# Maximum number of polls before giving up (60 x 10s = 10 minutes).
MAX_POLL_ATTEMPTS = 60


class RunwayVideoGenerator(BaseVideoGenerator):
	"""
    Video generator using Runway Gen-4 Turbo via the Runway ML REST API.

    Supports both Gen-4 Turbo and Gen-3 Alpha Turbo models via the ``model``
    config key.  Uses async HTTP polling to wait for job completion.

    Config keys:
        api_key / RUNWAY_API_KEY ,  Runway ML API key (**required**).
        model    ,  Runway model name (default: ``"gen4_turbo"``).
                   Options: ``"gen4_turbo"``, ``"gen3a_turbo"``.
        duration ,  Video duration in seconds: 5 or 10 (default: 10).
        ratio    ,  Aspect ratio string (default: ``"1280:720"``).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Runway ML client and set up auth headers.

        Args:
            config: Configuration dict.  See class docstring for keys.

        Raises:
            ValueError: If RUNWAY_API_KEY is absent.
        """
		super().__init__(config)
		self.api_key = config.get("api_key") or os.getenv("RUNWAY_API_KEY")
		if not self.api_key:
			raise ValueError("RUNWAY_API_KEY is missing from config and environment variables.")

		self.model = config.get("model", "gen4_turbo")
		self.duration = config.get("duration", 10)
		self.ratio = config.get("ratio", "1280:720")

		# Auth and version headers are reused for every API call.
		self.headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
			# Pin the API contract version to avoid unexpected breaking changes.
			"X-Runway-Version": "2024-11-06",
		}

	async def _poll_for_completion(self, client: httpx.AsyncClient, task_id: str) -> str:
		"""
        Poll the Runway task endpoint until the video generation succeeds or fails.

        Runs up to ``MAX_POLL_ATTEMPTS`` iterations, sleeping ``POLL_INTERVAL_SECONDS``
        between each check.  Returns the video URL on success; raises on failure
        or timeout.

        Args:
            client:  The shared ``httpx.AsyncClient`` from ``generate_scene``.
            task_id: The Runway task ID returned by the initial generation request.

        Returns:
            The URL of the generated video (``output[0]`` from the Runway response).

        Raises:
            RuntimeError: If the task fails, is cancelled, or has no output URL.
            RuntimeError: If ``MAX_POLL_ATTEMPTS`` are exhausted without completion.
        """
		url = f"{RUNWAY_API_URL}/tasks/{task_id}"

		for _ in range(MAX_POLL_ATTEMPTS):
			response = await client.get(url, headers=self.headers)
			response.raise_for_status()
			data = response.json()

			status = data.get("status")
			if status == "SUCCEEDED":
				output = data.get("output", [])
				if not output:
					raise RuntimeError(f"Task {task_id} succeeded but no output URL found.")
				return output[0]  # First (and usually only) video URL.

			if status in ("FAILED", "CANCELLED"):
				failure = data.get("failure", "Unknown reason")
				raise RuntimeError(
					f"Runway task {task_id} ended with status {status}: {failure}"
				)

			# Task is still processing ,  wait before polling again.
			await asyncio.sleep(POLL_INTERVAL_SECONDS)

		raise RuntimeError(
			f"Runway task {task_id} timed out after "
			f"{MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS}s."
		)

	async def generate_scene(self, scene: ProductionScene, output_dir: str) -> str:
		"""
        Generate a single video scene using Runway Gen-4 and save it to disk.

        Creates the output directory, submits the generation request, polls for
        completion, downloads the MP4, and returns the local file path.

        Args:
            scene:      ProductionScene with prompt and metadata.
            output_dir: Directory to save the downloaded MP4.

        Returns:
            Path to the saved MP4 file.

        Raises:
            RuntimeError: If Runway doesn't return a task ID, the job fails, or
                          polling times out.
        """
		prompt = scene.final_video_prompt or scene.visual_prompt

		Path(output_dir).mkdir(parents=True, exist_ok=True)
		output_path = os.path.join(output_dir, f"scene_{scene.scene_number}_runway.mp4")

		payload = {
			"model": self.model,
			"promptText": prompt,   # Runway uses "promptText" for text-only generation.
			"ratio": self.ratio,
			"duration": self.duration,
		}

		# Use a single AsyncClient for both submission and polling to reuse the
		# underlying TCP connection.  30s timeout for the initial POST.
		async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
			response = await client.post(
				f"{RUNWAY_API_URL}/image_to_video",
				headers=self.headers,
				json=payload,
			)
			response.raise_for_status()
			task_id = response.json().get("id")
			if not task_id:
				raise RuntimeError(
					f"Runway did not return a task ID for scene {scene.scene_number}."
				)

			# Block (async) until the video is ready.
			video_url = await self._poll_for_completion(client, task_id)

			# Download the generated video from Runway's CDN.
			dl_response = await client.get(video_url)
			dl_response.raise_for_status()
			# Use aiofiles for non-blocking file I/O.
			async with aiofiles.open(output_path, "wb") as f:
				await f.write(dl_response.content)

		return output_path
