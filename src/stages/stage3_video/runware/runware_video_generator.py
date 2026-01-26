"""
Stage 3 Video Generation ,  Runware Backend
===========================================
Uses the Runware API for GPU-accelerated video generation from text prompts.

Runware is an inference platform specializing in fast, cost-efficient GPU-based
AI generation.  The video inference API accepts a list of task objects and returns
video URLs either immediately (if the result is ready) or via a polling endpoint.

Runware API pattern:
    Unlike most other backends (which have separate "submit" and "poll" URLs),
    Runware accepts tasks at ``/v1/request`` and may return results inline *or*
    require polling.  We handle both cases:
        1. Check the immediate response for a ``videoURL`` field.
        2. If absent, poll ``GET /v1/request/{taskUUID}`` until the video is ready.

Task UUID:
    We use a deterministic ``"scene-{scene_number}"`` UUID so that if the same
    scene is re-submitted, the task UUID is consistent.  This is a design choice
    ,  Runware does not require UUID uniqueness per se.

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

# Runware REST API base URL.
RUNWARE_API_URL = "https://api.runware.ai/v1"
# Polling interval between status checks.
POLL_INTERVAL_SECONDS = 5
# Max polls: 120 x 5s = 10 minutes.
MAX_POLL_ATTEMPTS = 120


class RunwareVideoGenerator(BaseVideoGenerator):
	"""
    Video generator using the Runware API for fast GPU inference.

    Submits a ``videoInference`` task to Runware and downloads the result.
    Handles both immediate and async (polled) result delivery.

    Config keys:
        api_key / RUNWARE_API_KEY ,  Runware API key (**required**).
        model      ,  Runware model URN (default: ``"runware:101@1"``).
        width      ,  Output width in pixels (default: 1280).
        height     ,  Output height in pixels (default: 720).
        num_frames ,  Number of frames to generate (default: 80, ~3.3s at 24 fps).
    """

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the Runware client with API credentials and video parameters.

        Args:
            config: Configuration dict.  See class docstring for keys.

        Raises:
            ValueError: If RUNWARE_API_KEY is absent.
        """
		super().__init__(config)
		self.api_key = config.get("api_key") or os.getenv("RUNWARE_API_KEY")
		if not self.api_key:
			raise ValueError("RUNWARE_API_KEY is missing from config and environment variables.")

		# Model and output parameters.
		self.model = config.get("model", "runware:101@1")   # Default Runware Video model.
		self.width = config.get("width", 1280)               # 720p HD width.
		self.height = config.get("height", 720)              # 720p HD height.
		self.num_frames = config.get("num_frames", 80)       # ~3.3 seconds at 24 fps.

		# Authorization headers for all API requests.
		self.headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
		}

	async def generate_scene(self, scene: ProductionScene, output_dir: str) -> str:
		"""
        Generate a single video scene using Runware and save it to disk.

        Submits a ``videoInference`` task and handles either an immediate response
        (video URL present in the initial response) or an async result (requires
        polling via ``_poll_for_completion``).

        Args:
            scene:      ProductionScene with prompt and metadata.
            output_dir: Directory to save the output MP4.

        Returns:
            Path to the saved MP4 file.
        """
		prompt = scene.final_video_prompt or scene.visual_prompt

		Path(output_dir).mkdir(parents=True, exist_ok=True)
		output_path = os.path.join(output_dir, f"scene_{scene.scene_number}_runware.mp4")

		# Runware accepts a list of task dicts ,  one task per video generation request.
		# ``taskType: "videoInference"`` is the Runware task type for video generation.
		# ``taskUUID`` is a client-supplied identifier for polling and deduplication.
		payload = [
			{
				"taskType": "videoInference",
				"taskUUID": f"scene-{scene.scene_number}",  # Deterministic ID for this scene.
				"positivePrompt": prompt,
				"model": self.model,
				"width": self.width,
				"height": self.height,
				"numberFrames": self.num_frames,
			}
		]

		async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
			response = await client.post(
				f"{RUNWARE_API_URL}/request",
				headers=self.headers,
				json=payload,
			)
			response.raise_for_status()
			data = response.json()

			# Runware may return the video URL immediately in the initial response,
			# or it may require polling if the job isn't done yet.
			results = data.get("data", [])
			if results and results[0].get("videoURL"):
				# Video is already available ,  no polling needed.
				video_url = results[0]["videoURL"]
			else:
				# Video not ready yet ,  poll until it is.
				task_uuid = data.get("taskUUID") or f"scene-{scene.scene_number}"
				video_url = await self._poll_for_completion(client, task_uuid)

			# Download the generated video to disk.
			dl_response = await client.get(video_url)
			dl_response.raise_for_status()
			async with aiofiles.open(output_path, "wb") as f:
				await f.write(dl_response.content)

		return output_path

	async def _poll_for_completion(self, client: httpx.AsyncClient, task_uuid: str) -> str:
		"""
        Poll the Runware task status endpoint until the video is ready.

        Checks ``GET /v1/request/{task_uuid}`` every ``POLL_INTERVAL_SECONDS``
        seconds.  Returns the video URL when available; raises on failure or timeout.

        Args:
            client:    Shared ``httpx.AsyncClient`` from ``generate_scene``.
            task_uuid: The task UUID used when submitting the video inference job.

        Returns:
            The video URL from the completed task result.

        Raises:
            RuntimeError: If the task fails or times out.
        """
		for _ in range(MAX_POLL_ATTEMPTS):
			response = await client.get(
				f"{RUNWARE_API_URL}/request/{task_uuid}",
				headers=self.headers,
			)
			response.raise_for_status()
			data = response.json()
			results = data.get("data", [])

			if results and results[0].get("videoURL"):
				# Video is now ready.
				return results[0]["videoURL"]

			status = data.get("status")
			if status in ("FAILED", "ERROR"):
				raise RuntimeError(f"Runware task {task_uuid} failed: {data}")

			# Still processing ,  wait before polling again.
			await asyncio.sleep(POLL_INTERVAL_SECONDS)

		raise RuntimeError(f"Runware task {task_uuid} timed out.")
