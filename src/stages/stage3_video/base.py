"""
Stage 3: Video Generation ,  Abstract Base Class
================================================
This module defines the interface for all text-to-video generation backends and
provides the shared concurrency-management infrastructure.

Stage 3 is the third step in the D&D audio-to-video pipeline:

    ProductionScript  ->  [Stage 3: VideoGenerator]  ->  MP4 files (one per scene)

Each scene's ``final_video_prompt`` (from Stage 2) is sent to a video generation
API or local model.  Scenes are generated in parallel, with a configurable
concurrency limit to respect API rate limits and avoid overwhelming the service.

Concurrency model:
    All generators use ``asyncio.Semaphore`` to cap the number of simultaneously
    in-flight video generation requests.  ``generate_all_scenes`` gathers all
    tasks with ``asyncio.gather(..., return_exceptions=True)`` so that individual
    scene failures don't abort the entire batch.

Models not yet integrated (lack public API or practical inference path):
    - Google Veo 2        ,  allowlisted Vertex AI preview only; no public API
    - OpenAI Sora         ,  no API; only available through chatgpt.com
    - Open-Sora           ,  local only; requires 8x A100 80 GB GPUs
    - NVIDIA Cosmos       ,  early-access program only; no public API
    - Pyramid Flow        ,  local only; requires 1x A100 80 GB GPU

    Implement and add to _get_video_generator() in pipeline.py once an API
    becomes available or a practical local-inference path exists.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from src.shared.schemas import ProductionScene


class BaseVideoGenerator(ABC):
	"""
    Abstract Base Class for Stage 3 text-to-video generation backends.

    Implements shared concurrency control via an asyncio Semaphore and provides
    the ``generate_all_scenes`` orchestration method.  Concrete subclasses only
    need to implement ``__init__`` (API client setup) and ``generate_scene``
    (single-scene generation + download).

    Class attribute:
        MAX_CONCURRENT_SCENES (int): Default maximum number of video generation
            requests that may be in-flight simultaneously.  Individual backends
            should keep this <= their API's rate limit to avoid throttling.
    """

	# Conservative default: 3 concurrent scenes to stay within typical API limits.
	MAX_CONCURRENT_SCENES: int = 3

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize the video generator with a rate-limiting semaphore.

        Subclasses must call ``super().__init__(config)`` before setting up their
        own API clients so the semaphore is always initialized.

        Args:
            config: Configuration dict.  Recognized key:
                    ``max_concurrent`` ,  override the concurrency limit
                    (default: ``MAX_CONCURRENT_SCENES``).
        """
		# Allow per-instance concurrency override for APIs with tighter rate limits.
		concurrency = config.get("max_concurrent", self.MAX_CONCURRENT_SCENES)
		# asyncio.Semaphore ensures at most `concurrency` coroutines run _generate_scene
		# simultaneously, preventing API rate-limit errors on large scene counts.
		self._semaphore = asyncio.Semaphore(concurrency)

	@abstractmethod
	async def generate_scene(self, scene: ProductionScene, output_dir: str) -> str:
		"""
        Generate a single video scene from the scene's prompt and save it to disk.

        Implementations must:
            - Use ``scene.final_video_prompt`` as the primary prompt, falling back
              to ``scene.visual_prompt`` if the former is absent.
            - Create ``output_dir`` if it doesn't exist.
            - Save the downloaded MP4 to a deterministic filename like
              ``scene_{scene.scene_number}.mp4`` within ``output_dir``.
            - Be fully async (no blocking I/O on the event loop).

        Args:
            scene:      The ``ProductionScene`` containing prompt and metadata.
            output_dir: Directory path where the output MP4 will be saved.

        Returns:
            Absolute path to the downloaded MP4 file.

        Raises:
            RuntimeError: If the video generation service reports a failure.
            TimeoutError: If polling for completion exceeds the allowed window.
        """
		pass

	async def generate_all_scenes(
		self, scenes: List[ProductionScene], output_dir: str,
		scene_callback=None,
	) -> Tuple[List[str], List[Dict[str, Any]]]:
		"""
        Generate all scenes in parallel with semaphore-based rate limiting.

        Each scene is wrapped in a ``_guarded`` coroutine that acquires the
        semaphore before calling ``generate_scene``.  All wrapped coroutines are
        gathered concurrently ,  ``return_exceptions=True`` means that a failed
        scene's exception is captured rather than immediately propagating, allowing
        the remaining scenes to continue.

        After gathering, results are split into successes (file paths) and failures
        (dicts with scene_number + error string) for the caller to act on.

        Args:
            scenes: List of ``ProductionScene`` objects from Stage 2.
            output_dir: Directory where all output MP4 files will be saved.
            scene_callback: Optional async callable invoked immediately after each
                scene succeeds, before the full batch completes.
                Signature: ``await scene_callback(scene, video_path)``

        Returns:
            A 2-tuple:
                - ``video_paths`` (List[str]):  Paths of successfully generated videos,
                  in the same order as successful scenes.
                - ``failures`` (List[Dict]):    One entry per failed scene with keys:
                  ``"scene_number"`` (int) and ``"error"`` (str).
        """
		async def _guarded(scene: ProductionScene) -> str:
			"""Acquire the rate-limit semaphore, then generate the scene."""
			async with self._semaphore:
				result = await self.generate_scene(scene, output_dir)
				if scene_callback:
					await scene_callback(scene, result)
				return result

		# Launch all scene generation tasks concurrently.
		# return_exceptions=True prevents one failure from cancelling others.
		tasks = [_guarded(scene) for scene in scenes]
		results = await asyncio.gather(*tasks, return_exceptions=True)

		final_paths: List[str] = []
		failures: List[Dict[str, Any]] = []

		for index, result in enumerate(results):
			if isinstance(result, Exception):
				# Record the failure with its scene number for downstream reporting.
				failures.append({
					"scene_number": scenes[index].scene_number,
					"error": str(result),
				})
			else:
				final_paths.append(result)

		return final_paths, failures
