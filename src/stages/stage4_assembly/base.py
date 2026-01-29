"""
Stage 4: Assembly ,  Abstract Base Class
========================================
This module defines the interface for all video assembly backends.

Assembly is the final stage of the D&D audio-to-video pipeline:

    MP4 clips (one per scene)  ->  [Stage 4: Assembler]  ->  Final video

The assembler combines the independently generated scene clips into a single
cohesive video, then optionally overlays the original session audio.  Two
separate concerns are captured in this interface:

    1. **Video stitching** ,  Concatenating scene clips (optionally with crossfade
       transitions) and normalizing them to a common resolution.
    2. **Audio overlay**   ,  Replacing or adding the original D&D session audio
       track so players can hear the actual game alongside the generated visuals.

The primary concrete implementation is ``FFmpegAssembler``, which handles both
operations using FFmpeg subprocess calls.
"""

from abc import ABC, abstractmethod
from typing import List
from src.shared.schemas import ProductionScript, Transcript


class BaseAssembler(ABC):
	"""
    Abstract Base Class for Stage 4 video assembly tools.

    Implementations receive a list of scene MP4 paths and produce a single
    cohesive output video, optionally with the original game audio overlaid.
    """

	@abstractmethod
	def __init__(self):
		"""
        Initialize internal configuration for the assembler.

        Implementations should verify that required tools or dependencies
        (e.g. FFmpeg) are available at initialization time, raising ``RuntimeError``
        if they are not, so failures are caught early rather than mid-pipeline.
        """
		pass

	@abstractmethod
	def stitch_videos(
		self,
		video_paths: List[str],
		final_output_path: str,
		add_transitions: bool = True,
	) -> str:
		"""
        Concatenate a list of scene MP4 clips into a single final video.

        Implementations should:
            - Accept clips of varying resolutions and normalize them to a common
              size (e.g. 1920x1080) before concatenation.
            - When ``add_transitions=True``, apply crossfade transitions between
              clips for a polished cinematic look.
            - When ``add_transitions=False``, perform a hard cut between clips
              (faster to encode; useful for drafts or testing).
            - Write the output to ``final_output_path`` and return it.

        Args:
            video_paths: Ordered list of paths to scene MP4 clips.
                The order determines the final sequence.
            final_output_path: Path for the stitched output video.
            add_transitions: Whether to apply crossfade transitions (default: True).

        Returns:
            The absolute path to the stitched output file (``final_output_path``).

        Raises:
            ValueError: If ``video_paths`` is empty.
        """
		pass

	@abstractmethod
	def overlay_audio(
		self,
		video_path: str,
		original_audio_path: str,
		synced_output_path: str,
	) -> str:
		"""
        Overlay the original game audio track onto the stitched video.

        The pipeline generates AI video clips that are mute (no audio).
        This method replaces the silent video track with the actual D&D session
        recording so viewers can hear the players while watching the generated
        visuals.

        Implementations should:
            - Keep the video stream unchanged (copy, not re-encode if possible).
            - Encode or copy the audio stream to a compatible container format.
            - Truncate to the shorter of video or audio (``-shortest`` in FFmpeg).
            - Write the output to ``synced_output_path`` and return it.

        Args:
            video_path:         Path to the stitched (mute) video.
            original_audio_path: Path to the original D&D session audio file.
            synced_output_path:  Path for the output video with audio.

        Returns:
            The absolute path to the output file (``synced_output_path``).
        """
		pass
