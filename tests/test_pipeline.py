"""
Unit tests for the central orchestrator pipeline.
Verifies factory functions and the end-to-end asynchronous pipeline flow.
"""
import pytest

from unittest.mock import patch, MagicMock

from src.orchestrator.pipeline import (
	_get_transcriber,
	_get_llm_processor,
	_get_video_generator,
	run_pipeline
)

from src.stages.stage1_transcription.assembly_ai.assembly_ai_transcriber import AssemblyAITranscriber
from src.stages.stage1_transcription.deepgram.deepgram_transcriber import DeepgramTranscriber
from src.stages.stage1_transcription.rev_ai.rev_ai_transcriber import RevAiTranscriber
from src.stages.stage1_transcription.whisper_local.whisper_transcriber import WhisperTranscriber

from src.stages.stage2_llm.openai_gpt.openai_processor import OpenAIGPTProcessor
from src.stages.stage2_llm.anthropic_claude.claude_processor import ClaudeProcessor
from src.stages.stage2_llm.local_llama.local_llama_processor import LocalLlamaProcessor

from src.stages.stage3_video.luma_dream_machine.luma_video_generator import LumaVideoGenerator
from src.stages.stage3_video.replicate_pixverse.replicate_video_generator import ReplicateVideoGenerator
from src.stages.stage3_video.minimax_hailuo.hailuo_video_generator import HailuoVideoGenerator

from src.shared.schemas import Transcript, Storyboard, ProductionScript

# Test Factory Functions
def test_get_transcriber():
	"""Test the factory function for creating transcriber instances."""

	assert isinstance(_get_transcriber("assemblyai"), AssemblyAITranscriber)
	assert isinstance(_get_transcriber("deepgram"), DeepgramTranscriber)
	assert isinstance(_get_transcriber("revai"), RevAiTranscriber)
	assert isinstance(_get_transcriber("whisper"), WhisperTranscriber)

	with pytest.raises(ValueError):
		_get_transcriber("invalid")


def test_get_llm_processor():
	"""Test the factory function for creating LLM processor instances."""

	assert isinstance(_get_llm_processor("openai"), OpenAIGPTProcessor)
	assert isinstance(_get_llm_processor("anthropic"), ClaudeProcessor)
	assert isinstance(_get_llm_processor("llama"), LocalLlamaProcessor)

	with pytest.raises(ValueError):
		_get_llm_processor("invalid")


def test_get_video_generator():
	"""Test the factory function for creating video generator instances."""

	assert isinstance(_get_video_generator("luma"), LumaVideoGenerator)
	assert isinstance(_get_video_generator("replicate"), ReplicateVideoGenerator)
	assert isinstance(_get_video_generator("minimax"), HailuoVideoGenerator)

	with pytest.raises(ValueError):
		_get_video_generator("invalid")


# Test Pipeline Run with mocks
@pytest.mark.asyncio
@patch("src.orchestrator.pipeline._get_transcriber")
@patch("src.orchestrator.pipeline._get_llm_processor")
@patch("src.orchestrator.pipeline._get_video_generator")
@patch("src.orchestrator.pipeline.FFmpegAssembler")
@patch("pathlib.Path.mkdir")
@patch("builtins.open", new_callable=MagicMock)
async def test_run_pipeline(mock_open, mock_mkdir, mock_assembler_cls, mock_get_video, mock_get_llm, mock_get_transcriber):
	"""Test the full pipeline execution with all components mocked."""

	# Setup Mocks
	mock_transcriber = MagicMock()
	mock_get_transcriber.return_value = mock_transcriber
	
	mock_llm_processor = MagicMock()
	mock_get_llm.return_value = mock_llm_processor
	
	mock_video_generator = MagicMock()
	mock_get_video.return_value = mock_video_generator
	
	mock_assembler = MagicMock()
	mock_assembler_cls.return_value = mock_assembler

	# Mock return values for stages
	mock_transcript = Transcript(audio_duration=0.0, status="completed", utterances=[], full_text="test text")
	mock_transcriber.transcribe.return_value = mock_transcript
	
	mock_speaker_map = {"Speaker A": "Speaker A"}
	mock_llm_processor.map_speakers.return_value = mock_speaker_map
	
	mock_storyboard = Storyboard(scenes=[])
	mock_llm_processor.generate_storyboard.return_value = mock_storyboard
	
	mock_production_script = ProductionScript(scenes=[])
	mock_llm_processor.generate_production_script.return_value = mock_production_script
	
	async def mock_generate(*args, **kwargs):
		"""Returns a mock successful API response dict."""
		return (["path1.mp4", "path2.mp4"], [])
	mock_video_generator.generate_all_scenes = mock_generate

	# Run Pipeline
	await run_pipeline(
		audio_path="test_audio.mp3",
		transcriber_name="assemblyai",
		llm_name="openai",
		video_name="luma",
		output_dir="test_out"
	)

	# Asserts
	mock_mkdir.assert_called()
	mock_transcriber.transcribe.assert_called_once_with("test_audio.mp3")
	
	mock_llm_processor.map_speakers.assert_called_once_with(mock_transcript)
	mock_llm_processor.generate_storyboard.assert_called_once_with(mock_transcript, mock_speaker_map)
	mock_llm_processor.generate_production_script.assert_called_once_with(mock_storyboard, mock_transcript)
	
	# Check assembler calls
	mock_assembler.stitch_videos.assert_called_once()
	mock_assembler.overlay_audio.assert_called_once()

	# Check file writings (transcript, storyboard, production script)
	assert mock_open.call_count == 3
