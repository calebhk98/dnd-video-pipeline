"""
Provider Factory Functions
===========================
Lazy-loading factories for each pipeline stage's provider implementations.
Only the selected provider's dependencies are imported at runtime.
"""

import hashlib
import os

from src.stages.stage1_transcription.base import BaseTranscriber
from src.stages.stage2_llm.base import BaseLLMProcessor
from src.stages.stage3_video.base import BaseVideoGenerator


def _compute_file_hash(filepath: str, chunk_size: int = 65536) -> str:
	"""Return the SHA-256 hex digest of a file, reading in chunks."""
	h = hashlib.sha256()
	with open(filepath, "rb") as f:
		while chunk := f.read(chunk_size):
			h.update(chunk)
	return h.hexdigest()


def _get_transcriber(name: str) -> BaseTranscriber:
	"""Instantiate and return the transcriber implementation for the given name."""
	if name == "assemblyai":
		from src.stages.stage1_transcription.assembly_ai.assembly_ai_transcriber import AssemblyAITranscriber
		return AssemblyAITranscriber({"ASSEMBLYAI_API_KEY": os.getenv("ASSEMBLYAI_API_KEY")})
	elif name == "deepgram":
		from src.stages.stage1_transcription.deepgram.deepgram_transcriber import DeepgramTranscriber
		return DeepgramTranscriber({"DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY")})
	elif name == "revai":
		from src.stages.stage1_transcription.rev_ai.rev_ai_transcriber import RevAiTranscriber
		return RevAiTranscriber({"api_key": os.getenv("REV_AI_API_KEY")})
	elif name == "google_cloud":
		from src.stages.stage1_transcription.google_cloud_stt.google_cloud_transcriber import GoogleCloudTranscriber
		return GoogleCloudTranscriber({"api_key": os.getenv("GOOGLE_CLOUD_API_KEY")})
	elif name == "amazon_transcribe":
		from src.stages.stage1_transcription.amazon_transcribe.amazon_transcriber import AmazonTranscriber
		return AmazonTranscriber({
			"aws_access_key_id":     os.getenv("AWS_ACCESS_KEY_ID"),
			"aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
			"aws_region":            os.getenv("AWS_REGION"),
			"s3_bucket":             os.getenv("AWS_S3_BUCKET"),
		})
	elif name == "whisper":
		from src.stages.stage1_transcription.whisper_local.whisper_transcriber import WhisperTranscriber
		return WhisperTranscriber({})
	elif name == "whisperx":
		from src.stages.stage1_transcription.whisperx_local.whisperx_transcriber import WhisperXTranscriber
		return WhisperXTranscriber({"hf_token": os.getenv("HUGGING_FACE_TOKEN")})
	elif name == "nemo":
		from src.stages.stage1_transcription.nvidia_nemo.nemo_transcriber import NvidiaNemoTranscriber
		return NvidiaNemoTranscriber({})
	else:
		raise ValueError(f"Unknown transcriber: {name}")


def _get_llm_processor(name: str) -> BaseLLMProcessor:
	"""Instantiate and return the LLM processor implementation for the given name."""
	if name == "openai":
		from src.stages.stage2_llm.openai_gpt.openai_processor import OpenAIGPTProcessor
		return OpenAIGPTProcessor({"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"), "model": os.getenv("OPENAI_MODEL", "gpt-4o")})
	elif name == "anthropic":
		from src.stages.stage2_llm.anthropic_claude.claude_processor import ClaudeProcessor
		return ClaudeProcessor({"api_key": os.getenv("ANTHROPIC_API_KEY"), "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")})
	elif name == "gemini":
		from src.stages.stage2_llm.google_gemini.gemini_processor import GeminiProcessor
		return GeminiProcessor({"api_key": os.getenv("GOOGLE_API_KEY"), "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")})
	elif name == "deepseek":
		from src.stages.stage2_llm.deepseek.deepseek_processor import DeepseekProcessor
		return DeepseekProcessor({"api_key": os.getenv("DEEPSEEK_API_KEY"), "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat")})
	elif name == "llama":
		from src.stages.stage2_llm.local_llama.local_llama_processor import LocalLlamaProcessor
		return LocalLlamaProcessor({"model": "llama3.1"})
	elif name == "qwen":
		from src.stages.stage2_llm.local_llama.local_llama_processor import LocalLlamaProcessor
		return LocalLlamaProcessor({"model": "qwen2.5"})
	elif name == "gemma":
		from src.stages.stage2_llm.local_llama.local_llama_processor import LocalLlamaProcessor
		return LocalLlamaProcessor({"model": "gemma2"})
	elif name == "mistral":
		from src.stages.stage2_llm.local_llama.local_llama_processor import LocalLlamaProcessor
		return LocalLlamaProcessor({"model": "mistral"})
	elif name == "dolphin":
		from src.stages.stage2_llm.local_llama.local_llama_processor import LocalLlamaProcessor
		return LocalLlamaProcessor({"model": "dolphin-mixtral"})
	else:
		raise ValueError(f"Unknown LLM processor: {name}")


def _get_video_generator(name: str) -> BaseVideoGenerator:
	"""Instantiate and return the video generator implementation for the given name."""
	if name == "luma":
		from src.stages.stage3_video.luma_dream_machine.luma_video_generator import LumaVideoGenerator
		return LumaVideoGenerator({
			"api_key": os.getenv("LUMA_API_KEY"),
			"model":   os.getenv("LUMA_MODEL", "ray-2"),
		})
	elif name == "kling":
		from src.stages.stage3_video.kling_ai.kling_video_generator import KlingVideoGenerator
		return KlingVideoGenerator({"api_key": os.getenv("FAL_KEY")})
	elif name == "runway":
		from src.stages.stage3_video.runway.runway_video_generator import RunwayVideoGenerator
		return RunwayVideoGenerator({"api_key": os.getenv("RUNWAY_API_KEY"), "model": os.getenv("RUNWAY_MODEL", "gen4_turbo")})
	elif name == "pika":
		from src.stages.stage3_video.pika_labs.pika_video_generator import PikaVideoGenerator
		return PikaVideoGenerator({"replicate_api_token": os.getenv("REPLICATE_API_TOKEN")})
	elif name == "minimax":
		from src.stages.stage3_video.minimax_hailuo.hailuo_video_generator import HailuoVideoGenerator
		return HailuoVideoGenerator({"api_key": os.getenv("FAL_KEY")})
	elif name == "hunyuan":
		from src.stages.stage3_video.hunyuan_video.hunyuan_video_generator import HunyuanVideoGenerator
		return HunyuanVideoGenerator({"replicate_api_token": os.getenv("REPLICATE_API_TOKEN")})
	elif name == "ltx":
		from src.stages.stage3_video.ltx_video.ltx_video_generator import LTXVideoGenerator
		return LTXVideoGenerator({"replicate_api_token": os.getenv("REPLICATE_API_TOKEN")})
	elif name == "cogvideox":
		from src.stages.stage3_video.cogvideox.cogvideox_generator import CogVideoXGenerator
		return CogVideoXGenerator({"replicate_api_token": os.getenv("REPLICATE_API_TOKEN")})
	elif name == "mochi":
		from src.stages.stage3_video.mochi.mochi_generator import MochiGenerator
		return MochiGenerator({"replicate_api_token": os.getenv("REPLICATE_API_TOKEN")})
	elif name == "runware":
		from src.stages.stage3_video.runware.runware_video_generator import RunwareVideoGenerator
		return RunwareVideoGenerator({"api_key": os.getenv("RUNWARE_API_KEY")})
	elif name == "replicate":
		from src.stages.stage3_video.replicate_pixverse.replicate_video_generator import ReplicateVideoGenerator
		return ReplicateVideoGenerator({"replicate_api_token": os.getenv("REPLICATE_API_TOKEN")})
	else:
		raise ValueError(f"Unknown video generator: {name}")
