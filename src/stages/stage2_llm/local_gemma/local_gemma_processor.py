"""
Stage 2 LLM Processing ,  Local Gemma via Ollama Backend
========================================================
Thin subclass of ``LocalLlamaProcessor`` that defaults to the Google Gemma 3 model.

What is Gemma?
    Gemma is Google's family of open-weight language models, derived from the same
    research that produced Gemini.  Gemma 3 (the default here) offers a good
    balance of quality and speed for creative text generation tasks on consumer
    hardware.

Prerequisites:
    Ollama must be running locally and the model must be pulled first:
        ollama pull gemma3

All pipeline logic is inherited from ``LocalLlamaProcessor``.
"""

from typing import Dict, Any
from src.stages.stage2_llm.local_llama.local_llama_processor import LocalLlamaProcessor


class LocalGemmaProcessor(LocalLlamaProcessor):
	"""
    Concrete implementation of BaseLLMProcessor using Google Gemma via Ollama.

    Inherits all behavior from ``LocalLlamaProcessor`` and sets the default
    Ollama model to ``"gemma3"``.

    Requires Ollama running locally with the Gemma model pulled:
        ollama pull gemma3
    """

	# Default to Gemma 3; can be overridden in config (e.g. "gemma2", "gemma:7b").
	DEFAULT_OLLAMA_MODEL = "gemma3"

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize with Gemma 3 as the default Ollama model.

        Args:
            config: Configuration dict passed through to ``LocalLlamaProcessor``.
        """
		config.setdefault("model", self.DEFAULT_OLLAMA_MODEL)
		super().__init__(config)
