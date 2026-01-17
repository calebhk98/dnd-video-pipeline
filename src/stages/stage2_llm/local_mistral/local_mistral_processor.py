"""
Stage 2 LLM Processing ,  Local Mistral via Ollama Backend
==========================================================
Thin subclass of ``LocalLlamaProcessor`` that defaults to the Mistral model.

Mistral 7B is a compact, high-quality open-source model from Mistral AI that
runs well on consumer hardware.  It follows instructions reliably and tends to
produce clean JSON output, making it a good choice for the structured generation
tasks in Stage 2.

Prerequisites:
    Ollama must be running locally and the model must be pulled first:
        ollama pull mistral

All pipeline logic (chunking, JSON extraction, fallbacks) is inherited from
``LocalLlamaProcessor``.  The only difference is the default model name.
"""

from typing import Dict, Any
from src.stages.stage2_llm.local_llama.local_llama_processor import LocalLlamaProcessor


class LocalMistralProcessor(LocalLlamaProcessor):
	"""
    Concrete implementation of BaseLLMProcessor using Mistral via Ollama.

    Inherits all behavior from ``LocalLlamaProcessor`` and sets the default
    Ollama model to ``"mistral"``.  Callers can override the model via config.

    Requires Ollama running locally with the Mistral model pulled:
        ollama pull mistral
    """

	# Default model tag for Ollama.  Can be overridden in config.
	DEFAULT_OLLAMA_MODEL = "mistral"

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize with Mistral as the default Ollama model.

        Uses ``config.setdefault`` so that if a caller explicitly passes a
        ``"model"`` key in ``config``, that value takes precedence over the
        default here.

        Args:
            config: Configuration dict passed through to ``LocalLlamaProcessor``.
        """
		# Only set the default if the caller didn't specify a model.
		config.setdefault("model", self.DEFAULT_OLLAMA_MODEL)
		super().__init__(config)
