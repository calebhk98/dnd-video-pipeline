"""
Stage 2 LLM Processing ,  Local Dolphin (uncensored Mistral) via Ollama Backend
================================================================================
Thin subclass of ``LocalLlamaProcessor`` that defaults to the Dolphin-Mistral model.

What is Dolphin?
    Dolphin is an uncensored fine-tune of Mistral 7B trained on the Dolphin dataset
    (cleaned, filtered ORCA conversations).  The "uncensored" variant removes refusal
    behaviors for role-playing and fictional content, which can be beneficial for
    D&D-style creative writing tasks where the model might otherwise refuse to
    describe combat or dark narrative themes.

Prerequisites:
    Ollama must be running locally and the model must be pulled first:
        ollama pull dolphin-mistral

All pipeline logic is inherited from ``LocalLlamaProcessor``.
"""

from typing import Dict, Any
from src.stages.stage2_llm.local_llama.local_llama_processor import LocalLlamaProcessor


class LocalDolphinProcessor(LocalLlamaProcessor):
	"""
    Concrete implementation of BaseLLMProcessor using Dolphin-Mistral via Ollama.

    Inherits all behavior from ``LocalLlamaProcessor`` and sets the default
    Ollama model to ``"dolphin-mistral"``.

    Requires Ollama running locally with the Dolphin-Mistral model pulled:
        ollama pull dolphin-mistral
    """

	# Ollama model tag for the Dolphin uncensored Mistral fine-tune.
	DEFAULT_OLLAMA_MODEL = "dolphin-mistral"

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize with Dolphin-Mistral as the default Ollama model.

        Args:
            config: Configuration dict passed through to ``LocalLlamaProcessor``.
        """
		# Respect any explicit "model" override from the caller.
		config.setdefault("model", self.DEFAULT_OLLAMA_MODEL)
		super().__init__(config)
