"""
Stage 2 LLM Processing ,  Local Qwen via Ollama Backend
=======================================================
Thin subclass of ``LocalLlamaProcessor`` that defaults to the Alibaba Qwen 2.5 model.

What is Qwen?
    Qwen (Tongyi Qianwen) is Alibaba Cloud's family of open-weight language models.
    Qwen 2.5 offers strong multilingual performance and excels at instruction
    following and structured output generation, making it a solid choice for the
    JSON-heavy tasks in Stage 2.

Prerequisites:
    Ollama must be running locally and the model must be pulled first:
        ollama pull qwen2.5

All pipeline logic is inherited from ``LocalLlamaProcessor``.
"""

from typing import Dict, Any
from src.stages.stage2_llm.local_llama.local_llama_processor import LocalLlamaProcessor


class LocalQwenProcessor(LocalLlamaProcessor):
	"""
    Concrete implementation of BaseLLMProcessor using Qwen 2.5 via Ollama.

    Inherits all behavior from ``LocalLlamaProcessor`` and sets the default
    Ollama model to ``"qwen2.5"``.

    Requires Ollama running locally with the Qwen model pulled:
        ollama pull qwen2.5
    """

	# Default to Qwen 2.5; can be overridden (e.g. "qwen2.5:14b" for higher quality).
	DEFAULT_OLLAMA_MODEL = "qwen2.5"

	def __init__(self, config: Dict[str, Any]):
		"""
        Initialize with Qwen 2.5 as the default Ollama model.

        Args:
            config: Configuration dict passed through to ``LocalLlamaProcessor``.
        """
		config.setdefault("model", self.DEFAULT_OLLAMA_MODEL)
		super().__init__(config)
