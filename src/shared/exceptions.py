"""
Shared provider exception hierarchy.

These exceptions are raised by AI provider implementations (transcription, LLM,
video generation) to signal specific failure modes that the pipeline orchestrator
can translate into actionable user-facing messages.
"""


class ProviderError(Exception):
	"""Base class for all AI provider errors.

    Attributes:
        provider_name: Human-readable provider name (e.g. "Anthropic Claude").
        error_type:    Machine-readable category. One of:
                           "insufficient_credits" - account has no remaining credits
                           "invalid_key"          - API key is missing or rejected
                           "unavailable"          - service unreachable / 5xx
        help_url:      Optional URL to the provider's billing or status page.
    """

	def __init__(self, message: str, provider_name: str = "", error_type: str = "", help_url: str = ""):
		"""Initializes a ProviderError with specific provider and error metadata."""

		super().__init__(message)
		self.provider_name = provider_name
		self.error_type = error_type
		self.help_url = help_url


class InsufficientCreditsError(ProviderError):
	"""Raised when the provider rejects a request due to insufficient credits."""

	def __init__(self, message: str, provider_name: str = "", help_url: str = ""):
		"""Initializes an InsufficientCreditsError."""

		super().__init__(message, provider_name=provider_name, error_type="insufficient_credits", help_url=help_url)


class InvalidAPIKeyError(ProviderError):
	"""Raised when the provider rejects a request due to an invalid or missing API key."""

	def __init__(self, message: str, provider_name: str = "", help_url: str = ""):
		"""Initializes an InvalidAPIKeyError."""

		super().__init__(message, provider_name=provider_name, error_type="invalid_key", help_url=help_url)


class ProviderUnavailableError(ProviderError):
	"""Raised when the provider is unreachable or returns a server-side error."""

	def __init__(self, message: str, provider_name: str = "", help_url: str = ""):
		"""Initializes a ProviderUnavailableError."""

		super().__init__(message, provider_name=provider_name, error_type="unavailable", help_url=help_url)
