"""
OpenAI Error Translation
=========================
Translates OpenAI SDK exceptions into typed domain errors so that upstream
callers (routes, UI) can render meaningful error messages.
"""

import openai
from src.shared.exceptions import InsufficientCreditsError, InvalidAPIKeyError, ProviderUnavailableError


def call_api(fn):
	"""Call an OpenAI API function and translate SDK errors into typed ProviderErrors.

    Args:
        fn: A zero-argument callable that performs the OpenAI API call.

    Returns:
        The return value of ``fn()``.

    Raises:
        InvalidAPIKeyError:       On authentication failure.
        InsufficientCreditsError: On quota/billing rate-limit errors.
        ProviderUnavailableError: On connection or server errors.
    """
	try:
		return fn()
	except openai.AuthenticationError as e:
		raise InvalidAPIKeyError(
			f"OpenAI API key is invalid or missing: {e}",
			provider_name="OpenAI GPT",
			help_url="https://platform.openai.com/api-keys",
		) from e
	except openai.RateLimitError as e:
		msg = str(e).lower()
		if "quota" in msg or "billing" in msg or "insufficient" in msg:
			raise InsufficientCreditsError(
				f"Your OpenAI account has exceeded its quota or has insufficient credits: {e}",
				provider_name="OpenAI GPT",
				help_url="https://platform.openai.com/account/billing",
			) from e
		raise
	except (openai.APIConnectionError, openai.InternalServerError) as e:
		raise ProviderUnavailableError(
			f"OpenAI service is currently unavailable: {e}",
			provider_name="OpenAI GPT",
			help_url="https://status.openai.com",
		) from e
