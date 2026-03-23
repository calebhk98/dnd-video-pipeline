"""
retry.py ,  async exponential-backoff retry utility.

Many operations in this application call external APIs (transcription services,
LLM endpoints, video generation APIs) that can fail transiently due to network
hiccups, rate limiting, or momentary service unavailability.  Rather than
duplicating retry logic in every call site, this module provides a single
reusable wrapper: `retry_async`.

Usage pattern
-------------
    from shared.utils.retry import retry_async

    # Retry up to 3 times on any exception (default behaviour):
    result = await retry_async(lambda: some_api_call(arg1, arg2))

    # Retry only on specific transient errors, with more attempts:
    result = await retry_async(
        lambda: some_api_call(arg1, arg2),
        max_attempts=5,
        base_delay=1.0,
        exceptions=(TimeoutError, ConnectionError),
    )
"""

import asyncio
import logging
from typing import Callable, Awaitable, Tuple, Type

logger = logging.getLogger(__name__)


async def retry_async(
	coro_fn: Callable[[], Awaitable],
	max_attempts: int = 3,
	base_delay: float = 2.0,
	exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
	"""
    Retry an async coroutine with exponential backoff.

    Calls `coro_fn()` up to `max_attempts` times.  If the coroutine raises one
    of the specified `exceptions`, the call is retried after a delay.  The delay
    grows exponentially between retries to avoid hammering a struggling service:

        delay after attempt 1 = base_delay ^ 1  (e.g. 2.0 ^ 1 = 2.0 s)
        delay after attempt 2 = base_delay ^ 2  (e.g. 2.0 ^ 2 = 4.0 s)
        delay after attempt 3 = base_delay ^ 3  (e.g. 2.0 ^ 3 = 8.0 s)
        ...

    No delay is added after the final failed attempt ,  the exception is
    re-raised immediately so the caller receives it without unnecessary waiting.

    Args:
        coro_fn: A zero-argument callable that returns an awaitable coroutine.
            Typically a lambda that captures the arguments for the async function
            being retried, e.g. `lambda: fetch_transcript(audio_id)`.
            A new coroutine is created on each call so that the coroutine object
            is not reused across attempts (coroutines cannot be restarted).
        max_attempts: Total number of attempts to make before giving up and
            re-raising the last exception.  Must be >= 1.  The default of 3
            means one initial try plus two retries.
        base_delay: The base number of seconds used to compute the exponential
            backoff delay.  The actual delay before retry N is `base_delay ** N`.
            Defaults to 2.0, giving delays of 2 s, 4 s, 8 s, ... for successive
            retries.
        exceptions: A tuple of exception *types* that should trigger a retry.
            Any exception not in this tuple will propagate immediately without
            retrying.  Defaults to `(Exception,)`, which retries on any
            exception.  Narrow this to specific transient error types (e.g.
            `(TimeoutError, aiohttp.ClientError)`) to avoid masking bugs.

    Returns:
        The return value of `coro_fn()` on the first successful attempt.

    Raises:
        The last exception raised by `coro_fn()` if all `max_attempts`
        attempts fail.  The original traceback is preserved via a bare `raise`.

    Examples:
        # Basic usage ,  retry on any exception:
        result = await retry_async(lambda: some_async_call(arg1, arg2))

        # Selective retry ,  only transient network errors, 5 attempts:
        result = await retry_async(
            lambda: fetch_data(url),
            max_attempts=5,
            base_delay=1.0,
            exceptions=(TimeoutError, ConnectionResetError),
        )
    """
	# Use 1-based indexing for `attempt` so that log messages read naturally
	# ("Attempt 1/3 failed") rather than the confusing "Attempt 0/3 failed".
	for attempt in range(1, max_attempts + 1):
		try:
			# Call coro_fn() to create a fresh coroutine and immediately await
			# it.  A new coroutine must be created on every attempt because a
			# coroutine object that has already raised cannot be resumed.
			return await coro_fn()
		except exceptions as e:
			if attempt == max_attempts:
				# This was the last allowed attempt ,  do NOT sleep, just
				# re-raise so the caller receives the exception promptly.
				# A bare `raise` re-raises the caught exception while
				# preserving the original traceback (unlike `raise e`).
				raise

			# Compute the exponential backoff delay for this attempt.
			# Using `base_delay ** attempt` (not `** (attempt - 1)`) means
			# the first retry waits base_delay^1 seconds, the second waits
			# base_delay^2, and so on ,  giving increasingly longer gaps as
			# failures accumulate and the service is likely under stress.
			delay = base_delay ** attempt
			logger.warning(
				f"Attempt {attempt}/{max_attempts} failed: {e}. "
				f"Retrying in {delay:.1f}s..."
			)
			# Yield control back to the event loop during the backoff window
			# so other coroutines can make progress while we wait.
			await asyncio.sleep(delay)
