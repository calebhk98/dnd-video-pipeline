"""
routers/settings.py - API key settings endpoints.

Provides GET and POST endpoints for reading and writing the project's
.env file.  Keys are masked before being returned to the browser so the
actual values are never exposed in plain text.
"""

import json
import os
import logging
from typing import Dict

from fastapi import APIRouter
import dotenv

from core import ENV_PATH, EXAMPLE_PATH, FRONTEND_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["settings"])

# ---------------------------------------------------------------------------
# Load known keys from providers.json (single source of truth shared with the
# frontend).  HF_TOKEN is backend-only and appended separately.
# ---------------------------------------------------------------------------
_PROVIDERS_JSON = os.path.join(FRONTEND_DIR, "modules", "providers.json")


def _load_known_keys() -> list[str]:
	"""Loads environment variable keys for all supported providers."""

	with open(_PROVIDERS_JSON) as f:
		data = json.load(f)
	keys = [p["envKey"] for group in data for p in group["providers"]]
	keys.append("HF_TOKEN")  # Hugging Face token for local Whisper models
	return keys


KNOWN_KEYS = _load_known_keys()

# Non-secret config values returned unmasked
CONFIG_KEYS = [
	"AUTO_RUN", "DEFAULT_TRANSCRIBER", "DEFAULT_LLM", "DEFAULT_VIDEO_GEN",
	"LUMA_MODEL", "RUNWAY_MODEL",
	"OPENAI_MODEL", "ANTHROPIC_MODEL", "GEMINI_MODEL", "DEEPSEEK_MODEL",
]


@router.get("/settings")
async def get_settings():
	"""Return masked API key values from the .env file.

    - If the value is unset, returns ``""`` (frontend shows "Not configured").
    - If the value matches the .env.example placeholder, returns the example
      value so the user can see they haven't set a real key yet.
    - Otherwise returns ``"<first3>...<last3> [OK]"`` to confirm the key is set
      without leaking the actual secret.

    Config keys (e.g. AUTO_RUN) are returned as-is without masking.
    """
	if not os.path.exists(ENV_PATH):
		logger.warning("Settings .env not found at %s", ENV_PATH)
		return {}

	env_vars = dotenv.dotenv_values(ENV_PATH)
	example_vars = dotenv.dotenv_values(EXAMPLE_PATH) if os.path.exists(EXAMPLE_PATH) else {}
	masked = {}

	for key in KNOWN_KEYS:
		value = env_vars.get(key, "")
		example = example_vars.get(key, "")
		if not value:
			masked[key] = ""
		elif value == example:
			masked[key] = example
		else:
			prefix = value[:3]
			suffix = value[-3:] if len(value) > 6 else ""
			masked[key] = f"{prefix}...{suffix} \u2713"

	for key in CONFIG_KEYS:
		masked[key] = env_vars.get(key, "")

	return masked


@router.post("/settings")
async def update_settings(settings: Dict[str, str]):
	"""Persist new API key values to the .env file.

    Only non-empty values are written; omitted or empty keys are left
    unchanged so partial updates don't wipe existing configuration.
    """
	from core import ROOT_DIR
	os.makedirs(ROOT_DIR, exist_ok=True)

	# Create the .env file if it doesn't exist yet
	if not os.path.exists(ENV_PATH):
		open(ENV_PATH, 'a').close()

	for key, value in settings.items():
		if value:
			dotenv.set_key(ENV_PATH, key, value)
			logger.info("Updated setting: %s", key)

	return {"status": "success"}
