"""
Tests for the settings API and environment variable management.
Verifies masking of API keys and correct reading/writing of .env files.
"""
import pytest

import os
import sys
from fastapi.testclient import TestClient
import tempfile
import dotenv

# Add the project root to sys.path to resolve internal modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Web.app import app  # noqa: E402 - side-effect: adds Web/ to sys.path
import routers.settings as settings_router  # same module object the app registered

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_env_path(monkeypatch, tmp_path):
	"""Fixture to use temporary .env and .env.example files for testing."""
	temp_env_path = str(tmp_path / ".env")
	temp_example_path = str(tmp_path / ".env.example")

	# Initialize empty .env
	open(temp_env_path, 'a').close()

	# Write a .env.example with known placeholder values
	with open(temp_example_path, 'w') as f:
		f.write('OPENAI_API_KEY="your-openai-key"\n')
		f.write('DEEPGRAM_API_KEY="your-deepgram-key"\n')
		f.write('LUMA_API_KEY="your-luma-key"\n')

	monkeypatch.setattr(settings_router, "ENV_PATH", temp_env_path)
	monkeypatch.setattr(settings_router, "EXAMPLE_PATH", temp_example_path)
	yield temp_env_path

def test_get_settings_empty(mock_env_path):
	"""Test GET /api/settings when .env is empty."""
	response = client.get("/api/settings")
	assert response.status_code == 200
	data = response.json()
	assert data["OPENAI_API_KEY"] == ""
	assert data["DEEPGRAM_API_KEY"] == ""

def test_get_settings_placeholder_value(mock_env_path):
	"""Test GET /api/settings returns example value when key matches .env.example."""
	dotenv.set_key(mock_env_path, "OPENAI_API_KEY", "your-openai-key")

	response = client.get("/api/settings")
	assert response.status_code == 200
	data = response.json()
	assert data["OPENAI_API_KEY"] == "your-openai-key"

def test_post_settings_and_read_back(mock_env_path):
	"""Test POST /api/settings updates the file and GET masks it properly."""

	# 1. Update settings
	payload = {
		"OPENAI_API_KEY": "sk-real-test-key-12345",
		"DEEPGRAM_API_KEY": "test-deepgram"
	}
	response = client.post("/api/settings", json=payload)
	assert response.status_code == 200
	assert response.json() == {"status": "success"}

	# Verify the file was written to
	env_vars = dotenv.dotenv_values(mock_env_path)
	assert env_vars["OPENAI_API_KEY"] == "sk-real-test-key-12345"
	assert env_vars["DEEPGRAM_API_KEY"] == "test-deepgram"

	# 2. GET settings and verify they are masked
	response = client.get("/api/settings")
	assert response.status_code == 200
	data = response.json()

	assert data["OPENAI_API_KEY"] == "sk-...345 \u2713"   # first 3 + last 3 + checkmark
	assert data["DEEPGRAM_API_KEY"] == "tes...ram \u2713"

	# Empty inputs shouldn't fail
	assert data["LUMA_API_KEY"] == ""

def test_post_settings_ignores_empty(mock_env_path):
	"""Test POST /api/settings doesn't overwrite with empty values if previously set."""

	# Set initial value
	dotenv.set_key(mock_env_path, "OPENAI_API_KEY", "initial-key")

	# Post empty value
	payload = {
		"OPENAI_API_KEY": "",
		"FAL_KEY": "new-fal-key"
	}
	response = client.post("/api/settings", json=payload)
	assert response.status_code == 200

	# Check that initial-key wasn't erased
	env_vars = dotenv.dotenv_values(mock_env_path)
	assert env_vars["OPENAI_API_KEY"] == "initial-key"
	assert env_vars["FAL_KEY"] == "new-fal-key"
