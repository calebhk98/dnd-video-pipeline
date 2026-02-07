"""
Integration tests for the FastAPI web server.
Covers file uploads, job status polling, and history retrieval.
"""
import pytest

import sys
import os
# Add Web directory to path to import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Web')))

from fastapi.testclient import TestClient
from app import app, jobs_db

client = TestClient(app)

def test_api_upload_success():
	"""Test successful audio file upload and job creation."""

	# Mocking file upload
	file_content = b"fake audio data"
	response = client.post(
		"/api/upload",
		files={"file": ("test.mp3", file_content, "audio/mpeg")},
		data={"num_speakers": 2} # Form data
	)
	assert response.status_code == 200
	data = response.json()
	assert "job_id" in data
	
	# Store job ID for further tests
	pytest.shared_job_id = data["job_id"]

def test_api_upload_invalid_extension():
	"""Test that uploading a non-audio file results in a 400 error."""

	file_content = b"fake data"
	response = client.post(
		"/api/upload",
		files={"file": ("test.txt", file_content, "text/plain")},
		data={"num_speakers": 0}
	)
	assert response.status_code == 400

def test_api_get_transcript():
	"""Test retrieving the transcript for a created job."""

	job_id = getattr(pytest, "shared_job_id", None)
	if not job_id:
		pytest.skip("No job_id found")
		
	response = client.get(f"/api/transcript/{job_id}")
	assert response.status_code == 200
	data = response.json()
	assert "transcript" in data
	assert isinstance(data["speakers_detected"], list)

def test_api_map_speakers():
	"""Test submitting a speaker name mapping for a job."""

	job_id = getattr(pytest, "shared_job_id", None)
	if not job_id:
		pytest.skip("No job_id found")
		
	mapping = {"Speaker A": "Magnus", "Speaker B": "Merle"}
	response = client.post(f"/api/map_speakers/{job_id}", json=mapping)
	assert response.status_code == 200
	assert response.json()["status"] == "success"

def test_api_get_videos():
	"""Test polling for video generation status."""

	job_id = getattr(pytest, "shared_job_id", None)
	if not job_id:
		pytest.skip("No job_id found")
		
	response = client.get(f"/api/videos/{job_id}")
	assert response.status_code == 200
	data = response.json()
	assert data["status"] in ("completed", "uploaded", "processing", "error")
	assert "status" in data

def test_api_get_history():
	"""Test retrieving the global job history."""

	response = client.get("/api/history")
	assert response.status_code == 200
	data = response.json()
	assert "sessions" in data
	assert isinstance(data["sessions"], list)
