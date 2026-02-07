"""
Integration tests for the performance comparison runner.
Verifies that permutations are correctly executed and metrics are collected.
"""
import pytest

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.evaluation.run_comparisons import run_permutation, process_permutations

@pytest.fixture
def mock_transcriber():
	"""Provides a mocked transcriber for testing."""

	mock = MagicMock()
	mock.transcribe.return_value = MagicMock(segments=[])
	return mock

@pytest.fixture
def mock_llm():
	"""Provides a mocked LLM processor for testing."""

	mock = MagicMock()
	mock.map_speakers.return_value = {}
	mock.generate_storyboard.return_value = MagicMock(scenes=[])
	mock.generate_production_script.return_value = MagicMock(scenes=[])
	return mock

@pytest.fixture
def mock_video():
	"""Provides a mocked video generator for testing."""

	mock = AsyncMock()
	mock.generate_all_scenes.return_value = ["dummy_scene.mp4"]
	return mock

@pytest.fixture
def mock_assembler():
	"""Provides a mocked FFmpeg assembler for testing."""

	mock = MagicMock()
	return mock

@pytest.fixture
def test_permutation():
	"""Provides a sample model permutation for testing."""

	return {
		"name": "Test-Run-1",
		"transcriber": "mock_transcriber",
		"llm": "mock_llm",
		"video": "mock_video",
		"assembler": "mock_assembler"
	}

@patch('src.evaluation.run_comparisons.MODEL_REGISTRY')
@pytest.mark.asyncio
async def test_run_permutation_success(mock_registry, test_permutation, mock_transcriber, mock_llm, mock_video, mock_assembler, tmp_path):
	"""Test a successful permutation run and metric collection."""

	# Setup the registry mock to return our mocked classes
	mock_transcriber_class = MagicMock(return_value=mock_transcriber)
	mock_llm_class = MagicMock(return_value=mock_llm)
	mock_video_class = MagicMock(return_value=mock_video)
	mock_assembler_class = MagicMock(return_value=mock_assembler)
	
	mock_registry.__getitem__.side_effect = lambda k: {
		"mock_transcriber": mock_transcriber_class,
		"mock_llm": mock_llm_class,
		"mock_video": mock_video_class,
		"mock_assembler": mock_assembler_class
	}[k]

	metrics = await run_permutation(test_permutation, "dummy_audio.mp3", tmp_path)
	
	# Check that metrics were collected
	assert metrics.permutation_name == "Test-Run-1"
	assert metrics.status == "Success"
	assert metrics.stage1_time >= 0
	assert metrics.stage2_time >= 0
	assert metrics.stage3_time >= 0
	assert metrics.stage4_time >= 0
	assert metrics.total_time > 0
	
	# Check that the stages were actually called
	mock_transcriber.transcribe.assert_called_once()
	mock_llm.generate_production_script.assert_called_once()
	mock_video.generate_all_scenes.assert_called_once()
	mock_assembler.stitch_videos.assert_called_once()
	mock_assembler.overlay_audio.assert_called_once()

@patch('src.evaluation.run_comparisons.MODEL_REGISTRY')
@pytest.mark.asyncio
async def test_run_permutation_failure(mock_registry, test_permutation, mock_transcriber, mock_llm, mock_video, mock_assembler, tmp_path):
	"""Test that failures in a stage are correctly handled and reported."""

	# Setup registry
	mock_transcriber_class = MagicMock(return_value=mock_transcriber)
	
	# Make LLM throw an error
	mock_llm.generate_storyboard.side_effect = Exception("API Timeout")
	mock_llm_class = MagicMock(return_value=mock_llm)
	
	mock_video_class = MagicMock(return_value=mock_video)
	mock_assembler_class = MagicMock(return_value=mock_assembler)
	
	mock_registry.__getitem__.side_effect = lambda k: {
		"mock_transcriber": mock_transcriber_class,
		"mock_llm": mock_llm_class,
		"mock_video": mock_video_class,
		"mock_assembler": mock_assembler_class
	}[k]

	metrics = await run_permutation(test_permutation, "dummy_audio.mp3", tmp_path)
	
	# Should catch error, mark as failed
	assert metrics.permutation_name == "Test-Run-1"
	assert metrics.status == "Failed"
	assert "API Timeout" in metrics.error_traceback
	
	# Video generation should NOT be called since LLM failed
	mock_video.generate_all_scenes.assert_not_called()

@patch('src.evaluation.run_comparisons.MODEL_REGISTRY')
@pytest.mark.asyncio
async def test_process_permutations_file_outputs(mock_registry, test_permutation, mock_transcriber, mock_llm, mock_video, mock_assembler, tmp_path):
	"""Verify that processing multiple permutations correctly writes results to files."""

	mock_registry.__getitem__.side_effect = lambda k: {
		"mock_transcriber": MagicMock(return_value=mock_transcriber),
		"mock_llm": MagicMock(return_value=mock_llm),
		"mock_video": MagicMock(return_value=mock_video),
		"mock_assembler": MagicMock(return_value=mock_assembler)
	}[k]

	results_dir = tmp_path / "results"
	
	await process_permutations([test_permutation], "dummy_audio.mp3", str(results_dir))
	
	json_path = results_dir / "evaluation_metrics.json"
	csv_path = results_dir / "evaluation_summary.csv"
	
	assert json_path.exists()
	assert csv_path.exists()
	
	import json
	with open(json_path, "r") as f:
		data = json.load(f)
		assert len(data) == 1
		assert data[0]["permutation_name"] == "Test-Run-1"
		assert data[0]["status"] == "Success"
		
	import csv
	with open(csv_path, "r") as f:
		reader = csv.DictReader(f)
		rows = list(reader)
		assert len(rows) == 1
		assert rows[0]["permutation_name"] == "Test-Run-1"
		assert rows[0]["status"] == "Success"
