"""
Unit and integration tests for the Report Generator.
Verifies word count enforcement, text sanitization, and OpenAI-based generation.
"""
import os

import json
import pytest
from unittest.mock import patch, MagicMock
from src.orchestrator.report_generator import ReportGenerator, WordCountError

@pytest.fixture
def mock_generator():
	"""Provides a ReportGenerator instance with a test key."""

	# Use test_key so it doesn't try to read env vars or crash
	return ReportGenerator(api_key="test_key")

def test_word_count(mock_generator):
	"""Test basic word counting logic."""

	text = "This is a simple test of five words."
	print("Words manually split:", text.split())
	# Note: re.findall(r'\b\w+\b') counts 8 items: This, is, a, simple, test, of, five, words
	assert mock_generator.count_words(text) == 8

def test_enforce_word_count_success(mock_generator):
	"""Verify that text within the +/- 5% range passes enforcement."""

	text = "word " * 100
	mock_generator.enforce_word_count(text, 100) # Exactly 100
	mock_generator.enforce_word_count(text, 105) # Lower limit for 105 is int(105*0.95)=99, text has 100
	
	# 100 words text
	mock_generator.enforce_word_count("w "*105, 100) # 105 words is exactly upper bound for 100
	mock_generator.enforce_word_count("w "*95, 100) # 95 words is exactly lower bound for 100

def test_enforce_word_count_failure(mock_generator):
	"""Verify that text outside the allowed range raises a WordCountError."""

	text_short = "word " * 90
	text_long = "word " * 110
	
	with pytest.raises(WordCountError):
		mock_generator.enforce_word_count(text_short, 100)
		
	with pytest.raises(WordCountError):
		mock_generator.enforce_word_count(text_long, 100)

def test_sanitize_text(mock_generator):
	"""Verify that redundant AI-isms and unwanted punctuation are removed."""

	text = "The AI model, which is large--was fast. In conclusion, it worked well. Furthermore, it is a tapestry of patterns."
	sanitized = mock_generator.sanitize_text(text)
	
	assert ", " not in sanitized
	assert "--" not in sanitized
	assert "In conclusion" not in sanitized
	assert "Furthermore" not in sanitized
	assert "tapestry" not in sanitized.lower()
	
	expected = "The AI model, which is large, was fast. , it worked well.  , it is a  of patterns."
	assert "The AI model, which is large, was fast." in sanitized

@patch("src.orchestrator.report_generator.OpenAI")
def test_generate_section_integration(mock_openai, mock_generator, tmp_path):
	"""Integration test for generating a report section via mocked OpenAI API."""

	# Setup mock openai client
	mock_client = MagicMock()
	mock_openai.return_value = mock_client
	
	# Needs to fail first, then succeed
	mock_response_fail = MagicMock()
	mock_response_fail.choices[0].message.content = "word " * 50 # Fail target 100
	
	mock_response_success = MagicMock()
	mock_response_success.choices[0].message.content = "word " * 100 # Succeed target 100
	
	mock_client.chat.completions.create.side_effect = [
		mock_response_fail,
		mock_response_success
	]
	
	mock_generator.client = mock_client
	
	config_file = tmp_path / "test_config.json"
	config_data = [
		{
			"chapter": "Test Chapter",
			"sub_section": "Test Section",
			"target_words": 100
		}
	]
	with open(config_file, "w") as f:
		json.dump(config_data, f)
		
	final_output = mock_generator.run(str(config_file))
	
	assert "## Test Chapter" in final_output
	assert "### Test Section" in final_output
	# 2 calls, 1 fail, 1 success
	assert mock_client.chat.completions.create.call_count == 2
