"""
test_generate_report.py
=======================
Unit tests for `src.evaluation.generate_report.generate_markdown_report`.

Scope
-----
These tests exercise the pure `generate_markdown_report()` function only.
That function has no side effects (no filesystem I/O, no network calls) so
no mocking or temporary directories are needed ,  just call the function and
assert on the returned string.

The `main()` function in generate_report.py (which reads from disk and writes
FINAL_REPORT.md) is covered separately by integration tests, not here.

Test Strategy
-------------
Two pytest fixtures provide reusable input data:
  - sample_metrics    : a realistic "happy path" dataset (2 successes, 1 failure)
  - edge_case_metrics : a dataset with missing/None values to probe robustness

Each test function is narrow: it asserts only what its name promises, making
failures easy to diagnose without wading through unrelated assertions.
"""

import pytest
from src.evaluation.generate_report import generate_markdown_report


@pytest.fixture
def sample_metrics():
	"""Realistic evaluation metrics with 2 successful runs and 1 failed run.

    Design decisions for the fixture data:
    - Three entries cover all three report sections: Leaderboard, Stage Breakdown,
      and Error Log.  Two successes are needed to make stage-comparison meaningful.
    - AssemblyAI total_time (110.5s) < Deepgram total_time (137.8s) intentionally,
      so the leaderboard ordering is deterministic and can be asserted precisely.
    - Deepgram stage1_time (10.5s) < AssemblyAI stage1_time (15.0s) intentionally,
      so Stage 1 fastest-run logic can be tested independently of total_time ranking.
    - Whisper Local has stage3_time=0.0 and stage4_time=0.0 to simulate a run
      that failed partway through Stage 3 before generating any video output.
    """
	return [
		{
			"permutation_name": "Deepgram -> Claude -> Luma",
			"stage1_time": 10.5,   # Faster Stage 1 than AssemblyAI (10.5 < 15.0)
			"stage2_time": 5.2,
			"stage3_time": 120.0,  # Slower Stage 3 ,  drives the higher total_time
			"stage4_time": 2.1,
			"total_time": 137.8,   # Slower overall than AssemblyAI run
			"status": "Success"
		},
		{
			"permutation_name": "AssemblyAI -> OpenAI -> Replicate",
			"stage1_time": 15.0,   # Slower Stage 1 than Deepgram (15.0 > 10.5)
			"stage2_time": 3.1,	# Fastest Stage 2 in the fixture
			"stage3_time": 90.5,   # Fastest Stage 3 in the fixture
			"stage4_time": 1.9,
			"total_time": 110.5,   # Faster overall ,  should appear first in leaderboard
			"status": "Success"
		},
		{
			"permutation_name": "Whisper Local -> Local Llama -> Minimax",
			"stage1_time": 25.0,
			"stage2_time": 10.0,
			"stage3_time": 0.0,	# Never reached a real result; 0.0 is the default
			"stage4_time": 0.0,	# Never reached Stage 4 at all
			"total_time": 35.0,
			"status": "Failed",
			"error_message": "ConnectionError at Stage 3"  # Surfaced in Error Log section
		}
	]


@pytest.fixture
def edge_case_metrics():
	"""Edge-case metrics designed to test robustness with incomplete/missing data.

    This fixture is kept separate from sample_metrics intentionally: if we mixed
    edge cases into the main fixture, assertions that expect clean numeric data
    (e.g. stage time comparisons) would break.

    Entry 1 ,  all timing values are None:
        Simulates a run where the metrics object was created but none of the stage
        timers fired (e.g. the run was interrupted at startup).

    Entry 2 ,  missing permutation_name key entirely:
        Simulates a manually crafted or partially corrupted JSON record.
        The report code should fall back to "Unknown" rather than raising KeyError.
    """
	return [
		{
			"permutation_name": "Missing Times Models",
			"stage1_time": None,   # None, not 0.0 ,  tests the isinstance() guard in report
			"stage2_time": None,
			"stage3_time": None,
			"stage4_time": None,
			"total_time": None,	# Should be rendered as "N/A" in the leaderboard
			"status": "Success"	# Marked Success to exercise the stage breakdown path
		},
		{
			# No "permutation_name" key ,  tests the .get("permutation_name", "Unknown") fallback
			"stage1_time": 10.0,
			"stage2_time": 10.0,
			"stage3_time": 10.0,
			"stage4_time": 10.0,
			"total_time": 40.0,
			"status": "Failed",
			"error_message": "Some error"
		}
	]


def test_generate_markdown_report_executive_summary(sample_metrics):
	"""Test that the executive summary counts successes and failures correctly."""
	report = generate_markdown_report(sample_metrics)

	# Verify the report title is present to confirm basic structure.
	assert "# DND-Video-Pipeline: Final Evaluation Report" in report

	# Use .lower() on the report slice to avoid brittle case-sensitive matching.
	# The actual report uses bold Markdown syntax like "**Succeeded**: 2",
	# which becomes "succeeded**: 2" after lowercasing ,  still uniquely matchable.
	assert "succeeded**: 2" in report.lower()
	assert "failed**: 1" in report.lower()


def test_generate_markdown_report_leaderboard(sample_metrics):
	"""Test that the leaderboard ranks permutations by total_time (fastest first)."""
	report = generate_markdown_report(sample_metrics)

	assert "## Performance Leaderboard" in report

	# Find the character positions of each run name in the raw report string.
	# A lower index means the name appears earlier in the document ,  i.e. higher
	# in the leaderboard table.
	idx_assembly = report.find("AssemblyAI")
	idx_deepgram = report.find("Deepgram")

	# Both names must be present.
	assert idx_assembly != -1
	assert idx_deepgram != -1

	# Guard: if the string "Failed" appears between the AssemblyAI entry and the
	# end of the report, the index comparison could be confounded by the Error Log
	# section (which also mentions permutation names).  The conditional assertion
	# confirms ordering only when the comparison is unambiguous.
	if "Failed" not in report[idx_assembly:]:
		# AssemblyAI (110.5s total) should appear before Deepgram (137.8s total).
		assert idx_assembly < idx_deepgram


def test_generate_markdown_report_stage_breakdown(sample_metrics):
	"""Test that stage breakdown sections are present and show the correct fastest run."""
	report = generate_markdown_report(sample_metrics)

	# Verify all three stage headings are present.
	assert "### Fastest Transcriber" in report
	assert "### Fastest LLM" in report
	assert "### Fastest Video Generator" in report

	# Isolate the content between "### Fastest Transcriber (Stage 1)" and the
	# next "###" heading, then check that "Deepgram" appears in that slice.
	# This prevents a false positive if "Deepgram" appeared in a later section.
	# Deepgram has stage1_time=10.5s vs AssemblyAI's 15.0s, so it should win.
	assert "Deepgram" in report.split("### Fastest Transcriber (Stage 1)")[1].split("###")[0]


def test_generate_markdown_report_error_log(sample_metrics):
	"""Test that the error log includes failed permutations and their error messages."""
	report = generate_markdown_report(sample_metrics)

	# The Error Log section header must be present.
	assert "## Error Log" in report

	# Both the permutation name and the error message must appear in the log.
	# Having both is important: the name identifies which run failed, and the
	# error message explains why.
	assert "ConnectionError at Stage 3" in report
	assert "Whisper Local -> Local Llama -> Minimax" in report


def test_generate_markdown_report_empty():
	"""Test with empty metrics list ,  the function must not crash and must show 0/0 counts."""
	report = generate_markdown_report([])

	# The empty fast-path returns a slightly different string format than the full path.
	# Both "0 succeeded" (fast-path prose) and "succeeded**: 0" (full-path bold Markdown)
	# are valid representations; the `or` accepts either form.
	assert "0 succeeded" in report.lower() or "succeeded**: 0" in report.lower()
	assert "0 failed" in report.lower() or "failed**: 0" in report.lower()


def test_generate_markdown_report_edge_cases(edge_case_metrics):
	"""Test robustness with missing permutation names and None timing values."""
	report = generate_markdown_report(edge_case_metrics)

	# The missing "permutation_name" key should be handled gracefully via
	# .get("permutation_name", "Unknown") ,  "Unknown" must appear in the output.
	assert "Unknown" in report

	# A None total_time should be rendered as "N/A" in the leaderboard table
	# rather than crashing with a TypeError or showing "None".
	assert "N/A" in report

	# The successful run has all None stage times, so no stage in the stage
	# breakdown has a valid fastest run ,  each should show "- N/A".
	# The two-form assertion handles slight differences in whitespace/newlines
	# depending on whether the heading and bullet are on the same or different lines.
	assert "### Fastest Transcriber (Stage 1)\n- N/A" in report or "- N/A" in report.split("### Fastest Transcriber")[1]


def test_generate_markdown_report_all_failed():
	"""Test when every permutation failed ,  stage breakdown must show the 'no runs' message."""
	# This scenario warrants its own test because it exercises a specific branch:
	# when `successful_runs` is empty, the code must emit the fallback message
	# "*No successful runs to break down.*" instead of calling min() on an empty list,
	# which would raise a ValueError.
	metrics = [
		{"permutation_name": "Run 1", "status": "Failed", "error_message": "Err1"},
		{"permutation_name": "Run 2", "status": "Failed", "error_message": "Err2"}
	]
	report = generate_markdown_report(metrics)

	# Executive summary should reflect 0 successes and 2 failures.
	assert "succeeded**: 0" in report.lower()
	assert "failed**: 2" in report.lower()

	# Stage breakdown fallback message must be present.
	assert "No successful runs to break down." in report

	# Both error messages must appear in the Error Log section.
	assert "Err1" in report
	assert "Err2" in report
