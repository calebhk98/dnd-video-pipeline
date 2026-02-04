"""
generate_report.py
==================
Step 2 of the two-step evaluation workflow:
  1. run_comparisons.py  -> executes pipeline permutations, writes evaluation_metrics.json
  2. generate_report.py  -> reads evaluation_metrics.json, writes FINAL_REPORT.md  <- (this file)

Responsibility
--------------
Convert the raw JSON metrics produced by run_comparisons into a human-readable
GitHub-Flavored Markdown report. The report contains:
  - An Executive Summary (total runs, success/failure counts)
  - A Performance Leaderboard (all runs ranked by total execution time)
  - A Stage Breakdown (fastest model per pipeline stage, across successful runs only)
  - An Error Log (full error messages for every failed run)

Usage
-----
Run directly as a script from the project root:
    python -m src.evaluation.generate_report

Or import `generate_markdown_report` for testing/programmatic use:
    from src.evaluation.generate_report import generate_markdown_report
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure module-level logging using a standard format that includes timestamps
# and severity level, making it easy to correlate log lines with wall-clock time.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_markdown_report(metrics_data: List[Dict[str, Any]]) -> str:
	"""
    Generates a Markdown string summarizing the evaluation metrics.

    Args:
        metrics_data: A list of dictionaries, each representing a permutation run's metrics.
                      Expected keys per dict:
                        - permutation_name (str)
                        - stage1_time, stage2_time, stage3_time, stage4_time (float | None)
                        - total_time (float | None)
                        - status ("Success" | "Failed")
                        - error_message (str, present only on failed runs)

    Returns:
        A string containing the formatted Github-Flavored Markdown report.
        The string uses "\n" line endings and is ready to write directly to a .md file.
    """
	# Early return for empty input: avoids creating a report with empty sections
	# that would be confusing (e.g. a leaderboard table with no rows).
	if not metrics_data:
		return "# DND-Video-Pipeline: Final Evaluation Report\n\n## Executive Summary\n- 0 succeeded\n- 0 failed\n"

	# Partition runs into successes and failures up front.
	# We reference both lists in multiple sections (executive summary, stage breakdown,
	# error log), so computing them once here avoids repeated list comprehensions.
	successful_runs = [m for m in metrics_data if m.get("status") == "Success"]
	failed_runs = [m for m in metrics_data if m.get("status") == "Failed"]

	# Build the report as a list of strings and join at the end.
	# String concatenation in a loop creates many intermediate objects; appending to
	# a list and calling "\n".join() once is more memory-efficient.

	# --- Executive Summary ---
	# High-level pass/fail counts so readers know at a glance how the evaluation went.
	md_lines = [
		"# DND-Video-Pipeline: Final Evaluation Report",
		"",
		"## Executive Summary",
		f"- **Total Permutations Run**: {len(metrics_data)}",
		f"- **Succeeded**: {len(successful_runs)}",
		f"- **Failed**: {len(failed_runs)}",
		""
	]

	# --- Performance Leaderboard ---
	# Ranks ALL runs (successes and failures) by total execution time, fastest first.
	# Including failures in the leaderboard lets reviewers see how much time was spent
	# before a run crashed, which is useful for cost estimation.
	md_lines.extend([
		"## Performance Leaderboard",
		"Ranked by overall execution time (Fastest to Slowest).",
		"",
		"| Rank | Permutation | Total Time (s) | Status |",
		"| :--- | :--- | :--- | :--- |"
	])

	# Sort all runs by total_time ascending.
	# Runs with missing or non-numeric total_time are sorted to the bottom via the
	# float('inf') sentinel ,  they failed before a meaningful time could be recorded.
	sorted_all_runs = sorted(
		metrics_data,
		key=lambda x: x.get("total_time", float('inf')) if isinstance(x.get("total_time"), (int, float)) else float('inf')
	)

	for i, run in enumerate(sorted_all_runs, start=1):
		name = run.get("permutation_name", "Unknown")
		t_time = run.get("total_time", "N/A")
		# Format floats to 2 decimal places for a clean table; leave non-numeric as-is.
		if isinstance(t_time, float):
			t_time = f"{t_time:.2f}"
		status = run.get("status", "Unknown")
		# Emoji prefix gives an immediate visual signal when scanning the table in a
		# rendered Markdown viewer (e.g. GitHub PR, VS Code preview).
		status_icon = "[PASS] Success" if status == "Success" else "[FAIL] Failed"
		md_lines.append(f"| {i} | {name} | {t_time} | {status_icon} |")

	md_lines.append("")

	# --- Stage Breakdown ---
	# Shows which model was fastest for each of the three differentiating stages.
	# Stage 4 (FFmpeg assembly) is intentionally excluded: it runs on local hardware
	# and its time is dominated by video file size rather than the choice of assembler,
	# making cross-permutation comparison less meaningful for that stage.
	md_lines.extend([
		"## Stage Breakdown",
		"Comparing individual stage execution times across successful models.",
		""
	])

	if successful_runs:
		# Nested helper: closes over `successful_runs` from the enclosing scope.
		# Returns the run dict with the minimum time for the given stage key, or None
		# if no successful run recorded a valid (numeric) time for that stage.
		# We intentionally restrict to successful runs: failed runs may have a default
		# value of 0.0 for stages they never reached, which would incorrectly "win".
		def get_fastest_for_stage(stage_key: str) -> Dict[str, Any]:
			"""Helper to find the best performing run for a specific pipeline stage."""

			# Filter to runs that actually completed this stage and recorded a real time.
			valid_runs = [r for r in successful_runs if isinstance(r.get(stage_key), (int, float))]
			if not valid_runs:
				return None
			return min(valid_runs, key=lambda x: x[stage_key])

		fastest_stage1 = get_fastest_for_stage("stage1_time")
		fastest_stage2 = get_fastest_for_stage("stage2_time")
		fastest_stage3 = get_fastest_for_stage("stage3_time")

		# Stage 1 Transcription
		# Which transcriber service converted the audio to text fastest?
		md_lines.append("### Fastest Transcriber (Stage 1)")
		if fastest_stage1:
			md_lines.append(f"- **{fastest_stage1.get('permutation_name')}**: {fastest_stage1.get('stage1_time'):.2f}s")
		else:
			md_lines.append("- N/A")
		md_lines.append("")

		# Stage 2 LLM Processing
		# Which LLM produced the storyboard and production script fastest?
		md_lines.append("### Fastest LLM (Stage 2)")
		if fastest_stage2:
			md_lines.append(f"- **{fastest_stage2.get('permutation_name')}**: {fastest_stage2.get('stage2_time'):.2f}s")
		else:
			md_lines.append("- N/A")
		md_lines.append("")

		# Stage 3 Video Generation
		# Which video generation service rendered all scenes fastest?
		# This is typically the longest stage due to API-side rendering time.
		md_lines.append("### Fastest Video Generator (Stage 3)")
		if fastest_stage3:
			md_lines.append(f"- **{fastest_stage3.get('permutation_name')}**: {fastest_stage3.get('stage3_time'):.2f}s")
		else:
			md_lines.append("- N/A")
		md_lines.append("")
	else:
		# All runs failed there is no meaningful stage data to compare.
		md_lines.append("*No successful runs to break down.*")
		md_lines.append("")

	# --- Error Log ---
	# Only emitted when at least one run failed.  Each failed run gets its own
	# sub-section with the error message in a fenced code block so that multi-line
	# stack traces render correctly in GitHub/GitLab without collapsing whitespace.
	if failed_runs:
		md_lines.extend([
			"## Error Log",
			"Details on failed permutations.",
			""
		])
		for run in failed_runs:
			name = run.get("permutation_name", "Unknown")
			# error_message is written by run_comparisons (the full traceback string).
			# Fall back gracefully if the key is absent (e.g. manually crafted JSON).
			err_msg = run.get("error_message", "No error message provided.")
			md_lines.extend([
				f"### {name}",
				"```",
				err_msg,
				"```",
				""
			])

	# Join all lines with newlines to produce the final Markdown document.
	return "\n".join(md_lines)

def main():
	"""Reads metrics JSON, generates the Markdown report, and writes to disk."""
	# Resolve the project root by traversing upward from this script's location.
	# Expected directory structure:
	#   <project_root>/
	#     src/
	#       evaluation/
	#         generate_report.py   <- __file__
	# Therefore: __file__ -> evaluation/ -> src/ -> project_root/
	current_file = Path(__file__).resolve()
	project_root = current_file.parent.parent.parent

	results_dir = project_root / "results"
	metrics_file = results_dir / "evaluation_metrics.json"
	report_file = results_dir / "FINAL_REPORT.md"

	# Fail fast with a clear error if the metrics file doesn't exist rather than
	# producing an empty or misleading report.
	if not metrics_file.exists():
		logger.error(f"Metrics file not found: {metrics_file}")
		return

	# Read failure is handled separately from write failure because the recovery
	# actions differ: a read failure means the source data is corrupt or missing
	# (nothing useful can be done), while a write failure may indicate a permissions
	# issue with the results directory (the report content itself is fine).
	try:
		with open(metrics_file, "r", encoding="utf-8") as f:
			metrics_data = json.load(f)
	except Exception as e:
		logger.error(f"Failed to read or parse metrics JSON. Error: {e}")
		return

	logger.info(f"Loaded {len(metrics_data)} records from {metrics_file.name}")

	markdown_content = generate_markdown_report(metrics_data)

	# Ensure the results directory exists before writing.
	# parents=True handles the case where the entire path is missing;
	# exist_ok=True makes the call idempotent ,  safe to re-run without error.
	results_dir.mkdir(parents=True, exist_ok=True)

	try:
         with open(report_file, "w", encoding="utf-8") as f:
             f.write(markdown_content)
         logger.info(f"Successfully wrote evaluation report to {report_file}")
	except Exception as e:
         logger.error(f"Failed to write report. Error: {e}")

if __name__ == "__main__":
	main()
