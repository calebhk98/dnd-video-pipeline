"""
Pipeline Reporting Helpers
===========================
Console and JSON reporting for pipeline run results.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _print_scene_report(scenes, failures):
	"""Print a formatted scene generation summary table to the console.

    Args:
        scenes:   List of Scene objects from the production script (all scenes,
                  both succeeded and failed).
        failures: List of dicts with keys "scene_number" and "error" describing
                  scenes that could not be generated.
    """
	# Build a lookup from scene_number -> error string for O(1) checks per scene
	failed_map = {f["scene_number"]: f["error"] for f in failures}
	total = len(scenes)
	succeeded = total - len(failures)

	logger.info("=" * 60)
	logger.info("SCENE GENERATION REPORT")
	logger.info("=" * 60)
	# Column header ,  left-aligned with fixed widths to form a readable table
	logger.info(f"{'Scene':<8} {'Status':<10} {'Detail'}")
	logger.info("-" * 60)
	for scene in scenes:
		if scene.scene_number in failed_map:
			# Truncate long error messages to 60 characters to keep the table
			# readable; full errors are available in run_report.json
			error_snippet = failed_map[scene.scene_number][:60]
			logger.info(f"{scene.scene_number:<8} {'FAILED':<10} {error_snippet}")
		else:
			logger.info(f"{scene.scene_number:<8} {'OK':<10}")
	logger.info("-" * 60)
	logger.info(f"Total: {succeeded}/{total} succeeded, {len(failures)} failed/skipped")
	logger.info("=" * 60)


def _save_run_report(out_path: Path, scenes, failures):
	"""Write a machine-readable run_report.json to the session output directory.

    The report contains a high-level summary and per-scene details including
    success/failure status, error messages, and time ranges.  Consumers such as
    dashboards or CI scripts can parse this file to assess pipeline health without
    scanning log output.

    Args:
        out_path: Path to the session output directory.
        scenes:   List of all Scene objects from the production script.
        failures: List of dicts with keys "scene_number" and "error".
    """
	# Map scene_number -> error string for O(1) lookups when building records
	failed_map = {f["scene_number"]: f["error"] for f in failures}
	scene_records = []
	for scene in scenes:
		if scene.scene_number in failed_map:
			scene_records.append({
				"scene_number": scene.scene_number,
				"status": "failed",
				"error": failed_map[scene.scene_number],
				"start_time": scene.start_time,
				"end_time": scene.end_time,
			})
		else:
			scene_records.append({
				"scene_number": scene.scene_number,
				"status": "success",
				"start_time": scene.start_time,
				"end_time": scene.end_time,
			})

	report = {
		"summary": {
			"total_scenes": len(scenes),
			"succeeded": len(scenes) - len(failures),
			"failed": len(failures),
		},
		"scenes": scene_records,
	}

	report_path = out_path / "run_report.json"
	with open(report_path, "w") as f:
		json.dump(report, f, indent=2)
	logger.info(f"Run report saved to {report_path}")
