"""Stage Runner Entry Points - Re-export Facade
================================================
Each stage's logic now lives in its own module.  This file re-exports all four
``run_stageN`` functions so that existing callers (e.g. ``pipeline.py``) can
continue to import from ``src.orchestrator.stage_runners`` without changes.
"""

from src.orchestrator.stage1_runner import run_stage1
from src.orchestrator.stage2_runner import run_stage2
from src.orchestrator.stage3_runner import run_stage3
from src.orchestrator.stage4_runner import run_stage4

__all__ = ["run_stage1", "run_stage2", "run_stage3", "run_stage4"]
