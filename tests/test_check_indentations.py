"""Tests for the indentations check utility."""
import pytest

from pathlib import Path
import sys

# Add scripts directory to module search path
scripts_path = Path(__file__).parent.parent / "scripts"
sys.path.append(str(scripts_path))

from check_indentations import get_max_indentation, is_ignored, find_deep_files

def test_get_max_indentation_empty(tmp_path):
	"""Verify that an empty file returns 0 indentation."""

	f = tmp_path / "empty.py"
	f.write_text("")
	assert get_max_indentation(f) == 0

def test_get_max_indentation_spaces(tmp_path):
	"""Verify indentation calculation for space-indented files."""

	f = tmp_path / "spaces.py"
	content = """def main():
    if True:
        print("Hello")
"""
	f.write_text(content)
	assert get_max_indentation(f, spaces_per_tab=4) == 2

def test_get_max_indentation_tabs(tmp_path):
	"""Verify indentation calculation for tab-indented files."""

	f = tmp_path / "tabs.py"
	content = "def main():\n\tif True:\n\t\tprint('Hello')"
	f.write_text(content)
	assert get_max_indentation(f, spaces_per_tab=4) == 2

def test_get_max_indentation_mixed(tmp_path):
	"""Verify handling of mixed spaces and tabs (normalized to space count)."""

	f = tmp_path / "mixed.py"
	# mixed spaces and tabs
	content = "def main():\n  \tprint('Mixed')"
	f.write_text(content)
	# The tab expands to 4 spaces, plus 2 spaces = 6 spaces. 6 // 4 = 1.
	assert get_max_indentation(f, spaces_per_tab=4) == 1

def test_get_max_indentation_custom_spaces(tmp_path):
	"""Verify calculation with non-standard space-per-tab setting."""

	f = tmp_path / "custom.py"
	content = "def main():\n  if True:\n    print('Hello')"
	f.write_text(content)
	assert get_max_indentation(f, spaces_per_tab=2) == 2

def test_is_ignored():
	"""Verify directory and extension filtering logic."""

	assert is_ignored(Path("venv/foo.py")) == True
	assert is_ignored(Path("foo.jpg")) == True
	assert is_ignored(Path("src/foo.py")) == False

def test_find_deep_files(tmp_path):
	"""Verify recursive scan and depth threshold filtering."""

	(tmp_path / "src").mkdir()
	f1 = tmp_path / "src" / "deep.py"
	# 3 tabs deep
	f1.write_text("def a():\n\tdef b():\n\t\tdef c():\n\t\t\tpass")
	
	f2 = tmp_path / "src" / "shallow.py"
	# 1 tab deep
	f2.write_text("def a():\n\tpass")
	
	# 3 tabs deep inside ignored dir
	(tmp_path / "venv").mkdir()
	f3 = tmp_path / "venv" / "deep_ignored.py"
	f3.write_text("def a():\n\tdef b():\n\t\tdef c():\n\t\t\tpass")

	ignored_dirs = {'venv'}
	ignored_exts = {'.jpg'}
	
	# threshold 2: f1 should be flagged
	files = find_deep_files(tmp_path, threshold=2, spaces_per_tab=4, ignored_dirs=ignored_dirs, ignored_exts=ignored_exts)
	
	assert len(files) == 1
	assert files[0][0] == f1
	assert files[0][1] == 3
