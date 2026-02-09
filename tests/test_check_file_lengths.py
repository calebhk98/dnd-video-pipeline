"""Tests for the file length checker utility."""
import pytest

import json
from pathlib import Path
from scripts.check_file_lengths import get_file_line_count, is_ignored, find_large_files, DEFAULT_IGNORED_DIRS, DEFAULT_IGNORED_EXTS

def test_get_file_line_count(tmp_path):
	"""Test counting lines in a valid text file."""
	test_file = tmp_path / "test.txt"
	# 3 lines: "a\nb\nc" (2 newlines, but 3 lines of text)
	test_file.write_text("line1\nline2\nline3", encoding="utf-8")
	assert get_file_line_count(test_file) == 3

def test_get_file_line_count_newlines(tmp_path):
	"""Test counting lines with only newlines."""
	test_file = tmp_path / "newlines.txt"
	test_file.write_text("\n" * 5, encoding="utf-8")
	# \n\n\n\n\n -> 5 lines
	assert get_file_line_count(test_file) == 5

def test_get_file_line_count_binary(tmp_path):
	"""Test that binary files handle gracefully."""
	test_file = tmp_path / "test.bin"
	test_file.write_bytes(b"\x00\xFF\x00")
	count = get_file_line_count(test_file)
	assert isinstance(count, int)

def test_is_ignored_standard_dirs():
	"""Test that standard ignored directories are correctly identified."""
	assert is_ignored(Path(".git/config")) is True
	assert is_ignored(Path("venv/include/python3.10/pyconfig.h")) is True
	assert is_ignored(Path("__pycache__/main.cpython-310.pyc")) is True
	assert is_ignored(Path("src/main.py")) is False

def test_is_ignored_custom_dirs():
	"""Test that custom ignored directories are correctly identified."""
	custom_dirs = {".git", "custom_ignore"}
	assert is_ignored(Path("custom_ignore/file.txt"), ignored_dirs=custom_dirs) is True
	assert is_ignored(Path("other/file.txt"), ignored_dirs=custom_dirs) is False

def test_is_ignored_extensions():
	"""Test that ignored extensions are correctly identified."""
	assert is_ignored(Path("image.png")) is True
	assert is_ignored(Path("audio.mp3")) is True
	assert is_ignored(Path("script.py")) is False

def test_find_large_files(tmp_path):
	"""Test finding large files in a temporary directory structure."""
	root = tmp_path / "root"
	root.mkdir()
	
	large = root / "large.py"
	large.write_text("\n" * 10) # 10 lines
	
	small = root / "small.py"
	small.write_text("\n" * 2) # 2 lines
	
	ignored_dir = root / "ignored"
	ignored_dir.mkdir()
	ignored_large = ignored_dir / "large.py"
	ignored_large.write_text("\n" * 10)
	
	image = root / "image.png"
	image.write_text("\n" * 10)
	
	ignored_dirs = set(DEFAULT_IGNORED_DIRS) | {"ignored"}
	ignored_exts = set(DEFAULT_IGNORED_EXTS)
	
	results = find_large_files(root, threshold=5, ignored_dirs=ignored_dirs, ignored_exts=ignored_exts)
	
	# Should only find root/large.py
	assert len(results) == 1
	assert results[0][0].name == "large.py"
	assert results[0][1] == 10
	
def test_cli_json_output(tmp_path, capsys, monkeypatch):
	"""Test that the script can output JSON."""
	from scripts.check_file_lengths import main
	import sys
	
	root = tmp_path / "json_root"
	root.mkdir()
	large = root / "large.py"
	large.write_text("\n" * 10)
	
	# Mock sys.argv
	test_args = ["check_file_lengths.py", "--threshold", "5", "--root", str(root), "--json"]
	monkeypatch.setattr(sys, "argv", test_args)
	
	main()
	
	captured = capsys.readouterr()
	output = json.loads(captured.out)
	
	assert len(output) == 1
	assert output[0]["lines"] == 10
	assert output[0]["path"] == "large.py"

def test_cli_file_output(tmp_path, monkeypatch):
	"""Test that the script can save output to a file."""
	from scripts.check_file_lengths import main
	import sys
	
	root = tmp_path / "file_output_root"
	root.mkdir()
	large = root / "large.py"
	large.write_text("\n" * 10)
	
	output_filename = "test_log.txt"
	
	# Mock sys.argv
	test_args = ["check_file_lengths.py", "--threshold", "5", "--root", str(root), "--output", output_filename]
	monkeypatch.setattr(sys, "argv", test_args)
	
	main()
	
	logs_dir = root / "logs"
	output_file = logs_dir / output_filename
	
	assert logs_dir.exists()
	assert output_file.exists()
	
	content = output_file.read_text(encoding="utf-8")
	assert "Found 1 large files" in content
	assert "large.py" in content
