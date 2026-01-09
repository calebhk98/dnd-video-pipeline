"""
Scans the project for deeply nested code blocks (default > 5 levels).
Helps maintain code readability by encouraging early escapes and function extraction.
"""
import os

import argparse
import json
from pathlib import Path

# Default Configurations
DEFAULT_THRESHOLD = 5
DEFAULT_SPACES_PER_TAB = 4

# Typical ignores
DEFAULT_IGNORED_DIRS = {
	'.git', '.venv', 'venv', 'node_modules', '__pycache__', 
	'outputs', 'logs', 'dist', 'build', '.mypy_cache', '.pytest_cache', 
	'.gemini', '.idea', '.vscode'
}

DEFAULT_IGNORED_EXTS = {
	'.pyc', '.png', '.jpg', '.jpeg', '.gif', '.mp3', '.mp4', '.wav', 
	'.css', '.html', '.svg', '.json', '.lock', '.pdf', '.docx', '.csv', '.bin'
}

def get_max_indentation(file_path: Path, spaces_per_tab: int = 4) -> tuple[int, int]:
	"""
    Read a file and return (max_indentation_level, line_number_of_deepest_line).
    Line number is 1-indexed. Returns (-1, -1) on read error.
    """
	max_indent = 0
	max_line_num = -1
	try:
		with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
			for line_num, line in enumerate(f, start=1):
				stripped = line.lstrip(' \t')

				# Ignore empty lines or pure newlines
				if not stripped or stripped.startswith('\n'):
					continue

				leading_ws = line[:len(line) - len(stripped)]

				# Normalize tabs -> spaces so levels are consistent regardless of tab width
				spaces_count = len(leading_ws.replace('\t', ' ' * spaces_per_tab))

				level = spaces_count // spaces_per_tab
				if level > max_indent:
					max_indent = level
					max_line_num = line_num

		return max_indent, max_line_num
	except Exception as e:
		# Log and skip unreadable files (binary blobs, permission errors, etc.)
		print(f"Warning: Failed to read {file_path}: {e}")
		return -1, -1

def is_ignored(path: Path, ignored_dirs: set[str] = None, ignored_exts: set[str] = None) -> bool:
	"""Check if a path should be ignored based on directories or extension."""
	dirs = ignored_dirs if ignored_dirs is not None else DEFAULT_IGNORED_DIRS
	exts = ignored_exts if ignored_exts is not None else DEFAULT_IGNORED_EXTS

	if path.suffix.lower() in exts:
		return True
	
	# Check parts of the path
	for part in path.parts:
		if part in dirs:
			return True
			
	return False

def find_deep_files(root_dir: Path, threshold: int, spaces_per_tab: int, ignored_dirs: set[str], ignored_exts: set[str]) -> list[tuple[Path, int, int]]:
	"""Recursively find files exceeding the threshold indentation level.

    Returns a list of (file_path, max_indent_level, deepest_line_number) tuples.
    """
	deep_files = []

	for root, dirs, files in os.walk(root_dir):
		current_dir = Path(root)

		# Modify dirs in-place so os.walk skips ignored directories entirely
		dirs[:] = [d for d in dirs if d not in ignored_dirs]

		if is_ignored(current_dir, ignored_dirs, ignored_exts):
			continue

		for file in files:
			file_path = current_dir / file

			if is_ignored(file_path, ignored_dirs, ignored_exts):
				continue

			max_indent, deepest_line = get_max_indentation(file_path, spaces_per_tab)
			if max_indent > threshold:
				deep_files.append((file_path, max_indent, deepest_line))

	return deep_files

def main():
	"""Main execution point for checking code indentation levels."""

	parser = argparse.ArgumentParser(description="Scan for files exceeding an indentation depth threshold.")
	parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help=f"Maximum allowed indentation depth (default: {DEFAULT_THRESHOLD})")
	parser.add_argument("--spaces-per-tab", type=int, default=DEFAULT_SPACES_PER_TAB, help=f"Spaces per tab for counting (default: {DEFAULT_SPACES_PER_TAB})")
	parser.add_argument("--root", type=str, default=".", help="Root directory to scan (default: workspace root)")
	parser.add_argument("--json", action="store_true", help="Output results in JSON format")
	parser.add_argument("--output", "-o", type=str, help="Save output to a file in the 'logs' folder")
	parser.add_argument("--exclude-dir", action="append", help="Additional directories to exclude")
	parser.add_argument("--exclude-ext", action="append", help="Additional extensions to exclude (with dot, e.g., .md)")
	
	args = parser.parse_args()

	# Consolidate configuration
	ignored_dirs = set(DEFAULT_IGNORED_DIRS)
	if args.exclude_dir:
		ignored_dirs.update(args.exclude_dir)
		
	ignored_exts = set(DEFAULT_IGNORED_EXTS)
	if args.exclude_ext:
		ignored_exts.update(args.exclude_ext)

	if args.root == ".":
		script_dir = Path(__file__).resolve().parent
		workspace_root = script_dir.parent
	else:
		workspace_root = Path(args.root).resolve()

	if not args.json:
		print(f"Scanning {workspace_root.as_posix()} for files with > {args.threshold} indentation levels...\n", flush=True)
	
	deep_files = find_deep_files(workspace_root, args.threshold, args.spaces_per_tab, ignored_dirs, ignored_exts)

	# Sort files by largest indentation descending
	deep_files.sort(key=lambda x: x[1], reverse=True)

	# Format output
	output_lines = []
	if args.json:
		results = [
			{
				"path": Path(f[0].relative_to(workspace_root)).as_posix() if workspace_root in f[0].parents else f[0].as_posix(),
				"max_indentation": f[1],
				"deepest_line": f[2],
			}
			for f in deep_files
		]
		content = json.dumps(results, indent=2)
		output_lines.append(content)
	else:
		if not deep_files:
			output_lines.append(f"No files found with > {args.threshold} indentation levels. Great work following rule #2!")
		else:
			output_lines.append(f"Found {len(deep_files)} deeply indented files (> {args.threshold} levels):")

			indent_col_w = max(len(str(f[1])) for f in deep_files)
			indent_col_w = max(indent_col_w, len("Indent"))

			line_col_w = max(len(str(f[2])) for f in deep_files)
			line_col_w = max(line_col_w, len("Line #"))

			header = f"{'Indent'.rjust(indent_col_w)} | {'Line #'.rjust(line_col_w)} | File Path"
			output_lines.append(header)
			output_lines.append("-" * (indent_col_w + line_col_w + 3 + 3 + 40))

			for file_path, max_indent, deepest_line in deep_files:
				try:
					rel_path = file_path.relative_to(workspace_root)
				except ValueError:
					rel_path = file_path
				output_lines.append(
					f"{str(max_indent).rjust(indent_col_w)} | {str(deepest_line).rjust(line_col_w)} | {Path(rel_path).as_posix()}"
				)

			output_lines.append("-" * (indent_col_w + line_col_w + 3 + 3 + 40))

	# Print results
	for line in output_lines:
		print(line, flush=True)

	# Save to file
	if args.output:
		logs_dir = workspace_root / "logs"
		logs_dir.mkdir(exist_ok=True)
		
		output_file = logs_dir / args.output
		try:
			with open(output_file, "w", encoding="utf-8") as f:
				f.write("\n".join(output_lines) + "\n")
			print(f"\nResults saved to: {output_file.as_posix()}", flush=True)
		except Exception as e:
			print(f"\nError: Failed to save results to {output_file}: {e}", flush=True)

if __name__ == "__main__":
	main()
