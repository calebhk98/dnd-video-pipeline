"""
Scans the project to identify files that exceed a specified line count limit.
Used to maintain code modularity by flagging overly large files.
"""
import os

import argparse
import json
from pathlib import Path

# Default Thresholds and Ignores Config
DEFAULT_THRESHOLD = 500
DEFAULT_IGNORED_DIRS = {
	'.git', '.venv', 'venv', 'node_modules', '__pycache__', 
	'outputs', 'logs', 'dist', 'build', '.mypy_cache', '.pytest_cache', 
	'.gemini', '.idea', '.vscode'
}

DEFAULT_IGNORED_EXTS = {
	'.pyc', '.png', '.jpg', '.jpeg', '.gif', '.mp3', '.mp4', '.wav', 
	'.css', '.html', '.svg', '.json', '.lock', '.pdf', '.docx', '.csv'
}

def get_file_line_count(file_path: Path) -> int:
	"""Read a file and return its line count. Returns -1 on error (e.g., binary files)."""
	try:
		with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
			return sum(1 for _ in f)
	except Exception as e:
		print(f"Warning: Failed to read {file_path}: {e}")
		return -1

def is_ignored(path: Path, ignored_dirs: set[str] = None, ignored_exts: set[str] = None) -> bool:
	"""Check if a path should be ignored based on directories or extension."""
	dirs = ignored_dirs if ignored_dirs is not None else DEFAULT_IGNORED_DIRS
	exts = ignored_exts if ignored_exts is not None else DEFAULT_IGNORED_EXTS

	if path.suffix.lower() in exts:
		return True
	
	# Check if any part of the path is in our ignore list
	for part in path.parts:
		if part in dirs:
			return True
			
	return False

def find_large_files(root_dir: Path, threshold: int, ignored_dirs: set[str], ignored_exts: set[str]) -> list[tuple[Path, int]]:
	"""Recursively find files exceeding the threshold line count."""
	large_files = []
	
	for root, dirs, files in os.walk(root_dir):
		current_dir = Path(root)
		
		# Modify dirs in-place to skip ignored directories immediately
		dirs[:] = [d for d in dirs if d not in ignored_dirs]
		
		# Skip if the current directory path itself contains an ignored directory (secondary check)
		if is_ignored(current_dir, ignored_dirs, ignored_exts):
			continue
			
		for file in files:
			file_path = current_dir / file
			
			if is_ignored(file_path, ignored_dirs, ignored_exts):
				continue
				
			line_count = get_file_line_count(file_path)
			if line_count >= threshold:
				large_files.append((file_path, line_count))
				
	return large_files

def main():
	"""Main execution point for scanning file lengths."""

	parser = argparse.ArgumentParser(description="Scan for files exceeding a line count threshold.")
	parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help=f"Line count threshold (default: {DEFAULT_THRESHOLD})")
	parser.add_argument("--root", type=str, default=".", help="Root directory to scan (default: workspace root)")
	parser.add_argument("--json", action="store_true", help="Output results in JSON format")
	parser.add_argument("--output", "-o", type=str, help="Save output to a file in the 'logs' folder")
	parser.add_argument("--exclude-dir", action="append", help="Additional directories to exclude")
	parser.add_argument("--exclude-ext", action="append", help="Additional extensions to exclude (with dot, e.g., .md)")
	
	args = parser.parse_args()

	# Consolidate ignores
	ignored_dirs = set(DEFAULT_IGNORED_DIRS)
	if args.exclude_dir:
		ignored_dirs.update(args.exclude_dir)
		
	ignored_exts = set(DEFAULT_IGNORED_EXTS)
	if args.exclude_ext:
		ignored_exts.update(args.exclude_ext)

	# Determine workspace root
	if args.root == ".":
		# Assuming script is in root/scripts, find the workspace root
		script_dir = Path(__file__).resolve().parent
		workspace_root = script_dir.parent
	else:
		workspace_root = Path(args.root).resolve()

	if not args.json:
		print(f"Scanning {workspace_root.as_posix()} for files with {args.threshold} or more lines...\n", flush=True)
	
	large_files = find_large_files(workspace_root, args.threshold, ignored_dirs, ignored_exts)
	
	# Sort files by line count in descending order
	large_files.sort(key=lambda x: x[1], reverse=True)

	# Prepare output content
	output_lines = []
	if args.json:
		results = [
			{"path": Path(f[0].relative_to(workspace_root)).as_posix() if workspace_root in f[0].parents else f[0].as_posix(), "lines": f[1]}
			for f in large_files
		]
		content = json.dumps(results, indent=2)
		output_lines.append(content)
	else:
		if not large_files:
			output_lines.append(f"No files found with >= {args.threshold} lines. Nice work!")
		else:
			output_lines.append(f"Found {len(large_files)} large files (>= {args.threshold} lines):")
			
			# Calculate column widths
			max_line_width = max(len(str(f[1])) for f in large_files)
			max_line_width = max(max_line_width, len("Lines"))
			
			header_lines = "Lines".rjust(max_line_width)
			header_path = "File Path"
			output_lines.append(f"{header_lines} | {header_path}")
			output_lines.append("-" * (max_line_width + 3 + 40))
			
			for file_path, line_count in large_files:
				try:
					rel_path = file_path.relative_to(workspace_root)
				except ValueError:
					rel_path = file_path
				output_lines.append(f"{str(line_count).rjust(max_line_width)} | {Path(rel_path).as_posix()}")
			
			output_lines.append("-" * (max_line_width + 3 + 40))

	# Output to terminal
	for line in output_lines:
		print(line, flush=True)

	# Output to file if requested
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
