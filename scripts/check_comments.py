"""
This script scans the codebase for files with insufficient comments, missing docstrings, or TODOs.
It calculates a comment-to-code ratio and identifies specific lines where Python docstrings are missing.
The results can be output to the console or saved to a JSON/text report in the logs directory.
"""
import os
import argparse
import json
import ast
from pathlib import Path


# Default Configurations
DEFAULT_MIN_RATIO = 0.10

# Typical ignores
DEFAULT_IGNORED_DIRS = {
	'.git', '.venv', 'venv', 'node_modules', '__pycache__', 
	'outputs', 'logs', 'dist', 'build', '.mypy_cache', '.pytest_cache', 
	'.gemini', '.idea', '.vscode'
}

DEFAULT_IGNORED_EXTS = {
	'.pyc', '.png', '.jpg', '.jpeg', '.gif', '.mp3', '.mp4', '.wav', 
	'.css', '.html', '.svg', '.json', '.lock', '.pdf', '.docx', '.csv', '.bin',
	'.md', '.txt'
}

COMMENT_PREFIXES = {
	'.py': ('#',),
	'.js': ('//', '/*', '*', '*/'),
	'.ts': ('//', '/*', '*', '*/'),
}

def analyze_comments(file_path: Path) -> dict:
	"""
	Analyzes a single file for comments, docstrings, and TODOs.

	Args:
		file_path: Path to the file to analyze.

	Returns:
		A dictionary containing statistics: total_lines, comment_lines, ratio, 
		todo_count, func_count, and missing_docs_lines.
	"""

	total_lines = 0
	comment_lines = 0
	todo_count = 0
	func_count = 0
	missing_docs_lines = []

	ext = file_path.suffix.lower()
	prefixes = COMMENT_PREFIXES.get(ext, ('#', '//', '/*', '*'))

	try:
		with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
			source = f.read()

		lines = source.splitlines()
		total_lines = len(lines)

		if total_lines == 0:
			return None

		in_block_comment = False
		in_py_docstring = False

		for line in lines:
			stripped = line.strip()
			if not stripped:
				continue

			if any(stripped.startswith(p) for p in prefixes) and "TODO" in line:
				todo_count += 1


			if ext in ['.js', '.ts']:
				if stripped.startswith('/*') and not stripped.endswith('*/'):
					in_block_comment = True
					comment_lines += 1
					continue
				elif in_block_comment:
					comment_lines += 1
					if '*/' in stripped:
						in_block_comment = False
					continue
			
			elif ext == '.py':
				if (stripped.startswith('"""') and stripped.count('"""') == 1) or \
				   (stripped.startswith("'''") and stripped.count("'''") == 1):
					in_py_docstring = not in_py_docstring
					comment_lines += 1
					continue
				elif in_py_docstring:
					comment_lines += 1
					continue

			if any(stripped.startswith(p) for p in prefixes):
				comment_lines += 1
			elif ext == '.py' and ((stripped.startswith('"""') and stripped.endswith('"""')) or (stripped.startswith("'''") and stripped.endswith("'''"))):
				comment_lines += 1
			# Special handling for inline comments like `x = 1 # comment`
			elif ext == '.py' and '#' in line and not stripped.startswith('#'):
				# Adding partial credit for lines to be generous
				comment_lines += 1
			elif ext in ['.js', '.ts'] and '//' in line and not stripped.startswith('//'):
				comment_lines += 1

		if ext == '.py':
			try:
				tree = ast.parse(source)
				
				# Check for file/module-level docstring
				if not ast.get_docstring(tree):
					missing_docs_lines.append(1)
					
				for node in ast.walk(tree):
					if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
						func_count += 1
						if not ast.get_docstring(node):
							missing_docs_lines.append(node.lineno)
			except SyntaxError:
				pass

		return {
			"total_lines": total_lines,
			"comment_lines": comment_lines,
			"ratio": float(comment_lines) / total_lines if total_lines > 0 else 1.0,
			"todo_count": todo_count,
			"func_count": func_count if ext == '.py' else None,
			"missing_docs_lines": missing_docs_lines if ext == '.py' else None
		}

	except Exception as e:
		print(f"Warning: Failed to read {file_path}: {e}")
		return None

def is_ignored(path: Path, ignored_dirs: set, ignored_exts: set) -> bool:
	"""
	Checks if a given path should be ignored based on its directory or extension.

	Args:
		path: The file or directory path to check.
		ignored_dirs: A set of directory names to ignore.
		ignored_exts: A set of file extensions to ignore.

	Returns:
		True if the path should be ignored, False otherwise.
	"""

	if path.suffix.lower() in ignored_exts:
		return True
	
	for part in path.parts:
		if part in ignored_dirs:
			return True
			
	return False

def find_insufficient_comments(root_dir: Path, min_ratio: float, ignored_dirs: set, ignored_exts: set) -> list:
	"""
	Recursively scans a directory for files that do not meet comment standards.

	Args:
		root_dir: The root directory to start the scan.
		min_ratio: The minimum acceptable comment-to-line ratio.
		ignored_dirs: Directories to skip during the scan.
		ignored_exts: File extensions to skip.

	Returns:
		A list of tuples (file_path, stats, flag_reasons) for each flagged file.
	"""

	flagged_files = []

	for root, dirs, files in os.walk(root_dir):
		current_dir = Path(root)

		# Modify dirs in-place to skip ignored directories
		dirs[:] = [d for d in dirs if d not in ignored_dirs]

		if is_ignored(current_dir, ignored_dirs, ignored_exts):
			continue

		for file in files:
			file_path = current_dir / file

			if is_ignored(file_path, ignored_dirs, ignored_exts):
				continue

			stats = analyze_comments(file_path)
			if stats:
				flag_reasons = []
				
				if stats["ratio"] < min_ratio:
					flag_reasons.append(f"Low Ratio: {stats['ratio']:.1%}")
				
				if stats["missing_docs_lines"]:
					flag_reasons.append(f"Missing Py Docs ({len(stats['missing_docs_lines'])})")
				
				if stats["todo_count"] > 0:
					flag_reasons.append(f"TODOs ({stats['todo_count']})")

				# Always flag if it misses docstrings or is under min ratio. Also flag if has TODOs
				if stats["ratio"] < min_ratio or stats["missing_docs_lines"] or stats["todo_count"] > 0:
					flagged_files.append((file_path, stats, flag_reasons))

	return flagged_files


def main():
	"""Main execution point for scanning the project for comment quality."""

	parser = argparse.ArgumentParser(description="Scan for files with insufficient comments, missing docstrings, or TODOs.")
	parser.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_RATIO, help=f"Minimum allowed comment ratio (default: {DEFAULT_MIN_RATIO})")
	parser.add_argument("--root", type=str, default=".", help="Root directory to scan (default: workspace root)")
	parser.add_argument("--json", action="store_true", help="Output results in JSON format")
	parser.add_argument("--output", "-o", type=str, help="Save output to a file in the 'logs' folder")
	parser.add_argument("--exclude-dir", action="append", help="Additional directories to exclude")
	parser.add_argument("--exclude-ext", action="append", help="Additional extensions to exclude (with dot, e.g., .md)")
	
	args = parser.parse_args()

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
		print(f"Scanning {workspace_root.as_posix()} for comments...\n", flush=True)
	
	flagged_files = find_insufficient_comments(workspace_root, args.min_ratio, ignored_dirs, ignored_exts)

	# Sort files by comment ratio ascending (worst first)
	flagged_files.sort(key=lambda x: x[1]['ratio'])

	def get_rel_path(p):
		"""Returns the relative path from the workspace root for display."""

		try:
			return Path(p).relative_to(workspace_root).as_posix()
		except ValueError:
			return Path(p).as_posix()

	output_lines = []
	if args.json or (args.output and args.output.endswith('.json')):
		results = [
			{
				"file": get_rel_path(f[0]),
				"ratio": round(f[1]['ratio'], 3),
				"flags": f[2],
				"missing_docs_lines": f[1]['missing_docs_lines']
			}
			for f in flagged_files
		]
		content = json.dumps(results, indent=2)
		output_lines.append(content)
	else:
		if not flagged_files:
			output_lines.append(f"No files flagged! Codebase comment check passed.")
		else:
			output_lines.append(f"Found {len(flagged_files)} files needing review:")

			ratio_col = max((len(f"{stats['ratio']:.1%}") for _, stats, _ in flagged_files), default=7)
			ratio_col = max(ratio_col, len("Ratio"))

			reason_col = max((len(", ".join(reasons)) for _, _, reasons in flagged_files), default=10)
			reason_col = max(reason_col, len("Flags"))

			header = f"{'Ratio'.rjust(ratio_col)} | {'Flags'.ljust(reason_col)} | File Path"
			output_lines.append(header)
			output_lines.append("-" * (ratio_col + reason_col + 3 + 3 + 40))

			for file_path, stats, reasons in flagged_files:
				rel_path = get_rel_path(file_path)
				output_lines.append(
					f"{f'{stats['ratio']:.1%}'.rjust(ratio_col)} | {', '.join(reasons).ljust(reason_col)} | {rel_path}"
				)

			output_lines.append("-" * (ratio_col + reason_col + 3 + 3 + 40))

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
