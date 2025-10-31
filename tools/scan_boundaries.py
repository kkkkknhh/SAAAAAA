#!/usr/bin/env python3
"""
AST Scanner for Core Module Boundary Violations

Scans Python modules for:
1. I/O operations (open, json.load/dump, pickle, pandas read_*, etc.)
2. __main__ blocks
3. Side effects on import

Usage:
    python tools/scan_boundaries.py core/

Exit code 0 if clean, 1 if violations found.
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple


class BoundaryViolationVisitor(ast.NodeVisitor):
    """AST visitor to detect boundary violations in core modules."""

    # I/O function names to detect
    IO_FUNCTIONS = {
        'open', 'read', 'write',
        'load', 'dump', 'loads', 'dumps',
        'read_csv', 'read_excel', 'read_json', 'read_sql', 'read_parquet',
        'to_csv', 'to_excel', 'to_json', 'to_sql', 'to_parquet',
    }

    # Module names that indicate I/O
    IO_MODULES = {
        'pickle', 'json', 'yaml', 'toml',
    }

    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[Dict[str, any]] = []
        self.has_main_block = False
        self.io_calls: List[Tuple[int, str]] = []

    def visit_If(self, node: ast.If) -> None:
        """Detect if __name__ == '__main__' blocks."""
        # Check for __name__ == '__main__' pattern
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.left, ast.Name):
                if node.test.left.id == '__name__':
                    for comparator in node.test.comparators:
                        if isinstance(comparator, ast.Constant):
                            if comparator.value == '__main__':
                                self.has_main_block = True
                                self.violations.append({
                                    'type': 'main_block',
                                    'line': node.lineno,
                                    'message': f'__main__ block found at line {node.lineno}'
                                })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect I/O function calls."""
        # Direct function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in self.IO_FUNCTIONS:
                self.io_calls.append((node.lineno, node.func.id))
                self.violations.append({
                    'type': 'io_call',
                    'line': node.lineno,
                    'function': node.func.id,
                    'message': f'I/O operation {node.func.id}() at line {node.lineno}'
                })

        # Module.function calls (e.g., json.load)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                func_name = node.func.attr
                if module_name in self.IO_MODULES or func_name in self.IO_FUNCTIONS:
                    self.io_calls.append((node.lineno, f'{module_name}.{func_name}'))
                    self.violations.append({
                        'type': 'io_call',
                        'line': node.lineno,
                        'function': f'{module_name}.{func_name}',
                        'message': f'I/O operation {module_name}.{func_name}() at line {node.lineno}'
                    })

        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Detect 'with open(...)' patterns."""
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                if isinstance(item.context_expr.func, ast.Name):
                    if item.context_expr.func.id == 'open':
                        self.io_calls.append((node.lineno, 'open (with)'))
                        self.violations.append({
                            'type': 'io_call',
                            'line': node.lineno,
                            'function': 'open',
                            'message': f'I/O operation: with open(...) at line {node.lineno}'
                        })
        self.generic_visit(node)


def scan_file(filepath: Path) -> Dict[str, any]:
    """Scan a single Python file for boundary violations."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
    except Exception as e:
        return {
            'file': str(filepath),
            'error': f'Could not read file: {e}',
            'violations': [],
            'clean': False
        }

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return {
            'file': str(filepath),
            'error': f'Syntax error: {e}',
            'violations': [],
            'clean': False
        }

    visitor = BoundaryViolationVisitor(str(filepath))
    visitor.visit(tree)

    return {
        'file': str(filepath),
        'violations': visitor.violations,
        'has_main_block': visitor.has_main_block,
        'io_call_count': len(visitor.io_calls),
        'clean': len(visitor.violations) == 0,
        'error': None
    }


def scan_directory(directory: Path, pattern: str = '*.py') -> List[Dict[str, any]]:
    """Scan all Python files in a directory."""
    results = []
    for filepath in sorted(directory.rglob(pattern)):
        # Skip __pycache__ and test files
        if '__pycache__' in str(filepath) or 'test_' in filepath.name:
            continue
        results.append(scan_file(filepath))
    return results


def print_report(results: List[Dict[str, any]]) -> int:
    """Print scan results and return exit code."""
    total_files = len(results)
    clean_files = sum(1 for r in results if r['clean'])
    total_violations = sum(len(r['violations']) for r in results)

    print("=" * 80)
    print("CORE MODULE BOUNDARY SCAN REPORT")
    print("=" * 80)
    print(f"\nFiles scanned: {total_files}")
    print(f"Clean files: {clean_files}")
    print(f"Files with violations: {total_files - clean_files}")
    print(f"Total violations: {total_violations}")
    print()

    if total_violations == 0:
        print("✅ All files are clean! No boundary violations detected.")
        return 0

    print("❌ Violations found:\n")

    for result in results:
        if not result['clean']:
            print(f"\n{result['file']}")
            if result.get('error'):
                print(f"  ERROR: {result['error']}")
            else:
                for violation in result['violations']:
                    print(f"  Line {violation['line']}: {violation['message']}")

    print("\n" + "=" * 80)
    print("REMEDIATION:")
    print("- Move all __main__ blocks to examples/ directory")
    print("- Move all I/O operations to orchestrator/factory.py")
    print("- Core modules should be pure libraries receiving data via contracts")
    print("=" * 80)

    return 1


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python tools/scan_boundaries.py <directory>")
        print("Example: python tools/scan_boundaries.py core/")
        return 1

    target_path = Path(sys.argv[1])
    if not target_path.exists():
        print(f"Error: Directory {target_path} does not exist")
        return 1

    results = scan_directory(target_path)
    return print_report(results)


if __name__ == '__main__':
    sys.exit(main())
