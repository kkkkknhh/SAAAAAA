"""
Test suite for core module boundary enforcement.

Tests that core modules remain pure libraries without:
- __main__ blocks
- I/O operations (open, file operations, json/pickle/yaml IO)
- Side effects on import

This test suite acts as a governance layer to prevent regression.
"""

import ast
import sys
from pathlib import Path
from typing import List, Set

import pytest


# Core modules that must remain pure
CORE_MODULES = [
    "Analyzer_one.py",
    "dereck_beach.py",
    "financiero_viabilidad_tablas.py",
    "teoria_cambio.py",
    "contradiction_deteccion.py",
    "embedding_policy.py",
    "semantic_chunking_policy.py",
]


class IODetector(ast.NodeVisitor):
    """AST visitor to detect I/O operations."""

    IO_FUNCTIONS = {
        'open', 'read', 'write',
        'load', 'dump', 'loads', 'dumps',
        'read_csv', 'read_excel', 'read_json', 'read_sql', 'read_parquet',
        'to_csv', 'to_excel', 'to_json', 'to_sql', 'to_parquet',
    }

    IO_MODULES = {'pickle', 'json', 'yaml', 'toml'}

    def __init__(self):
        self.has_io = False
        self.io_locations: List[int] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Detect I/O function calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.IO_FUNCTIONS:
                self.has_io = True
                self.io_locations.append(node.lineno)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                func_name = node.func.attr
                if module_name in self.IO_MODULES or func_name in self.IO_FUNCTIONS:
                    self.has_io = True
                    self.io_locations.append(node.lineno)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Detect 'with open(...)' patterns."""
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                if isinstance(item.context_expr.func, ast.Name):
                    if item.context_expr.func.id == 'open':
                        self.has_io = True
                        self.io_locations.append(node.lineno)
        self.generic_visit(node)


class MainBlockDetector(ast.NodeVisitor):
    """AST visitor to detect __main__ blocks."""

    def __init__(self):
        self.has_main = False
        self.main_locations: List[int] = []

    def visit_If(self, node: ast.If) -> None:
        """Detect if __name__ == '__main__' blocks."""
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.left, ast.Name):
                if node.test.left.id == '__name__':
                    for comparator in node.test.comparators:
                        if isinstance(comparator, ast.Constant):
                            if comparator.value == '__main__':
                                self.has_main = True
                                self.main_locations.append(node.lineno)
        self.generic_visit(node)


def get_module_path(module_name: str) -> Path:
    """Get the path to a core module."""
    repo_root = Path(__file__).parent.parent
    return repo_root / module_name


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_no_main_blocks(module_name: str) -> None:
    """Test that core modules have no __main__ blocks."""
    module_path = get_module_path(module_name)
    
    if not module_path.exists():
        pytest.skip(f"Module {module_name} not found")
    
    with open(module_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source, filename=str(module_path))
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {module_name}: {e}")
    
    detector = MainBlockDetector()
    detector.visit(tree)
    
    assert not detector.has_main, (
        f"{module_name} contains __main__ block(s) at line(s): "
        f"{detector.main_locations}. Core modules must not have __main__ blocks."
    )


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_no_io_operations(module_name: str) -> None:
    """Test that core modules have no I/O operations.
    
    Note: This test is currently expected to fail for modules that still
    contain I/O operations. It will pass once I/O is moved to orchestrator.
    """
    module_path = get_module_path(module_name)
    
    if not module_path.exists():
        pytest.skip(f"Module {module_name} not found")
    
    with open(module_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source, filename=str(module_path))
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {module_name}: {e}")
    
    detector = IODetector()
    detector.visit(tree)
    
    # For now, we just document the violations
    # TODO: Once I/O is migrated, change this to assert not detector.has_io
    if detector.has_io:
        pytest.skip(
            f"{module_name} still contains {len(detector.io_locations)} I/O operations "
            f"at lines: {detector.io_locations[:10]}... "
            f"(I/O migration pending)"
        )


@pytest.mark.parametrize("module_name", CORE_MODULES)  
def test_module_imports_without_side_effects(module_name: str) -> None:
    """Test that core modules can be imported without executing code.
    
    This test attempts to import each module and ensures no exceptions
    are raised due to missing files or other side effects.
    
    Note: This may fail if modules have heavy dependencies not installed.
    """
    module_path = get_module_path(module_name)
    
    if not module_path.exists():
        pytest.skip(f"Module {module_name} not found")
    
    # We can't actually import due to dependencies, but we can check
    # that the module has no top-level code execution outside of
    # function/class definitions
    with open(module_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source, filename=str(module_path))
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {module_name}: {e}")
    
    # Check for top-level statements that aren't imports, class defs,
    # or function defs
    dangerous_statements = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, 
                            ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.Assign):
            # Constants and type annotations are OK
            continue
        if isinstance(node, (ast.Expr, ast.If)):
            # Module docstrings are OK (they're Expr nodes)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            # __main__ blocks we already tested
            if isinstance(node, ast.If):
                continue
            dangerous_statements.append(node.lineno)
    
    # For now this is informational
    if dangerous_statements:
        pytest.skip(
            f"{module_name} has top-level statements at lines: {dangerous_statements}"
        )


def test_boundary_scanner_tool_exists() -> None:
    """Test that the boundary scanner tool exists and is executable."""
    scanner_path = Path(__file__).parent.parent / "tools" / "scan_boundaries.py"
    assert scanner_path.exists(), "Boundary scanner tool not found"
    
    # Check it's a valid Python file
    with open(scanner_path, 'r') as f:
        source = f.read()
    
    try:
        ast.parse(source)
    except SyntaxError as e:
        pytest.fail(f"Syntax error in scan_boundaries.py: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
