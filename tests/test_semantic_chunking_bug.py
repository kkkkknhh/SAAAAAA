"""
Test suite for semantic_chunking_policy module.

Ensures that the syntax error fix (duplicate lines 555-562) stays fixed
and tests basic functionality.
"""

import ast
from pathlib import Path

import pytest


def test_semantic_chunking_syntax():
    """Test that semantic_chunking_policy.py has valid syntax."""
    module_path = Path(__file__).parent.parent / "semantic_chunking_policy.py"
    
    if not module_path.exists():
        pytest.skip("semantic_chunking_policy.py not found")
    
    with open(module_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # This will raise SyntaxError if the file has syntax errors
    try:
        ast.parse(source, filename=str(module_path))
    except SyntaxError as e:
        pytest.fail(
            f"Syntax error in semantic_chunking_policy.py at line {e.lineno}: {e.msg}\n"
            f"This suggests the duplicate lines bug (555-562) may have been reintroduced."
        )


def test_no_duplicate_return_statements():
    """Test that there are no duplicate return statements in _extract_key_excerpts.
    
    The original bug had duplicate lines 555-562:
    - Duplicate list comprehension closing
    - Duplicate return statement
    
    This test ensures that pattern doesn't reoccur.
    """
    module_path = Path(__file__).parent.parent / "semantic_chunking_policy.py"
    
    if not module_path.exists():
        pytest.skip("semantic_chunking_policy.py not found")
    
    with open(module_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source, filename=str(module_path))
    except SyntaxError:
        pytest.fail("File has syntax errors")
    
    # Find the _extract_key_excerpts method
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == '_extract_key_excerpts':
                # Count return statements in this function
                returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
                
                # Should have exactly 1 return statement
                assert len(returns) == 1, (
                    f"_extract_key_excerpts should have exactly 1 return statement, "
                    f"found {len(returns)}. This may indicate the duplicate return bug."
                )
                
                # Check that the return statement is a dict
                ret_value = returns[0].value
                assert isinstance(ret_value, ast.Name), (
                    f"Return value should be a simple name (excerpts variable), "
                    f"got {type(ret_value).__name__}"
                )
                break
    else:
        pytest.skip("_extract_key_excerpts method not found")


def test_extract_key_excerpts_method_structure():
    """Test the structure of _extract_key_excerpts to catch similar bugs."""
    module_path = Path(__file__).parent.parent / "semantic_chunking_policy.py"
    
    if not module_path.exists():
        pytest.skip("semantic_chunking_policy.py not found")
    
    with open(module_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Look for the method definition
    method_start = None
    for i, line in enumerate(lines):
        if 'def _extract_key_excerpts(' in line:
            method_start = i
            break
    
    if method_start is None:
        pytest.skip("_extract_key_excerpts method not found")
    
    # Find the method end (next def or class at same/lower indentation)
    method_end = len(lines)
    base_indent = len(lines[method_start]) - len(lines[method_start].lstrip())
    
    for i in range(method_start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.strip().startswith('#'):
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent and (line.strip().startswith('def ') or 
                                         line.strip().startswith('class ')):
                method_end = i
                break
    
    method_lines = lines[method_start:method_end]
    
    # Check for duplicate return statements
    return_count = sum(1 for line in method_lines if 'return excerpts' in line)
    assert return_count == 1, (
        f"Found {return_count} 'return excerpts' statements in _extract_key_excerpts. "
        f"Expected exactly 1. This may indicate the duplicate lines bug."
    )
    
    # Check for duplicate list comprehension closings
    list_comp_closing_count = sum(
        1 for line in method_lines 
        if line.strip().startswith('for c in top_chunks')
    )
    assert list_comp_closing_count == 1, (
        f"Found {list_comp_closing_count} 'for c in top_chunks' lines. "
        f"Expected exactly 1. This may indicate duplicate list comprehension."
    )


def test_no_main_block():
    """Test that semantic_chunking_policy.py has no __main__ block."""
    module_path = Path(__file__).parent.parent / "semantic_chunking_policy.py"
    
    if not module_path.exists():
        pytest.skip("semantic_chunking_policy.py not found")
    
    with open(module_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Simple check for __main__ block
    assert 'if __name__ == "__main__"' not in source, (
        "semantic_chunking_policy.py should not have a __main__ block"
    )
    assert "if __name__ == '__main__'" not in source, (
        "semantic_chunking_policy.py should not have a __main__ block"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
