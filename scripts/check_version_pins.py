#!/usr/bin/env python3
"""
Check if all dependencies have exact version pins (no open ranges).

This script verifies that requirement files don't contain open ranges like >=, ~=, or *.
Used in CI to enforce reproducible builds.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def check_file_for_open_ranges(filepath: Path) -> Tuple[bool, List[str]]:
    """
    Check a requirements file for open version ranges.
    
    Returns:
        Tuple of (has_open_ranges, list_of_violations)
    """
    if not filepath.exists():
        return False, []
    
    violations = []
    # Match version specifiers more precisely: package_name OPERATOR version
    # This avoids false positives from package names or comments
    version_specifier_pattern = re.compile(r'^[a-zA-Z0-9_-]+\s*(>=|~=|<=|<|>|\*)')
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            original_line = line
            line = line.strip()
            
            # Skip empty lines, comments, and -r includes
            if not line or line.startswith('#') or line.startswith('-r '):
                continue
            
            # Check for open ranges using more precise pattern
            if version_specifier_pattern.match(line):
                violations.append(f"Line {line_num}: {original_line.strip()}")
    
    return len(violations) > 0, violations


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: check_version_pins.py <requirement-file> [<requirement-file> ...]")
        return 1
    
    print("="*80)
    print("VERSION PIN VALIDATOR")
    print("="*80)
    print("Checking for open version ranges (>=, ~=, *, <, >)")
    print()
    
    total_violations = 0
    files_with_violations = []
    
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        print(f"Checking {filepath}...")
        
        has_violations, violations = check_file_for_open_ranges(filepath)
        
        if has_violations:
            total_violations += len(violations)
            files_with_violations.append(filepath)
            print(f"  ❌ Found {len(violations)} violation(s):")
            for violation in violations:
                print(f"    {violation}")
        else:
            print(f"  ✅ All versions exactly pinned")
        print()
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files checked: {len(sys.argv) - 1}")
    print(f"Files with violations: {len(files_with_violations)}")
    print(f"Total violations: {total_violations}")
    
    if total_violations > 0:
        print("\n❌ FAILED: Open version ranges detected!")
        print("\nFor reproducible builds, all versions must be exactly pinned with ==")
        print("Example: numpy==2.2.1 (not numpy>=2.0.0)")
        return 1
    else:
        print("\n✅ SUCCESS: All versions exactly pinned")
        return 0


if __name__ == "__main__":
    sys.exit(main())
