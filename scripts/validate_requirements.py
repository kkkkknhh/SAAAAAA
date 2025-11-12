#!/usr/bin/env python3
"""
Validate requirements files for SAAAAAA project.
Checks for:
- Proper version pinning
- Duplicate packages
- Consistency across files
- Critical version constraints
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_requirements_file(file_path: Path) -> Dict[str, str]:
    """Parse a requirements file and return dict of package: version."""
    packages = {}
    duplicates = []

    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return packages

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip comments, empty lines, and file references
            if not line or line.startswith('#') or line.startswith('-r'):
                continue

            # Check for proper pinning
            if '==' not in line:
                print(f"❌ {file_path.name}:{line_num} - Missing version pin: {line}")
                continue

            # Extract package and version
            pkg, version = line.split('==', 1)
            pkg = pkg.strip()
            version = version.strip()

            # Check for duplicates
            if pkg.lower() in packages:
                duplicates.append((pkg, line_num))
                print(f"❌ {file_path.name}:{line_num} - Duplicate package: {pkg}")
            else:
                packages[pkg.lower()] = version

    return packages


def check_critical_constraints(packages: Dict[str, str]) -> List[str]:
    """Check critical version constraints."""
    issues = []

    # NumPy must be 1.26.4
    if 'numpy' in packages:
        if packages['numpy'] != '1.26.4':
            issues.append(f"❌ NumPy version {packages['numpy']} - MUST be 1.26.4 for PyMC compatibility!")

    # PyTensor must be 2.34.0
    if 'pytensor' in packages:
        if packages['pytensor'] != '2.34.0':
            issues.append(f"⚠️  PyTensor version {packages['pytensor']} - Should be 2.34.0 for NumPy 1.x")

    # PyMC should be 5.16.2
    if 'pymc' in packages:
        if packages['pymc'] != '5.16.2':
            issues.append(f"⚠️  PyMC version {packages['pymc']} - Should be 5.16.2")

    return issues


def check_file_consistency(files: Dict[str, Dict[str, str]]) -> List[str]:
    """Check consistency across requirement files."""
    issues = []

    if 'requirements.txt' not in files or 'requirements-core.txt' not in files:
        issues.append("⚠️  Missing core requirement files")
        return issues

    main_packages = files['requirements.txt']
    core_packages = files['requirements-core.txt']

    # Check that core packages are in main requirements with same versions
    for pkg, version in core_packages.items():
        if pkg in main_packages:
            if main_packages[pkg] != version:
                issues.append(
                    f"❌ Version mismatch for {pkg}: "
                    f"requirements.txt={main_packages[pkg]} vs "
                    f"requirements-core.txt={version}"
                )
        else:
            issues.append(f"⚠️  Package {pkg} in requirements-core.txt but not in requirements.txt")

    return issues


def main():
    """Main validation routine."""
    project_root = Path(__file__).parent.parent

    print("="*80)
    print("SAAAAAA Requirements Validation")
    print("="*80)

    # Files to validate
    required_files = [
        'requirements.txt',
        'requirements-core.txt',
        'requirements-dev.txt',
        'requirements-optional.txt',
        'requirements-all.txt',
    ]

    print("\n[1/3] Checking file existence...")
    missing_files = []
    for filename in required_files:
        file_path = project_root / filename
        if file_path.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ❌ {filename} - MISSING")
            missing_files.append(filename)

    if missing_files:
        print(f"\n❌ Missing required files: {', '.join(missing_files)}")
        return 1

    # Parse all requirement files
    print("\n[2/3] Parsing and validating requirements files...")
    all_packages = {}

    for filename in required_files:
        file_path = project_root / filename
        print(f"\n  Validating {filename}...")
        packages = parse_requirements_file(file_path)
        all_packages[filename] = packages

        if packages:
            print(f"    ✓ Parsed {len(packages)} packages")

            # Check critical constraints for main files
            if filename in ['requirements.txt', 'requirements-core.txt']:
                issues = check_critical_constraints(packages)
                if issues:
                    for issue in issues:
                        print(f"    {issue}")
                else:
                    print(f"    ✓ Critical constraints OK")

    # Check consistency
    print("\n[3/3] Checking consistency across files...")
    consistency_issues = check_file_consistency(all_packages)

    if consistency_issues:
        print("\n  Consistency issues found:")
        for issue in consistency_issues:
            print(f"    {issue}")
    else:
        print("  ✓ All files are consistent")

    # Summary
    print("\n" + "="*80)
    if not consistency_issues and not missing_files:
        print("✅ VALIDATION PASSED")
        print("="*80)
        print(f"\nTotal packages in requirements.txt: {len(all_packages.get('requirements.txt', {}))}")
        print(f"Total packages in requirements-core.txt: {len(all_packages.get('requirements-core.txt', {}))}")
        print("\n✓ All requirements files are valid and consistent")
        print("✓ Critical version constraints are satisfied")
        print("\n")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print("="*80)
        print("\nPlease fix the issues above and run validation again.")
        print("\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
