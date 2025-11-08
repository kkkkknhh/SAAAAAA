#!/usr/bin/env python3
"""Analyze calibration gaps between catalog and calibration_registry."""

import json
import sys
from pathlib import Path

# Add src to path using portable path utilities
_script_dir = Path(__file__).parent
_proj_root = _script_dir.parent
sys.path.insert(0, str(_proj_root / 'src'))

from saaaaaa.core.orchestrator.calibration_registry import CALIBRATIONS

# Load catalog
catalog_path = _proj_root / 'config' / 'rules' / 'METODOS' / 'catalogo_completo_canonico.json'
catalog = json.load(open(catalog_path))

print("=" * 80)
print("CALIBRATION GAP ANALYSIS")
print("=" * 80)
print()

# Get all methods from catalog
catalog_methods = set()
if 'methods_catalog' in catalog:
    for method in catalog['methods_catalog']:
        class_name = method.get('class', 'Unknown')
        method_name = method.get('method_name', 'unknown')
        catalog_methods.add((class_name, method_name))

print(f"Methods in catalog: {len(catalog_methods)}")
print(f"Methods with calibration: {len(CALIBRATIONS)}")
print()

# Get calibrated method keys
calibrated_methods = set(CALIBRATIONS.keys())

# Find methods in catalog but not calibrated
not_calibrated = catalog_methods - calibrated_methods
# Find calibrated methods not in catalog
not_in_catalog = calibrated_methods - catalog_methods

print("=" * 80)
print("METHODS NEEDING CALIBRATION")
print("=" * 80)
print(f"\nTotal: {len(not_calibrated)} methods")
print()

if not_calibrated:
    # Group by class
    by_class = {}
    for class_name, method_name in sorted(not_calibrated):
        if class_name not in by_class:
            by_class[class_name] = []
        by_class[class_name].append(method_name)
    
    print("By class:")
    for class_name in sorted(by_class.keys())[:20]:
        methods = by_class[class_name]
        print(f"\n  {class_name} ({len(methods)} methods):")
        for method in methods[:5]:
            print(f"    - {method}")
        if len(methods) > 5:
            print(f"    ... and {len(methods) - 5} more")

print()
print("=" * 80)
print("CALIBRATED BUT NOT IN CATALOG")
print("=" * 80)
print(f"\nTotal: {len(not_in_catalog)} methods")
print()

if not_in_catalog:
    # Group by class
    by_class = {}
    for class_name, method_name in sorted(not_in_catalog):
        if class_name not in by_class:
            by_class[class_name] = []
        by_class[class_name].append(method_name)
    
    print("By class (first 10):")
    for class_name in sorted(by_class.keys())[:10]:
        methods = by_class[class_name]
        print(f"\n  {class_name} ({len(methods)} methods):")
        for method in methods[:3]:
            print(f"    - {method}")

print()
print("=" * 80)
print("CALIBRATION COVERAGE STATISTICS")
print("=" * 80)
print()

total_methods = len(catalog_methods)
calibrated_count = len(catalog_methods & calibrated_methods)
coverage = (calibrated_count / total_methods * 100) if total_methods > 0 else 0

print(f"Coverage: {calibrated_count}/{total_methods} ({coverage:.1f}%)")
print(f"Gap: {len(not_calibrated)} methods need calibration")
print()

# Identify priority classes that need calibration
print("=" * 80)
print("PRIORITY CLASSES FOR CALIBRATION")
print("=" * 80)
print()

priority_keywords = ['Executor', 'Aggregat', 'Scor', 'Bayesian', 'Analyzer', 
                     'Extractor', 'Processor', 'Engine']

priority_missing = []
for class_name, method_name in not_calibrated:
    if any(keyword in class_name for keyword in priority_keywords):
        priority_missing.append((class_name, method_name))

print(f"Priority methods needing calibration: {len(priority_missing)}")
print()

if priority_missing:
    by_class = {}
    for class_name, method_name in sorted(priority_missing):
        if class_name not in by_class:
            by_class[class_name] = []
        by_class[class_name].append(method_name)
    
    for class_name in sorted(by_class.keys())[:15]:
        methods = by_class[class_name]
        print(f"\n  {class_name} ({len(methods)} methods):")
        for method in methods[:5]:
            print(f"    - {method}")
        if len(methods) > 5:
            print(f"    ... and {len(methods) - 5} more")

print()
print("=" * 80)
