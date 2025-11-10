#!/usr/bin/env python3
"""
Populate intrinsic_calibration.json with all methods from canonical catalog.

Per canonic_calibration_methods.md requirement:
"Every method in canonical_method_catalog.json MUST be in exactly one of:
- intrinsic_calibration.json with a valid intrinsic profile, or
- intrinsic_calibration.json with 'calibration_status': 'excluded' and reason"

This script adds missing methods with initial calibration entries.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone


def load_json(path: Path) -> dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    """Save JSON file with formatting"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')  # Add trailing newline


def get_initial_calibration(method_info: dict) -> dict:
    """
    Generate initial calibration entry for a method.
    
    Strategy:
    - Utility/simple methods: mark as excluded (not in evaluation pipeline)
    - Core pipeline methods: provide conservative initial calibration
    """
    canonical_name = method_info.get('canonical_name', '')
    layer = method_info.get('layer', 'unknown')
    requires_calibration = method_info.get('requires_calibration', False)
    
    # Determine if method should be excluded
    # Exclude methods that are not in evaluation/scoring pipelines
    excluded_patterns = [
        '__init__',
        '__str__',
        '__repr__',
        '__eq__',
        '__hash__',
        '_get_',
        '_set_',
        '_load_',
        '_save_',
        'visit_',  # AST visitors
        'is_test_file',
        'scan_',
        'generate_report',
    ]
    
    # Check if method matches exclusion patterns
    method_name = method_info.get('method_name', '')
    is_utility = any(pattern in method_name for pattern in excluded_patterns)
    is_private_utility = method_name.startswith('_') and not method_name.startswith('__')
    
    # Utility layers are typically excluded
    is_utility_layer = layer in ['unknown', 'utility']
    
    if is_utility or (is_private_utility and is_utility_layer) or (is_utility_layer and not requires_calibration):
        return {
            "method_id": canonical_name,
            "calibration_status": "excluded",
            "reason": "Utility method not in evaluation/scoring pipeline",
            "layer": layer,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "approved_by": "automated_population"
        }
    
    # For methods requiring calibration, provide conservative initial scores
    # These should be reviewed and updated with actual evidence
    return {
        "method_id": canonical_name,
        "b_theory": 0.50,  # Conservative default - needs review
        "b_impl": 0.50,    # Conservative default - needs review
        "b_deploy": 0.50,  # Conservative default - needs review
        "evidence": {
            "theory_sources": ["Pending manual review"],
            "implementation_metrics": {
                "note": "Initial conservative scores - requires evidence gathering"
            },
            "deployment_history": {
                "note": "Pending production validation"
            }
        },
        "calibration_status": "initial",  # Mark as needing review
        "layer": layer,
        "requires_calibration": requires_calibration,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "approved_by": "automated_population_pending_review"
    }


def populate_calibrations():
    """Main function to populate intrinsic calibrations"""
    repo_root = Path(__file__).parent.parent
    catalog_path = repo_root / "config" / "canonical_method_catalog.json"
    intrinsic_path = repo_root / "config" / "intrinsic_calibration.json"
    
    print("Loading canonical catalog...")
    catalog = load_json(catalog_path)
    
    print("Loading current intrinsic calibrations...")
    intrinsic = load_json(intrinsic_path)
    
    # Get existing method IDs (excluding metadata and template)
    existing_methods = set()
    for method_id in intrinsic.get("methods", {}).keys():
        if not method_id.startswith("_"):
            existing_methods.add(method_id)
    
    print(f"Current calibrated methods: {len(existing_methods)}")
    
    # Get all methods from catalog
    all_catalog_methods = {}
    for layer_name, methods in catalog.get("layers", {}).items():
        for method_info in methods:
            canonical_name = method_info.get("canonical_name", "")
            if canonical_name:
                all_catalog_methods[canonical_name] = method_info
    
    print(f"Total methods in catalog: {len(all_catalog_methods)}")
    
    # Find missing methods
    missing_methods = set(all_catalog_methods.keys()) - existing_methods
    print(f"Missing methods to add: {len(missing_methods)}")
    
    if not missing_methods:
        print("✓ All methods already have calibration entries!")
        return 0
    
    # Add missing methods
    print("\nPopulating missing methods...")
    added_count = 0
    excluded_count = 0
    
    for method_id in sorted(missing_methods):
        method_info = all_catalog_methods[method_id]
        calibration_entry = get_initial_calibration(method_info)
        
        intrinsic["methods"][method_id] = calibration_entry
        
        if calibration_entry.get("calibration_status") == "excluded":
            excluded_count += 1
        else:
            added_count += 1
        
        if (added_count + excluded_count) % 100 == 0:
            print(f"  Processed {added_count + excluded_count}/{len(missing_methods)} methods...")
    
    # Update metadata
    intrinsic["_metadata"]["last_populated"] = datetime.now(timezone.utc).isoformat()
    intrinsic["_metadata"]["population_summary"] = {
        "total_methods": len(all_catalog_methods),
        "manually_calibrated": len(existing_methods),
        "auto_populated_initial": added_count,
        "auto_excluded": excluded_count,
        "note": "Auto-populated methods marked 'initial' require manual review and evidence"
    }
    
    # Save updated file
    print(f"\nSaving updated intrinsic_calibration.json...")
    save_json(intrinsic_path, intrinsic)
    
    print("\n" + "=" * 80)
    print("POPULATION COMPLETE")
    print("=" * 80)
    print(f"Total methods in catalog: {len(all_catalog_methods)}")
    print(f"Previously calibrated: {len(existing_methods)}")
    print(f"Added with initial calibration: {added_count}")
    print(f"Added as excluded: {excluded_count}")
    print(f"Total now in intrinsic_calibration.json: {len(intrinsic['methods']) - 1}")  # -1 for template
    print("\n⚠️  IMPORTANT:")
    print("   Methods marked 'calibration_status': 'initial' have conservative scores")
    print("   and require manual review with proper evidence gathering.")
    print("   Methods marked 'calibration_status': 'excluded' should be reviewed")
    print("   to confirm they are truly outside evaluation pipelines.")
    
    return 0


if __name__ == "__main__":
    sys.exit(populate_calibrations())
