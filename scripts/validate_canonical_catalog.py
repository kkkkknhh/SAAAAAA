#!/usr/bin/env python3
"""
Validate Canonical Method Catalog

This script validates the structure, integrity, and completeness of the
canonical method catalog (config/canonical_method_catalog.json).

Validations performed:
1. JSON structure is valid
2. All required metadata fields present
3. All methods have required fields
4. Layer taxonomy is complete and correct
5. Calibration references are valid
6. Statistics match actual data
7. No duplicate method entries

Exit codes:
    0 - Catalog is valid
    1 - Catalog has validation errors
    2 - Critical error (file not found, JSON parse error, etc.)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter

# Repository root
REPO_ROOT = Path(__file__).parent.parent.absolute()
CANONICAL_CATALOG = REPO_ROOT / "config" / "canonical_method_catalog.json"
CALIBRATION_REGISTRY = REPO_ROOT / "src" / "saaaaaa" / "core" / "orchestrator" / "calibration_registry.py"


class CatalogValidator:
    """Validates canonical method catalog structure and integrity."""

    def __init__(self, catalog_path: Path):
        self.catalog_path = catalog_path
        self.errors = []
        self.warnings = []
        self.catalog = None

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("=" * 80)
        print("CANONICAL METHOD CATALOG VALIDATION")
        print("=" * 80)
        print(f"Catalog: {self.catalog_path}")
        print()

        # Load catalog
        if not self.load_catalog():
            return False

        # Run validations
        self.validate_metadata()
        self.validate_layer_taxonomy()
        self.validate_methods()
        self.validate_statistics()
        self.validate_calibration_references()
        self.check_duplicates()

        # Print results
        self.print_results()

        return len(self.errors) == 0

    def load_catalog(self) -> bool:
        """Load and parse catalog JSON."""
        if not self.catalog_path.exists():
            self.errors.append(f"Catalog file not found: {self.catalog_path}")
            print(f"✗ Catalog file not found: {self.catalog_path}")
            return False

        try:
            with open(self.catalog_path) as f:
                self.catalog = json.load(f)
            print(f"✓ Catalog loaded successfully")
            return True
        except json.JSONDecodeError as e:
            self.errors.append(f"JSON parse error: {e}")
            print(f"✗ JSON parse error: {e}")
            return False

    def validate_metadata(self):
        """Validate metadata section."""
        print("\n--- Validating Metadata ---")
        
        required_fields = [
            "version",
            "generated_at",
            "purpose",
            "total_methods",
            "layer_taxonomy",
            "migration_status"
        ]

        metadata = self.catalog.get("metadata", {})
        
        for field in required_fields:
            if field not in metadata:
                self.errors.append(f"Missing required metadata field: {field}")
                print(f"✗ Missing metadata field: {field}")
            else:
                print(f"✓ Metadata field present: {field}")

        # Validate layer_taxonomy structure
        if "layer_taxonomy" in metadata:
            taxonomy = metadata["layer_taxonomy"]
            required_layers = {"Q", "D", "P", "C", "M"}
            found_layers = set(taxonomy.keys())
            
            if required_layers == found_layers:
                print(f"✓ Layer taxonomy complete (5 layers)")
            else:
                missing = required_layers - found_layers
                extra = found_layers - required_layers
                if missing:
                    self.errors.append(f"Missing layers in taxonomy: {missing}")
                    print(f"✗ Missing layers: {missing}")
                if extra:
                    self.warnings.append(f"Extra layers in taxonomy: {extra}")
                    print(f"⚠ Extra layers: {extra}")

    def validate_layer_taxonomy(self):
        """Validate that all methods have valid layer assignments."""
        print("\n--- Validating Layer Assignments ---")
        
        valid_layers = {"Q", "D", "P", "C", "M"}
        methods = self.catalog.get("methods", {})
        
        invalid_layers = []
        for method_name, method_data in methods.items():
            layer = method_data.get("layer")
            if not layer:
                invalid_layers.append((method_name, "missing"))
            elif layer not in valid_layers:
                invalid_layers.append((method_name, layer))

        if invalid_layers:
            for method, layer in invalid_layers[:5]:  # Show first 5
                self.errors.append(f"Invalid layer '{layer}' for method: {method}")
                print(f"✗ Invalid layer '{layer}': {method}")
            if len(invalid_layers) > 5:
                print(f"  ... and {len(invalid_layers) - 5} more")
        else:
            print(f"✓ All {len(methods)} methods have valid layer assignments")

    def validate_methods(self):
        """Validate method entries."""
        print("\n--- Validating Method Entries ---")
        
        required_fields = [
            "canonical_id",
            "class",
            "method_name",
            "module",
            "file",
            "layer",
            "flags",
            "calibration_status",
            "calibration_ref"
        ]

        methods = self.catalog.get("methods", {})
        
        if not methods:
            self.errors.append("No methods found in catalog")
            print("✗ No methods found")
            return

        missing_fields_count = 0
        for method_name, method_data in methods.items():
            missing = [f for f in required_fields if f not in method_data]
            if missing:
                missing_fields_count += 1
                if missing_fields_count <= 3:  # Show first 3
                    self.errors.append(f"Method '{method_name}' missing fields: {missing}")
                    print(f"✗ Method '{method_name}' missing: {missing}")

        if missing_fields_count > 3:
            print(f"  ... and {missing_fields_count - 3} more methods with missing fields")

        if missing_fields_count == 0:
            print(f"✓ All {len(methods)} methods have required fields")

    def validate_statistics(self):
        """Validate statistics match actual data."""
        print("\n--- Validating Statistics ---")
        
        statistics = self.catalog.get("statistics", {})
        methods = self.catalog.get("methods", {})
        
        # Count actual layers
        actual_layer_counts = Counter()
        actual_complexity_counts = Counter()
        actual_priority_counts = Counter()
        actual_calibration_status_counts = Counter()
        
        for method_data in methods.values():
            actual_layer_counts[method_data.get("layer", "unknown")] += 1
            actual_complexity_counts[method_data.get("complexity", "unknown")] += 1
            actual_priority_counts[method_data.get("priority", "unknown")] += 1
            actual_calibration_status_counts[method_data.get("calibration_status", "unknown")] += 1

        # Validate total_calibrated
        reported_total = statistics.get("total_calibrated", 0)
        actual_total = len(methods)
        
        if reported_total == actual_total:
            print(f"✓ Total calibrated matches: {actual_total}")
        else:
            self.errors.append(f"Total calibrated mismatch: reported {reported_total}, actual {actual_total}")
            print(f"✗ Total mismatch: reported {reported_total}, actual {actual_total}")

        # Validate by_layer
        reported_layers = statistics.get("by_layer", {})
        for layer, count in reported_layers.items():
            actual_count = actual_layer_counts.get(layer, 0)
            if count != actual_count:
                self.warnings.append(f"Layer {layer} count mismatch: reported {count}, actual {actual_count}")
                print(f"⚠ Layer {layer}: reported {count}, actual {actual_count}")

    def validate_calibration_references(self):
        """Validate that calibration references exist in calibration_registry.py."""
        print("\n--- Validating Calibration References ---")
        
        if not CALIBRATION_REGISTRY.exists():
            self.warnings.append("calibration_registry.py not found - skipping reference validation")
            print("⚠ calibration_registry.py not found - skipping")
            return

        # Read registry
        registry_content = CALIBRATION_REGISTRY.read_text()
        
        methods = self.catalog.get("methods", {})
        missing_refs = []
        
        for method_name, method_data in methods.items():
            calib_ref = method_data.get("calibration_ref")
            if not calib_ref:
                continue
            
            # Check if reference exists in registry
            # Format: ("ClassName", "method_name"):
            class_name = method_data.get("class", "")
            method_only = method_data.get("method_name", "")
            
            search_pattern = f'("{class_name}", "{method_only}"):'
            if search_pattern not in registry_content:
                missing_refs.append(method_name)

        if missing_refs:
            for ref in missing_refs[:5]:  # Show first 5
                self.warnings.append(f"Calibration reference not found in registry: {ref}")
                print(f"⚠ Reference not found: {ref}")
            if len(missing_refs) > 5:
                print(f"  ... and {len(missing_refs) - 5} more")
        else:
            print(f"✓ All calibration references validated")

    def check_duplicates(self):
        """Check for duplicate method entries."""
        print("\n--- Checking for Duplicates ---")
        
        methods = self.catalog.get("methods", {})
        
        # Check for duplicate canonical_ids
        canonical_ids = [m.get("canonical_id") for m in methods.values() if m.get("canonical_id")]
        canonical_id_counts = Counter(canonical_ids)
        duplicates = {cid: count for cid, count in canonical_id_counts.items() if count > 1}
        
        if duplicates:
            for cid, count in list(duplicates.items())[:5]:
                self.errors.append(f"Duplicate canonical_id: {cid} (appears {count} times)")
                print(f"✗ Duplicate: {cid} ({count} times)")
            if len(duplicates) > 5:
                print(f"  ... and {len(duplicates) - 5} more duplicates")
        else:
            print(f"✓ No duplicate canonical_ids found")

        # Check for duplicate method names
        method_names = list(methods.keys())
        method_name_counts = Counter(method_names)
        dup_names = {name: count for name, count in method_name_counts.items() if count > 1}
        
        if dup_names:
            for name, count in list(dup_names.items())[:5]:
                self.errors.append(f"Duplicate method name: {name} (appears {count} times)")
                print(f"✗ Duplicate method: {name} ({count} times)")
        else:
            print(f"✓ No duplicate method names found")

    def print_results(self):
        """Print validation results."""
        print()
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print()

        if self.errors:
            print("ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
            print()

        if self.warnings:
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()

        if len(self.errors) == 0:
            print("✓✓✓ CATALOG IS VALID ✓✓✓")
            return True
        else:
            print("✗✗✗ CATALOG VALIDATION FAILED ✗✗✗")
            print(f"Fix {len(self.errors)} error(s) to make catalog valid.")
            return False


def main():
    """Main entry point."""
    validator = CatalogValidator(CANONICAL_CATALOG)
    
    try:
        success = validator.validate_all()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
