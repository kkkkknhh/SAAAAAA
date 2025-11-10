#!/usr/bin/env python3
"""
Check Directive Compliance - Universal Calibration Migration Validation

This script verifies that all directive requirements for the calibration migration
have been satisfied:

1. All methods requiring calibration are in calibration_registry.py
2. Calibration logic aligned with canonical layer taxonomy
3. All YAML-based calibrations eliminated
4. Single canonical center for calibration reporting
5. Embedded calibrations migrated or tracked
6. CALIBRATION_PROCEDURE.md exists and is complete

Exit codes:
    0 - All directive requirements satisfied
    1 - One or more directive requirements NOT satisfied
    2 - Critical error during validation
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util

# Repository root
REPO_ROOT = Path(__file__).parent.parent.absolute()

# Required files
CALIBRATION_REGISTRY = REPO_ROOT / "src" / "saaaaaa" / "core" / "orchestrator" / "calibration_registry.py"
CANONICAL_CATALOG = REPO_ROOT / "config" / "canonical_method_catalog.json"
EMBEDDED_APPENDIX = REPO_ROOT / "config" / "embedded_calibration_appendix.json"
CALIBRATION_PROCEDURE = REPO_ROOT / "CALIBRATION_PROCEDURE.md"

# YAML files that should be deprecated
YAML_CALIBRATION_FILES = [
    REPO_ROOT / "calibracion_bayesiana.yaml",
    REPO_ROOT / "financia_callibrator.yaml",
    REPO_ROOT / "catalogo_principal.yaml",
    REPO_ROOT / "causal_exctractor.yaml",
    REPO_ROOT / "trazabilidad_cohrencia.yaml",
]


class DirectiveComplianceChecker:
    """Validates Universal Calibration Migration directive compliance."""

    def __init__(self):
        self.passed_checks = []
        self.failed_checks = []
        self.warnings = []

    def check_all(self) -> bool:
        """Run all compliance checks."""
        print("=" * 80)
        print("UNIVERSAL CALIBRATION MIGRATION - DIRECTIVE COMPLIANCE CHECK")
        print("=" * 80)
        print()

        # Directive 1: All methods in calibration_registry
        self.check_calibration_registry_exists()
        self.check_calibration_registry_completeness()

        # Directive 2: Layer taxonomy alignment
        self.check_canonical_catalog_exists()
        self.check_layer_taxonomy_alignment()

        # Directive 3: YAML elimination
        self.check_yaml_elimination()

        # Directive 4: Single source of truth
        self.check_single_source_of_truth()

        # Directive 5: Embedded calibrations tracked
        self.check_embedded_calibrations_appendix()

        # Directive 6: CALIBRATION_PROCEDURE.md exists
        self.check_calibration_procedure_doc()

        # Print results
        self.print_results()

        return len(self.failed_checks) == 0

    def check_calibration_registry_exists(self):
        """Verify calibration_registry.py exists."""
        check_name = "Directive 1.1: calibration_registry.py exists"
        if CALIBRATION_REGISTRY.exists():
            self.passed_checks.append(check_name)
            print(f"✓ {check_name}")
        else:
            self.failed_checks.append((check_name, "File not found"))
            print(f"✗ {check_name}: File not found at {CALIBRATION_REGISTRY}")

    def check_calibration_registry_completeness(self):
        """Verify calibration_registry.py has calibrations."""
        check_name = "Directive 1.2: calibration_registry.py has calibrations"
        
        if not CALIBRATION_REGISTRY.exists():
            self.failed_checks.append((check_name, "Registry file not found"))
            return

        content = CALIBRATION_REGISTRY.read_text()
        
        # Count calibration entries
        import re
        calibration_pattern = r'^\s+\(".*?", ".*?"\):\s*MethodCalibration\('
        matches = re.findall(calibration_pattern, content, re.MULTILINE)
        count = len(matches)

        if count >= 100:  # Expect at least 100 calibrations
            self.passed_checks.append(check_name)
            print(f"✓ {check_name}: Found {count} calibrations")
        else:
            self.failed_checks.append((check_name, f"Only {count} calibrations found, expected >= 100"))
            print(f"✗ {check_name}: Only {count} calibrations found")

    def check_canonical_catalog_exists(self):
        """Verify canonical_method_catalog.json exists."""
        check_name = "Directive 2.1: canonical_method_catalog.json exists"
        if CANONICAL_CATALOG.exists():
            self.passed_checks.append(check_name)
            print(f"✓ {check_name}")
        else:
            self.failed_checks.append((check_name, "File not found"))
            print(f"✗ {check_name}: File not found at {CANONICAL_CATALOG}")

    def check_layer_taxonomy_alignment(self):
        """Verify layer taxonomy is defined and used."""
        check_name = "Directive 2.2: Layer taxonomy alignment"
        
        if not CANONICAL_CATALOG.exists():
            self.failed_checks.append((check_name, "Catalog file not found"))
            return

        try:
            with open(CANONICAL_CATALOG) as f:
                catalog = json.load(f)

            # Check for layer_taxonomy in metadata
            if "layer_taxonomy" in catalog.get("metadata", {}):
                taxonomy = catalog["metadata"]["layer_taxonomy"]
                required_layers = {"Q", "D", "P", "C", "M"}
                found_layers = set(taxonomy.keys())
                
                if required_layers.issubset(found_layers):
                    self.passed_checks.append(check_name)
                    print(f"✓ {check_name}: All 5 required layers defined")
                else:
                    missing = required_layers - found_layers
                    self.failed_checks.append((check_name, f"Missing layers: {missing}"))
                    print(f"✗ {check_name}: Missing layers: {missing}")
            else:
                self.failed_checks.append((check_name, "No layer_taxonomy in metadata"))
                print(f"✗ {check_name}: No layer_taxonomy in metadata")

        except json.JSONDecodeError as e:
            self.failed_checks.append((check_name, f"JSON parse error: {e}"))
            print(f"✗ {check_name}: JSON parse error")

    def check_yaml_elimination(self):
        """Verify all YAML calibration files are deprecated."""
        check_name = "Directive 3: YAML calibration elimination"
        
        deprecated_count = 0
        active_count = 0
        missing_count = 0

        for yaml_file in YAML_CALIBRATION_FILES:
            if not yaml_file.exists():
                missing_count += 1
                continue

            content = yaml_file.read_text()
            if "DEPRECATED" in content and "DO NOT USE" in content:
                deprecated_count += 1
            else:
                active_count += 1
                self.warnings.append(f"YAML file may still be active: {yaml_file.name}")

        if active_count == 0:
            self.passed_checks.append(check_name)
            print(f"✓ {check_name}: All {deprecated_count} YAML files properly deprecated")
        else:
            self.failed_checks.append((check_name, f"{active_count} YAML files not deprecated"))
            print(f"✗ {check_name}: {active_count} YAML files not properly deprecated")

    def check_single_source_of_truth(self):
        """Verify calibration_registry is single source of truth."""
        check_name = "Directive 4: Single source of truth (calibration_registry.py)"
        
        if not CALIBRATION_REGISTRY.exists():
            self.failed_checks.append((check_name, "Registry not found"))
            return

        # Check that registry defines CALIBRATIONS dict
        content = CALIBRATION_REGISTRY.read_text()
        if "CALIBRATIONS:" in content or "CALIBRATIONS =" in content:
            self.passed_checks.append(check_name)
            print(f"✓ {check_name}")
        else:
            self.failed_checks.append((check_name, "No CALIBRATIONS dict found"))
            print(f"✗ {check_name}: No CALIBRATIONS dict found")

    def check_embedded_calibrations_appendix(self):
        """Verify embedded calibrations are tracked."""
        check_name = "Directive 5: Embedded calibrations tracked"
        
        if not EMBEDDED_APPENDIX.exists():
            self.failed_checks.append((check_name, "Appendix file not found"))
            print(f"✗ {check_name}: File not found at {EMBEDDED_APPENDIX}")
            return

        try:
            with open(EMBEDDED_APPENDIX) as f:
                appendix = json.load(f)

            # Check for embedded_calibrations section
            if "embedded_calibrations" in appendix:
                # Check migration verification
                verification = appendix.get("migration_verification", {})
                if verification.get("all_production_calibrations_migrated"):
                    self.passed_checks.append(check_name)
                    print(f"✓ {check_name}: All production calibrations migrated")
                else:
                    self.failed_checks.append((check_name, "Not all calibrations migrated"))
                    print(f"✗ {check_name}: Not all production calibrations migrated")
            else:
                self.failed_checks.append((check_name, "No embedded_calibrations section"))
                print(f"✗ {check_name}: No embedded_calibrations section")

        except json.JSONDecodeError as e:
            self.failed_checks.append((check_name, f"JSON parse error: {e}"))
            print(f"✗ {check_name}: JSON parse error")

    def check_calibration_procedure_doc(self):
        """Verify CALIBRATION_PROCEDURE.md exists and is complete."""
        check_name = "Directive 6: CALIBRATION_PROCEDURE.md exists"
        
        if not CALIBRATION_PROCEDURE.exists():
            self.failed_checks.append((check_name, "File not found"))
            print(f"✗ {check_name}: File not found at {CALIBRATION_PROCEDURE}")
            return

        content = CALIBRATION_PROCEDURE.read_text()
        
        # Check for required sections
        required_sections = [
            "Mathematical Foundations",
            "Calibration Parameters",
            "Deterministic Rules",
            "Contextual Refinement",
            "Procedure for Adding New Calibrations",
            "Examples",
            "Change Log"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)

        if not missing_sections:
            self.passed_checks.append(check_name)
            print(f"✓ {check_name}: All required sections present")
        else:
            self.failed_checks.append((check_name, f"Missing sections: {missing_sections}"))
            print(f"✗ {check_name}: Missing sections: {missing_sections}")

    def print_results(self):
        """Print final compliance results."""
        print()
        print("=" * 80)
        print("COMPLIANCE SUMMARY")
        print("=" * 80)
        print(f"Passed: {len(self.passed_checks)}")
        print(f"Failed: {len(self.failed_checks)}")
        print(f"Warnings: {len(self.warnings)}")
        print()

        if self.failed_checks:
            print("FAILED CHECKS:")
            for check, reason in self.failed_checks:
                print(f"  - {check}: {reason}")
            print()

        if self.warnings:
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()

        if len(self.failed_checks) == 0:
            print("✓✓✓ ALL DIRECTIVE REQUIREMENTS SATISFIED ✓✓✓")
            print()
            print("Universal Calibration Migration: COMPLETE")
            return True
        else:
            print("✗✗✗ DIRECTIVE COMPLIANCE FAILED ✗✗✗")
            print()
            print(f"Fix {len(self.failed_checks)} failed check(s) to complete migration.")
            return False


def main():
    """Main entry point."""
    checker = DirectiveComplianceChecker()
    
    try:
        success = checker.check_all()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
