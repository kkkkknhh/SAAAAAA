#!/usr/bin/env python3
"""
Canonical Systems Verification Script

Quick verification that all canonical systems components are operational.

Run this after making changes to ensure system integrity.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def check_catalog():
    """Verify catalog module loads"""
    try:
        from saaaaaa.core.orchestrator.catalogo_completo_canonico import CATALOG
        print(f"✅ Catalog loaded: {CATALOG.total_methods} methods")
        return True
    except Exception as e:
        print(f"❌ Catalog failed to load: {e}")
        return False


def check_registry():
    """Verify calibration registry loads"""
    try:
        from saaaaaa.core.orchestrator.calibration_registry import CALIBRATIONS
        print(f"✅ Calibration registry loaded: {len(CALIBRATIONS)} calibrations")
        return True
    except Exception as e:
        print(f"❌ Calibration registry failed to load: {e}")
        return False


def check_ontology():
    """Verify canonical ontology file exists"""
    ontology_path = repo_root / "config" / "canonical_ontologies" / "policy_areas_and_dimensions.json"
    if ontology_path.exists():
        print(f"✅ Canonical ontology exists")
        return True
    else:
        print(f"❌ Canonical ontology not found at {ontology_path}")
        return False


def check_artifacts():
    """Verify all required artifacts exist"""
    artifacts = [
        "config/method_usage_intelligence.json",
        "config/calibration_decisions.json",
        "config/alignment_audit_report.json",
    ]
    
    all_exist = True
    for artifact in artifacts:
        path = repo_root / artifact
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"✅ {artifact} ({size_kb:.1f} KB)")
        else:
            print(f"❌ {artifact} missing")
            all_exist = False
    
    return all_exist


def check_scripts():
    """Verify scripts are executable"""
    scripts = [
        "scripts/build_method_usage_intelligence.py",
        "scripts/build_calibration_decisions.py",
        "scripts/audit_catalog_registry_alignment.py",
    ]
    
    all_exec = True
    for script in scripts:
        path = repo_root / script
        if path.exists() and path.stat().st_mode & 0o111:
            print(f"✅ {script} is executable")
        else:
            print(f"❌ {script} not executable or missing")
            all_exec = False
    
    return all_exec


def main():
    print("="*80)
    print("CANONICAL SYSTEMS VERIFICATION")
    print("="*80)
    
    results = []
    
    print("\n[1/5] Checking catalog module...")
    results.append(check_catalog())
    
    print("\n[2/5] Checking calibration registry...")
    results.append(check_registry())
    
    print("\n[3/5] Checking canonical ontology...")
    results.append(check_ontology())
    
    print("\n[4/5] Checking artifacts...")
    results.append(check_artifacts())
    
    print("\n[5/5] Checking scripts...")
    results.append(check_scripts())
    
    print("\n" + "="*80)
    if all(results):
        print("✅ ALL CHECKS PASSED")
        print("="*80)
        print("\nCanonical systems infrastructure is operational.")
        print("\nNext steps:")
        print("  1. Run alignment audit: python scripts/audit_catalog_registry_alignment.py")
        print("  2. Review defects in config/alignment_audit_report.json")
        print("  3. Resolve defects to achieve system integrity")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("="*80)
        print("\nFix errors above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
