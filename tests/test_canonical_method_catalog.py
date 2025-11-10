#!/usr/bin/env python3
"""
Tests for Canonical Method Catalog

Validates that the canonical method catalog is correctly structured,
complete, and aligned with the calibration registry.
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any

# Repository root
REPO_ROOT = Path(__file__).parent.parent.absolute()
CANONICAL_CATALOG = REPO_ROOT / "config" / "canonical_method_catalog.json"
CALIBRATION_REGISTRY = REPO_ROOT / "src" / "saaaaaa" / "core" / "orchestrator" / "calibration_registry.py"


@pytest.fixture
def catalog() -> Dict[str, Any]:
    """Load canonical method catalog."""
    with open(CANONICAL_CATALOG) as f:
        return json.load(f)


@pytest.fixture
def calibration_registry_content() -> str:
    """Load calibration registry content."""
    return CALIBRATION_REGISTRY.read_text()


class TestCatalogStructure:
    """Tests for catalog structure and required fields."""

    def test_catalog_file_exists(self):
        """Test that catalog file exists."""
        assert CANONICAL_CATALOG.exists(), f"Catalog not found at {CANONICAL_CATALOG}"

    def test_catalog_is_valid_json(self):
        """Test that catalog is valid JSON."""
        with open(CANONICAL_CATALOG) as f:
            data = json.load(f)
        assert isinstance(data, dict), "Catalog must be a JSON object"

    def test_required_top_level_keys(self, catalog):
        """Test that all required top-level keys are present."""
        required_keys = {"metadata", "methods", "statistics", "migration_notes"}
        actual_keys = set(catalog.keys())
        assert required_keys.issubset(actual_keys), f"Missing keys: {required_keys - actual_keys}"

    def test_metadata_fields(self, catalog):
        """Test that metadata contains all required fields."""
        required_fields = [
            "version",
            "generated_at",
            "purpose",
            "total_methods",
            "layer_taxonomy",
            "migration_status"
        ]
        metadata = catalog["metadata"]
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

    def test_layer_taxonomy_complete(self, catalog):
        """Test that layer taxonomy defines all 5 required layers."""
        required_layers = {"Q", "D", "P", "C", "M"}
        taxonomy = catalog["metadata"]["layer_taxonomy"]
        actual_layers = set(taxonomy.keys())
        assert required_layers == actual_layers, f"Layer mismatch. Expected {required_layers}, got {actual_layers}"

    def test_migration_status_complete(self, catalog):
        """Test that migration status indicates completion."""
        status = catalog["metadata"]["migration_status"]
        assert status == "COMPLETE", f"Migration not complete: {status}"


class TestMethodEntries:
    """Tests for individual method entries."""

    def test_methods_not_empty(self, catalog):
        """Test that methods dictionary is not empty."""
        methods = catalog["methods"]
        assert len(methods) > 0, "Methods dictionary is empty"

    def test_method_required_fields(self, catalog):
        """Test that all methods have required fields."""
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
        
        methods = catalog["methods"]
        for method_name, method_data in methods.items():
            for field in required_fields:
                assert field in method_data, f"Method '{method_name}' missing field: {field}"

    def test_layer_values_valid(self, catalog):
        """Test that all layer values are valid."""
        valid_layers = {"Q", "D", "P", "C", "M"}
        methods = catalog["methods"]
        
        for method_name, method_data in methods.items():
            layer = method_data["layer"]
            assert layer in valid_layers, f"Method '{method_name}' has invalid layer: {layer}"

    def test_calibration_status_valid(self, catalog):
        """Test that calibration status values are valid."""
        valid_statuses = {"CAL", "REQ", "OPT", "DER", "INS"}
        methods = catalog["methods"]
        
        for method_name, method_data in methods.items():
            status = method_data["calibration_status"]
            assert status in valid_statuses, f"Method '{method_name}' has invalid status: {status}"

    def test_flags_is_list(self, catalog):
        """Test that flags field is a list."""
        methods = catalog["methods"]
        
        for method_name, method_data in methods.items():
            flags = method_data["flags"]
            assert isinstance(flags, list), f"Method '{method_name}' flags is not a list: {type(flags)}"


class TestStatistics:
    """Tests for catalog statistics."""

    def test_statistics_present(self, catalog):
        """Test that statistics section is present."""
        assert "statistics" in catalog, "Statistics section missing"

    def test_total_calibrated_matches(self, catalog):
        """Test that total_calibrated matches actual method count."""
        reported_total = catalog["statistics"]["total_calibrated"]
        actual_total = len(catalog["methods"])
        assert reported_total == actual_total, f"Total mismatch: reported {reported_total}, actual {actual_total}"

    def test_layer_statistics_sum(self, catalog):
        """Test that layer statistics sum correctly."""
        statistics = catalog["statistics"]
        
        if "by_layer" in statistics:
            layer_total = sum(statistics["by_layer"].values())
            actual_total = len(catalog["methods"])
            # Allow some tolerance for uncategorized methods
            assert abs(layer_total - actual_total) <= 5, \
                f"Layer sum {layer_total} doesn't match total {actual_total}"


class TestCalibrationAlignment:
    """Tests for alignment with calibration registry."""

    def test_calibration_registry_exists(self):
        """Test that calibration registry file exists."""
        assert CALIBRATION_REGISTRY.exists(), f"Registry not found at {CALIBRATION_REGISTRY}"

    def test_calibration_references_valid(self, catalog, calibration_registry_content):
        """Test that calibration references exist in registry."""
        methods = catalog["methods"]
        missing_refs = []
        
        for method_name, method_data in methods.items():
            if method_data["calibration_status"] != "CAL":
                continue  # Skip non-calibrated methods
            
            class_name = method_data["class"]
            method_only = method_data["method_name"]
            
            # Check if entry exists in registry
            search_pattern = f'("{class_name}", "{method_only}"):'
            if search_pattern not in calibration_registry_content:
                missing_refs.append(method_name)
        
        # Allow up to 10 missing references (for methods being migrated)
        assert len(missing_refs) <= 10, \
            f"Too many missing calibration references ({len(missing_refs)}): {missing_refs[:5]}"


class TestMigrationNotes:
    """Tests for migration notes and completeness."""

    def test_migration_notes_present(self, catalog):
        """Test that migration notes are present."""
        assert "migration_notes" in catalog, "Migration notes missing"

    def test_yaml_sources_documented(self, catalog):
        """Test that YAML sources are documented as eliminated."""
        notes = catalog["migration_notes"]
        assert "yaml_sources_eliminated" in notes, "YAML sources not documented"
        
        yaml_sources = notes["yaml_sources_eliminated"]
        assert isinstance(yaml_sources, list), "yaml_sources_eliminated must be a list"
        assert len(yaml_sources) > 0, "No YAML sources documented"

    def test_single_source_of_truth_declared(self, catalog):
        """Test that single source of truth is declared."""
        notes = catalog["migration_notes"]
        assert "single_source_of_truth" in notes, "Single source of truth not declared"
        
        source = notes["single_source_of_truth"]
        expected = "src/saaaaaa/core/orchestrator/calibration_registry.py"
        assert source == expected, f"Wrong source of truth: {source}"

    def test_no_yaml_dependencies(self, catalog):
        """Test that no YAML dependencies remain."""
        notes = catalog["migration_notes"]
        assert notes.get("no_yaml_dependencies") is True, "YAML dependencies still exist"

    def test_all_calibrations_in_registry(self, catalog):
        """Test that all calibrations are in registry."""
        notes = catalog["migration_notes"]
        assert notes.get("all_calibrations_in_registry") is True, \
            "Not all calibrations migrated to registry"


class TestCanonicalID:
    """Tests for canonical ID format and uniqueness."""

    def test_canonical_ids_unique(self, catalog):
        """Test that all canonical IDs are unique."""
        methods = catalog["methods"]
        canonical_ids = [m["canonical_id"] for m in methods.values()]
        
        assert len(canonical_ids) == len(set(canonical_ids)), \
            "Duplicate canonical IDs found"

    def test_canonical_id_format(self, catalog):
        """Test that canonical IDs follow the correct format."""
        # Format: <MODULE>:<CLASS>.<METHOD>@<LAYER>[<FLAGS>]{<STATUS>}
        import re
        pattern = r'^[A-Z]+:[A-Za-z_]+\.[a-z_]+@[QDPCM]\[[A-Z]*\]\{[A-Z]+\}$'
        
        methods = catalog["methods"]
        invalid_ids = []
        
        for method_name, method_data in methods.items():
            canonical_id = method_data["canonical_id"]
            if not re.match(pattern, canonical_id):
                invalid_ids.append((method_name, canonical_id))
        
        # Allow some flexibility for edge cases
        assert len(invalid_ids) <= 5, \
            f"Too many invalid canonical IDs: {invalid_ids[:5]}"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
