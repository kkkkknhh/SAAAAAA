#!/usr/bin/env python3
"""
Example demonstrating I/O-free orchestrator initialization using factory pattern.

This shows the recommended way to initialize the Orchestrator:
1. Use factory.py to load all data from disk (I/O layer)
2. Pass pre-loaded data to Orchestrator constructor (business logic layer)
3. Orchestrator remains pure and testable without file dependencies
"""

from pathlib import Path

from saaaaaa.core.orchestrator import Orchestrator, get_questionnaire_provider
from saaaaaa.core.orchestrator.factory import (
    CoreModuleFactory,
    load_catalog,
    load_method_map,
    load_questionnaire_monolith,
    load_schema,
)

def main():
    print("=== I/O-Free Orchestrator Initialization Example ===\n")

    # Step 1: Load all data using factory (I/O layer)
    print("Step 1: Loading data from disk using factory...")
    try:
        # Load catalog
        catalog_path = Path("rules/METODOS/metodos_completos_nivel3.json")
        if catalog_path.exists():
            catalog = load_catalog(catalog_path)
            print(f"  ✓ Loaded catalog with {len(catalog)} methods")
        else:
            print(f"  ⚠ Catalog not found at {catalog_path}, using empty dict")
            catalog = {}

        # Load questionnaire monolith
        monolith_path = Path("questionnaire_monolith.json")
        if monolith_path.exists():
            monolith = load_questionnaire_monolith(monolith_path)
            print("  ✓ Loaded questionnaire monolith")
            # Initialize global provider for backward compatibility
            get_questionnaire_provider().set_data(monolith)
        else:
            print(f"  ⚠ Questionnaire not found at {monolith_path}, using empty dict")
            monolith = {"blocks": {"micro_questions": [], "meso_questions": [], "macro_question": {}}}

        # Load method map (optional)
        method_map_path = Path("COMPLETE_METHOD_CLASS_MAP.json")
        if method_map_path.exists():
            method_map = load_method_map(method_map_path)
            print("  ✓ Loaded method map")
        else:
            print(f"  ⚠ Method map not found at {method_map_path}, using None")
            method_map = None

        # Load schema (optional)
        schema_path = Path("schemas/questionnaire.schema.json")
        if schema_path.exists():
            schema = load_schema(schema_path)
            print("  ✓ Loaded schema")
        else:
            print(f"  ⚠ Schema not found at {schema_path}, using None")
            schema = None

    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return

    print()

    # Step 2: Initialize Orchestrator with pre-loaded data (I/O-free)
    print("Step 2: Initializing Orchestrator with pre-loaded data (I/O-free)...")
    try:
        orchestrator = Orchestrator(
            catalog=catalog,
            monolith=monolith,
            method_map=method_map,
            schema=schema,
        )
        print("  ✓ Orchestrator initialized successfully (no file I/O)")

        # Verify orchestrator is functional
        health = orchestrator.health_check()
        print(f"  ✓ Health check: score={health['score']:.1f}")

    except Exception as e:
        print(f"  ✗ Error initializing orchestrator: {e}")
        return

    print()

    # Step 3: Alternative - Use CoreModuleFactory
    print("Step 3: Alternative approach using CoreModuleFactory...")
    try:
        factory = CoreModuleFactory()
        # Factory handles all I/O and caching
        factory.get_questionnaire()
        print("  ✓ Factory loaded questionnaire (cached)")

    except Exception as e:
        print(f"  ⚠ Factory approach not available: {e}")

    print()
    print("=== Example Complete ===")
    print("\nKey Benefits of I/O-Free Initialization:")
    print("  • Testability: Can test with mock data without file dependencies")
    print("  • Performance: Data loaded once and reused")
    print("  • Flexibility: Easy to inject different data sources")
    print("  • Separation: Clear boundary between I/O and business logic")

if __name__ == "__main__":
    main()
