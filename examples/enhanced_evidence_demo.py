"""
Example demonstrating the enhanced Evidence Registry features:
1. Canonical JSON serialization with EvidenceRecord.create()
2. Chain replay logic with _assert_chain()
3. JSONContractLoader for loading configurations
"""

import sys
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import (
    EvidenceRecord,
    EvidenceRegistry,
    JSONContractLoader,
)


def demo_canonical_serialization():
    """Demonstrate canonical JSON serialization."""
    print("=" * 60)
    print("1. Canonical JSON Serialization")
    print("=" * 60)
    
    # Create records with same data but different key order
    print("\nCreating two records with same data, different key order...")
    
    record1 = EvidenceRecord.create(
        evidence_type="analysis",
        payload={"zebra": 1, "apple": 2, "mango": 3},  # Unordered keys
    )
    
    record2 = EvidenceRecord.create(
        evidence_type="analysis",
        payload={"apple": 2, "mango": 3, "zebra": 1},  # Different order
    )
    
    print(f"Record 1 hash: {record1.content_hash}")
    print(f"Record 2 hash: {record2.content_hash}")
    print(f"Hashes match: {record1.content_hash == record2.content_hash}")
    
    # Show canonical dump
    canonical = record1._canonical_dump(record1.payload)
    print(f"\nCanonical JSON: {canonical}")
    print("Keys are always sorted, no whitespace")
    
    print("\n✓ Canonical serialization ensures deterministic hashing\n")


def demo_chain_validation():
    """Demonstrate chain validation with _assert_chain()."""
    print("=" * 60)
    print("2. Chain Validation & Replay Logic")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "evidence.jsonl"
        
        # Create a chain of evidence
        print("\nCreating evidence chain...")
        registry = EvidenceRegistry(storage_path=storage_path)
        
        ids = []
        for i in range(5):
            eid = registry.record_evidence(
                evidence_type="step",
                payload={"step": i, "data": f"value_{i}"},
                source_method=f"Process.step_{i}",
            )
            ids.append(eid)
            print(f"  - Recorded step {i}: {eid[:16]}...")
        
        # Verify chain integrity
        print("\nVerifying chain integrity...")
        is_valid, errors = registry.verify_chain_integrity()
        print(f"Chain valid: {is_valid}")
        print(f"Errors: {len(errors)}")
        
        # Show chain linkage
        print("\nChain linkage:")
        for i, eid in enumerate(ids):
            evidence = registry.get_evidence(eid)
            prev = evidence.previous_hash
            if prev:
                print(f"  Step {i}: previous_hash = {prev[:16]}...")
            else:
                print(f"  Step {i}: (first record, no previous_hash)")
        
        # Reload to demonstrate _assert_chain()
        print("\nReloading registry to test _assert_chain()...")
        registry2 = EvidenceRegistry(storage_path=storage_path)
        print(f"Loaded {len(registry2.hash_index)} records")
        print(f"Last entry index: {registry2.last_entry.payload['step']}")
        
        print("\n✓ Chain validation ensures ordering and integrity\n")


def demo_contract_loader():
    """Demonstrate JSONContractLoader features."""
    print("=" * 60)
    print("3. JSONContractLoader - Path Resolution & Error Aggregation")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        
        # Create test directory structure
        configs_dir = base_path / "configs"
        configs_dir.mkdir()
        
        schemas_dir = configs_dir / "schemas"
        schemas_dir.mkdir()
        
        # Create valid config files
        (configs_dir / "database.json").write_text(json.dumps({
            "host": "localhost",
            "port": 5432,
            "name": "mydb"
        }))
        
        (configs_dir / "api.json").write_text(json.dumps({
            "endpoint": "https://api.example.com",
            "timeout": 30
        }))
        
        # Create schema files
        (schemas_dir / "user.json").write_text(json.dumps({
            "type": "object",
            "properties": {"id": {"type": "string"}}
        }))
        
        # Create invalid file for error demonstration
        (configs_dir / "invalid.json").write_text("{invalid json}")
        
        print("\nDirectory structure created:")
        print("  configs/")
        print("    database.json")
        print("    api.json")
        print("    invalid.json")
        print("    schemas/")
        print("      user.json")
        
        # Load with JSONContractLoader
        print("\n--- Loading all JSON files from configs/ ---")
        loader = JSONContractLoader(base_paths=[base_path])
        
        result = loader.load_directory(
            "configs",
            pattern="*.json",
            recursive=False,
            aggregate_errors=True
        )
        
        print(f"\nSuccess: {result.success}")
        print(f"Files loaded: {len(result.files_loaded)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.data:
            print("\nLoaded configurations:")
            for key, value in result.data.items():
                print(f"  {key}: {list(value.keys())}")
        
        if result.errors:
            print("\nErrors encountered:")
            for error in result.errors:
                print(f"  {error}")
        
        # Demonstrate recursive loading
        print("\n--- Loading recursively with pattern 'user*.json' ---")
        result2 = loader.load_directory(
            "configs",
            pattern="user*.json",
            recursive=True,
            aggregate_errors=True
        )
        
        print(f"Files matched: {len(result2.files_loaded)}")
        if result2.data:
            print(f"Loaded: {list(result2.data.keys())}")
        
        # Demonstrate path resolution
        print("\n--- Testing path resolution ---")
        single_result = loader.load_file("configs/database.json")
        print(f"Loaded single file: {single_result.success}")
        if single_result.data:
            print(f"Database config: {single_result.data}")
        
        print("\n✓ JSONContractLoader provides robust file loading\n")


def demo_integration():
    """Demonstrate integration of all features."""
    print("=" * 60)
    print("4. Integration: Config Loading + Evidence Registry")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        
        # Setup configs
        configs_dir = base_path / "configs"
        configs_dir.mkdir()
        (configs_dir / "analysis.json").write_text(json.dumps({
            "method": "bayesian",
            "confidence": 0.95
        }))
        
        # Load configs
        print("\nLoading configuration...")
        loader = JSONContractLoader(base_paths=[base_path])
        config_result = loader.load_directory("configs")
        
        if config_result.success:
            print(f"✓ Loaded configs: {list(config_result.data.keys())}")
        
        # Use config in evidence registry
        print("\nRecording evidence with loaded config...")
        registry = EvidenceRegistry(
            storage_path=base_path / "evidence.jsonl"
        )
        
        for config_name, config_data in config_result.data.items():
            evidence_id = registry.record_evidence(
                evidence_type="configuration",
                payload={
                    "config_name": config_name,
                    "config_data": config_data,
                    "loaded_at": "demo_time"
                },
                source_method="ConfigLoader.load",
            )
            print(f"  Recorded {config_name}: {evidence_id[:16]}...")
        
        # Verify the chain
        is_valid, errors = registry.verify_chain_integrity()
        print(f"\nChain validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
        
        # Export provenance
        if registry.dag:
            dag_export = registry.export_provenance_dag(format="dict")
            print(f"Provenance DAG: {dag_export['stats']['total_nodes']} nodes")
        
        print("\n✓ All features work together seamlessly\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Evidence Registry Enhancement Demo")
    print("=" * 60 + "\n")
    
    try:
        demo_canonical_serialization()
        demo_chain_validation()
        demo_contract_loader()
        demo_integration()
        
        print("=" * 60)
        print("✅ All demos completed successfully!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
