"""
Unit Tests for Evidence Registry

Tests the evidence registry's append-only JSONL storage, hash chain integrity,
provenance tracking, and cryptographic verification capabilities.

Preconditions:
- Write access to /tmp for test files
- Valid JSON payload structures

Invariants:
- Hash chain integrity maintained
- Append-only semantics preserved
- Evidence immutability enforced
- Provenance DAG is acyclic

Postconditions:
- All evidence is recoverable
- Hash chain is verifiable
- Provenance is traceable
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.evidence_registry import (
    EvidenceRecord,
    EvidenceRegistry,
)


def test_evidence_record_creation():
    """Test creating evidence records with hash generation."""
    payload = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
    
    record = EvidenceRecord(
        evidence_id="test_001",
        evidence_type="method_result",
        payload=payload,
        source_method="TestModule.test_method",
    )
    
    # Verify record fields
    assert record.evidence_id == "test_001"
    assert record.evidence_type == "method_result"
    assert record.payload == payload
    assert record.source_method == "TestModule.test_method"
    
    # Verify hash generation
    assert record.content_hash is not None
    assert len(record.content_hash) == 64  # SHA-256 hex length
    assert record.entry_hash is not None
    assert len(record.entry_hash) == 64
    
    print(f"✓ Evidence record created with hashes")
    print(f"  content_hash: {record.content_hash[:16]}...")
    print(f"  entry_hash: {record.entry_hash[:16]}...")


def test_evidence_record_hash_reproducibility():
    """Test that same content produces same hash."""
    payload = {"data": "test", "value": 123}
    
    record1 = EvidenceRecord(
        evidence_id="test_001",
        evidence_type="test",
        payload=payload,
    )
    
    record2 = EvidenceRecord(
        evidence_id="test_001",
        evidence_type="test",
        payload=payload,
    )
    
    # Same payload should produce same content hash
    assert record1.content_hash == record2.content_hash
    print(f"✓ Hash reproducibility verified")


def test_evidence_record_hash_uniqueness():
    """Test that different content produces different hashes."""
    payload1 = {"data": "test1"}
    payload2 = {"data": "test2"}
    
    record1 = EvidenceRecord(
        evidence_id="test_001",
        evidence_type="test",
        payload=payload1,
    )
    
    record2 = EvidenceRecord(
        evidence_id="test_002",
        evidence_type="test",
        payload=payload2,
    )
    
    # Different payloads should produce different hashes
    assert record1.content_hash != record2.content_hash
    assert record1.entry_hash != record2.entry_hash
    print(f"✓ Hash uniqueness verified")


def test_registry_initialization():
    """Test registry initialization with JSONL file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.jsonl"
        
        registry = EvidenceRegistry(storage_path=registry_path)
        
        assert registry.storage_path == registry_path
        # Registry doesn't create file until first append
        assert len(registry.hash_index) == 0
        
        print(f"✓ Registry initialized at {registry_path}")


def test_registry_append_single_evidence():
    """Test appending single evidence to registry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.jsonl"
        registry = EvidenceRegistry(storage_path=registry_path)
        
        payload = {"test": "data", "value": 42}
        
        evidence_id = registry.record_evidence(
            evidence_type="method_result",
            payload=payload,
            source_method="TestModule.test_method",
        )
        
        assert len(registry.hash_index) == 1
        assert evidence_id in registry.hash_index
        print(f"✓ Single evidence appended with ID: {evidence_id[:16]}...")


def test_registry_append_multiple_evidence():
    """Test appending multiple evidence records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.jsonl"
        registry = EvidenceRegistry(storage_path=registry_path)
        
        evidence_ids = []
        for i in range(5):
            eid = registry.record_evidence(
                evidence_type="method_result",
                payload={"index": i, "data": f"value_{i}"},
                source_method=f"TestModule.method_{i}",
            )
            evidence_ids.append(eid)
        
        assert len(registry.hash_index) == 5
        for eid in evidence_ids:
            assert eid in registry.hash_index
        print(f"✓ Multiple evidence records appended")


def test_registry_retrieve_evidence():
    """Test retrieving evidence by ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.jsonl"
        registry = EvidenceRegistry(storage_path=registry_path)
        
        # Append evidence
        payload = {"test": "retrieve", "value": 99}
        evidence_id = registry.record_evidence(
            evidence_type="method_result",
            payload=payload,
        )
        
        # Retrieve evidence
        retrieved = registry.get_evidence(evidence_id)
        
        assert retrieved is not None
        assert retrieved.evidence_id == evidence_id
        assert retrieved.payload == payload
        print(f"✓ Evidence retrieved successfully")


def test_registry_retrieve_nonexistent():
    """Test retrieving nonexistent evidence returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.jsonl"
        registry = EvidenceRegistry(storage_path=registry_path)
        
        retrieved = registry.get_evidence("nonexistent_id")
        assert retrieved is None
        print(f"✓ Nonexistent evidence returns None")


def test_registry_hash_chain_integrity():
    """Test that hash chain is maintained across appends."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.jsonl"
        registry = EvidenceRegistry(storage_path=registry_path)
        
        # Append multiple records
        for i in range(3):
            registry.record_evidence(
                evidence_type="test",
                payload={"index": i},
            )
        
        # Verify hash chain
        is_valid, errors = registry.verify_chain_integrity()
        
        assert is_valid, f"Hash chain validation failed: {errors}"
        print(f"✓ Hash chain integrity verified")


def test_registry_detect_tampering():
    """Test that tampering is detected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.jsonl"
        registry = EvidenceRegistry(storage_path=registry_path)
        
        # Append evidence
        registry.record_evidence(
            evidence_type="test",
            payload={"original": "data"},
        )
        
        # Tamper with JSONL file
        with open(registry_path, "r") as f:
            lines = f.readlines()
        
        # Modify payload in file
        tampered_data = json.loads(lines[0])
        tampered_data["payload"]["original"] = "TAMPERED"
        
        with open(registry_path, "w") as f:
            f.write(json.dumps(tampered_data) + "\n")
        
        # Reload registry and verify
        new_registry = EvidenceRegistry(storage_path=registry_path)
        is_valid, errors = new_registry.verify_chain_integrity()
        
        assert not is_valid, "Tampering should be detected"
        assert len(errors) > 0
        print(f"✓ Tampering detected: {errors[0]}")


def test_registry_provenance_tracking():
    """Test provenance tracking with parent dependencies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.jsonl"
        registry = EvidenceRegistry(storage_path=registry_path)
        
        # Create parent evidence
        parent1_id = registry.record_evidence(
            evidence_type="extraction",
            payload={"data": "parent1"},
        )
        
        parent2_id = registry.record_evidence(
            evidence_type="extraction",
            payload={"data": "parent2"},
        )
        
        # Create child evidence with parent dependencies
        child_id = registry.record_evidence(
            evidence_type="analysis",
            payload={"data": "child", "result": "combined"},
            parent_evidence_ids=[parent1_id, parent2_id],
        )
        
        # Retrieve and verify
        retrieved = registry.get_evidence(child_id)
        assert retrieved is not None
        assert len(retrieved.parent_evidence_ids) == 2
        assert parent1_id in retrieved.parent_evidence_ids
        assert parent2_id in retrieved.parent_evidence_ids
        
        print(f"✓ Provenance tracking verified")


def test_registry_export_provenance_dag():
    """Test exporting provenance DAG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.jsonl"
        registry = EvidenceRegistry(storage_path=registry_path)
        
        # Create evidence chain
        node1_id = registry.record_evidence(
            evidence_type="extraction",
            payload={"data": "node1"},
        )
        
        node2_id = registry.record_evidence(
            evidence_type="analysis",
            payload={"data": "node2"},
            parent_evidence_ids=[node1_id],
        )
        
        node3_id = registry.record_evidence(
            evidence_type="synthesis",
            payload={"data": "node3"},
            parent_evidence_ids=[node1_id, node2_id],
        )
        
        # Export DAG
        dag = registry.export_provenance_dag()
        
        assert "nodes" in dag
        assert "edges" in dag
        assert len(dag["nodes"]) == 3
        # node_002 has 1 parent, node_003 has 2 parents
        assert len(dag["edges"]) == 3
        
        print(f"✓ Provenance DAG exported")
        print(f"  Nodes: {len(dag['nodes'])}")
        print(f"  Edges: {len(dag['edges'])}")


def test_registry_persistence():
    """Test that registry persists across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.jsonl"
        
        # Create registry and append evidence
        registry1 = EvidenceRegistry(storage_path=registry_path)
        evidence_id = registry1.record_evidence(
            evidence_type="test",
            payload={"data": "persistent"},
        )
        
        # Create new registry instance with same path
        registry2 = EvidenceRegistry(storage_path=registry_path)
        
        # Verify evidence persisted
        retrieved = registry2.get_evidence(evidence_id)
        assert retrieved is not None
        assert retrieved.payload["data"] == "persistent"
        
        print(f"✓ Registry persistence verified")


def test_registry_immutability():
    """Test that evidence cannot be modified after append."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "test_registry.jsonl"
        registry = EvidenceRegistry(storage_path=registry_path)
        
        # Append evidence
        original_payload = {"data": "original"}
        evidence_id = registry.record_evidence(
            evidence_type="test",
            payload=original_payload,
        )
        
        # Attempt to modify original payload (should not affect registry)
        original_payload["data"] = "MODIFIED"
        
        # Retrieve and verify unchanged
        retrieved = registry.get_evidence(evidence_id)
        assert retrieved.payload["data"] == "original"
        
        print(f"✓ Evidence immutability verified")


def run_all_tests():
    """Run all evidence registry tests."""
    print("\n=== Running Evidence Registry Tests ===\n")
    
    tests = [
        test_evidence_record_creation,
        test_evidence_record_hash_reproducibility,
        test_evidence_record_hash_uniqueness,
        test_registry_initialization,
        test_registry_append_single_evidence,
        test_registry_append_multiple_evidence,
        test_registry_retrieve_evidence,
        test_registry_retrieve_nonexistent,
        test_registry_hash_chain_integrity,
        test_registry_detect_tampering,
        test_registry_provenance_tracking,
        test_registry_export_provenance_dag,
        test_registry_persistence,
        test_registry_immutability,
    ]
    
    failed = 0
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Total: {len(tests)}")
    print(f"Passed: {len(tests) - failed}")
    print(f"Failed: {failed}")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    import json
    success = run_all_tests()
    sys.exit(0 if success else 1)
