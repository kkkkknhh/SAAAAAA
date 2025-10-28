"""Tests for hash chain integrity verification."""
import sys
from pathlib import Path
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.evidence_registry import (
    EvidenceRecord,
    EvidenceRegistry,
)


def test_hash_chain_creation():
    """Test that hash chain is properly created when recording evidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = EvidenceRegistry(storage_path=Path(tmpdir) / "test.jsonl")
        
        # Record first evidence (should have no previous_hash)
        e1_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "first"},
        )
        
        e1 = registry.get_evidence(e1_id)
        assert e1 is not None
        assert e1.previous_hash is None  # First entry has no predecessor
        assert e1.entry_hash is not None
        
        # Record second evidence (should have previous_hash from e1)
        e2_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "second"},
        )
        
        e2 = registry.get_evidence(e2_id)
        assert e2 is not None
        assert e2.previous_hash == e1.entry_hash  # Should link to e1
        assert e2.entry_hash is not None
        assert e2.entry_hash != e1.entry_hash  # Different entries have different hashes
        
        # Record third evidence (should link to e2)
        e3_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "third"},
        )
        
        e3 = registry.get_evidence(e3_id)
        assert e3 is not None
        assert e3.previous_hash == e2.entry_hash  # Should link to e2
        
        print("✓ Hash chain creation works")


def test_chain_verification_with_valid_chain():
    """Test that chain verification passes for valid chain."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = EvidenceRegistry(storage_path=Path(tmpdir) / "test.jsonl")
        
        # Create a chain of evidence
        for i in range(5):
            registry.record_evidence(
                evidence_type="test",
                payload={"data": f"entry-{i}"},
            )
        
        # Verify chain integrity
        is_valid, errors = registry.verify_chain_integrity()
        assert is_valid, f"Chain should be valid but got errors: {errors}"
        assert len(errors) == 0
        
        print("✓ Chain verification passes for valid chain")


def test_chain_verification_detects_previous_hash_tampering():
    """Test that verification detects when previous_hash is tampered."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test.jsonl"
        registry = EvidenceRegistry(storage_path=storage_path)
        
        # Create a chain
        e1_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "first"},
        )
        
        e2_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "second"},
        )
        
        # Tamper with the previous_hash in the stored file
        # Read all lines
        with open(storage_path, 'r') as f:
            lines = f.readlines()
        
        # Modify the second entry's previous_hash
        if len(lines) >= 2:
            data = json.loads(lines[1])
            data['previous_hash'] = "tampered_hash_value"
            lines[1] = json.dumps(data) + '\n'
        
        # Write back the tampered data
        with open(storage_path, 'w') as f:
            f.writelines(lines)
        
        # Create new registry to load tampered data
        registry2 = EvidenceRegistry(storage_path=storage_path)
        
        # Verify chain should detect the tampering
        is_valid, errors = registry2.verify_chain_integrity()
        assert not is_valid, "Chain verification should detect previous_hash tampering"
        assert len(errors) > 0
        assert "previous_hash mismatch" in errors[0] or "Chain broken" in errors[0]
        
        print("✓ Chain verification detects previous_hash tampering")


def test_chain_verification_detects_entry_hash_corruption():
    """Test that verification detects when entry data is corrupted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test.jsonl"
        registry = EvidenceRegistry(storage_path=storage_path)
        
        # Create a chain
        e1_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "first"},
        )
        
        e2_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "second"},
        )
        
        # Tamper with payload but keep hashes - this should fail content hash check
        with open(storage_path, 'r') as f:
            lines = f.readlines()
        
        # Modify the first entry's payload
        if len(lines) >= 1:
            data = json.loads(lines[0])
            data['payload']['data'] = "tampered"  # Change payload
            # Don't update hashes - this simulates corruption
            lines[0] = json.dumps(data) + '\n'
        
        # Write back
        with open(storage_path, 'w') as f:
            f.writelines(lines)
        
        # Create new registry
        registry2 = EvidenceRegistry(storage_path=storage_path)
        
        # Verify chain should detect corruption
        is_valid, errors = registry2.verify_chain_integrity()
        assert not is_valid, "Chain verification should detect payload corruption"
        assert len(errors) > 0
        assert "Hash integrity check failed" in errors[0] or "previous_hash mismatch" in errors[0]
        
        print("✓ Chain verification detects entry hash corruption")


def test_individual_evidence_verification_with_chain():
    """Test verifying individual evidence with chain linkage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = EvidenceRegistry(storage_path=Path(tmpdir) / "test.jsonl")
        
        # Create chain
        e1_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "first"},
        )
        
        e2_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "second"},
        )
        
        # Verify e1 (no previous)
        assert registry.verify_evidence(e1_id, verify_chain=True)
        
        # Verify e2 with chain
        assert registry.verify_evidence(e2_id, verify_chain=True)
        
        # Tamper with e2's previous_hash in memory
        e2 = registry.get_evidence(e2_id)
        original_previous_hash = e2.previous_hash
        e2.previous_hash = "tampered"
        
        # Verification should fail
        assert not registry.verify_evidence(e2_id, verify_chain=True)
        
        # Restore and verify again
        e2.previous_hash = original_previous_hash
        assert registry.verify_evidence(e2_id, verify_chain=True)
        
        print("✓ Individual evidence verification with chain works")


def test_entry_hash_changes_with_previous_hash():
    """Test that entry_hash changes when previous_hash changes."""
    # Create two evidence with same payload but different previous_hash
    e1 = EvidenceRecord(
        evidence_id="test1",
        evidence_type="test",
        payload={"data": "same"},
        previous_hash=None,
    )
    
    e2 = EvidenceRecord(
        evidence_id="test2",
        evidence_type="test",
        payload={"data": "same"},
        previous_hash="some_hash",
    )
    
    # Content hash should be the same (same payload)
    assert e1.content_hash == e2.content_hash
    
    # But entry hash should be different (different previous_hash)
    assert e1.entry_hash != e2.entry_hash
    
    print("✓ Entry hash changes with previous_hash")


def test_backward_compatibility_with_old_records():
    """Test that old records without previous_hash/entry_hash still work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test.jsonl"
        
        # Create an old-style record without previous_hash and entry_hash
        old_record = {
            "evidence_id": "old-id",
            "evidence_type": "test",
            "payload": {"data": "old"},
            "source_method": None,
            "parent_evidence_ids": [],
            "question_id": None,
            "document_id": None,
            "timestamp": 1234567890.0,
            "execution_time_ms": 0.0,
            "content_hash": None,
            "metadata": {},
        }
        
        # Write old record to file
        with open(storage_path, 'w') as f:
            f.write(json.dumps(old_record) + '\n')
        
        # Load registry - should handle old records gracefully
        registry = EvidenceRegistry(storage_path=storage_path)
        
        # Should have loaded the record
        assert len(registry.hash_index) == 1
        
        # Now add a new record - should create proper chain
        new_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "new"},
        )
        
        new_record = registry.get_evidence(new_id)
        # New record should have previous_hash and entry_hash
        assert new_record.previous_hash is not None
        assert new_record.entry_hash is not None
        
        print("✓ Backward compatibility with old records works")


if __name__ == "__main__":
    print("Running hash chain integrity tests...\n")
    
    try:
        test_hash_chain_creation()
        test_chain_verification_with_valid_chain()
        test_chain_verification_detects_previous_hash_tampering()
        test_chain_verification_detects_entry_hash_corruption()
        test_individual_evidence_verification_with_chain()
        test_entry_hash_changes_with_previous_hash()
        test_backward_compatibility_with_old_records()
        
        print("\n✅ All hash chain integrity tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
