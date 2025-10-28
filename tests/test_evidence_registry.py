"""Tests for evidence registry with JSONL storage and provenance DAG."""
import sys
from pathlib import Path
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.evidence_registry import (
    EvidenceRecord,
    ProvenanceNode,
    ProvenanceDAG,
    EvidenceRegistry,
    get_global_registry,
)


def test_evidence_record_creation():
    """Test creating evidence records."""
    evidence = EvidenceRecord(
        evidence_id="test-id",
        evidence_type="analysis",
        payload={"result": "data"},
        source_method="TestClass.test_method",
        question_id="D1-Q1",
    )
    
    assert evidence.evidence_id == "test-id"
    assert evidence.evidence_type == "analysis"
    assert evidence.payload == {"result": "data"}
    assert evidence.source_method == "TestClass.test_method"
    assert evidence.question_id == "D1-Q1"
    assert evidence.content_hash is not None
    
    print("✓ Evidence record creation works")


def test_content_hash_computation():
    """Test content hash computation for evidence."""
    payload = {"key": "value", "number": 42}
    
    evidence1 = EvidenceRecord(
        evidence_id="",
        evidence_type="test",
        payload=payload,
    )
    
    evidence2 = EvidenceRecord(
        evidence_id="",
        evidence_type="test",
        payload=payload,
    )
    
    # Same payload should produce same hash
    assert evidence1.content_hash == evidence2.content_hash
    
    # Different payload should produce different hash
    evidence3 = EvidenceRecord(
        evidence_id="",
        evidence_type="test",
        payload={"different": "data"},
    )
    
    assert evidence1.content_hash != evidence3.content_hash
    
    print("✓ Content hash computation works")


def test_evidence_integrity_verification():
    """Test evidence integrity verification."""
    evidence = EvidenceRecord(
        evidence_id="",
        evidence_type="test",
        payload={"data": "value"},
    )
    
    # Should verify successfully
    assert evidence.verify_integrity() is True
    
    # Tamper with payload
    evidence.payload["data"] = "tampered"
    
    # Should fail verification
    assert evidence.verify_integrity() is False
    
    print("✓ Evidence integrity verification works")


def test_evidence_serialization():
    """Test evidence record serialization."""
    evidence = EvidenceRecord(
        evidence_id="test-id",
        evidence_type="analysis",
        payload={"result": "data"},
        source_method="TestClass.method",
    )
    
    # To dict
    evidence_dict = evidence.to_dict()
    assert evidence_dict["evidence_id"] == "test-id"
    assert evidence_dict["payload"] == {"result": "data"}
    
    # From dict
    restored = EvidenceRecord.from_dict(evidence_dict)
    assert restored.evidence_id == evidence.evidence_id
    assert restored.payload == evidence.payload
    
    print("✓ Evidence serialization works")


def test_provenance_dag_creation():
    """Test creating provenance DAG."""
    dag = ProvenanceDAG()
    
    # Create evidence chain
    evidence1 = EvidenceRecord(
        evidence_id="e1",
        evidence_type="extraction",
        payload={"data": "raw"},
    )
    
    evidence2 = EvidenceRecord(
        evidence_id="e2",
        evidence_type="analysis",
        payload={"data": "processed"},
        parent_evidence_ids=["e1"],
    )
    
    # Add to DAG
    dag.add_evidence(evidence1)
    dag.add_evidence(evidence2)
    
    assert len(dag.nodes) == 2
    assert "e1" in dag.nodes
    assert "e2" in dag.nodes
    
    # Check parent-child relationship
    assert "e2" in dag.nodes["e1"].children
    assert "e1" in dag.nodes["e2"].parents
    
    print("✓ Provenance DAG creation works")


def test_provenance_lineage_tracking():
    """Test lineage tracking in DAG."""
    dag = ProvenanceDAG()
    
    # Create evidence chain: e1 -> e2 -> e3
    for i in range(1, 4):
        evidence = EvidenceRecord(
            evidence_id=f"e{i}",
            evidence_type="test",
            payload={"step": i},
            parent_evidence_ids=[f"e{i-1}"] if i > 1 else [],
        )
        dag.add_evidence(evidence)
    
    # Get ancestors of e3
    ancestors = dag.get_ancestors("e3")
    assert "e1" in ancestors
    assert "e2" in ancestors
    assert len(ancestors) == 2
    
    # Get descendants of e1
    descendants = dag.get_descendants("e1")
    assert "e2" in descendants
    assert "e3" in descendants
    assert len(descendants) == 2
    
    # Get lineage
    lineage = dag.get_lineage("e2")
    assert lineage["evidence_id"] == "e2"
    assert lineage["ancestor_count"] == 1
    assert lineage["descendant_count"] == 1
    
    print("✓ Provenance lineage tracking works")


def test_provenance_dag_export_dot():
    """Test exporting DAG to DOT format."""
    dag = ProvenanceDAG()
    
    evidence1 = EvidenceRecord(
        evidence_id="e1",
        evidence_type="extraction",
        payload={},
        source_method="Extract.method",
    )
    
    evidence2 = EvidenceRecord(
        evidence_id="e2",
        evidence_type="analysis",
        payload={},
        parent_evidence_ids=["e1"],
    )
    
    dag.add_evidence(evidence1)
    dag.add_evidence(evidence2)
    
    # Export to DOT
    dot = dag.export_dot()
    
    assert "digraph ProvenanceDAG" in dot
    assert '"e1"' in dot
    assert '"e2"' in dot
    assert '"e1" -> "e2"' in dot
    
    print("✓ Provenance DAG DOT export works")


def test_evidence_registry_initialization():
    """Test evidence registry initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_evidence.jsonl"
        
        registry = EvidenceRegistry(
            storage_path=storage_path,
            enable_dag=True,
        )
        
        assert registry.storage_path == storage_path
        assert registry.enable_dag is True
        assert len(registry.hash_index) == 0
        assert registry.dag is not None
        
        print("✓ Evidence registry initialization works")


def test_evidence_recording():
    """Test recording evidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_evidence.jsonl"
        
        registry = EvidenceRegistry(storage_path=storage_path)
        
        # Record evidence
        evidence_id = registry.record_evidence(
            evidence_type="analysis",
            payload={"result": "value"},
            source_method="TestClass.analyze",
            question_id="D1-Q1",
        )
        
        assert evidence_id is not None
        assert len(evidence_id) == 64  # SHA-256 hex length
        
        # Check indexed
        assert evidence_id in registry.hash_index
        assert evidence_id in registry.type_index["analysis"]
        assert evidence_id in registry.method_index["TestClass.analyze"]
        assert evidence_id in registry.question_index["D1-Q1"]
        
        # Check persisted to storage
        assert storage_path.exists()
        with open(storage_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
        
        print("✓ Evidence recording works")


def test_evidence_retrieval():
    """Test retrieving evidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = EvidenceRegistry(storage_path=Path(tmpdir) / "test.jsonl")
        
        # Record evidence
        payload = {"data": "test"}
        evidence_id = registry.record_evidence(
            evidence_type="test",
            payload=payload,
        )
        
        # Retrieve evidence
        retrieved = registry.get_evidence(evidence_id)
        
        assert retrieved is not None
        assert retrieved.evidence_id == evidence_id
        assert retrieved.payload == payload
        
        print("✓ Evidence retrieval works")


def test_evidence_queries():
    """Test querying evidence by type, method, question."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = EvidenceRegistry(storage_path=Path(tmpdir) / "test.jsonl")
        
        # Record various evidence
        registry.record_evidence(
            evidence_type="extraction",
            payload={"data": "1"},
            source_method="Extract.method",
            question_id="D1-Q1",
        )
        
        registry.record_evidence(
            evidence_type="extraction",
            payload={"data": "2"},
            source_method="Extract.method",
            question_id="D1-Q1",
        )
        
        registry.record_evidence(
            evidence_type="analysis",
            payload={"data": "3"},
            source_method="Analyze.method",
            question_id="D1-Q2",
        )
        
        # Query by type
        extractions = registry.query_by_type("extraction")
        assert len(extractions) == 2
        
        # Query by method
        extract_evidence = registry.query_by_method("Extract.method")
        assert len(extract_evidence) == 2
        
        # Query by question
        q1_evidence = registry.query_by_question("D1-Q1")
        assert len(q1_evidence) == 2
        
        print("✓ Evidence queries work")


def test_evidence_verification():
    """Test evidence verification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = EvidenceRegistry(storage_path=Path(tmpdir) / "test.jsonl")
        
        # Record evidence
        evidence_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "value"},
        )
        
        # Verify evidence
        assert registry.verify_evidence(evidence_id) is True
        
        # Tamper with evidence (simulate corruption)
        evidence = registry.hash_index[evidence_id]
        evidence.payload["data"] = "tampered"
        
        # Verification should fail
        assert registry.verify_evidence(evidence_id) is False
        
        print("✓ Evidence verification works")


def test_provenance_retrieval():
    """Test retrieving provenance for evidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = EvidenceRegistry(
            storage_path=Path(tmpdir) / "test.jsonl",
            enable_dag=True,
        )
        
        # Record evidence chain
        e1_id = registry.record_evidence(
            evidence_type="extraction",
            payload={"data": "raw"},
        )
        
        e2_id = registry.record_evidence(
            evidence_type="analysis",
            payload={"data": "processed"},
            parent_evidence_ids=[e1_id],
        )
        
        # Get provenance
        provenance = registry.get_provenance(e2_id)
        
        assert provenance is not None
        assert provenance["evidence_id"] == e2_id
        assert e1_id in provenance["ancestors"]
        assert provenance["ancestor_count"] == 1
        
        print("✓ Provenance retrieval works")


def test_dag_export():
    """Test exporting provenance DAG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = EvidenceRegistry(
            storage_path=Path(tmpdir) / "test.jsonl",
            enable_dag=True,
        )
        
        # Record evidence
        e1_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "1"},
        )
        
        e2_id = registry.record_evidence(
            evidence_type="test",
            payload={"data": "2"},
            parent_evidence_ids=[e1_id],
        )
        
        # Export as dict
        dag_dict = registry.export_provenance_dag(format="dict")
        assert "nodes" in dag_dict
        assert "stats" in dag_dict
        assert len(dag_dict["nodes"]) == 2
        
        # Export as DOT
        dag_dot = registry.export_provenance_dag(format="dot")
        assert "digraph" in dag_dot
        
        # Export to file
        output_path = Path(tmpdir) / "provenance.json"
        registry.export_provenance_dag(format="json", output_path=output_path)
        assert output_path.exists()
        
        print("✓ DAG export works")


def test_registry_persistence():
    """Test that evidence persists across registry instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test.jsonl"
        
        # Create first registry and record evidence
        registry1 = EvidenceRegistry(storage_path=storage_path)
        evidence_id = registry1.record_evidence(
            evidence_type="test",
            payload={"data": "persistent"},
            source_method="Test.method",
        )
        
        # Create second registry (should load from storage)
        registry2 = EvidenceRegistry(storage_path=storage_path)
        
        # Evidence should be loaded
        assert evidence_id in registry2.hash_index
        retrieved = registry2.get_evidence(evidence_id)
        assert retrieved.payload == {"data": "persistent"}
        
        print("✓ Registry persistence works")


def test_registry_statistics():
    """Test registry statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = EvidenceRegistry(storage_path=Path(tmpdir) / "test.jsonl")
        
        # Record some evidence
        registry.record_evidence(
            evidence_type="extraction",
            payload={"data": "1"},
            source_method="Extract.method",
        )
        
        registry.record_evidence(
            evidence_type="analysis",
            payload={"data": "2"},
            source_method="Analyze.method",
        )
        
        # Get statistics
        stats = registry.get_statistics()
        
        assert stats["total_evidence"] == 2
        assert stats["by_type"]["extraction"] == 1
        assert stats["by_type"]["analysis"] == 1
        assert stats["by_method"]["Extract.method"] == 1
        assert stats["dag_enabled"] is True
        
        print("✓ Registry statistics work")


def test_global_registry():
    """Test global registry singleton."""
    registry1 = get_global_registry()
    registry2 = get_global_registry()
    
    assert registry1 is registry2
    
    print("✓ Global registry singleton works")


if __name__ == "__main__":
    print("Running evidence registry tests...\n")
    
    try:
        test_evidence_record_creation()
        test_content_hash_computation()
        test_evidence_integrity_verification()
        test_evidence_serialization()
        test_provenance_dag_creation()
        test_provenance_lineage_tracking()
        test_provenance_dag_export_dot()
        test_evidence_registry_initialization()
        test_evidence_recording()
        test_evidence_retrieval()
        test_evidence_queries()
        test_evidence_verification()
        test_provenance_retrieval()
        test_dag_export()
        test_registry_persistence()
        test_registry_statistics()
        test_global_registry()
        
        print("\n✅ All evidence registry tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
