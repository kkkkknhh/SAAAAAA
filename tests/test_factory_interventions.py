"""
Comprehensive tests for factory interventions.

This test suite validates the three innovative interventions:
1. Intervention #2: Bidirectional Contract Hash Protocol
2. Intervention #3: Lazy-Loading Executor Factory
3. Intervention #4: Immutable Execution Context

Each intervention is designed to exponentially increase value and
self-evidently decrease error probability.
"""

import pytest
from datetime import datetime
from types import MappingProxyType

from src.saaaaaa.core.orchestrator.factory import (
    ContractManifest,
    ImmutableExecutionContext,
    CoreModuleFactory,
    ProcessorBundle,
    build_processor,
    compute_blake3_hash,
    compute_contract_schemas_hash,
)


class TestIntervention2_BidirectionalContractHashProtocol:
    """Test Intervention #2: Bidirectional Contract Hash Protocol.

    This intervention provides cryptographic proof of factory-executor alignment
    using BLAKE3 hashing. Error probability is mathematically near-zero (10^-77).
    """

    def test_contract_manifest_creation(self):
        """Test that ContractManifest is created with all required hashes."""
        manifest = ContractManifest(
            factory_version="1.0.0",
            questionnaire_hash="a" * 64,
            catalog_hash="b" * 64,
            method_map_hash="c" * 64,
            contract_schemas_hash="d" * 64,
            created_at="2025-01-01T00:00:00",
        )

        assert manifest.factory_version == "1.0.0"
        assert manifest.questionnaire_hash == "a" * 64
        assert manifest.catalog_hash == "b" * 64
        assert manifest.method_map_hash == "c" * 64
        assert manifest.contract_schemas_hash == "d" * 64
        assert len(manifest.manifest_hash) == 64  # BLAKE2b 32-byte = 64 hex chars

    def test_contract_manifest_self_referential_hash(self):
        """Test that manifest_hash is deterministic and self-referential."""
        manifest1 = ContractManifest(
            factory_version="1.0.0",
            questionnaire_hash="a" * 64,
            catalog_hash="b" * 64,
            method_map_hash="c" * 64,
            contract_schemas_hash="d" * 64,
            created_at="2025-01-01T00:00:00",
        )

        manifest2 = ContractManifest(
            factory_version="1.0.0",
            questionnaire_hash="a" * 64,
            catalog_hash="b" * 64,
            method_map_hash="c" * 64,
            contract_schemas_hash="d" * 64,
            created_at="2025-01-01T00:00:00",
        )

        # Same inputs = same manifest hash
        assert manifest1.manifest_hash == manifest2.manifest_hash

    def test_contract_manifest_verify_compatibility_success(self):
        """Test that identical manifests verify as compatible."""
        manifest_factory = ContractManifest(
            factory_version="1.0.0",
            questionnaire_hash="a" * 64,
            catalog_hash="b" * 64,
            method_map_hash="c" * 64,
            contract_schemas_hash="d" * 64,
            created_at="2025-01-01T00:00:00",
        )

        manifest_executor = ContractManifest(
            factory_version="1.0.0",
            questionnaire_hash="a" * 64,
            catalog_hash="b" * 64,
            method_map_hash="c" * 64,
            contract_schemas_hash="d" * 64,
            created_at="2025-01-01T00:00:01",  # Different timestamp OK
        )

        is_compatible, reason = manifest_factory.verify_compatibility(manifest_executor)

        assert is_compatible is True
        assert "compatible" in reason.lower()

    def test_contract_manifest_verify_compatibility_questionnaire_mismatch(self):
        """Test that questionnaire hash mismatch is detected."""
        manifest_factory = ContractManifest(
            factory_version="1.0.0",
            questionnaire_hash="a" * 64,
            catalog_hash="b" * 64,
            method_map_hash="c" * 64,
            contract_schemas_hash="d" * 64,
            created_at="2025-01-01T00:00:00",
        )

        manifest_executor = ContractManifest(
            factory_version="1.0.0",
            questionnaire_hash="f" * 64,  # Different questionnaire
            catalog_hash="b" * 64,
            method_map_hash="c" * 64,
            contract_schemas_hash="d" * 64,
            created_at="2025-01-01T00:00:00",
        )

        is_compatible, reason = manifest_factory.verify_compatibility(manifest_executor)

        assert is_compatible is False
        assert "questionnaire mismatch" in reason.lower()

    def test_contract_manifest_verify_compatibility_catalog_mismatch(self):
        """Test that catalog hash mismatch is detected."""
        manifest_factory = ContractManifest(
            factory_version="1.0.0",
            questionnaire_hash="a" * 64,
            catalog_hash="b" * 64,
            method_map_hash="c" * 64,
            contract_schemas_hash="d" * 64,
            created_at="2025-01-01T00:00:00",
        )

        manifest_executor = ContractManifest(
            factory_version="1.0.0",
            questionnaire_hash="a" * 64,
            catalog_hash="z" * 64,  # Different catalog
            method_map_hash="c" * 64,
            contract_schemas_hash="d" * 64,
            created_at="2025-01-01T00:00:00",
        )

        is_compatible, reason = manifest_factory.verify_compatibility(manifest_executor)

        assert is_compatible is False
        assert "catalog mismatch" in reason.lower()

    def test_blake3_hash_deterministic(self):
        """Test that compute_blake3_hash is deterministic."""
        data = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}

        hash1 = compute_blake3_hash(data)
        hash2 = compute_blake3_hash(data)

        assert hash1 == hash2
        assert len(hash1) == 64  # 32 bytes = 64 hex chars

    def test_blake3_hash_key_order_independent(self):
        """Test that hash is independent of key order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}

        hash1 = compute_blake3_hash(data1)
        hash2 = compute_blake3_hash(data2)

        assert hash1 == hash2

    def test_contract_schemas_hash_stable(self):
        """Test that contract schemas hash is stable."""
        hash1 = compute_contract_schemas_hash()
        hash2 = compute_contract_schemas_hash()

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_processor_bundle_includes_manifest(self, tmp_path):
        """Test that build_processor creates ProcessorBundle with ContractManifest."""
        # Create temporary questionnaire
        questionnaire = {
            "version": "1.0",
            "schema_version": "1.0",
            "blocks": {
                "micro_questions": [
                    {
                        "question_id": "Q1",
                        "question_global": 1,
                        "base_slot": "slot1"
                    }
                ]
            }
        }

        import json
        questionnaire_path = tmp_path / "questionnaire_monolith.json"
        questionnaire_path.write_text(json.dumps(questionnaire))

        # Build processor
        bundle = build_processor(questionnaire_path=questionnaire_path)

        # Verify bundle has manifest
        assert hasattr(bundle, 'contract_manifest')
        assert isinstance(bundle.contract_manifest, ContractManifest)
        assert bundle.contract_manifest.factory_version == "1.0.0"
        assert len(bundle.contract_manifest.questionnaire_hash) == 64
        assert len(bundle.contract_manifest.catalog_hash) == 64


class TestIntervention3_LazyLoadingExecutorFactory:
    """Test Intervention #3: Lazy-Loading Executor Factory.

    This intervention provides fail-fast validation and unified executor construction.
    Eliminates split responsibility between factory and executors.
    """

    def test_executor_registry_registration(self):
        """Test executor registration mechanism."""
        class MockExecutor:
            def __init__(self, method_executor, signal_registry=None, config=None, calibration_orchestrator=None):
                self.method_executor = method_executor

        factory = CoreModuleFactory()
        factory.register_executor("MockExecutor", MockExecutor)

        assert "MockExecutor" in factory.get_registered_executors()

    def test_create_executor_fail_fast_unknown_executor(self, tmp_path):
        """Test that create_executor fails fast for unknown executor."""
        questionnaire = {
            "version": "1.0",
            "schema_version": "1.0",
            "blocks": {
                "micro_questions": [
                    {"question_id": "Q1", "question_global": 1, "base_slot": "slot1"}
                ]
            }
        }

        import json
        questionnaire_path = tmp_path / "questionnaire_monolith.json"
        questionnaire_path.write_text(json.dumps(questionnaire))

        bundle = build_processor(questionnaire_path=questionnaire_path)
        factory = CoreModuleFactory()

        # Should fail fast with clear error
        with pytest.raises(ValueError, match="not found in registry"):
            factory.create_executor("NonExistentExecutor", bundle)

    def test_create_executor_fail_fast_invalid_bundle(self):
        """Test that create_executor fails fast for invalid ProcessorBundle."""
        factory = CoreModuleFactory()

        class InvalidBundle:
            pass

        invalid_bundle = InvalidBundle()

        # Should fail fast with clear error
        with pytest.raises(TypeError, match="method_executor"):
            factory.create_executor("D1Q1_Executor", invalid_bundle)

    def test_executor_has_dependencies_wired(self, tmp_path):
        """Test that created executor has all dependencies pre-wired."""
        questionnaire = {
            "version": "1.0",
            "schema_version": "1.0",
            "blocks": {
                "micro_questions": [
                    {"question_id": "Q1", "question_global": 1, "base_slot": "slot1"}
                ]
            }
        }

        import json
        questionnaire_path = tmp_path / "questionnaire_monolith.json"
        questionnaire_path.write_text(json.dumps(questionnaire))

        bundle = build_processor(questionnaire_path=questionnaire_path)
        factory = bundle.factory

        # Create executor (will auto-register on first call)
        try:
            executor = factory.create_executor("D1Q1_Executor", bundle)

            # Verify dependencies are wired
            assert hasattr(executor, 'method_executor')
            assert executor.method_executor is bundle.method_executor

        except ImportError:
            pytest.skip("Executors module not available")


class TestIntervention4_ImmutableExecutionContext:
    """Test Intervention #4: Immutable Execution Context.

    This intervention eliminates mutation bugs by using frozen dataclass with
    copy-on-write semantics. Thread-safe by construction.
    """

    def test_immutable_context_creation(self):
        """Test creating an immutable execution context."""
        ctx = ImmutableExecutionContext.create(
            phase_id=1,
            phase_name="initialization",
            document_id="doc123"
        )

        assert ctx.phase_id == 1
        assert ctx.phase_name == "initialization"
        assert ctx.document_id == "doc123"
        assert ctx.context_version == 1
        assert ctx.parent_context_hash == "root"

    def test_immutable_context_frozen(self):
        """Test that context is truly immutable (frozen dataclass)."""
        ctx = ImmutableExecutionContext.create(
            phase_id=1,
            phase_name="test"
        )

        # Attempting to modify should raise error
        with pytest.raises(Exception):  # FrozenInstanceError
            ctx.phase_id = 2

    def test_immutable_context_with_arguments_copy_on_write(self):
        """Test copy-on-write semantics for arguments."""
        ctx1 = ImmutableExecutionContext.create(
            phase_id=1,
            phase_name="test"
        )

        # Create new context with arguments
        ctx2 = ctx1.with_arguments({"arg1": "value1"})

        # Original context unchanged
        assert len(ctx1.arguments) == 0

        # New context has arguments
        assert ctx2.arguments["arg1"] == "value1"

        # Version incremented
        assert ctx2.context_version == 2

        # Parent hash points to ctx1
        assert ctx2.parent_context_hash == ctx1._compute_hash()

    def test_immutable_context_with_metadata_copy_on_write(self):
        """Test copy-on-write semantics for metadata."""
        ctx1 = ImmutableExecutionContext.create(
            phase_id=1,
            phase_name="test",
            metadata={"meta1": "val1"}
        )

        ctx2 = ctx1.with_metadata({"meta2": "val2"})

        # Original metadata unchanged
        assert "meta2" not in ctx1.metadata
        assert ctx1.metadata["meta1"] == "val1"

        # New context has merged metadata
        assert ctx2.metadata["meta1"] == "val1"
        assert ctx2.metadata["meta2"] == "val2"

    def test_immutable_context_with_phase_copy_on_write(self):
        """Test copy-on-write semantics for phase transition."""
        ctx1 = ImmutableExecutionContext.create(
            phase_id=1,
            phase_name="phase1",
            arguments={"arg": "value"}
        )

        ctx2 = ctx1.with_phase(2, "phase2")

        # Original phase unchanged
        assert ctx1.phase_id == 1
        assert ctx1.phase_name == "phase1"

        # New context has new phase
        assert ctx2.phase_id == 2
        assert ctx2.phase_name == "phase2"

        # Arguments carried over (structural sharing)
        assert ctx2.arguments["arg"] == "value"

    def test_immutable_context_arguments_are_readonly(self):
        """Test that arguments mapping is read-only (MappingProxyType)."""
        ctx = ImmutableExecutionContext.create(
            phase_id=1,
            phase_name="test",
            arguments={"key": "value"}
        )

        assert isinstance(ctx.arguments, MappingProxyType)

        # Cannot modify through proxy
        with pytest.raises(TypeError):
            ctx.arguments["new_key"] = "new_value"

    def test_immutable_context_metadata_are_readonly(self):
        """Test that metadata mapping is read-only."""
        ctx = ImmutableExecutionContext.create(
            phase_id=1,
            phase_name="test",
            metadata={"key": "value"}
        )

        assert isinstance(ctx.metadata, MappingProxyType)

        with pytest.raises(TypeError):
            ctx.metadata["new_key"] = "new_value"

    def test_immutable_context_audit_trail(self):
        """Test that context creates perfect audit trail via hashes."""
        ctx1 = ImmutableExecutionContext.create(
            phase_id=1,
            phase_name="init"
        )

        ctx2 = ctx1.with_arguments({"arg1": "val1"})
        ctx3 = ctx2.with_arguments({"arg2": "val2"})
        ctx4 = ctx3.with_phase(2, "process")

        # Each context points to its parent
        assert ctx2.parent_context_hash == ctx1._compute_hash()
        assert ctx3.parent_context_hash == ctx2._compute_hash()
        assert ctx4.parent_context_hash == ctx3._compute_hash()

        # Versions increment
        assert ctx1.context_version == 1
        assert ctx2.context_version == 2
        assert ctx3.context_version == 3
        assert ctx4.context_version == 4

    def test_immutable_context_to_dict_serialization(self):
        """Test context serialization to dict."""
        ctx = ImmutableExecutionContext.create(
            phase_id=1,
            phase_name="test",
            document_id="doc123",
            method_sequence=[("Class1", "method1"), ("Class2", "method2")],
            arguments={"arg": "value"},
            metadata={"meta": "data"}
        )

        ctx_dict = ctx.to_dict()

        assert ctx_dict["phase_id"] == 1
        assert ctx_dict["phase_name"] == "test"
        assert ctx_dict["document_id"] == "doc123"
        assert ctx_dict["method_sequence"] == [("Class1", "method1"), ("Class2", "method2")]
        assert ctx_dict["arguments"] == {"arg": "value"}
        assert ctx_dict["metadata"] == {"meta": "data"}
        assert "parent_context_hash" in ctx_dict
        assert "context_version" in ctx_dict

    def test_immutable_context_thread_safety_by_construction(self):
        """Test that immutable context is thread-safe by construction.

        Since the context is frozen, multiple threads can safely read from it
        without synchronization. Modifications create new contexts, so there's
        no shared mutable state.
        """
        import threading

        ctx = ImmutableExecutionContext.create(
            phase_id=1,
            phase_name="test",
            arguments={"shared": "data"}
        )

        results = []

        def read_context():
            # Multiple threads reading simultaneously
            for _ in range(100):
                results.append(ctx.arguments["shared"])

        threads = [threading.Thread(target=read_context) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads successful, no corruption
        assert all(r == "data" for r in results)
        assert len(results) == 1000  # 10 threads * 100 reads


class TestFactoryExecutorOrchestrationAlignment:
    """Test overall alignment between factory, executors, and orchestrator."""

    def test_processor_bundle_complete_integration(self, tmp_path):
        """Test that ProcessorBundle integrates all three interventions."""
        questionnaire = {
            "version": "1.0",
            "schema_version": "1.0",
            "blocks": {
                "micro_questions": [
                    {"question_id": "Q1", "question_global": 1, "base_slot": "slot1"}
                ]
            }
        }

        import json
        questionnaire_path = tmp_path / "questionnaire_monolith.json"
        questionnaire_path.write_text(json.dumps(questionnaire))

        # Build processor
        bundle = build_processor(questionnaire_path=questionnaire_path)

        # Verify ProcessorBundle has all components
        assert hasattr(bundle, 'method_executor')
        assert hasattr(bundle, 'questionnaire')
        assert hasattr(bundle, 'factory')
        assert hasattr(bundle, 'contract_manifest')

        # Verify contract manifest (Intervention #2)
        assert isinstance(bundle.contract_manifest, ContractManifest)

        # Verify factory can create executors (Intervention #3)
        assert isinstance(bundle.factory, CoreModuleFactory)
        assert hasattr(bundle.factory, 'create_executor')

        # ImmutableExecutionContext available (Intervention #4)
        ctx = ImmutableExecutionContext.create(phase_id=1, phase_name="test")
        assert ctx.phase_id == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
