"""
Tests for CanonicalQuestionnaire and questionnaire integrity enforcement.

This test module validates:
1. CanonicalQuestionnaire loading and validation
2. Immutability enforcement via MappingProxyType
3. Hash verification
4. Structure validation
5. Provider integration
"""

import json
import hashlib
from pathlib import Path
from types import MappingProxyType
from collections import OrderedDict

import pytest


# Test data path
REPO_ROOT = Path(__file__).parent.parent
QUESTIONNAIRE_PATH = REPO_ROOT / "data" / "questionnaire_monolith.json"

# Expected values
EXPECTED_HASH = "f4a48932f6a3c408e65589680de334d54e69d4a43adb787bb91571788a91feb8"
EXPECTED_COUNT = 300


class TestCanonicalQuestionnaireStructure:
    """Test CanonicalQuestionnaire dataclass structure and validation."""
    
    def test_questionnaire_file_exists(self):
        """Verify questionnaire file exists at expected location."""
        assert QUESTIONNAIRE_PATH.exists(), f"Questionnaire not found at {QUESTIONNAIRE_PATH}"
    
    def test_questionnaire_hash(self):
        """Verify questionnaire file hash matches expected value."""
        raw_content = QUESTIONNAIRE_PATH.read_bytes()
        actual_hash = hashlib.sha256(raw_content).hexdigest()
        assert actual_hash == EXPECTED_HASH, (
            f"Questionnaire hash mismatch!\n"
            f"Expected: {EXPECTED_HASH}\n"
            f"Actual:   {actual_hash}\n"
            f"If this is a legitimate change, update EXPECTED_QUESTIONNAIRE_HASH in factory.py"
        )
    
    def test_questionnaire_question_count(self):
        """Verify questionnaire has exactly 300 micro questions."""
        content = QUESTIONNAIRE_PATH.read_text(encoding='utf-8')
        data = json.loads(content)
        question_count = len(data['blocks']['micro_questions'])
        assert question_count == EXPECTED_COUNT, (
            f"Expected {EXPECTED_COUNT} questions, got {question_count}"
        )
    
    def test_questionnaire_structure(self):
        """Verify questionnaire has required top-level structure."""
        content = QUESTIONNAIRE_PATH.read_text(encoding='utf-8')
        data = json.loads(content)
        
        # Check required keys
        assert 'version' in data, "Missing 'version' key"
        assert 'schema_version' in data, "Missing 'schema_version' key"
        assert 'blocks' in data, "Missing 'blocks' key"
        
        # Check blocks structure
        blocks = data['blocks']
        assert 'micro_questions' in blocks, "Missing 'micro_questions' in blocks"
        assert isinstance(blocks['micro_questions'], list), "micro_questions must be a list"


class TestCanonicalQuestionnaireLoading:
    """Test canonical questionnaire loading functionality."""
    
    def test_load_questionnaire_basic(self):
        """Test basic loading of canonical questionnaire."""
        try:
            from saaaaaa.core.orchestrator.factory import (
                load_questionnaire,
                CanonicalQuestionnaire,
            )
        except ImportError as e:
            pytest.skip(f"Cannot import factory module: {e}")
        
        # Load questionnaire
        q = load_questionnaire()
        
        # Verify type
        assert isinstance(q, CanonicalQuestionnaire), (
            f"Expected CanonicalQuestionnaire, got {type(q).__name__}"
        )
        
        # Verify attributes
        assert hasattr(q, 'data'), "Missing 'data' attribute"
        assert hasattr(q, 'sha256'), "Missing 'sha256' attribute"
        assert hasattr(q, 'micro_questions'), "Missing 'micro_questions' attribute"
        assert hasattr(q, 'question_count'), "Missing 'question_count' attribute"
        assert hasattr(q, 'version'), "Missing 'version' attribute"
        assert hasattr(q, 'schema_version'), "Missing 'schema_version' attribute"
    
    def test_canonical_questionnaire_hash(self):
        """Test that loaded questionnaire has correct hash."""
        try:
            from saaaaaa.core.orchestrator.factory import load_questionnaire
        except ImportError as e:
            pytest.skip(f"Cannot import factory module: {e}")
        
        q = load_questionnaire()
        assert q.sha256 == EXPECTED_HASH, (
            f"Hash mismatch: expected {EXPECTED_HASH}, got {q.sha256}"
        )
    
    def test_canonical_questionnaire_count(self):
        """Test that loaded questionnaire has correct question count."""
        try:
            from saaaaaa.core.orchestrator.factory import load_questionnaire
        except ImportError as e:
            pytest.skip(f"Cannot import factory module: {e}")
        
        q = load_questionnaire()
        assert q.question_count == EXPECTED_COUNT, (
            f"Question count mismatch: expected {EXPECTED_COUNT}, got {q.question_count}"
        )
    
    def test_canonical_questionnaire_version(self):
        """Test that loaded questionnaire has version information."""
        try:
            from saaaaaa.core.orchestrator.factory import load_questionnaire
        except ImportError as e:
            pytest.skip(f"Cannot import factory module: {e}")
        
        q = load_questionnaire()
        assert q.version is not None, "Version should not be None"
        assert q.schema_version is not None, "Schema version should not be None"


class TestCanonicalQuestionnaireImmutability:
    """Test immutability enforcement of CanonicalQuestionnaire."""
    
    def test_data_is_mapping_proxy(self):
        """Test that data is MappingProxyType (immutable)."""
        try:
            from saaaaaa.core.orchestrator.factory import load_questionnaire
        except ImportError as e:
            pytest.skip(f"Cannot import factory module: {e}")
        
        q = load_questionnaire()
        assert isinstance(q.data, MappingProxyType), (
            f"data must be MappingProxyType, got {type(q.data).__name__}"
        )
    
    def test_data_is_immutable(self):
        """Test that data cannot be modified."""
        try:
            from saaaaaa.core.orchestrator.factory import load_questionnaire
        except ImportError as e:
            pytest.skip(f"Cannot import factory module: {e}")
        
        q = load_questionnaire()
        
        with pytest.raises(TypeError):
            q.data['new_key'] = 'value'  # Should raise TypeError
    
    def test_micro_questions_is_tuple(self):
        """Test that micro_questions is a tuple (immutable)."""
        try:
            from saaaaaa.core.orchestrator.factory import load_questionnaire
        except ImportError as e:
            pytest.skip(f"Cannot import factory module: {e}")
        
        q = load_questionnaire()
        assert isinstance(q.micro_questions, tuple), (
            f"micro_questions must be tuple, got {type(q.micro_questions).__name__}"
        )
    
    def test_micro_questions_items_are_mapping_proxy(self):
        """Test that each micro question is MappingProxyType (immutable)."""
        try:
            from saaaaaa.core.orchestrator.factory import load_questionnaire
        except ImportError as e:
            pytest.skip(f"Cannot import factory module: {e}")
        
        q = load_questionnaire()
        
        for i, mq in enumerate(q.micro_questions[:5]):  # Check first 5
            assert isinstance(mq, MappingProxyType), (
                f"micro_questions[{i}] must be MappingProxyType, got {type(mq).__name__}"
            )
    
    def test_micro_questions_items_are_immutable(self):
        """Test that micro questions cannot be modified."""
        try:
            from saaaaaa.core.orchestrator.factory import load_questionnaire
        except ImportError as e:
            pytest.skip(f"Cannot import factory module: {e}")
        
        q = load_questionnaire()
        
        with pytest.raises(TypeError):
            q.micro_questions[0]['new_key'] = 'value'  # Should raise TypeError
    
    def test_canonical_questionnaire_is_frozen(self):
        """Test that CanonicalQuestionnaire dataclass is frozen."""
        try:
            from saaaaaa.core.orchestrator.factory import load_questionnaire
        except ImportError as e:
            pytest.skip(f"Cannot import factory module: {e}")
        
        q = load_questionnaire()
        
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            q.question_count = 999  # Should raise error


class TestQuestionnaireProvider:
    """Test questionnaire provider integration."""
    
    def test_provider_get_canonical(self):
        """Test that provider can return CanonicalQuestionnaire."""
        try:
            from saaaaaa.core.orchestrator import get_questionnaire_provider
            from saaaaaa.core.orchestrator.factory import CanonicalQuestionnaire
        except ImportError as e:
            pytest.skip(f"Cannot import provider module: {e}")
        
        provider = get_questionnaire_provider()
        
        # Get canonical should auto-load if needed
        q = provider.get_canonical()
        
        assert isinstance(q, CanonicalQuestionnaire), (
            f"Expected CanonicalQuestionnaire, got {type(q).__name__}"
        )
    
    def test_provider_set_canonical(self):
        """Test that provider can store CanonicalQuestionnaire."""
        try:
            from saaaaaa.core.orchestrator import get_questionnaire_provider
            from saaaaaa.core.orchestrator.factory import load_questionnaire
        except ImportError as e:
            pytest.skip(f"Cannot import provider module: {e}")
        
        provider = get_questionnaire_provider()
        q = load_questionnaire()
        
        # Set data
        provider.set_data(q)
        
        # Retrieve it
        retrieved = provider.get_data()
        assert retrieved is q, "Retrieved data should be the same instance"


class TestBackwardCompatibility:
    """Test backward compatibility with legacy dict-based loading."""
    
    def test_load_questionnaire_monolith_still_works(self):
        """Test that deprecated function still works for backward compatibility."""
        try:
            from saaaaaa.core.orchestrator.factory import load_questionnaire_monolith
        except ImportError as e:
            pytest.skip(f"Cannot import factory module: {e}")
        
        # This should work but log a warning
        data = load_questionnaire_monolith()
        
        # Should return dict
        assert isinstance(data, dict), (
            f"load_questionnaire_monolith should return dict, got {type(data).__name__}"
        )
        
        # Should have expected structure
        assert 'blocks' in data, "Missing 'blocks' key"
        assert 'micro_questions' in data['blocks'], "Missing 'micro_questions'"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
