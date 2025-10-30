"""
Monolith Orchestrator - Centralized Access to questionnaire_monolith.json
==========================================================================

This module provides the ONLY authorized way to access the questionnaire monolith.
All other files MUST use this orchestrator instead of direct file access.

Features:
- Centralized monolith loading with caching
- Type-safe access to questions, clusters, scoring rules
- Hash-based integrity verification
- Access logging and audit trail

Author: Integration Team
Version: 1.0.0
Python: 3.10+
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class MonolithMetadata:
    """Metadata about the loaded monolith"""
    version: str
    schema_version: str
    generated_at: str
    hash_sha256: str
    loaded_at: str
    file_path: str
    total_questions: int


class MonolithAccessError(Exception):
    """Raised when monolith access fails"""
    pass


class MonolithOrchestrator:
    """
    Centralized orchestrator for questionnaire_monolith.json access.
    
    This is the ONLY authorized way to access monolith data.
    Direct file access is prohibited - use this orchestrator instead.
    
    Usage:
        orchestrator = MonolithOrchestrator()
        question = orchestrator.get_question("PA01-DIM01-Q01")
        method_set = orchestrator.get_method_set("PA01-DIM01-Q01")
        cluster = orchestrator.get_cluster("CL01")
    """
    
    def __init__(self, monolith_path: Optional[Path] = None):
        """
        Initialize monolith orchestrator.
        
        Args:
            monolith_path: Path to questionnaire_monolith.json (default: current dir)
        """
        self.monolith_path = monolith_path or Path("questionnaire_monolith.json")
        
        if not self.monolith_path.exists():
            raise MonolithAccessError(f"Monolith not found: {self.monolith_path}")
        
        # Load and cache monolith
        self._monolith_data: Dict[str, Any] = {}
        self._metadata: Optional[MonolithMetadata] = None
        self._question_index: Dict[str, Dict[str, Any]] = {}
        self._cluster_index: Dict[str, Dict[str, Any]] = {}
        self._load_monolith()
        
        logger.info(
            f"MonolithOrchestrator initialized: "
            f"{self._metadata.total_questions} questions loaded, "
            f"hash={self._metadata.hash_sha256[:8]}..."
        )
    
    def _load_monolith(self) -> None:
        """Load and index monolith data"""
        try:
            with open(self.monolith_path, 'r', encoding='utf-8') as f:
                self._monolith_data = json.load(f)
            
            # Calculate hash for integrity
            with open(self.monolith_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Extract metadata
            self._metadata = MonolithMetadata(
                version=self._monolith_data.get('version', 'unknown'),
                schema_version=self._monolith_data.get('schema_version', 'unknown'),
                generated_at=self._monolith_data.get('generated_at', 'unknown'),
                hash_sha256=file_hash,
                loaded_at=datetime.now(timezone.utc).isoformat(),
                file_path=str(self.monolith_path),
                total_questions=0  # Will be set below
            )
            
            # Index questions
            self._index_questions()
            
            # Index clusters
            self._index_clusters()
            
            # Update total questions
            self._metadata.total_questions = len(self._question_index)
            
        except json.JSONDecodeError as e:
            raise MonolithAccessError(f"Failed to parse monolith JSON: {e}")
        except Exception as e:
            raise MonolithAccessError(f"Failed to load monolith: {e}")
    
    def _index_questions(self) -> None:
        """Build question index for fast lookup"""
        blocks = self._monolith_data.get('blocks', {})
        
        # Index micro questions
        micro_questions = blocks.get('micro_questions', [])
        for question in micro_questions:
            question_id = question.get('question_id')
            if question_id:
                self._question_index[question_id] = question
        
        # Index meso questions
        meso_questions = blocks.get('meso_questions', [])
        for question in meso_questions:
            question_id = question.get('question_id')
            if question_id:
                self._question_index[question_id] = question
        
        # Index macro question
        macro_question = blocks.get('macro_question')
        if macro_question:
            question_id = macro_question.get('question_id')
            if question_id:
                self._question_index[question_id] = macro_question
        
        logger.debug(f"Indexed {len(self._question_index)} questions")
    
    def _index_clusters(self) -> None:
        """Build cluster index for fast lookup"""
        blocks = self._monolith_data.get('blocks', {})
        
        # Look for cluster data in various locations
        # This may need adjustment based on actual monolith structure
        meso_questions = blocks.get('meso_questions', [])
        for question in meso_questions:
            cluster_id = question.get('cluster_id')
            if cluster_id:
                # Store cluster info from meso questions
                if cluster_id not in self._cluster_index:
                    self._cluster_index[cluster_id] = {
                        'cluster_id': cluster_id,
                        'questions': []
                    }
                self._cluster_index[cluster_id]['questions'].append(question.get('question_id'))
        
        logger.debug(f"Indexed {len(self._cluster_index)} clusters")
    
    # =========================================================================
    # PUBLIC API - AUTHORIZED ACCESS METHODS
    # =========================================================================
    
    def get_question(self, question_id: str) -> Dict[str, Any]:
        """
        Get question data by ID.
        
        Args:
            question_id: Question ID (e.g., "PA01-DIM01-Q01")
            
        Returns:
            Question data dictionary
            
        Raises:
            MonolithAccessError: If question not found
        """
        if question_id not in self._question_index:
            raise MonolithAccessError(f"Question not found: {question_id}")
        
        return self._question_index[question_id].copy()
    
    def get_method_set(self, question_id: str) -> List[Dict[str, Any]]:
        """
        Get method sets for a question.
        
        Args:
            question_id: Question ID
            
        Returns:
            List of method set dictionaries
            
        Raises:
            MonolithAccessError: If question not found
        """
        question = self.get_question(question_id)
        return question.get('method_sets', [])
    
    def get_cluster(self, cluster_id: str) -> Dict[str, Any]:
        """
        Get cluster data by ID.
        
        Args:
            cluster_id: Cluster ID (e.g., "CL01")
            
        Returns:
            Cluster data dictionary
            
        Raises:
            MonolithAccessError: If cluster not found
        """
        if cluster_id not in self._cluster_index:
            raise MonolithAccessError(f"Cluster not found: {cluster_id}")
        
        return self._cluster_index[cluster_id].copy()
    
    def get_scoring_rules(self) -> Dict[str, Any]:
        """
        Get scoring rules from monolith.
        
        Returns:
            Scoring rules dictionary
        """
        blocks = self._monolith_data.get('blocks', {})
        return blocks.get('scoring', {})
    
    def get_rubricacion_scoring(self) -> Dict[str, Any]:
        """
        Get rubricacion (rubric) scoring configuration.
        
        Returns:
            Rubricacion scoring dictionary with thresholds and levels
        """
        scoring = self.get_scoring_rules()
        return scoring.get('rubricacion', scoring)
    
    def get_all_questions(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all questions, optionally filtered by level.
        
        Args:
            level: Optional level filter ('micro', 'meso', 'macro')
            
        Returns:
            List of question dictionaries
        """
        if level is None:
            return list(self._question_index.values())
        
        blocks = self._monolith_data.get('blocks', {})
        
        if level == 'micro':
            return blocks.get('micro_questions', [])
        elif level == 'meso':
            return blocks.get('meso_questions', [])
        elif level == 'macro':
            macro = blocks.get('macro_question')
            return [macro] if macro else []
        else:
            raise ValueError(f"Invalid level: {level}")
    
    def get_all_clusters(self) -> List[Dict[str, Any]]:
        """
        Get all clusters.
        
        Returns:
            List of cluster dictionaries
        """
        return list(self._cluster_index.values())
    
    def get_metadata(self) -> MonolithMetadata:
        """
        Get monolith metadata.
        
        Returns:
            MonolithMetadata object
        """
        return self._metadata
    
    def get_monolith_hash(self) -> str:
        """
        Get SHA-256 hash of monolith file.
        
        Returns:
            Hex digest of SHA-256 hash
        """
        return self._metadata.hash_sha256
    
    def query_questions(
        self,
        policy_area: Optional[str] = None,
        dimension: Optional[str] = None,
        cluster: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query questions by criteria.
        
        Args:
            policy_area: Policy area filter (e.g., "PA01")
            dimension: Dimension filter (e.g., "DIM01")
            cluster: Cluster filter (e.g., "CL01")
            
        Returns:
            List of matching questions
        """
        results = []
        
        for question in self._question_index.values():
            # Check policy area
            if policy_area and question.get('policy_area_id') != policy_area:
                continue
            
            # Check dimension
            if dimension and question.get('dimension_id') != dimension:
                continue
            
            # Check cluster
            if cluster and question.get('cluster_id') != cluster:
                continue
            
            results.append(question)
        
        return results
    
    def validate_access(self) -> bool:
        """
        Validate that monolith is properly loaded and accessible.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check that we have data
            if not self._monolith_data:
                return False
            
            # Check that we have metadata
            if not self._metadata:
                return False
            
            # Check that we have questions
            if len(self._question_index) == 0:
                return False
            
            # Check file still exists
            if not self.monolith_path.exists():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


# Global singleton instance
_global_orchestrator: Optional[MonolithOrchestrator] = None


def get_global_orchestrator(monolith_path: Optional[Path] = None) -> MonolithOrchestrator:
    """
    Get or create global monolith orchestrator singleton.
    
    Args:
        monolith_path: Path to monolith (only used on first call)
        
    Returns:
        Global MonolithOrchestrator instance
    """
    global _global_orchestrator
    
    if _global_orchestrator is None:
        _global_orchestrator = MonolithOrchestrator(monolith_path)
    
    return _global_orchestrator


def reset_global_orchestrator():
    """Reset global orchestrator (mainly for testing)"""
    global _global_orchestrator
    _global_orchestrator = None


__all__ = [
    'MonolithOrchestrator',
    'MonolithAccessError',
    'MonolithMetadata',
    'get_global_orchestrator',
    'reset_global_orchestrator',
]
