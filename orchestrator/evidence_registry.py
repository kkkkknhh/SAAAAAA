"""
Evidence Registry: Append-Only JSONL Store with Hash Chain and Provenance DAG Export

This module implements a comprehensive evidence tracking system that:
1. Stores all evidence in append-only JSONL format for immutability
2. Maintains hash-based indexing for fast evidence lookup
3. Implements blockchain-style hash chaining for ledger integrity
4. Exports provenance DAG showing evidence lineage and dependencies
5. Provides cryptographic verification of evidence integrity

Architecture:
- JSONL Storage: One JSON object per line, append-only for audit trail
- Hash Index: SHA-256 hashes for content-addressable storage
- Hash Chain: Each entry links to previous via previous_hash and entry_hash
- Provenance DAG: Directed acyclic graph of evidence dependencies
- Verification: Cryptographic chain-of-custody validation with chain linkage checks

Hash Chain Security:
The registry implements a blockchain-style hash chain where each entry contains:
- content_hash: SHA-256 of the payload (for content verification)
- previous_hash: Hash of the previous entry's entry_hash (creates the chain)
- entry_hash: SHA-256 of (content_hash + previous_hash + metadata)

This ensures that:
1. Any tampering with payload is detected via content_hash mismatch
2. Any tampering with previous_hash is detected via chain verification
3. Entries cannot be reordered without breaking the chain
4. The entire ledger history can be cryptographically verified
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EvidenceRecord:
    """
    Immutable evidence record with provenance metadata and hash chain linkage.
    
    Each evidence record captures:
    - Unique identifier (hash-based)
    - Evidence payload (method result, analysis output, etc.)
    - Provenance metadata (source, dependencies, lineage)
    - Temporal metadata (timestamp, execution time)
    - Verification data (content hash, chain hashes)
    
    Hash Chain Fields:
    - content_hash: SHA-256 of payload (verifies content integrity)
    - previous_hash: entry_hash of previous record (creates chain linkage)
    - entry_hash: SHA-256 of (content + previous_hash + metadata) (unique entry ID)
    
    The hash chain ensures that:
    1. Tampering with payload breaks content_hash
    2. Tampering with previous_hash breaks chain verification
    3. Entire ledger history is cryptographically verifiable
    """
    
    # Identification
    evidence_id: str  # SHA-256 hash of content
    evidence_type: str  # Type of evidence (e.g., "method_result", "analysis", "extraction")
    
    # Payload
    payload: Dict[str, Any]
    
    # Provenance
    source_method: Optional[str] = None  # FQN of method that produced this evidence
    parent_evidence_ids: List[str] = field(default_factory=list)  # Dependencies
    question_id: Optional[str] = None
    document_id: Optional[str] = None
    
    # Temporal
    timestamp: float = field(default_factory=time.time)
    execution_time_ms: float = 0.0
    
    # Verification
    content_hash: Optional[str] = None  # Hash of payload for verification
    previous_hash: Optional[str] = None  # Hash of previous entry in chain (for ledger integrity)
    entry_hash: Optional[str] = None  # Hash of this entire entry including previous_hash
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate content hash and entry hash if not provided."""
        if self.content_hash is None:
            self.content_hash = self._compute_content_hash()
        if self.entry_hash is None:
            self.entry_hash = self._compute_entry_hash()
    
    def _compute_content_hash(self) -> str:
        """
        Compute SHA-256 hash of payload for content-addressable storage.
        
        Returns:
            Hex digest of SHA-256 hash
        """
        # Create deterministic JSON representation
        payload_json = json.dumps(self.payload, sort_keys=True, separators=(',', ':'))
        
        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(payload_json.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _compute_entry_hash(self) -> str:
        """
        Compute SHA-256 hash of the entire entry including previous_hash.
        This creates the hash chain linking entries together.
        
        Returns:
            Hex digest of SHA-256 hash
        """
        # Combine content hash with previous hash to create chain
        # Use empty string for first entry (no predecessor)
        chain_data = {
            'content_hash': self.content_hash,
            'previous_hash': self.previous_hash if self.previous_hash is not None else '',
            'evidence_type': self.evidence_type,
            'timestamp': self.timestamp,
        }
        chain_json = json.dumps(chain_data, sort_keys=True, separators=(',', ':'))
        hash_obj = hashlib.sha256(chain_json.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def verify_integrity(self, previous_record: Optional['EvidenceRecord'] = None) -> bool:
        """
        Verify evidence integrity by recomputing hashes and checking chain linkage.
        
        Args:
            previous_record: The record that should precede this one in the chain
        
        Returns:
            True if all integrity checks pass, False otherwise
        """
        # Verify content hash matches
        current_content_hash = self._compute_content_hash()
        if current_content_hash != self.content_hash:
            return False
        
        # Verify entry hash matches
        current_entry_hash = self._compute_entry_hash()
        if current_entry_hash != self.entry_hash:
            return False
        
        # If previous record is provided, verify the chain linkage
        if previous_record is not None:
            # Verify that our previous_hash matches the actual hash of the previous record
            if self.previous_hash != previous_record.entry_hash:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceRecord:
        """Create evidence record from dictionary."""
        return cls(**data)


@dataclass
class ProvenanceNode:
    """Node in provenance DAG."""
    
    evidence_id: str
    evidence_type: str
    source_method: Optional[str]
    timestamp: float
    children: List[str] = field(default_factory=list)  # Evidence IDs that depend on this
    parents: List[str] = field(default_factory=list)   # Evidence IDs this depends on
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProvenanceDAG:
    """
    Directed Acyclic Graph of evidence provenance.
    
    Captures the full lineage of evidence:
    - Which evidence produced which other evidence
    - Method invocation chains
    - Data flow dependencies
    """
    
    nodes: Dict[str, ProvenanceNode] = field(default_factory=dict)
    
    # Index for fast queries
    by_method: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    by_type: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    by_question: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    
    def add_evidence(
        self,
        evidence: EvidenceRecord
    ) -> None:
        """
        Add evidence to provenance DAG.
        
        Args:
            evidence: Evidence record to add
        """
        # Create node
        node = ProvenanceNode(
            evidence_id=evidence.evidence_id,
            evidence_type=evidence.evidence_type,
            source_method=evidence.source_method,
            timestamp=evidence.timestamp,
            parents=evidence.parent_evidence_ids.copy(),
        )
        
        # Add to nodes
        self.nodes[evidence.evidence_id] = node
        
        # Update parent-child relationships
        for parent_id in evidence.parent_evidence_ids:
            if parent_id in self.nodes:
                self.nodes[parent_id].children.append(evidence.evidence_id)
        
        # Update indices
        if evidence.source_method:
            self.by_method[evidence.source_method].append(evidence.evidence_id)
        self.by_type[evidence.evidence_type].append(evidence.evidence_id)
        if evidence.question_id:
            self.by_question[evidence.question_id].append(evidence.evidence_id)
    
    def get_ancestors(self, evidence_id: str) -> Set[str]:
        """
        Get all ancestor evidence IDs (transitive parents).
        
        Args:
            evidence_id: Evidence ID to trace
            
        Returns:
            Set of ancestor evidence IDs
        """
        ancestors = set()
        visited = set()
        
        def traverse(eid: str):
            if eid in visited:
                return
            visited.add(eid)
            
            if eid not in self.nodes:
                return
            
            node = self.nodes[eid]
            for parent_id in node.parents:
                ancestors.add(parent_id)
                traverse(parent_id)
        
        traverse(evidence_id)
        return ancestors
    
    def get_descendants(self, evidence_id: str) -> Set[str]:
        """
        Get all descendant evidence IDs (transitive children).
        
        Args:
            evidence_id: Evidence ID to trace
            
        Returns:
            Set of descendant evidence IDs
        """
        descendants = set()
        visited = set()
        
        def traverse(eid: str):
            if eid in visited:
                return
            visited.add(eid)
            
            if eid not in self.nodes:
                return
            
            node = self.nodes[eid]
            for child_id in node.children:
                descendants.add(child_id)
                traverse(child_id)
        
        traverse(evidence_id)
        return descendants
    
    def get_lineage(self, evidence_id: str) -> Dict[str, Any]:
        """
        Get complete lineage for evidence (ancestors + descendants).
        
        Args:
            evidence_id: Evidence ID to trace
            
        Returns:
            Dictionary with lineage information
        """
        return {
            "evidence_id": evidence_id,
            "ancestors": list(self.get_ancestors(evidence_id)),
            "descendants": list(self.get_descendants(evidence_id)),
            "ancestor_count": len(self.get_ancestors(evidence_id)),
            "descendant_count": len(self.get_descendants(evidence_id)),
        }
    
    def export_dot(self) -> str:
        """
        Export DAG in GraphViz DOT format.
        
        Returns:
            DOT format string
        """
        lines = ["digraph ProvenanceDAG {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box];")
        lines.append("")
        
        # Add nodes
        for eid, node in self.nodes.items():
            label = f"{node.evidence_type}\\n{eid[:8]}..."
            if node.source_method:
                label += f"\\n{node.source_method}"
            lines.append(f'  "{eid}" [label="{label}"];')
        
        lines.append("")
        
        # Add edges
        for eid, node in self.nodes.items():
            for child_id in node.children:
                lines.append(f'  "{eid}" -> "{child_id}";')
        
        lines.append("}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export DAG to dictionary."""
        return {
            "nodes": {eid: node.to_dict() for eid, node in self.nodes.items()},
            "stats": {
                "total_nodes": len(self.nodes),
                "by_method": {k: len(v) for k, v in self.by_method.items()},
                "by_type": {k: len(v) for k, v in self.by_type.items()},
                "by_question": {k: len(v) for k, v in self.by_question.items()},
            }
        }


class EvidenceRegistry:
    """
    Append-only evidence registry with hash indexing and provenance tracking.
    
    Features:
    - JSONL append-only storage for immutability
    - Content-addressable hash indexing
    - Provenance DAG for lineage tracking
    - Cryptographic verification
    - Fast queries by hash, type, method, question
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        enable_dag: bool = True,
    ):
        """
        Initialize evidence registry.
        
        Args:
            storage_path: Path to JSONL storage file (default: evidence_registry.jsonl)
            enable_dag: Enable provenance DAG tracking
        """
        self.storage_path = storage_path or Path("evidence_registry.jsonl")
        self.enable_dag = enable_dag
        
        # Hash index: hash -> evidence record
        self.hash_index: Dict[str, EvidenceRecord] = {}
        
        # Type index: type -> list of hashes
        self.type_index: Dict[str, List[str]] = defaultdict(list)
        
        # Method index: method FQN -> list of hashes
        self.method_index: Dict[str, List[str]] = defaultdict(list)
        
        # Question index: question ID -> list of hashes
        self.question_index: Dict[str, List[str]] = defaultdict(list)
        
        # Provenance DAG
        self.dag = ProvenanceDAG() if enable_dag else None
        
        # Track the last entry in the ledger chain for hash chaining
        self.last_entry: Optional[EvidenceRecord] = None
        
        # Load existing evidence
        self._load_from_storage()
        
        logger.info(
            f"EvidenceRegistry initialized with {len(self.hash_index)} records, "
            f"storage={self.storage_path}, dag={'enabled' if enable_dag else 'disabled'}"
        )
    
    def _load_from_storage(self) -> None:
        """Load evidence from JSONL storage."""
        if not self.storage_path.exists():
            logger.info(f"No existing evidence storage found at {self.storage_path}")
            return
        
        loaded_count = 0
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        evidence = EvidenceRecord.from_dict(data)
                        self._index_evidence(evidence, persist=False)
                        loaded_count += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to load evidence on line {line_num}: {e}")
            
            logger.info(f"Loaded {loaded_count} evidence records from storage")
            
        except Exception as e:
            logger.error(f"Failed to load evidence storage: {e}")
    
    def _index_evidence(
        self,
        evidence: EvidenceRecord,
        persist: bool = True
    ) -> None:
        """
        Index evidence record in all indices.
        
        Args:
            evidence: Evidence to index
            persist: If True, append to JSONL storage
        """
        # Hash index
        self.hash_index[evidence.evidence_id] = evidence
        
        # Type index
        self.type_index[evidence.evidence_type].append(evidence.evidence_id)
        
        # Method index
        if evidence.source_method:
            self.method_index[evidence.source_method].append(evidence.evidence_id)
        
        # Question index
        if evidence.question_id:
            self.question_index[evidence.question_id].append(evidence.evidence_id)
        
        # DAG
        if self.enable_dag and self.dag:
            self.dag.add_evidence(evidence)
        
        # Update last entry for hash chaining
        self.last_entry = evidence
        
        # Persist to storage
        if persist:
            self._append_to_storage(evidence)
    
    def _append_to_storage(self, evidence: EvidenceRecord) -> None:
        """
        Append evidence to JSONL storage.
        
        Args:
            evidence: Evidence to append
        """
        try:
            # Ensure parent directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to JSONL
            with open(self.storage_path, 'a', encoding='utf-8') as f:
                json_line = json.dumps(evidence.to_dict(), separators=(',', ':'))
                f.write(json_line + '\n')
                
        except Exception as e:
            logger.error(f"Failed to append evidence to storage: {e}")
            raise
    
    def record_evidence(
        self,
        evidence_type: str,
        payload: Dict[str, Any],
        source_method: Optional[str] = None,
        parent_evidence_ids: Optional[List[str]] = None,
        question_id: Optional[str] = None,
        document_id: Optional[str] = None,
        execution_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record new evidence in registry.
        
        Args:
            evidence_type: Type of evidence
            payload: Evidence data
            source_method: FQN of method that produced evidence
            parent_evidence_ids: List of parent evidence IDs
            question_id: Question ID this evidence relates to
            document_id: Document ID this evidence relates to
            execution_time_ms: Execution time
            metadata: Additional metadata
            
        Returns:
            Evidence ID (hash)
        """
        # Determine previous_hash from last entry in the chain
        previous_hash = self.last_entry.entry_hash if self.last_entry else None
        
        # Normalize metadata and ensure recorded_at timestamp
        metadata_dict: Dict[str, Any] = dict(metadata) if metadata else {}
        metadata_dict.setdefault(
            "recorded_at",
            datetime.now(timezone.utc).isoformat(),
        )

        # Create evidence record
        evidence = EvidenceRecord(
            evidence_id="",  # Will be set by hash
            evidence_type=evidence_type,
            payload=payload,
            source_method=source_method,
            parent_evidence_ids=parent_evidence_ids or [],
            question_id=question_id,
            document_id=document_id,
            execution_time_ms=execution_time_ms,
            metadata=metadata_dict,
            previous_hash=previous_hash,
        )
        
        # Set evidence_id to content hash
        evidence.evidence_id = evidence.content_hash or ""
        
        # Check for duplicate
        if evidence.evidence_id in self.hash_index:
            logger.debug(f"Evidence {evidence.evidence_id} already exists, skipping")
            return evidence.evidence_id
        
        # Index evidence
        self._index_evidence(evidence, persist=True)
        
        logger.debug(f"Recorded evidence {evidence.evidence_id} of type {evidence_type}")
        
        return evidence.evidence_id
    
    def get_evidence(self, evidence_id: str) -> Optional[EvidenceRecord]:
        """
        Retrieve evidence by ID.
        
        Args:
            evidence_id: Evidence hash
            
        Returns:
            EvidenceRecord or None
        """
        return self.hash_index.get(evidence_id)
    
    def query_by_type(self, evidence_type: str) -> List[EvidenceRecord]:
        """Query evidence by type."""
        evidence_ids = self.type_index.get(evidence_type, [])
        return [self.hash_index[eid] for eid in evidence_ids if eid in self.hash_index]
    
    def query_by_method(self, method_fqn: str) -> List[EvidenceRecord]:
        """Query evidence by source method."""
        evidence_ids = self.method_index.get(method_fqn, [])
        return [self.hash_index[eid] for eid in evidence_ids if eid in self.hash_index]
    
    def query_by_question(self, question_id: str) -> List[EvidenceRecord]:
        """Query evidence by question ID."""
        evidence_ids = self.question_index.get(question_id, [])
        return [self.hash_index[eid] for eid in evidence_ids if eid in self.hash_index]
    
    def verify_evidence(self, evidence_id: str, verify_chain: bool = True) -> bool:
        """
        Verify evidence integrity and optionally chain linkage.
        
        Args:
            evidence_id: Evidence hash
            verify_chain: If True, verify chain linkage with previous entry
            
        Returns:
            True if evidence is valid
        """
        evidence = self.get_evidence(evidence_id)
        if evidence is None:
            return False
        
        # Get previous record if chain verification is requested
        previous_record = None
        if verify_chain and evidence.previous_hash:
            # Find the record with entry_hash matching our previous_hash
            for record in self.hash_index.values():
                if record.entry_hash == evidence.previous_hash:
                    previous_record = record
                    break
        
        return evidence.verify_integrity(previous_record=previous_record)
    
    def verify_chain_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of the entire evidence chain.
        
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        # Build the chain by reading from storage in order
        if not self.storage_path.exists():
            return True, []  # Empty chain is valid
        
        try:
            previous_record = None
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        evidence = EvidenceRecord.from_dict(data)
                        
                        # Verify the record's integrity
                        if not evidence.verify_integrity(previous_record=previous_record):
                            if previous_record and evidence.previous_hash != previous_record.entry_hash:
                                errors.append(
                                    f"Line {line_num}: Chain broken - previous_hash mismatch. "
                                    f"Expected {previous_record.entry_hash}, got {evidence.previous_hash}"
                                )
                            else:
                                errors.append(
                                    f"Line {line_num}: Hash integrity check failed for evidence {evidence.evidence_id}"
                                )
                        
                        previous_record = evidence
                        
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: JSON parsing error - {e}")
                    except Exception as e:
                        errors.append(f"Line {line_num}: Verification error - {e}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Failed to verify chain: {e}"]
    
    def get_provenance(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """
        Get provenance information for evidence.
        
        Args:
            evidence_id: Evidence hash
            
        Returns:
            Provenance dictionary or None
        """
        if not self.enable_dag or self.dag is None:
            return None
        
        return self.dag.get_lineage(evidence_id)
    
    def export_provenance_dag(
        self,
        format: str = "dict",
        output_path: Optional[Path] = None
    ) -> Any:
        """
        Export provenance DAG.
        
        Args:
            format: Export format ("dict", "dot", "json")
            output_path: Optional path to write output
            
        Returns:
            Exported DAG in requested format
        """
        if not self.enable_dag or self.dag is None:
            raise ValueError("DAG tracking is not enabled")
        
        if format == "dot":
            result = self.dag.export_dot()
        elif format == "dict":
            result = self.dag.to_dict()
        elif format == "json":
            result = json.dumps(self.dag.to_dict(), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Write to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(result, str):
                output_path.write_text(result, encoding='utf-8')
            else:
                output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            logger.info(f"Exported provenance DAG to {output_path}")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_evidence": len(self.hash_index),
            "by_type": {k: len(v) for k, v in self.type_index.items()},
            "by_method": {k: len(v) for k, v in self.method_index.items()},
            "by_question": {k: len(v) for k, v in self.question_index.items()},
            "storage_path": str(self.storage_path),
            "dag_enabled": self.enable_dag,
        }
        
        if self.enable_dag and self.dag:
            stats["dag_nodes"] = len(self.dag.nodes)
        
        return stats


# Global registry instance
_global_registry: Optional[EvidenceRegistry] = None


def get_global_registry() -> EvidenceRegistry:
    """Get or create global evidence registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = EvidenceRegistry()
    return _global_registry


__all__ = [
    "EvidenceRecord",
    "ProvenanceNode",
    "ProvenanceDAG",
    "EvidenceRegistry",
    "get_global_registry",
]
