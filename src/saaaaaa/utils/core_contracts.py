"""
Core Module Contracts - Type-safe API boundaries for pure library modules.

This module defines InputContract and OutputContract TypedDicts for each
core module to establish clear API boundaries and enable dependency injection.

Architectural Principles:
- Core modules receive all data via InputContract parameters
- Core modules return data via OutputContract structures
- No I/O operations within core modules
- All I/O happens in orchestrator/factory.py
- Type-safe contracts with strict typing

Version: 1.0.0
Status: Initial contract definitions (to be expanded)
"""

from typing import TypedDict, List, Dict, Any, Optional
try:
    from typing import NotRequired  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired  # Python 3.9-3.10


# ============================================================================
# ANALYZER_ONE.PY CONTRACTS
# ============================================================================

class SemanticAnalyzerInputContract(TypedDict):
    """Input contract for SemanticAnalyzer methods.
    
    Example:
        {
            "text": "El plan de desarrollo municipal...",
            "segments": ["Segment 1", "Segment 2"],
            "ontology_params": {"domain": "municipal"}
        }
    """
    text: str
    segments: NotRequired[List[str]]
    ontology_params: NotRequired[Dict[str, Any]]


class SemanticAnalyzerOutputContract(TypedDict):
    """Output contract for SemanticAnalyzer methods."""
    semantic_cube: Dict[str, Any]
    coherence_score: float
    complexity_score: float
    domain_classification: Dict[str, float]


# ============================================================================
# DERECK_BEACH.PY CONTRACTS  
# ============================================================================

class CDAFFrameworkInputContract(TypedDict):
    """Input contract for CDAFFramework (Causal Deconstruction Audit Framework)."""
    document_text: str
    plan_metadata: Dict[str, Any]
    config: NotRequired[Dict[str, Any]]


class CDAFFrameworkOutputContract(TypedDict):
    """Output contract for CDAFFramework."""
    causal_mechanisms: List[Dict[str, Any]]
    evidential_tests: Dict[str, Any]
    bayesian_inference: Dict[str, Any]
    audit_results: Dict[str, Any]


# ============================================================================
# FINANCIERO_VIABILIDAD_TABLAS.PY CONTRACTS
# ============================================================================

class PDETAnalyzerInputContract(TypedDict):
    """Input contract for PDET (Programas de Desarrollo con Enfoque Territorial) Analyzer."""
    document_content: str
    extract_tables: NotRequired[bool]
    config: NotRequired[Dict[str, Any]]


class PDETAnalyzerOutputContract(TypedDict):
    """Output contract for PDET Analyzer."""
    extracted_tables: List[Dict[str, Any]]
    financial_indicators: Dict[str, float]
    viability_score: float
    quality_scores: Dict[str, float]


# ============================================================================
# TEORIA_CAMBIO.PY CONTRACTS
# ============================================================================

class TeoriaCambioInputContract(TypedDict):
    """Input contract for Theory of Change analysis."""
    document_text: str
    strategic_goals: NotRequired[List[str]]
    config: NotRequired[Dict[str, Any]]


class TeoriaCambioOutputContract(TypedDict):
    """Output contract for Theory of Change analysis."""
    causal_dag: Dict[str, Any]
    validation_results: Dict[str, Any]
    monte_carlo_results: NotRequired[Dict[str, Any]]
    graph_visualizations: NotRequired[List[Dict[str, Any]]]


# ============================================================================
# CONTRADICTION_DETECCION.PY CONTRACTS
# ============================================================================

class ContradictionDetectorInputContract(TypedDict):
    """Input contract for PolicyContradictionDetector."""
    text: str
    plan_name: str
    dimension: NotRequired[str]  # PolicyDimension enum value
    config: NotRequired[Dict[str, Any]]


class ContradictionDetectorOutputContract(TypedDict):
    """Output contract for PolicyContradictionDetector."""
    contradictions: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    temporal_conflicts: List[Dict[str, Any]]
    severity_scores: Dict[str, float]


# ============================================================================
# EMBEDDING_POLICY.PY CONTRACTS
# ============================================================================

class EmbeddingPolicyInputContract(TypedDict):
    """Input contract for embedding-based policy analysis."""
    text: str
    dimensions: NotRequired[List[str]]
    model_config: NotRequired[Dict[str, Any]]


class EmbeddingPolicyOutputContract(TypedDict):
    """Output contract for embedding policy analysis."""
    embeddings: List[List[float]]
    similarity_scores: Dict[str, float]
    bayesian_evaluation: Dict[str, Any]
    policy_metrics: Dict[str, float]


# ============================================================================
# SEMANTIC_CHUNKING_POLICY.PY CONTRACTS
# ============================================================================

class SemanticChunkingInputContract(TypedDict):
    """Input contract for semantic chunking and policy document analysis."""
    text: str
    preserve_structure: NotRequired[bool]
    config: NotRequired[Dict[str, Any]]


class SemanticChunkingOutputContract(TypedDict):
    """Output contract for semantic chunking."""
    chunks: List[Dict[str, Any]]
    causal_dimensions: Dict[str, Dict[str, Any]]
    key_excerpts: Dict[str, List[str]]
    summary: Dict[str, Any]


# ============================================================================
# POLICY_PROCESSOR.PY CONTRACTS
# ============================================================================

class PolicyProcessorInputContract(TypedDict):
    """Input contract for IndustrialPolicyProcessor."""
    data: Any
    text: str
    sentences: NotRequired[List[str]]
    tables: NotRequired[List[Dict[str, Any]]]
    config: NotRequired[Dict[str, Any]]


class PolicyProcessorOutputContract(TypedDict):
    """Output contract for IndustrialPolicyProcessor."""
    processed_data: Dict[str, Any]
    evidence_bundles: List[Dict[str, Any]]
    bayesian_scores: Dict[str, float]
    matched_patterns: List[Dict[str, Any]]


# ============================================================================
# SHARED DATA STRUCTURES
# ============================================================================

class DocumentData(TypedDict):
    """Standard document data structure from orchestrator.
    
    This is what the orchestrator/factory provides to core modules.
    """
    raw_text: str
    sentences: List[str]
    tables: List[Dict[str, Any]]
    metadata: Dict[str, Any]


__all__ = [
    # Analyzer_one
    'SemanticAnalyzerInputContract',
    'SemanticAnalyzerOutputContract',
    
    # dereck_beach
    'CDAFFrameworkInputContract',
    'CDAFFrameworkOutputContract',
    
    # financiero_viabilidad_tablas
    'PDETAnalyzerInputContract',
    'PDETAnalyzerOutputContract',
    
    # teoria_cambio
    'TeoriaCambioInputContract',
    'TeoriaCambioOutputContract',
    
    # contradiction_deteccion
    'ContradictionDetectorInputContract',
    'ContradictionDetectorOutputContract',
    
    # embedding_policy
    'EmbeddingPolicyInputContract',
    'EmbeddingPolicyOutputContract',
    
    # semantic_chunking_policy
    'SemanticChunkingInputContract',
    'SemanticChunkingOutputContract',
    
    # policy_processor
    'PolicyProcessorInputContract',
    'PolicyProcessorOutputContract',
    
    # Shared
    'DocumentData',
]
