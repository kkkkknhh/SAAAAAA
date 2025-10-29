"""
Scoring Module - TYPE_A through TYPE_F Modality Implementation

This module implements the scoring system for the SAAAAAA policy analysis framework.
It provides:
- Application of 6 scoring modalities (TYPE_A through TYPE_F)
- Validation of evidence structure vs modality
- Assignment of quality levels
- Structured logging with strict abortability
- Reproducible ScoredResult outputs

Preconditions:
- Evidence and modality must be declared
- Evidence structure must match modality requirements

Invariants:
- Score range is maintained per modality definition
- Evidence structure is validated before scoring

Postconditions:
- ScoredResult is reproducible with same inputs
- No fallback or partial heuristic scoring
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ScoringModality(Enum):
    """Scoring modality types."""
    TYPE_A = "TYPE_A"  # Bayesian: Numerical claims, gaps, risks
    TYPE_B = "TYPE_B"  # DAG: Causal chains, ToC completeness
    TYPE_C = "TYPE_C"  # Coherence: Inverted contradictions
    TYPE_D = "TYPE_D"  # Pattern: Baseline data, formalization
    TYPE_E = "TYPE_E"  # Financial: Budget traceability
    TYPE_F = "TYPE_F"  # Beach: Mechanism inference, plausibility


class QualityLevel(Enum):
    """Quality level classifications."""
    EXCELENTE = "EXCELENTE"
    BUENO = "BUENO"
    ACEPTABLE = "ACEPTABLE"
    INSUFICIENTE = "INSUFICIENTE"


class ScoringError(Exception):
    """Base exception for scoring errors."""
    pass


class ModalityValidationError(ScoringError):
    """Exception raised when evidence structure doesn't match modality requirements."""
    pass


class EvidenceStructureError(ScoringError):
    """Exception raised when evidence structure is invalid."""
    pass


@dataclass(frozen=True)
class ScoredResult:
    """
    Reproducible scored result for a question.
    
    Attributes:
        question_global: Global question number (1-300)
        base_slot: Question slot identifier
        policy_area: Policy area ID (PA01-PA10)
        dimension: Dimension ID (DIM01-DIM06)
        modality: Scoring modality used (TYPE_A through TYPE_F)
        score: Raw score value
        normalized_score: Normalized score (0-1)
        quality_level: Quality level classification
        evidence_hash: SHA-256 hash of evidence for reproducibility
        metadata: Additional scoring metadata
        timestamp: ISO timestamp of scoring
    """
    question_global: int
    base_slot: str
    policy_area: str
    dimension: str
    modality: str
    score: float
    normalized_score: float
    quality_level: str
    evidence_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @staticmethod
    def compute_evidence_hash(evidence: Dict[str, Any]) -> str:
        """
        Compute reproducible hash of evidence.
        
        Args:
            evidence: Evidence dictionary
            
        Returns:
            SHA-256 hash as hex string
        """
        canonical = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class ModalityConfig:
    """
    Configuration for a scoring modality.
    
    Attributes:
        name: Modality name
        description: Modality description
        score_range: Min and max score values
        rounding_mode: Rounding mode (half_up, bankers, truncate)
        rounding_precision: Decimal precision for rounding
        required_evidence_keys: Required keys in evidence
        expected_elements: Expected number of elements (if applicable)
        deterministic: Whether scoring is deterministic
    """
    name: str
    description: str
    score_range: Tuple[float, float]
    rounding_mode: str = "half_up"
    rounding_precision: int = 2
    required_evidence_keys: List[str] = field(default_factory=list)
    expected_elements: Optional[int] = None
    deterministic: bool = True
    
    def validate_evidence(self, evidence: Dict[str, Any]) -> None:
        """
        Validate evidence structure against modality requirements.
        
        Args:
            evidence: Evidence dictionary to validate
            
        Raises:
            EvidenceStructureError: If evidence is missing required keys
            ModalityValidationError: If evidence structure doesn't match modality
        """
        if not isinstance(evidence, dict):
            raise EvidenceStructureError(
                f"Evidence must be a dictionary, got {type(evidence).__name__}"
            )
        
        # Check required keys
        missing_keys = [key for key in self.required_evidence_keys if key not in evidence]
        if missing_keys:
            raise EvidenceStructureError(
                f"Evidence missing required keys for {self.name}: {missing_keys}"
            )
        
        # Validate expected elements if applicable
        if self.expected_elements is not None:
            elements = evidence.get("elements", [])
            if not isinstance(elements, list):
                raise ModalityValidationError(
                    f"{self.name} requires 'elements' to be a list, got {type(elements).__name__}"
                )


class ScoringValidator:
    """Validates evidence structure against modality requirements."""
    
    # Modality configurations
    MODALITY_CONFIGS: ClassVar[Dict[ScoringModality, ModalityConfig]] = {
        ScoringModality.TYPE_A: ModalityConfig(
            name="TYPE_A",
            description="Bayesian: Numerical claims, gaps, risks",
            score_range=(0.0, 4.0),
            required_evidence_keys=["elements", "confidence"],
            expected_elements=4,
        ),
        ScoringModality.TYPE_B: ModalityConfig(
            name="TYPE_B",
            description="DAG: Causal chains, ToC completeness",
            score_range=(0.0, 3.0),
            required_evidence_keys=["elements", "completeness"],
            expected_elements=3,
        ),
        ScoringModality.TYPE_C: ModalityConfig(
            name="TYPE_C",
            description="Coherence: Inverted contradictions",
            score_range=(0.0, 3.0),
            required_evidence_keys=["elements", "coherence_score"],
            expected_elements=2,
        ),
        ScoringModality.TYPE_D: ModalityConfig(
            name="TYPE_D",
            description="Pattern: Baseline data, formalization",
            score_range=(0.0, 3.0),
            required_evidence_keys=["elements", "pattern_matches"],
            expected_elements=3,
        ),
        ScoringModality.TYPE_E: ModalityConfig(
            name="TYPE_E",
            description="Financial: Budget traceability",
            score_range=(0.0, 3.0),
            required_evidence_keys=["elements", "traceability"],
        ),
        ScoringModality.TYPE_F: ModalityConfig(
            name="TYPE_F",
            description="Beach: Mechanism inference, plausibility",
            score_range=(0.0, 3.0),
            required_evidence_keys=["elements", "plausibility"],
        ),
    }
    
    @classmethod
    def validate(
        cls,
        evidence: Dict[str, Any],
        modality: ScoringModality,
    ) -> None:
        """
        Validate evidence structure against modality.
        
        Args:
            evidence: Evidence dictionary
            modality: Scoring modality
            
        Raises:
            ModalityValidationError: If validation fails
            
        Note:
            This function has strict abortability - any validation failure
            will raise an exception and halt processing.
        """
        config = cls.MODALITY_CONFIGS.get(modality)
        if not config:
            raise ModalityValidationError(f"Unknown modality: {modality}")
        
        logger.info(f"Validating evidence for {modality.value}")
        
        try:
            config.validate_evidence(evidence)
            logger.info(f"✓ Evidence validation passed for {modality.value}")
        except (EvidenceStructureError, ModalityValidationError) as e:
            logger.exception(f"✗ Evidence validation failed for {modality.value}: {e}")
            raise
    
    @classmethod
    def get_config(cls, modality: ScoringModality) -> ModalityConfig:
        """Get configuration for a modality."""
        config = cls.MODALITY_CONFIGS.get(modality)
        if not config:
            raise ModalityValidationError(f"Unknown modality: {modality}")
        return config


def apply_rounding(
    value: float,
    mode: str = "half_up",
    precision: int = 2,
) -> float:
    """
    Apply rounding to a numeric value.
    
    Args:
        value: Value to round
        mode: Rounding mode (half_up, bankers, truncate)
        precision: Decimal precision
        
    Returns:
        Rounded value
    """
    decimal_value = Decimal(str(value))
    
    if mode == "half_up":
        rounded = decimal_value.quantize(
            Decimal(10) ** -precision,
            rounding=ROUND_HALF_UP
        )
    elif mode == "bankers":
        # Python's default rounding is bankers rounding
        rounded = round(decimal_value, precision)
    elif mode == "truncate":
        # Truncate by converting to int and back
        factor = 10 ** precision
        rounded = Decimal(int(decimal_value * factor)) / factor
    else:
        raise ValueError(f"Unknown rounding mode: {mode}")
    
    return float(rounded)


def score_type_a(evidence: Dict[str, Any], config: ModalityConfig) -> Tuple[float, Dict[str, Any]]:
    """
    Score TYPE_A evidence: Bayesian numerical claims, gaps, risks.
    
    Expects:
    - elements: List of up to 4 elements
    - confidence: Bayesian confidence score (0-1)
    
    Scoring:
    - Count elements (max 4)
    - Weight by confidence
    - Scale to 0-4 range
    
    Args:
        evidence: Evidence dictionary
        config: Modality configuration
        
    Returns:
        Tuple of (score, metadata)
    """
    elements = evidence.get("elements", [])
    confidence = evidence.get("confidence", 0.0)
    
    if not isinstance(elements, list):
        raise ModalityValidationError("TYPE_A: 'elements' must be a list")
    
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
        raise ModalityValidationError("TYPE_A: 'confidence' must be a number between 0 and 1")
    
    # Count valid elements (up to expected)
    element_count = min(len(elements), config.expected_elements or 4)
    
    # Calculate raw score: count weighted by confidence, scale to range
    raw_score = (element_count / 4.0) * 4.0 * confidence
    
    # Clamp to valid range
    score = max(config.score_range[0], min(config.score_range[1], raw_score))
    
    metadata = {
        "element_count": element_count,
        "confidence": confidence,
        "raw_score": raw_score,
        "expected_elements": config.expected_elements,
    }
    
    logger.info(
        f"TYPE_A score: {score:.2f} "
        f"(elements={element_count}, confidence={confidence:.2f})"
    )
    
    return score, metadata


def score_type_b(evidence: Dict[str, Any], config: ModalityConfig) -> Tuple[float, Dict[str, Any]]:
    """
    Score TYPE_B evidence: DAG causal chains, ToC completeness.
    
    Expects:
    - elements: List of causal chain elements (up to 3)
    - completeness: DAG completeness score (0-1)
    
    Scoring:
    - Count causal elements (max 3)
    - Weight by completeness
    - Each element worth 1 point
    
    Args:
        evidence: Evidence dictionary
        config: Modality configuration
        
    Returns:
        Tuple of (score, metadata)
    """
    elements = evidence.get("elements", [])
    completeness = evidence.get("completeness", 0.0)
    
    if not isinstance(elements, list):
        raise ModalityValidationError("TYPE_B: 'elements' must be a list")
    
    if not isinstance(completeness, (int, float)) or not (0 <= completeness <= 1):
        raise ModalityValidationError("TYPE_B: 'completeness' must be a number between 0 and 1")
    
    # Count valid elements (up to expected)
    element_count = min(len(elements), config.expected_elements or 3)
    
    # Calculate raw score: each element worth 1 point, weighted by completeness
    raw_score = float(element_count) * completeness
    
    # Clamp to valid range
    score = max(config.score_range[0], min(config.score_range[1], raw_score))
    
    metadata = {
        "element_count": element_count,
        "completeness": completeness,
        "raw_score": raw_score,
        "expected_elements": config.expected_elements,
    }
    
    logger.info(
        f"TYPE_B score: {score:.2f} "
        f"(elements={element_count}, completeness={completeness:.2f})"
    )
    
    return score, metadata


def score_type_c(evidence: Dict[str, Any], config: ModalityConfig) -> Tuple[float, Dict[str, Any]]:
    """
    Score TYPE_C evidence: Coherence via inverted contradictions.
    
    Expects:
    - elements: List of coherence elements (up to 2)
    - coherence_score: Inverted contradiction score (0-1, higher is better)
    
    Scoring:
    - Count coherence elements (max 2)
    - Scale by coherence score
    - Scale to 0-3 range
    
    Args:
        evidence: Evidence dictionary
        config: Modality configuration
        
    Returns:
        Tuple of (score, metadata)
    """
    elements = evidence.get("elements", [])
    coherence_score = evidence.get("coherence_score", 0.0)
    
    if not isinstance(elements, list):
        raise ModalityValidationError("TYPE_C: 'elements' must be a list")
    
    if not isinstance(coherence_score, (int, float)) or not (0 <= coherence_score <= 1):
        raise ModalityValidationError("TYPE_C: 'coherence_score' must be a number between 0 and 1")
    
    # Count valid elements (up to expected)
    element_count = min(len(elements), config.expected_elements or 2)
    
    # Calculate raw score: scale elements to range, weighted by coherence
    raw_score = (element_count / 2.0) * 3.0 * coherence_score
    
    # Clamp to valid range
    score = max(config.score_range[0], min(config.score_range[1], raw_score))
    
    metadata = {
        "element_count": element_count,
        "coherence_score": coherence_score,
        "raw_score": raw_score,
        "expected_elements": config.expected_elements,
    }
    
    logger.info(
        f"TYPE_C score: {score:.2f} "
        f"(elements={element_count}, coherence={coherence_score:.2f})"
    )
    
    return score, metadata


def score_type_d(evidence: Dict[str, Any], config: ModalityConfig) -> Tuple[float, Dict[str, Any]]:
    """
    Score TYPE_D evidence: Pattern matching for baseline data.
    
    Expects:
    - elements: List of pattern matches (up to 3)
    - pattern_matches: Number of successful pattern matches
    
    Scoring:
    - Count pattern matches (max 3)
    - Weight by match quality if available
    - Scale to 0-3 range
    
    Args:
        evidence: Evidence dictionary
        config: Modality configuration
        
    Returns:
        Tuple of (score, metadata)
    """
    elements = evidence.get("elements", [])
    pattern_matches = evidence.get("pattern_matches", 0)
    
    if not isinstance(elements, list):
        raise ModalityValidationError("TYPE_D: 'elements' must be a list")
    
    if not isinstance(pattern_matches, (int, float)) or pattern_matches < 0:
        raise ModalityValidationError("TYPE_D: 'pattern_matches' must be a non-negative number")
    
    # Count valid elements (up to expected)
    element_count = min(len(elements), config.expected_elements or 3)
    
    # Use actual pattern matches if available, otherwise use element count
    match_count = min(pattern_matches, element_count) if pattern_matches > 0 else element_count
    
    # Calculate raw score: scale to 0-3 range
    raw_score = (match_count / 3.0) * 3.0
    
    # Clamp to valid range
    score = max(config.score_range[0], min(config.score_range[1], raw_score))
    
    metadata = {
        "element_count": element_count,
        "pattern_matches": match_count,
        "raw_score": raw_score,
        "expected_elements": config.expected_elements,
    }
    
    logger.info(
        f"TYPE_D score: {score:.2f} "
        f"(elements={element_count}, matches={match_count})"
    )
    
    return score, metadata


def score_type_e(evidence: Dict[str, Any], config: ModalityConfig) -> Tuple[float, Dict[str, Any]]:
    """
    Score TYPE_E evidence: Financial budget traceability.
    
    Expects:
    - elements: List of budget elements
    - traceability: Boolean or numeric traceability score
    
    Scoring:
    - Boolean presence check
    - If numeric traceability provided, use that
    - Scale to 0-3 range
    
    Args:
        evidence: Evidence dictionary
        config: Modality configuration
        
    Returns:
        Tuple of (score, metadata)
    """
    elements = evidence.get("elements", [])
    traceability = evidence.get("traceability", False)
    
    if not isinstance(elements, list):
        raise ModalityValidationError("TYPE_E: 'elements' must be a list")
    
    # Handle both boolean and numeric traceability
    if isinstance(traceability, bool):
        traceability_score = 1.0 if traceability else 0.0
    elif isinstance(traceability, (int, float)):
        if not (0 <= traceability <= 1):
            raise ModalityValidationError("TYPE_E: numeric 'traceability' must be between 0 and 1")
        traceability_score = float(traceability)
    else:
        raise ModalityValidationError("TYPE_E: 'traceability' must be boolean or numeric")
    
    # Count valid elements
    element_count = len(elements)
    has_elements = element_count > 0
    
    # Calculate raw score: presence check weighted by traceability
    raw_score = 3.0 * traceability_score if has_elements else 0.0
    
    # Clamp to valid range
    score = max(config.score_range[0], min(config.score_range[1], raw_score))
    
    metadata = {
        "element_count": element_count,
        "traceability": traceability_score,
        "raw_score": raw_score,
        "has_elements": has_elements,
    }
    
    logger.info(
        f"TYPE_E score: {score:.2f} "
        f"(elements={element_count}, traceability={traceability_score:.2f})"
    )
    
    return score, metadata


def score_type_f(evidence: Dict[str, Any], config: ModalityConfig) -> Tuple[float, Dict[str, Any]]:
    """
    Score TYPE_F evidence: Beach mechanism inference and plausibility.
    
    Expects:
    - elements: List of mechanism elements
    - plausibility: Plausibility score (0-1)
    
    Scoring:
    - Continuous scale based on plausibility
    - Weight by element presence
    - Scale to 0-3 range
    
    Args:
        evidence: Evidence dictionary
        config: Modality configuration
        
    Returns:
        Tuple of (score, metadata)
    """
    elements = evidence.get("elements", [])
    plausibility = evidence.get("plausibility", 0.0)
    
    if not isinstance(elements, list):
        raise ModalityValidationError("TYPE_F: 'elements' must be a list")
    
    if not isinstance(plausibility, (int, float)) or not (0 <= plausibility <= 1):
        raise ModalityValidationError("TYPE_F: 'plausibility' must be a number between 0 and 1")
    
    # Count valid elements
    element_count = len(elements)
    
    # Calculate raw score: continuous scale weighted by plausibility
    raw_score = 3.0 * plausibility if element_count > 0 else 0.0
    
    # Clamp to valid range
    score = max(config.score_range[0], min(config.score_range[1], raw_score))
    
    metadata = {
        "element_count": element_count,
        "plausibility": plausibility,
        "raw_score": raw_score,
    }
    
    logger.info(
        f"TYPE_F score: {score:.2f} "
        f"(elements={element_count}, plausibility={plausibility:.2f})"
    )
    
    return score, metadata


# Scoring function registry
SCORING_FUNCTIONS = {
    ScoringModality.TYPE_A: score_type_a,
    ScoringModality.TYPE_B: score_type_b,
    ScoringModality.TYPE_C: score_type_c,
    ScoringModality.TYPE_D: score_type_d,
    ScoringModality.TYPE_E: score_type_e,
    ScoringModality.TYPE_F: score_type_f,
}


def determine_quality_level(
    normalized_score: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> QualityLevel:
    """
    Determine quality level from normalized score.
    
    Args:
        normalized_score: Score normalized to 0-1 range
        thresholds: Optional custom thresholds
        
    Returns:
        Quality level
        
    Note:
        Default thresholds:
        - EXCELENTE: >= 0.85
        - BUENO: >= 0.70
        - ACEPTABLE: >= 0.55
        - INSUFICIENTE: < 0.55
    """
    if thresholds is None:
        thresholds = {
            "EXCELENTE": 0.85,
            "BUENO": 0.70,
            "ACEPTABLE": 0.55,
        }
    
    if normalized_score >= thresholds["EXCELENTE"]:
        return QualityLevel.EXCELENTE
    elif normalized_score >= thresholds["BUENO"]:
        return QualityLevel.BUENO
    elif normalized_score >= thresholds["ACEPTABLE"]:
        return QualityLevel.ACEPTABLE
    else:
        return QualityLevel.INSUFICIENTE


def apply_scoring(
    question_global: int,
    base_slot: str,
    policy_area: str,
    dimension: str,
    evidence: Dict[str, Any],
    modality: str,
    quality_thresholds: Optional[Dict[str, float]] = None,
) -> ScoredResult:
    """
    Apply scoring to evidence using specified modality.
    
    This is the main entry point for scoring. It:
    1. Validates evidence structure against modality
    2. Applies modality-specific scoring function
    3. Normalizes score to 0-1 range
    4. Determines quality level
    5. Returns reproducible ScoredResult
    
    Args:
        question_global: Global question number (1-300)
        base_slot: Question slot identifier
        policy_area: Policy area ID (PA01-PA10)
        dimension: Dimension ID (DIM01-DIM06)
        evidence: Evidence dictionary
        modality: Scoring modality (TYPE_A through TYPE_F)
        quality_thresholds: Optional custom quality thresholds
        
    Returns:
        ScoredResult
        
    Raises:
        ModalityValidationError: If evidence validation fails
        ScoringError: If scoring fails
        
    Note:
        This function has strict abortability. Any validation or scoring
        error will raise an exception and halt processing. No fallback
        or partial scoring is performed.
    """
    logger.info(
        f"Scoring question {question_global} ({base_slot}) "
        f"using {modality}"
    )
    
    # Parse modality
    try:
        modality_enum = ScoringModality(modality)
    except ValueError as e:
        raise ModalityValidationError(
            f"Invalid modality: {modality}. "
            f"Must be one of: {[m.value for m in ScoringModality]}"
        ) from e
    
    # Validate evidence structure
    ScoringValidator.validate(evidence, modality_enum)
    
    # Get modality configuration
    config = ScoringValidator.get_config(modality_enum)
    
    # Get scoring function
    scoring_func = SCORING_FUNCTIONS.get(modality_enum)
    if not scoring_func:
        raise ScoringError(f"No scoring function for {modality}")
    
    # Apply scoring
    try:
        score, metadata = scoring_func(evidence, config)
    except (ModalityValidationError, EvidenceStructureError, ScoringError) as e:
        logger.exception(f"Scoring failed for {modality}: {e}")
        raise ScoringError(f"Scoring failed for {modality}: {e}") from e
    except Exception as e:
        logger.exception(f"Unexpected error in scoring {modality}: {e}")
        raise ScoringError(f"Unexpected error in scoring {modality}: {e}") from e
    
    # Apply rounding
    rounded_score = apply_rounding(
        score,
        mode=config.rounding_mode,
        precision=config.rounding_precision,
    )
    
    # Normalize score to 0-1 range
    score_range = config.score_range
    normalized_score = (rounded_score - score_range[0]) / (score_range[1] - score_range[0])
    
    # Determine quality level
    quality_level = determine_quality_level(normalized_score, quality_thresholds)
    
    # Compute evidence hash for reproducibility
    evidence_hash = ScoredResult.compute_evidence_hash(evidence)
    
    # Build result
    result = ScoredResult(
        question_global=question_global,
        base_slot=base_slot,
        policy_area=policy_area,
        dimension=dimension,
        modality=modality,
        score=rounded_score,
        normalized_score=normalized_score,
        quality_level=quality_level.value,
        evidence_hash=evidence_hash,
        metadata={
            **metadata,
            "score_range": score_range,
            "rounding_mode": config.rounding_mode,
            "rounding_precision": config.rounding_precision,
        },
    )
    
    logger.info(
        f"✓ Scoring complete: score={rounded_score:.2f}, "
        f"normalized={normalized_score:.2f}, quality={quality_level.value}"
    )
    
    return result


__all__ = [
    "ScoringModality",
    "QualityLevel",
    "ScoringError",
    "ModalityValidationError",
    "EvidenceStructureError",
    "ScoredResult",
    "ModalityConfig",
    "ScoringValidator",
    "apply_scoring",
    "determine_quality_level",
]
