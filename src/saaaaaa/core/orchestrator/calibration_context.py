"""Context-aware calibration system for policy analysis executors.

This module implements a multi-dimensional calibration resolver that addresses
calibration gaps by considering:
- Question context (dimension + question number, e.g., "D1Q1", "D6Q3")
- Policy area (fiscal, social, health, infrastructure, etc.)
- Unit of analysis (baseline_gap, indicator, activity, impact, etc.)
- Method sequence/context

The system maintains backward compatibility while adding contextual refinement.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Literal

from .calibration_registry import MethodCalibration


class PolicyArea(str, Enum):
    """Policy domain classification."""
    FISCAL = "fiscal"
    SOCIAL = "social"
    HEALTH = "health"
    EDUCATION = "education"
    INFRASTRUCTURE = "infrastructure"
    ENVIRONMENT = "environment"
    AGRICULTURE = "agriculture"
    SECURITY = "security"
    GOVERNANCE = "governance"
    ECONOMIC = "economic"
    UNKNOWN = "unknown"


class UnitOfAnalysis(str, Enum):
    """Analysis focus classification."""
    BASELINE_GAP = "baseline_gap"
    INDICATOR = "indicator"
    ACTIVITY = "activity"
    PRODUCT = "product"
    RESULT = "result"
    IMPACT = "impact"
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    FINANCIAL = "financial"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"


class DocumentType(str, Enum):
    """Document type classification for policy analysis.
    
    Different document types require different calibration approaches:
    - Plan de desarrollo municipal: Comprehensive, multi-sector, long-term planning
    - Política pública: Focused intervention, specific sector
    - Plan sectorial: Sector-specific planning
    """
    PLAN_DESARROLLO_MUNICIPAL = "plan_desarrollo_municipal"
    POLITICA_PUBLICA = "politica_publica"
    PLAN_SECTORIAL = "plan_sectorial"
    PLAN_ESTRATEGICO = "plan_estrategico"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CalibrationContext:
    """Complete context for calibration resolution.
    
    Attributes:
        question_id: Question identifier (e.g., "D1Q1", "D6Q3")
        dimension: Dimension number (1-10)
        question_num: Question number within dimension
        policy_area: Policy domain being analyzed
        unit_of_analysis: Type of analysis unit
        document_type: Type of policy document being analyzed
        method_position: Position in method sequence (0-indexed)
        total_methods: Total methods in sequence
    """
    question_id: str
    dimension: int
    question_num: int
    policy_area: PolicyArea = PolicyArea.UNKNOWN
    unit_of_analysis: UnitOfAnalysis = UnitOfAnalysis.UNKNOWN
    document_type: DocumentType = DocumentType.UNKNOWN
    method_position: int = 0
    total_methods: int = 1
    
    @classmethod
    def from_question_id(cls, question_id: str) -> "CalibrationContext":
        """Create context from question ID like 'D1Q1'."""
        try:
            parts = question_id.upper().replace("D", "").split("Q")
            if len(parts) == 2:
                dimension = int(parts[0])
                question_num = int(parts[1])
                return cls(
                    question_id=question_id,
                    dimension=dimension,
                    question_num=question_num
                )
        except (ValueError, IndexError):
            pass
        
        # Fallback for invalid format
        return cls(question_id=question_id, dimension=0, question_num=0)
    
    def with_policy_area(self, policy_area: PolicyArea) -> "CalibrationContext":
        """Create new context with specified policy area."""
        return replace(self, policy_area=policy_area)
    
    def with_unit_of_analysis(self, unit: UnitOfAnalysis) -> "CalibrationContext":
        """Create new context with specified unit of analysis."""
        return replace(self, unit_of_analysis=unit)
    
    def with_document_type(self, doc_type: DocumentType) -> "CalibrationContext":
        """Create new context with specified document type."""
        return replace(self, document_type=doc_type)
    
    def with_method_position(self, position: int, total: int) -> "CalibrationContext":
        """Create new context with method sequence position."""
        return replace(self, method_position=position, total_methods=total)


@dataclass(frozen=True)
class CalibrationModifier:
    """Multiplicative modifiers for calibration parameters.
    
    All modifiers are multiplicative (1.0 = no change).
    """
    min_evidence_multiplier: float = 1.0
    max_evidence_multiplier: float = 1.0
    contradiction_tolerance_multiplier: float = 1.0
    uncertainty_penalty_multiplier: float = 1.0
    aggregation_weight_multiplier: float = 1.0
    sensitivity_multiplier: float = 1.0
    
    def _apply_evidence_multiplier(self, value: int, multiplier: float) -> int:
        """Apply multiplier to evidence snippet count with minimum bound (round up)."""
        import math
        return max(1, int(math.ceil(value * multiplier)))
    
    def apply(self, base: MethodCalibration) -> MethodCalibration:
        """Apply modifiers to base calibration."""
        return MethodCalibration(
            score_min=base.score_min,
            score_max=base.score_max,
            min_evidence_snippets=self._apply_evidence_multiplier(
                base.min_evidence_snippets, self.min_evidence_multiplier
            ),
            max_evidence_snippets=self._apply_evidence_multiplier(
                base.max_evidence_snippets, self.max_evidence_multiplier
            ),
            contradiction_tolerance=_clamp(
                base.contradiction_tolerance * self.contradiction_tolerance_multiplier, 0.0, 1.0
            ),
            uncertainty_penalty=_clamp(
                base.uncertainty_penalty * self.uncertainty_penalty_multiplier, 0.0, 1.0
            ),
            aggregation_weight=max(0.1, base.aggregation_weight * self.aggregation_weight_multiplier),
            sensitivity=_clamp(base.sensitivity * self.sensitivity_multiplier, 0.0, 1.0),
            requires_numeric_support=base.requires_numeric_support,
            requires_temporal_support=base.requires_temporal_support,
            requires_source_provenance=base.requires_source_provenance,
        )


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range [min_val, max_val]."""
    return max(min_val, min(max_val, value))


# =============================================================================
# DIMENSION-SPECIFIC MODIFIERS
# =============================================================================

_DIMENSION_MODIFIERS = {
    # D1: Baseline gaps - requires high evidence, quantitative focus
    1: CalibrationModifier(
        min_evidence_multiplier=1.3,  # Need more evidence for gap analysis
        uncertainty_penalty_multiplier=0.8,  # More tolerant of uncertainty
        sensitivity_multiplier=1.1,  # Higher sensitivity to detect gaps
    ),
    
    # D2: Indicators - quantitative, needs precision
    2: CalibrationModifier(
        min_evidence_multiplier=1.2,
        contradiction_tolerance_multiplier=0.7,  # Less tolerant of contradictions
        sensitivity_multiplier=1.15,
    ),
    
    # D3: Activities - operational, mid-range evidence
    3: CalibrationModifier(
        min_evidence_multiplier=1.0,
        aggregation_weight_multiplier=1.05,
    ),
    
    # D4: Products - concrete outputs, moderate requirements
    4: CalibrationModifier(
        min_evidence_multiplier=1.1,
        uncertainty_penalty_multiplier=0.9,
    ),
    
    # D5: Results - outcome focus, higher evidence needs
    5: CalibrationModifier(
        min_evidence_multiplier=1.25,
        sensitivity_multiplier=1.1,
    ),
    
    # D6: Logical framework - structural coherence, high sensitivity
    6: CalibrationModifier(
        contradiction_tolerance_multiplier=0.5,  # Very intolerant of contradictions
        aggregation_weight_multiplier=1.2,
        sensitivity_multiplier=1.2,
    ),
    
    # D7: Causal mechanisms - needs strong evidence chains
    7: CalibrationModifier(
        min_evidence_multiplier=1.4,
        contradiction_tolerance_multiplier=0.6,
        sensitivity_multiplier=1.25,
    ),
    
    # D8: Temporal consistency - timeline focus
    8: CalibrationModifier(
        min_evidence_multiplier=1.2,
        contradiction_tolerance_multiplier=0.7,
    ),
    
    # D9: Financial coherence - precision critical
    9: CalibrationModifier(
        min_evidence_multiplier=1.35,
        contradiction_tolerance_multiplier=0.5,
        uncertainty_penalty_multiplier=1.2,
        sensitivity_multiplier=1.3,
    ),
    
    # D10: Operationalization - implementation feasibility
    10: CalibrationModifier(
        min_evidence_multiplier=1.25,
        aggregation_weight_multiplier=1.15,
        sensitivity_multiplier=1.15,
    ),
}


# =============================================================================
# POLICY-AREA-SPECIFIC MODIFIERS
# =============================================================================

_POLICY_AREA_MODIFIERS = {
    PolicyArea.FISCAL: CalibrationModifier(
        min_evidence_multiplier=1.3,  # Financial requires more evidence
        contradiction_tolerance_multiplier=0.6,  # Less tolerance for contradictions
        uncertainty_penalty_multiplier=1.2,  # Higher penalty for uncertainty
        sensitivity_multiplier=1.2,
    ),
    
    PolicyArea.SOCIAL: CalibrationModifier(
        min_evidence_multiplier=1.1,
        contradiction_tolerance_multiplier=0.9,
        uncertainty_penalty_multiplier=0.9,  # More tolerant of qualitative uncertainty
    ),
    
    PolicyArea.HEALTH: CalibrationModifier(
        min_evidence_multiplier=1.25,
        sensitivity_multiplier=1.15,
    ),
    
    PolicyArea.EDUCATION: CalibrationModifier(
        min_evidence_multiplier=1.15,
        aggregation_weight_multiplier=1.05,
    ),
    
    PolicyArea.INFRASTRUCTURE: CalibrationModifier(
        min_evidence_multiplier=1.4,  # Infrastructure needs concrete evidence
        contradiction_tolerance_multiplier=0.5,
        sensitivity_multiplier=1.25,
    ),
    
    PolicyArea.ENVIRONMENT: CalibrationModifier(
        min_evidence_multiplier=1.2,
        uncertainty_penalty_multiplier=0.85,
    ),
    
    PolicyArea.AGRICULTURE: CalibrationModifier(
        min_evidence_multiplier=1.15,
        sensitivity_multiplier=1.1,
    ),
    
    PolicyArea.SECURITY: CalibrationModifier(
        min_evidence_multiplier=1.35,
        contradiction_tolerance_multiplier=0.6,
        sensitivity_multiplier=1.2,
    ),
    
    PolicyArea.GOVERNANCE: CalibrationModifier(
        min_evidence_multiplier=1.2,
        aggregation_weight_multiplier=1.1,
    ),
    
    PolicyArea.ECONOMIC: CalibrationModifier(
        min_evidence_multiplier=1.3,
        sensitivity_multiplier=1.15,
    ),
}


# =============================================================================
# UNIT-OF-ANALYSIS-SPECIFIC MODIFIERS
# =============================================================================

_UNIT_OF_ANALYSIS_MODIFIERS = {
    UnitOfAnalysis.BASELINE_GAP: CalibrationModifier(
        min_evidence_multiplier=1.3,
        sensitivity_multiplier=1.2,
    ),
    
    UnitOfAnalysis.INDICATOR: CalibrationModifier(
        min_evidence_multiplier=1.25,
        contradiction_tolerance_multiplier=0.7,
        sensitivity_multiplier=1.15,
    ),
    
    UnitOfAnalysis.ACTIVITY: CalibrationModifier(
        min_evidence_multiplier=1.1,
        aggregation_weight_multiplier=1.05,
    ),
    
    UnitOfAnalysis.PRODUCT: CalibrationModifier(
        min_evidence_multiplier=1.15,
    ),
    
    UnitOfAnalysis.RESULT: CalibrationModifier(
        min_evidence_multiplier=1.25,
        sensitivity_multiplier=1.1,
    ),
    
    UnitOfAnalysis.IMPACT: CalibrationModifier(
        min_evidence_multiplier=1.4,
        sensitivity_multiplier=1.2,
    ),
    
    UnitOfAnalysis.QUANTITATIVE: CalibrationModifier(
        contradiction_tolerance_multiplier=0.6,
        uncertainty_penalty_multiplier=1.2,
        sensitivity_multiplier=1.15,
    ),
    
    UnitOfAnalysis.QUALITATIVE: CalibrationModifier(
        min_evidence_multiplier=1.2,
        contradiction_tolerance_multiplier=1.1,
        uncertainty_penalty_multiplier=0.8,
    ),
    
    UnitOfAnalysis.FINANCIAL: CalibrationModifier(
        min_evidence_multiplier=1.35,
        contradiction_tolerance_multiplier=0.5,
        uncertainty_penalty_multiplier=1.3,
        sensitivity_multiplier=1.25,
    ),
    
    UnitOfAnalysis.TEMPORAL: CalibrationModifier(
        min_evidence_multiplier=1.2,
        contradiction_tolerance_multiplier=0.75,
    ),
}


# =============================================================================
# DOCUMENT-TYPE-SPECIFIC MODIFIERS
# =============================================================================

_DOCUMENT_TYPE_MODIFIERS = {
    DocumentType.PLAN_DESARROLLO_MUNICIPAL: CalibrationModifier(
        min_evidence_multiplier=1.4,  # Municipal plans are extensive, need more evidence
        contradiction_tolerance_multiplier=0.6,  # Less tolerance due to multi-sector complexity
        uncertainty_penalty_multiplier=1.1,  # Higher penalty for vague statements
        sensitivity_multiplier=1.25,  # High sensitivity to detect cross-sector issues
        aggregation_weight_multiplier=1.15,  # Higher weight due to strategic importance
    ),
    
    DocumentType.POLITICA_PUBLICA: CalibrationModifier(
        min_evidence_multiplier=1.25,  # Focused intervention needs clear evidence
        contradiction_tolerance_multiplier=0.7,
        uncertainty_penalty_multiplier=1.05,
        sensitivity_multiplier=1.15,
        aggregation_weight_multiplier=1.1,
    ),
    
    DocumentType.PLAN_SECTORIAL: CalibrationModifier(
        min_evidence_multiplier=1.3,  # Sector-specific needs domain evidence
        contradiction_tolerance_multiplier=0.65,
        sensitivity_multiplier=1.2,
        aggregation_weight_multiplier=1.05,
    ),
    
    DocumentType.PLAN_ESTRATEGICO: CalibrationModifier(
        min_evidence_multiplier=1.35,  # Strategic plans need high-level evidence
        contradiction_tolerance_multiplier=0.6,
        uncertainty_penalty_multiplier=1.15,
        sensitivity_multiplier=1.3,  # Very sensitive to strategic gaps
        aggregation_weight_multiplier=1.2,
    ),
}


# =============================================================================
# METHOD-POSITION-SPECIFIC MODIFIERS
# =============================================================================

def _get_position_modifier(position: int, total: int) -> CalibrationModifier:
    """Get modifier based on method position in sequence.
    
    Early methods: Build foundation, need more evidence
    Middle methods: Balance aggregation
    Late methods: Synthesis, higher weights
    """
    if total <= 1:
        return CalibrationModifier()  # No adjustment for single methods
    
    # Calculate position ratio: 0.0 (first) to 1.0 (last)
    position_ratio = position / (total - 1) if total > 1 else 0.0
    
    if position_ratio < 0.33:  # Early methods
        return CalibrationModifier(
            min_evidence_multiplier=1.15,
            aggregation_weight_multiplier=0.95,
        )
    elif position_ratio < 0.67:  # Middle methods
        return CalibrationModifier(
            aggregation_weight_multiplier=1.0,
        )
    else:  # Late methods (synthesis)
        return CalibrationModifier(
            min_evidence_multiplier=0.9,  # Can rely on earlier work
            aggregation_weight_multiplier=1.15,  # Higher weight for final synthesis
            sensitivity_multiplier=1.05,
        )


# =============================================================================
# CONTEXTUAL CALIBRATION RESOLVER
# =============================================================================

def resolve_contextual_calibration(
    base_calibration: MethodCalibration,
    context: Optional[CalibrationContext] = None,
) -> MethodCalibration:
    """Resolve calibration with contextual refinements.
    
    Args:
        base_calibration: Base calibration from registry
        context: Optional execution context for refinement
        
    Returns:
        Refined calibration with context-aware adjustments
        
    The refinement process:
    1. Start with base calibration
    2. Apply dimension-specific modifier (if dimension known)
    3. Apply policy-area modifier (if area known)
    4. Apply unit-of-analysis modifier (if unit known)
    5. Apply document-type modifier (if document type known)
    6. Apply method-position modifier (if position known)
    
    All modifiers are multiplicative and cumulative.
    """
    if context is None:
        return base_calibration
    
    calibration = base_calibration
    
    # Apply dimension modifier
    if context.dimension in _DIMENSION_MODIFIERS:
        modifier = _DIMENSION_MODIFIERS[context.dimension]
        calibration = modifier.apply(calibration)
    
    # Apply policy area modifier
    if context.policy_area in _POLICY_AREA_MODIFIERS:
        modifier = _POLICY_AREA_MODIFIERS[context.policy_area]
        calibration = modifier.apply(calibration)
    
    # Apply unit of analysis modifier
    if context.unit_of_analysis in _UNIT_OF_ANALYSIS_MODIFIERS:
        modifier = _UNIT_OF_ANALYSIS_MODIFIERS[context.unit_of_analysis]
        calibration = modifier.apply(calibration)
    
    # Apply document type modifier
    if context.document_type in _DOCUMENT_TYPE_MODIFIERS:
        modifier = _DOCUMENT_TYPE_MODIFIERS[context.document_type]
        calibration = modifier.apply(calibration)
    
    # Apply method position modifier
    if context.total_methods > 1:
        modifier = _get_position_modifier(context.method_position, context.total_methods)
        calibration = modifier.apply(calibration)
    
    return calibration


# =============================================================================
# QUESTION-ID TO CONTEXT INFERENCE
# =============================================================================

def infer_context_from_question_id(question_id: str) -> CalibrationContext:
    """Infer calibration context from question ID.
    
    This provides reasonable defaults for policy area and unit of analysis
    based on dimension and question patterns.
    """
    context = CalibrationContext.from_question_id(question_id)
    
    # Infer policy area and unit based on dimension
    if context.dimension == 1:
        context = context.with_unit_of_analysis(UnitOfAnalysis.BASELINE_GAP)
    elif context.dimension == 2:
        context = context.with_unit_of_analysis(UnitOfAnalysis.INDICATOR)
    elif context.dimension == 3:
        context = context.with_unit_of_analysis(UnitOfAnalysis.ACTIVITY)
    elif context.dimension == 4:
        context = context.with_unit_of_analysis(UnitOfAnalysis.PRODUCT)
    elif context.dimension == 5:
        context = context.with_unit_of_analysis(UnitOfAnalysis.RESULT)
    elif context.dimension == 6:
        # D6 is about logical framework - qualitative
        context = context.with_unit_of_analysis(UnitOfAnalysis.QUALITATIVE)
    elif context.dimension == 7:
        # D7 is causal mechanisms - qualitative
        context = context.with_unit_of_analysis(UnitOfAnalysis.QUALITATIVE)
    elif context.dimension == 8:
        context = context.with_unit_of_analysis(UnitOfAnalysis.TEMPORAL)
    elif context.dimension == 9:
        context = context.with_unit_of_analysis(UnitOfAnalysis.FINANCIAL)
    elif context.dimension == 10:
        # D10 is operationalization - impact focus
        context = context.with_unit_of_analysis(UnitOfAnalysis.IMPACT)
    
    return context
