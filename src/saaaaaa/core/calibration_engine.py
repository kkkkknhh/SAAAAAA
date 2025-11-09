"""
Method Calibration Engine

Implements Cal(M) = F_M({x_ℓ(M)}_ℓ∈L(M)) with full transparency and verification.

This module provides:
1. Calibration computation for methods based on layer scores
2. Automatic formula generation in canonical notation
3. Step-by-step arithmetic traces
4. Property verification (monotonicity, boundedness, etc.)
5. Machine-verifiable proofs
6. Endogenous determination of L(M) from method characteristics
7. Layer influence application

All calibration behavior is explicit - no hidden defaults or undocumented interactions.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from .layer_coexistence import (
    Layer, LayerScore, MethodSignature, FusionOperator,
    create_fusion_operator
)
from .layer_influence_model import (
    CANONICAL_LAYER_MODEL, LayerCoexistenceModel
)


@dataclass
class CalibrationResult:
    """
    Result of calibrating a method with full transparency.
    
    Attributes:
        method_id: Unique method identifier
        calibrated_score: Final Cal(M) value
        layer_scores: Individual layer scores used
        fusion_operator: Name of F_M used
        formula: Explicit mathematical formula
        computation_trace: Step-by-step arithmetic
        property_checks: Verification results
    """
    method_id: str
    calibrated_score: float
    layer_scores: List[LayerScore]
    fusion_operator: str
    formula: str
    computation_trace: List[str]
    property_checks: Dict[str, bool]
    layer_influences: List[str] = None  # NEW: Record of applied influences
    compatibility_score: float = 1.0    # NEW: Layer compatibility score
    
    def __post_init__(self):
        """Initialize derived fields"""
        if self.layer_influences is None:
            self.layer_influences = []
    
    def to_dict(self) -> Dict:
        """Export to dictionary"""
        return {
            'method_id': self.method_id,
            'calibrated_score': self.calibrated_score,
            'layer_scores': [
                {
                    'layer': score.layer.value,
                    'value': score.value,
                    'weight': score.weight,
                    'metadata': score.metadata
                }
                for score in self.layer_scores
            ],
            'fusion_operator': self.fusion_operator,
            'formula': self.formula,
            'computation_trace': self.computation_trace,
            'property_checks': self.property_checks,
            'layer_influences': self.layer_influences,
            'compatibility_score': self.compatibility_score
        }
    
    def __str__(self) -> str:
        """Human-readable representation"""
        lines = [
            f"Calibration Result for {self.method_id}",
            f"=" * 60,
            f"Calibrated Score: {self.calibrated_score:.4f}",
            f"",
            f"Formula: {self.formula}",
            f"",
            f"Layer Scores:",
        ]
        
        for score in self.layer_scores:
            lines.append(f"  {score.layer.value}: {score.value:.4f} (weight={score.weight:.4f})")
        
        lines.extend([
            f"",
            f"Computation Trace:",
        ])
        lines.extend([f"  {step}" for step in self.computation_trace])
        
        lines.extend([
            f"",
            f"Property Verification:",
        ])
        for prop, passed in self.property_checks.items():
            status = "✓" if passed else "✗"
            lines.append(f"  {status} {prop}")
        
        return "\n".join(lines)


class MethodCalibrationEngine:
    """
    Engine for computing calibrated scores with full transparency.
    
    This is the core implementation of:
        Cal(M) = F_M({x_ℓ(M)}_ℓ∈L(M))
    
    Enhanced with:
    - Endogenous L(M) determination from method characteristics
    - Layer influence application
    - Compatibility verification
    
    Usage:
        engine = MethodCalibrationEngine()
        
        # Option 1: Provide signature with pre-defined L(M)
        signature = MethodSignature(...)
        layer_scores = [LayerScore(...), ...]
        result = engine.calibrate(signature, layer_scores)
        
        # Option 2: Auto-derive L(M) from characteristics
        characteristics = {'operates_on_questions': True, ...}
        result = engine.calibrate_from_characteristics(
            method_id="Class.method",
            characteristics=characteristics,
            layer_scores=layer_scores
        )
    """
    
    def __init__(self, layer_model: Optional[LayerCoexistenceModel] = None):
        self.signatures: Dict[str, MethodSignature] = {}
        self.layer_model = layer_model or CANONICAL_LAYER_MODEL
    
    def register_method(self, signature: MethodSignature):
        """
        Register a method signature for calibration.
        
        Args:
            signature: Complete MethodSignature with all required fields
        """
        self.signatures[signature.method_id] = signature
    
    def calibrate(
        self, 
        signature: MethodSignature, 
        layer_scores: List[LayerScore],
        apply_influences: bool = True
    ) -> CalibrationResult:
        """
        Compute calibrated score for a method.
        
        Args:
            signature: Method signature defining F_M and L(M)
            layer_scores: Layer-specific scores {x_ℓ(M)}_ℓ∈L(M)
            apply_influences: Whether to apply layer influence model
            
        Returns:
            CalibrationResult with score, formula, trace, and verification
            
        Raises:
            ValueError: If layer_scores don't match signature.active_layers
        """
        # Validate that provided layers match signature
        provided_layers = {score.layer for score in layer_scores}
        if not provided_layers.issubset(signature.active_layers):
            raise ValueError(
                f"Layer mismatch for {signature.method_id}. "
                f"Expected layers: {signature.active_layers}, "
                f"Got: {provided_layers}"
            )
        
        # Check layer compatibility
        is_compatible, compat_score = self.layer_model.check_compatibility(
            signature.active_layers
        )
        if not is_compatible:
            raise ValueError(
                f"Incompatible layer set for {signature.method_id}: "
                f"{signature.active_layers} (compatibility={compat_score:.2f})"
            )
        
        # Apply layer influences if requested
        layer_influences_applied = []
        if apply_influences:
            layer_scores, layer_influences_applied = self._apply_layer_influences(
                layer_scores,
                signature.active_layers
            )
        
        # Create fusion operator
        fusion_op = create_fusion_operator(
            signature.fusion_operator_name,
            signature.fusion_parameters
        )
        
        # Compute calibrated score
        calibrated_score = fusion_op.fuse(layer_scores)
        
        # Get formula
        formula = fusion_op.get_formula()
        
        # Get computation trace
        trace = fusion_op.get_trace(layer_scores)
        
        # Verify properties
        property_checks = fusion_op.verify_properties()
        
        return CalibrationResult(
            method_id=signature.method_id,
            calibrated_score=calibrated_score,
            layer_scores=layer_scores,
            fusion_operator=signature.fusion_operator_name,
            formula=formula,
            computation_trace=trace,
            property_checks=property_checks,
            layer_influences=layer_influences_applied,
            compatibility_score=compat_score
        )
    
    def _apply_layer_influences(
        self,
        layer_scores: List[LayerScore],
        active_layers: Set[Layer]
    ) -> Tuple[List[LayerScore], List[str]]:
        """
        Apply layer influence model to adjust scores and weights.
        
        Args:
            layer_scores: Original layer scores
            active_layers: Set of active layers
            
        Returns:
            (adjusted_scores, influence_log)
        """
        # Convert to dict for easier manipulation
        score_dict = {score.layer: score for score in layer_scores}
        influence_log = []
        
        # Apply influences to each layer
        adjusted_scores = []
        for layer in active_layers:
            if layer not in score_dict:
                continue
            
            original_score = score_dict[layer]
            
            # Compute effective weight after influences
            effective_weight = self.layer_model.compute_effective_weight(
                target_layer=layer,
                base_weight=original_score.weight,
                layer_scores=score_dict,
                active_layers=active_layers
            )
            
            # Log if weight changed
            if abs(effective_weight - original_score.weight) > 1e-6:
                influence_log.append(
                    f"{layer.value}: weight {original_score.weight:.4f} → {effective_weight:.4f}"
                )
            
            # Create adjusted score
            adjusted_scores.append(LayerScore(
                layer=layer,
                value=original_score.value,
                weight=effective_weight,
                metadata={
                    **original_score.metadata,
                    'original_weight': original_score.weight,
                    'influences_applied': True
                }
            ))
        
        return adjusted_scores, influence_log
    
    def calibrate_from_characteristics(
        self,
        method_id: str,
        characteristics: Dict,
        fusion_operator_name: str,
        fusion_parameters: Optional[Dict] = None,
        raw_layer_scores: Optional[Dict[Layer, float]] = None
    ) -> CalibrationResult:
        """
        Calibrate a method by auto-deriving L(M) from characteristics.
        
        This implements the endogenous determination of active layers.
        
        Args:
            method_id: Unique method identifier
            characteristics: Method characteristics dict with keys like:
                - 'operates_on_questions': bool
                - 'aggregates_dimensions': bool
                - 'addresses_policy_areas': bool
                - 'uses_ensemble': bool
                - 'performs_meta_aggregation': bool
                - 'question_count': int
                - 'dimension_count': int
                - 'policy_area_count': int
            fusion_operator_name: Name of fusion operator to use
            fusion_parameters: Parameters for fusion operator
            raw_layer_scores: Optional dict of Layer -> score value
                If not provided, uses default 0.5 for all active layers
            
        Returns:
            CalibrationResult
        """
        # Determine active layers endogenously
        active_layers = self.layer_model.determine_active_layers(characteristics)
        
        if not active_layers:
            raise ValueError(
                f"No layers activated for {method_id} with characteristics: "
                f"{characteristics}"
            )
        
        # Create layer scores
        if raw_layer_scores is None:
            raw_layer_scores = {layer: 0.5 for layer in active_layers}
        
        layer_scores = [
            LayerScore(layer, raw_layer_scores.get(layer, 0.5), weight=1.0)
            for layer in active_layers
            if layer in raw_layer_scores or layer in active_layers
        ]
        
        # Create signature
        signature = MethodSignature(
            method_id=method_id,
            active_layers=active_layers,
            input_schema=characteristics,
            output_schema={'calibrated_score': 'float'},
            fusion_operator_name=fusion_operator_name,
            fusion_parameters=fusion_parameters or {},
            calibration_rule=(
                f"Auto-derived L(M)={{{', '.join(l.value for l in active_layers)}}} "
                f"from characteristics: {list(characteristics.keys())}"
            )
        )
        
        return self.calibrate(signature, layer_scores, apply_influences=True)
    
    def calibrate_by_id(
        self,
        method_id: str,
        layer_scores: List[LayerScore]
    ) -> CalibrationResult:
        """
        Calibrate a registered method by ID.
        
        Args:
            method_id: Registered method identifier
            layer_scores: Layer scores
            
        Returns:
            CalibrationResult
        """
        if method_id not in self.signatures:
            raise ValueError(f"Method {method_id} not registered")
        
        signature = self.signatures[method_id]
        return self.calibrate(signature, layer_scores)
    
    def export_signature_registry(self, output_path: Path):
        """
        Export all registered method signatures to JSON.
        
        Args:
            output_path: Path to write JSON file
        """
        registry = {
            method_id: signature.to_dict()
            for method_id, signature in self.signatures.items()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    
    def import_signature_registry(self, input_path: Path):
        """
        Import method signatures from JSON.
        
        Args:
            input_path: Path to JSON file
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        for method_id, signature_dict in registry.items():
            signature = MethodSignature.from_dict(signature_dict)
            self.register_method(signature)


def demonstrate_calibration():
    """
    Demonstration of the calibration system with endogenous L(M) determination.
    
    This shows the complete workflow:
    1. Define method characteristics
    2. Auto-derive active layers L(M)
    3. Apply layer influence model
    4. Compute calibration
    5. Display transparency information
    """
    print("="*70)
    print("DEMONSTRATION: Endogenous Layer Activation and Influence")
    print("="*70)
    
    # Example 1: Municipal plan analyzer
    print("\n[Example 1] Municipal Plan Analyzer")
    print("-" * 70)
    
    characteristics = {
        'operates_on_questions': True,
        'aggregates_dimensions': True,
        'addresses_policy_areas': True,
        'uses_ensemble': False,
        'performs_meta_aggregation': False,
        'question_count': 250,
        'dimension_count': 6,
        'policy_area_count': 8
    }
    
    layer_scores = {
        Layer.QUESTION: 0.75,
        Layer.DIMENSION: 0.82,
        Layer.POLICY_AREA: 0.68
    }
    
    engine = MethodCalibrationEngine()
    result = engine.calibrate_from_characteristics(
        method_id="PDETMunicipalPlanAnalyzer.analyze_municipal_plan",
        characteristics=characteristics,
        fusion_operator_name="WeightedAverage",
        raw_layer_scores=layer_scores
    )
    
    print(result)
    
    # Example 2: Question-only method
    print("\n" + "="*70)
    print("[Example 2] Question Scorer (fewer layers)")
    print("-" * 70)
    
    characteristics2 = {
        'operates_on_questions': True,
        'aggregates_dimensions': False,
        'question_count': 50
    }
    
    layer_scores2 = {
        Layer.QUESTION: 0.85
    }
    
    result2 = engine.calibrate_from_characteristics(
        method_id="QuestionScorer.score_questions",
        characteristics=characteristics2,
        fusion_operator_name="WeightedAverage",
        raw_layer_scores=layer_scores2
    )
    
    print(result2)
    
    # Example 3: Meta-aggregator with ensemble
    print("\n" + "="*70)
    print("[Example 3] Meta-Aggregator with Ensemble")
    print("-" * 70)
    
    characteristics3 = {
        'operates_on_questions': True,
        'aggregates_dimensions': True,
        'addresses_policy_areas': True,
        'uses_ensemble': True,
        'performs_meta_aggregation': True,
        'question_count': 300,
        'dimension_count': 6,
        'policy_area_count': 10,
        'ensemble_method_count': 3
    }
    
    layer_scores3 = {
        Layer.QUESTION: 0.78,
        Layer.DIMENSION: 0.84,
        Layer.POLICY_AREA: 0.72,
        Layer.CONGRUENCE: 0.90,
        Layer.META: 0.80
    }
    
    result3 = engine.calibrate_from_characteristics(
        method_id="MetaAggregator.aggregate_all_layers",
        characteristics=characteristics3,
        fusion_operator_name="OWA",
        fusion_parameters={'weights': [0.4, 0.3, 0.2, 0.1]},
        raw_layer_scores=layer_scores3
    )
    
    print(result3)


if __name__ == "__main__":
    demonstrate_calibration()
