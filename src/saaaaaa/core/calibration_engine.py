"""
Method Calibration Engine

Implements Cal(M) = F_M({x_ℓ(M)}_ℓ∈L(M)) with full transparency and verification.

This module provides:
1. Calibration computation for methods based on layer scores
2. Automatic formula generation in canonical notation
3. Step-by-step arithmetic traces
4. Property verification (monotonicity, boundedness, etc.)
5. Machine-verifiable proofs

All calibration behavior is explicit - no hidden defaults or undocumented interactions.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from .layer_coexistence import (
    Layer, LayerScore, MethodSignature, FusionOperator,
    create_fusion_operator
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
            'property_checks': self.property_checks
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
    
    Usage:
        engine = MethodCalibrationEngine()
        signature = MethodSignature(...)
        layer_scores = [LayerScore(...), ...]
        result = engine.calibrate(signature, layer_scores)
    """
    
    def __init__(self):
        self.signatures: Dict[str, MethodSignature] = {}
    
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
        layer_scores: List[LayerScore]
    ) -> CalibrationResult:
        """
        Compute calibrated score for a method.
        
        Args:
            signature: Method signature defining F_M and L(M)
            layer_scores: Layer-specific scores {x_ℓ(M)}_ℓ∈L(M)
            
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
            property_checks=property_checks
        )
    
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
    Demonstration of the calibration system with a sample method.
    
    This shows the complete workflow:
    1. Define method signature
    2. Provide layer scores
    3. Compute calibration
    4. Display transparency information
    """
    # Create method signature
    signature = MethodSignature(
        method_id="PDETMunicipalPlanAnalyzer.analyze_municipal_plan",
        active_layers={Layer.QUESTION, Layer.DIMENSION, Layer.POLICY_AREA},
        input_schema={
            'plan_document': 'PDF',
            'questionnaire': 'QuestionnaireMonolith'
        },
        output_schema={
            'analysis_report': 'Dict',
            'scores': 'Dict[str, float]'
        },
        fusion_operator_name="WeightedAverage",
        fusion_parameters={'normalize_weights': True},
        calibration_rule=(
            "Calibration combines question-level evidence (@q), "
            "dimension-level aggregation (@d), and policy area coherence (@p) "
            "using weighted average fusion with equal weights."
        )
    )
    
    # Create sample layer scores
    layer_scores = [
        LayerScore(Layer.QUESTION, 0.75, weight=1.0, metadata={'questions_answered': 250}),
        LayerScore(Layer.DIMENSION, 0.82, weight=1.0, metadata={'dimensions_evaluated': 6}),
        LayerScore(Layer.POLICY_AREA, 0.68, weight=1.0, metadata={'policy_areas_covered': 8}),
    ]
    
    # Create engine and calibrate
    engine = MethodCalibrationEngine()
    result = engine.calibrate(signature, layer_scores)
    
    # Display results
    print(result)
    print("\n" + "="*60)
    print("JSON Export:")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    demonstrate_calibration()
