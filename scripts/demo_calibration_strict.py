#!/usr/bin/env python3
"""Demo script showing strict calibration enforcement.

This script demonstrates the new calibration system:
1. Missing calibrations raise MissingCalibrationError
2. Context-aware calibration adjusts requirements based on document type
3. Calibration version and hash are tracked for reproducibility
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from saaaaaa.core.orchestrator.calibration_registry import (
    MissingCalibrationError,
    resolve_calibration,
    get_calibration_hash,
    CALIBRATION_VERSION,
    CALIBRATIONS,
)
from saaaaaa.core.orchestrator.calibration_context import (
    CalibrationContext,
    DocumentType,
    PolicyArea,
    UnitOfAnalysis,
    resolve_contextual_calibration,
)


def demo_strict_enforcement():
    """Demonstrate strict calibration enforcement."""
    print("=" * 80)
    print("DEMO: Strict Calibration Enforcement")
    print("=" * 80)
    print()
    
    print(f"Calibration System Version: {CALIBRATION_VERSION}")
    print(f"Calibration Hash: {get_calibration_hash()[:32]}...")
    print(f"Total Calibrations: {len(CALIBRATIONS)}")
    print()
    
    # Demo 1: Missing calibration raises error
    print("1. Missing Calibration (Strict Mode)")
    print("-" * 40)
    try:
        resolve_calibration("UncalibratedClass", "uncalibrated_method", strict=True)
        print("ERROR: Should have raised MissingCalibrationError")
    except MissingCalibrationError as e:
        print(f"✓ Execution blocked: {e}")
    print()
    
    # Demo 2: Successful resolution
    print("2. Valid Calibration Resolution")
    print("-" * 40)
    calib = resolve_calibration("BayesianEvidenceScorer", "compute_evidence_score", strict=True)
    print(f"✓ Method: BayesianEvidenceScorer.compute_evidence_score")
    print(f"  Evidence required: {calib.min_evidence_snippets}-{calib.max_evidence_snippets} snippets")
    print(f"  Sensitivity: {calib.sensitivity}")
    print(f"  Requires numeric support: {calib.requires_numeric_support}")
    print()
    
    # Demo 3: Context-aware calibration
    print("3. Context-Aware Calibration")
    print("-" * 40)
    
    # Base calibration
    base = resolve_calibration("BayesianNumericalAnalyzer", "evaluate_policy_metric", strict=True)
    print(f"Base calibration: {base.min_evidence_snippets} evidence snippets")
    
    # Municipal plan + financial dimension
    context_municipal = CalibrationContext(
        question_id="D9Q1",
        dimension=9,  # Financial dimension
        question_num=1,
        document_type=DocumentType.PLAN_DESARROLLO_MUNICIPAL,
        policy_area=PolicyArea.FISCAL,
        unit_of_analysis=UnitOfAnalysis.FINANCIAL,
    )
    
    calib_municipal = resolve_contextual_calibration(base, context_municipal)
    print(f"Municipal plan (D9, fiscal, financial): {calib_municipal.min_evidence_snippets} evidence snippets")
    print(f"  → {(calib_municipal.min_evidence_snippets / base.min_evidence_snippets - 1) * 100:.0f}% stricter requirements")
    
    # Simple policy
    context_policy = CalibrationContext(
        question_id="D3Q1",
        dimension=3,  # Activities
        question_num=1,
        document_type=DocumentType.POLITICA_PUBLICA,
    )
    
    calib_policy = resolve_contextual_calibration(base, context_policy)
    print(f"Policy (D3, activities): {calib_policy.min_evidence_snippets} evidence snippets")
    print()
    
    # Demo 4: Document type specificity
    print("4. Document Type Impact")
    print("-" * 40)
    
    for doc_type in [DocumentType.PLAN_DESARROLLO_MUNICIPAL, 
                     DocumentType.POLITICA_PUBLICA,
                     DocumentType.PLAN_SECTORIAL,
                     DocumentType.PLAN_ESTRATEGICO]:
        context = CalibrationContext(
            question_id="D1Q1",
            dimension=1,
            question_num=1,
            document_type=doc_type,
        )
        calib = resolve_contextual_calibration(base, context)
        print(f"  {doc_type.value:30s}: {calib.min_evidence_snippets:2d} evidence snippets")
    
    print()
    print("=" * 80)
    print("Demo Complete - Calibration system enforces rigorous requirements")
    print("=" * 80)


if __name__ == "__main__":
    demo_strict_enforcement()
