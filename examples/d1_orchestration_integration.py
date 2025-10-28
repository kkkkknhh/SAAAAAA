"""Example Integration: D1 Orchestration in Policy Analysis Pipeline.

This example demonstrates how to integrate the D1 orchestrator into the
existing policy analysis pipeline for deterministic, contract-enforced
diagnostic evaluation.

USAGE:
    python examples/d1_orchestration_integration.py <pdf_path>
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from orchestrator.d1_orchestrator import (
        D1Question,
        D1QuestionOrchestrator,
        D1OrchestrationError,
    )
    from orchestrator.canonical_registry import CANONICAL_METHODS
except ImportError as e:
    print(f"Error importing orchestration modules: {e}", file=sys.stderr)
    print("Note: This example requires the D1 orchestrator to be properly set up.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_context_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Prepare execution context from a PDF policy document.
    
    In a real implementation, this would:
    - Extract text from PDF using PyPDF2, pdfplumber, or similar
    - Parse tables using camelot or tabula
    - Extract metadata (plan name, year, municipality, etc.)
    - Pre-process and normalize text
    
    Args:
        pdf_path: Path to PDF policy document
        
    Returns:
        Execution context dictionary
    """
    # Mock implementation for demonstration
    logger.info(f"Preparing context from: {pdf_path}")
    
    context = {
        "pdf_path": str(pdf_path),
        "text": "Sample policy document text...",  # Would be extracted from PDF
        "metadata": {
            "plan_name": "Plan de Desarrollo Municipal - Example",
            "year": 2024,
            "municipality": "Example Municipality",
        },
        "data": {
            "tables": [],  # Would be extracted tables
            "figures": [],  # Would be extracted figures
        },
    }
    
    return context


def orchestrate_d1_diagnostic(context: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    """Orchestrate all D1 diagnostic questions with contract enforcement.
    
    Args:
        context: Execution context prepared from policy document
        strict: If True, abort on any method failure (SIN_CARRETA doctrine)
        
    Returns:
        Dictionary with orchestration results and audit report
    """
    logger.info("=" * 80)
    logger.info("D1 DIAGNOSTIC ORCHESTRATION")
    logger.info("SIN_CARRETA Doctrine: Strict Method Concurrence")
    logger.info("=" * 80)
    
    # Initialize orchestrator with canonical registry
    orchestrator = D1QuestionOrchestrator(canonical_registry=CANONICAL_METHODS)
    
    # Orchestrate each D1 question
    results = []
    all_success = True
    
    for question in D1Question:
        logger.info(f"\nOrchestrating {question.value}...")
        
        try:
            result = orchestrator.orchestrate_question(
                question,
                context,
                strict=strict
            )
            
            results.append(result)
            
            if result.success:
                logger.info(
                    f"✓ {question.value}: All {len(result.executed_methods)} methods succeeded "
                    f"({result.total_duration_ms:.2f}ms)"
                )
            else:
                all_success = False
                logger.warning(
                    f"✗ {question.value}: {len(result.failed_methods)} methods failed"
                )
                logger.warning(f"  Error: {result.error_summary}")
        
        except D1OrchestrationError as e:
            all_success = False
            logger.error(f"✗ {question.value}: Orchestration failed (strict mode)")
            logger.error(f"  Failed methods: {len(e.failed_methods)}")
            logger.error(f"  Error: {str(e)}")
            
            # Re-raise in strict mode to abort pipeline
            if strict:
                raise
    
    # Generate audit report
    audit_report = orchestrator.generate_audit_report(results)
    
    logger.info("\n" + "=" * 80)
    logger.info("ORCHESTRATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Questions Orchestrated: {audit_report['summary']['total_questions']}")
    logger.info(f"Successful Questions: {audit_report['summary']['successful_questions']}")
    logger.info(f"Failed Questions: {audit_report['summary']['failed_questions']}")
    logger.info(f"Total Methods: {audit_report['summary']['total_methods_executed']}")
    logger.info(f"Success Rate: {audit_report['summary']['overall_success_rate']:.1%}")
    
    logger.info("\nDoctrine Compliance:")
    for principle, satisfied in audit_report['doctrine_compliance'].items():
        status = "✓" if satisfied else "✗"
        logger.info(f"  {status} {principle.replace('_', ' ').title()}")
    
    return {
        "success": all_success,
        "results": results,
        "audit_report": audit_report,
    }


def extract_diagnostic_insights(orchestration_output: Dict[str, Any]) -> Dict[str, Any]:
    """Extract diagnostic insights from orchestration results.
    
    This would typically:
    - Aggregate method results across questions
    - Calculate diagnostic scores and confidence intervals
    - Identify gaps and weaknesses
    - Generate recommendations
    
    Args:
        orchestration_output: Output from orchestrate_d1_diagnostic
        
    Returns:
        Diagnostic insights dictionary
    """
    results = orchestration_output["results"]
    
    insights = {
        "diagnostic_coverage": {
            "baseline_quantification": None,  # From D1-Q1 results
            "source_normalization": None,     # From D1-Q2 results
            "resource_allocation": None,      # From D1-Q3 results
            "institutional_capacity": None,   # From D1-Q4 results
            "temporal_constraints": None,     # From D1-Q5 results
        },
        "overall_diagnostic_quality": None,
        "critical_gaps": [],
        "recommendations": [],
    }
    
    # Example: Extract insights from each question's results
    for result in results:
        if result.question_id == "D1-Q1" and result.success:
            # Extract baseline quantification insights
            method_results = result.method_results
            # Process BayesianNumericalAnalyzer.evaluate_policy_metric results
            # Calculate coverage scores
            insights["diagnostic_coverage"]["baseline_quantification"] = {
                "score": 0.0,  # Would be calculated from method results
                "confidence_interval": (0.0, 0.0),
                "evidence_count": len(method_results),
            }
    
    return insights


def main():
    """Main example entry point."""
    if len(sys.argv) < 2:
        print("Usage: python d1_orchestration_integration.py <pdf_path>")
        print("\nExample:")
        print("  python d1_orchestration_integration.py /path/to/pdm.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    try:
        # Step 1: Prepare execution context
        context = prepare_context_from_pdf(pdf_path)
        
        # Step 2: Orchestrate D1 diagnostic with strict contract enforcement
        orchestration_output = orchestrate_d1_diagnostic(
            context,
            strict=True  # Enforce SIN_CARRETA doctrine
        )
        
        # Step 3: Extract diagnostic insights
        insights = extract_diagnostic_insights(orchestration_output)
        
        # Step 4: Output results
        output = {
            "orchestration": orchestration_output["audit_report"],
            "insights": insights,
        }
        
        output_path = Path("d1_orchestration_output.json")
        output_path.write_text(json.dumps(output, indent=2))
        logger.info(f"\n✓ Results written to: {output_path}")
        
        if orchestration_output["success"]:
            logger.info("\n✓✓✓ D1 ORCHESTRATION COMPLETED SUCCESSFULLY ✓✓✓")
            sys.exit(0)
        else:
            logger.warning("\n⚠ D1 ORCHESTRATION COMPLETED WITH WARNINGS ⚠")
            sys.exit(0)
    
    except D1OrchestrationError as e:
        logger.error("\n✗✗✗ D1 ORCHESTRATION FAILED ✗✗✗")
        logger.error(f"Contract Violation: {str(e)}")
        logger.error(f"Failed Methods: {', '.join(e.failed_methods[:10])}")
        sys.exit(1)
    
    except Exception as e:
        logger.exception("Unexpected error during orchestration")
        sys.exit(2)


if __name__ == "__main__":
    main()
