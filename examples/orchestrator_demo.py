"""
Demonstration of the Orchestrator module.

This script shows how to use the Orchestrator to process
a policy document through all pipeline phases.
"""

import logging
from pathlib import Path

from orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    ProcessingPhase,
)


def demo_basic_orchestrator():
    """Demonstrate basic orchestrator usage."""
    print("=" * 70)
    print("ORCHESTRATOR BASIC DEMO")
    print("=" * 70)
    
    # Create orchestrator with default configuration
    orchestrator = Orchestrator()
    
    print(f"\n1. Orchestrator initialized")
    print(f"   - Max workers: {orchestrator.config.max_workers}")
    print(f"   - Default timeout: {orchestrator.config.default_question_timeout}s")
    print(f"   - Min completion rate: {orchestrator.config.min_completion_rate}")
    
    # Get initial status
    status = orchestrator.get_processing_status()
    print(f"\n2. Initial status:")
    print(f"   - Current phase: {status.current_phase.name}")
    print(f"   - Progress: {status.completion_percentage:.1f}%")
    print(f"   - Questions: {status.questions_completed}/{status.questions_total}")


def demo_custom_configuration():
    """Demonstrate custom configuration."""
    print("\n" + "=" * 70)
    print("ORCHESTRATOR CUSTOM CONFIGURATION DEMO")
    print("=" * 70)
    
    # Create custom configuration
    config = OrchestratorConfig(
        max_workers=25,
        default_question_timeout=120.0,
        min_completion_rate=0.85,
        log_level="DEBUG",
    )
    
    orchestrator = Orchestrator(config=config)
    
    print(f"\n1. Custom orchestrator created:")
    print(f"   - Max workers: {orchestrator.config.max_workers}")
    print(f"   - Question timeout: {orchestrator.config.default_question_timeout}s")
    print(f"   - Min completion: {orchestrator.config.min_completion_rate:.1%}")
    print(f"   - Log level: {orchestrator.config.log_level}")


def demo_validation_phase():
    """Demonstrate validation phase."""
    print("\n" + "=" * 70)
    print("ORCHESTRATOR VALIDATION PHASE DEMO")
    print("=" * 70)
    
    orchestrator = Orchestrator()
    
    print("\n1. Validating configuration...")
    
    try:
        result = orchestrator.validate_configuration()
        print(f"   ✓ Validation passed: {result}")
        
        # Show what was loaded
        if orchestrator.monolith:
            micro_count = len(orchestrator.monolith['blocks']['micro_questions'])
            print(f"   ✓ Loaded monolith: {micro_count} micro questions")
        
        if orchestrator.method_catalog:
            method_count = len(orchestrator.method_catalog)
            print(f"   ✓ Loaded catalog: {method_count} method packages")
        
        # Show validation metrics
        metrics = orchestrator.state.phase_metrics.get(
            ProcessingPhase.PHASE_0_VALIDATION
        )
        if metrics:
            print(f"\n2. Validation metrics:")
            print(f"   - Status: {metrics.status.value}")
            print(f"   - Duration: {metrics.duration_seconds:.3f}s")
            print(f"   - Errors: {len(metrics.errors)}")
            print(f"   - Warnings: {len(metrics.warnings)}")
        
    except Exception as e:
        print(f"   ✗ Validation failed: {e}")


def demo_metrics_tracking():
    """Demonstrate metrics tracking."""
    print("\n" + "=" * 70)
    print("ORCHESTRATOR METRICS TRACKING DEMO")
    print("=" * 70)
    
    orchestrator = Orchestrator()
    
    print("\n1. Getting comprehensive metrics...")
    
    metrics = orchestrator.get_metrics()
    
    print(f"\n2. System metrics:")
    print(f"   - Total questions: {metrics['total_questions']}")
    print(f"   - Completed: {metrics['completed_questions']}")
    print(f"   - Progress: {metrics['progress']:.1%}")
    print(f"   - Elapsed time: {metrics['elapsed_time']:.2f}s")
    
    if metrics['phase_metrics']:
        print(f"\n3. Phase metrics:")
        for phase_name, phase_data in metrics['phase_metrics'].items():
            print(f"   - {phase_name}:")
            print(f"     Status: {phase_data['status']}")
            print(f"     Duration: {phase_data['duration']:.3f}s")


def demo_abort_control():
    """Demonstrate abort control."""
    print("\n" + "=" * 70)
    print("ORCHESTRATOR ABORT CONTROL DEMO")
    print("=" * 70)
    
    orchestrator = Orchestrator()
    
    print("\n1. Abort control features:")
    print(f"   - Abort requested: {orchestrator.abort_requested}")
    
    print("\n2. Requesting abort...")
    orchestrator.request_abort()
    
    print(f"   - Abort requested: {orchestrator.abort_requested}")
    print("   ✓ Abort flag set successfully")
    
    print("\n3. Future phase executions will be cancelled")


def demo_phase_enumeration():
    """Demonstrate phase enumeration."""
    print("\n" + "=" * 70)
    print("ORCHESTRATOR PHASES DEMO")
    print("=" * 70)
    
    print("\n1. All pipeline phases:")
    for phase in ProcessingPhase:
        print(f"   {phase.value:2d}. {phase.name}")
    
    print("\n2. Phase descriptions:")
    phase_descriptions = {
        ProcessingPhase.PHASE_0_VALIDATION: "Validate configuration and contracts",
        ProcessingPhase.PHASE_1_INGESTION: "Ingest and preprocess document",
        ProcessingPhase.PHASE_2_EXECUTION: "Execute 300 micro questions",
        ProcessingPhase.PHASE_3_SCORING: "Score all question results",
        ProcessingPhase.PHASE_4_DIMENSION_AGG: "Aggregate into 60 dimensions",
        ProcessingPhase.PHASE_5_AREA_AGG: "Aggregate into 10 policy areas",
        ProcessingPhase.PHASE_6_CLUSTER_AGG: "Aggregate into 4 clusters (MESO)",
        ProcessingPhase.PHASE_7_MACRO_EVAL: "Perform macro evaluation",
        ProcessingPhase.PHASE_8_RECOMMENDATIONS: "Generate recommendations",
        ProcessingPhase.PHASE_9_ASSEMBLY: "Assemble complete report",
        ProcessingPhase.PHASE_10_FORMATTING: "Format outputs (JSON, HTML, PDF, Excel)",
    }
    
    for phase, description in phase_descriptions.items():
        print(f"   Phase {phase.value}: {description}")


def main():
    """Run all demonstrations."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 70)
    print("ORCHESTRATOR MODULE DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows the key features of the Orchestrator module:")
    print("- Phase coordination (0-10)")
    print("- Configuration management")
    print("- Validation and contract loading")
    print("- Metrics tracking")
    print("- Abort control")
    print("- Progressive integration interfaces")
    
    # Run demos
    demo_basic_orchestrator()
    demo_custom_configuration()
    demo_validation_phase()
    demo_metrics_tracking()
    demo_abort_control()
    demo_phase_enumeration()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nThe Orchestrator is ready for production use!")
    print("See orchestrator/orchestrator_core.py for implementation details.")
    print("See tests/test_orchestrator.py for usage examples.")


if __name__ == "__main__":
    main()
