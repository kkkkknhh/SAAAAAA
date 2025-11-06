#!/usr/bin/env python3
"""
Complete CPP-based pipeline execution for Plan_1.pdf

Stages:
1. CPP Ingestion (9 deterministic phases)
2. CPP Adaptation (to PreprocessedDocument)
3. Orchestrator Execution (11 phases)

Output:
- artifacts/plan1/phase_report.json
- artifacts/plan1/phase_report.md
- artifacts/plan1/execution_log.txt
"""

import asyncio
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Imports (exact paths as specified)
from saaaaaa.processing.cpp_ingestion.pipeline import CPPIngestionPipeline
from saaaaaa.utils.cpp_adapter import CPPAdapter
from saaaaaa.core.orchestrator.factory import build_processor
from saaaaaa.core.orchestrator.core import Orchestrator


class ExecutionLogger:
    """Logger for execution progress with file output."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.start_time = datetime.now()
        self.buffer = []
        
    def log(self, message: str):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        self.buffer.append(line)
        
    def section(self, title: str):
        """Log a section header."""
        line = f"\n--- {title} ---"
        print(line)
        self.buffer.append(line)
        
    def save(self):
        """Save log buffer to file."""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPLETE POLICY ANALYSIS PIPELINE - CPP INGESTION\n")
            f.write("=" * 80 + "\n")
            f.write(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Document: Plan_1\n")
            f.write(f"PDF: data/plans/Plan_1.pdf\n")
            f.write(f"Output: artifacts/plan1/\n")
            f.write("\n")
            f.write("\n".join(self.buffer))
            f.write("\n\n")


def summarize_value(value: Any) -> dict[str, Any]:
    """Create compact summary of phase data for reporting."""
    try:
        if is_dataclass(value):
            value = asdict(value)
    except Exception:
        pass
    
    if isinstance(value, dict):
        keys = list(value.keys())
        return {
            "type": "dict",
            "keys": len(value),
            "sample_keys": keys[:5] if len(keys) > 5 else keys
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": len(value),
            "sample_item": type(value[0]).__name__ if value else None
        }
    if isinstance(value, (str, bytes)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "preview": value[:50] if isinstance(value, str) else None
        }
    return {"type": type(value).__name__}


def stage1_cpp_ingestion(logger: ExecutionLogger) -> tuple[Any, float, dict]:
    """
    Stage 1: CPP Ingestion
    
    Returns:
        (cpp_document, duration_seconds, metrics)
    """
    logger.section("STAGE 1: CPP INGESTION")
    
    # Initialize pipeline
    logger.log("Initializing CPPIngestionPipeline")
    logger.log("Parameters: enable_ocr=True, ocr_confidence_threshold=0.85, chunk_overlap_threshold=0.15")
    
    pipeline = CPPIngestionPipeline(
        enable_ocr=True,
        ocr_confidence_threshold=0.85,
        chunk_overlap_threshold=0.15,
    )
    
    # Prepare paths
    input_path = Path("data/plans/Plan_1.pdf")
    output_dir = Path("artifacts/plan1/cpp_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get file size
    file_size_kb = input_path.stat().st_size / 1024
    logger.log(f"Input: {input_path} ({file_size_kb:.0f} KB)")
    
    # Execute ingestion
    start_time = time.perf_counter()
    outcome = pipeline.ingest(
        input_path=input_path,
        output_dir=output_dir
    )
    duration = time.perf_counter() - start_time
    
    # Validate outcome
    if outcome.status != "OK":
        error_msg = f"CPP ingestion failed: {outcome.diagnostics}"
        logger.log(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg)
    
    # Extract metrics
    cpp_document = outcome.cpp
    chunk_count = len(cpp_document.chunk_graph.chunks) if cpp_document and cpp_document.chunk_graph else 0
    
    metrics = {
        "provenance_completeness": outcome.metrics.provenance_completeness if outcome.metrics else 0.0,
        "chunk_count": chunk_count,
        "boundary_f1": outcome.metrics.boundary_f1 if outcome.metrics else 0.0,
        "kpi_linkage_rate": outcome.metrics.kpi_linkage_rate if outcome.metrics else 0.0,
        "budget_consistency": outcome.metrics.budget_consistency_score if outcome.metrics else 0.0,
        "structural_consistency": outcome.metrics.structural_consistency if outcome.metrics else 0.0,
    }
    
    logger.log(f"CPP ingestion complete: status=OK, chunks={chunk_count}, provenance_completeness={metrics['provenance_completeness']:.2f}")
    logger.log(f"Duration: {duration:.3f}s")
    
    return cpp_document, duration, metrics


def stage2_cpp_adaptation(cpp_document: Any, logger: ExecutionLogger) -> tuple[Any, float]:
    """
    Stage 2: CPP Adaptation
    
    Returns:
        (preprocessed_document, duration_seconds)
    """
    logger.section("STAGE 2: CPP ADAPTATION")
    
    logger.log("Initializing CPPAdapter")
    adapter = CPPAdapter()
    
    logger.log("Converting CanonPolicyPackage to PreprocessedDocument")
    start_time = time.perf_counter()
    
    preprocessed = adapter.to_preprocessed_document(
        cpp_document,
        document_id="Plan_1"
    )
    
    duration = time.perf_counter() - start_time
    
    # Validate PreprocessedDocument structure
    required_attrs = ["document_id", "raw_text", "sentences", "tables", "metadata"]
    for attr in required_attrs:
        if hasattr(preprocessed, attr):
            logger.log(f"Validation: {attr} ✓")
        else:
            raise RuntimeError(f"PreprocessedDocument missing required attribute: {attr}")
    
    # Extract adapter metrics
    chunk_count = preprocessed.metadata.get("chunk_count", 0)
    provenance_completeness = preprocessed.metadata.get("provenance_completeness", 0.0)
    
    logger.log(f"Adapter metrics: chunk_count={chunk_count}, provenance_completeness={provenance_completeness:.2f}")
    logger.log(f"Duration: {duration:.3f}s")
    
    return preprocessed, duration


async def stage3_orchestrator_execution(
    preprocessed_document: Any, 
    logger: ExecutionLogger
) -> tuple[list, float]:
    """
    Stage 3: Orchestrator Execution
    
    Returns:
        (phase_results, duration_seconds)
    """
    logger.section("STAGE 3: ORCHESTRATOR EXECUTION")
    
    # Build processor bundle
    logger.log("Building processor bundle via build_processor()")
    bundle = build_processor()
    
    # Get sizes for logging
    questionnaire_size = len(bundle.questionnaire.get("blocks", {}).get("micro_questions", []))
    catalog_size = len(bundle.factory.catalog.registry) if hasattr(bundle.factory, 'catalog') and hasattr(bundle.factory.catalog, 'registry') else 0
    
    logger.log(f"Bundle: questionnaire_size={questionnaire_size}, catalog_size={catalog_size}")
    
    # Initialize Orchestrator
    logger.log("Initializing Orchestrator(monolith=bundle.questionnaire, catalog=bundle.factory.catalog)")
    orchestrator = Orchestrator(
        monolith=bundle.questionnaire,
        catalog=bundle.factory.catalog
    )
    
    logger.log("Starting 11-phase execution")
    
    # Execute pipeline
    start_time = time.perf_counter()
    results = await orchestrator.process_development_plan_async(
        pdf_path="data/plans/Plan_1.pdf",
        preprocessed_document=preprocessed_document
    )
    duration = time.perf_counter() - start_time
    
    # Log each phase
    phase_labels = [label for _, _, _, label in Orchestrator.FASES]
    for i, result in enumerate(results):
        status_icon = "✓" if result.success else "✗"
        label = phase_labels[i] if i < len(phase_labels) else f"FASE {i}"
        logger.log(f"{label} ... {status_icon} ({result.duration_ms:.0f} ms)")
    
    # Summary
    successful_phases = sum(1 for r in results if r.success)
    logger.log(f"All {successful_phases}/{len(results)} phases completed")
    logger.log(f"Duration: {duration:.3f}s")
    
    return results, duration


def generate_reports(
    results: list,
    cpp_metrics: dict,
    timings: dict,
    output_dir: Path,
    logger: ExecutionLogger
):
    """Generate all three output files."""
    logger.section("GENERATING REPORTS")
    
    # Generate phase_report.json
    json_report = {
        "summary": {
            "document_id": "Plan_1",
            "pdf_path": "data/plans/Plan_1.pdf",
            "cpp_ingestion_seconds": timings["cpp_ingestion"],
            "cpp_adaptation_seconds": timings["cpp_adaptation"],
            "orchestration_seconds": timings["orchestration"],
            "total_seconds": sum(timings.values()),
            "phases_completed": sum(1 for r in results if r.success),
            "phases_total": len(results),
            "cpp_metrics": cpp_metrics
        },
        "phases": []
    }
    
    phase_labels = [label for _, _, _, label in Orchestrator.FASES]
    for i, result in enumerate(results):
        phase_info = {
            "index": i,
            "id": result.phase_id,
            "label": phase_labels[i] if i < len(phase_labels) else f"FASE {i}",
            "mode": result.mode,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "aborted": result.aborted,
            "error": str(result.error) if result.error else None,
            "data_summary": summarize_value(result.data)
        }
        json_report["phases"].append(phase_info)
    
    json_path = output_dir / "phase_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    logger.log(f"Generated: {json_path}")
    
    # Generate phase_report.md
    md_lines = [
        "# Orchestrator Phase Report (CPP Ingestion)",
        "",
        "## Summary",
        f"- Document: `Plan_1`",
        f"- PDF: `data/plans/Plan_1.pdf`",
        f"- CPP ingestion time: `{timings['cpp_ingestion']:.3f}s`",
        f"- CPP adaptation time: `{timings['cpp_adaptation']:.3f}s`",
        f"- Orchestration time: `{timings['orchestration']:.3f}s`",
        f"- Total time: `{sum(timings.values()):.3f}s`",
        f"- Phases: `{sum(1 for r in results if r.success)}/{len(results)}` completed",
        f"- CPP provenance completeness: `{cpp_metrics['provenance_completeness']:.2f}`",
        f"- CPP chunk count: `{cpp_metrics['chunk_count']}`",
        "",
        "## CPP Quality Metrics",
        f"- Boundary F1: `{cpp_metrics['boundary_f1']:.2f}`",
        f"- KPI Linkage Rate: `{cpp_metrics['kpi_linkage_rate']:.2f}`",
        f"- Budget Consistency: `{cpp_metrics['budget_consistency']:.2f}`",
        f"- Structural Consistency: `{cpp_metrics['structural_consistency']:.2f}`",
        "",
        "## Phases",
        ""
    ]
    
    for i, result in enumerate(results):
        icon = "✅" if result.success else "❌"
        label = phase_labels[i] if i < len(phase_labels) else f"FASE {i}"
        data_summary = summarize_value(result.data)
        
        md_lines.extend([
            f"### {icon} [{i}] {label}",
            f"- id: `{result.phase_id}`",
            f"- mode: `{result.mode}`",
            f"- duration: `{result.duration_ms:.0f} ms`",
            f"- data: {data_summary['type']} with `{data_summary.get('keys', data_summary.get('items', 'N/A'))}` {'keys' if 'keys' in data_summary else 'items'}",
            f"- success: `{result.success}`",
            ""
        ])
    
    md_path = output_dir / "phase_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    logger.log(f"Generated: {md_path}")
    
    # Add summary to execution log
    logger.buffer.extend([
        "",
        "--- SUMMARY ---",
        f"Total execution time: {sum(timings.values()):.3f}s",
        f"CPP ingestion: {timings['cpp_ingestion']:.3f}s ({timings['cpp_ingestion']/sum(timings.values())*100:.1f}%)",
        f"CPP adaptation: {timings['cpp_adaptation']:.3f}s ({timings['cpp_adaptation']/sum(timings.values())*100:.1f}%)",
        f"Orchestration: {timings['orchestration']:.3f}s ({timings['orchestration']/sum(timings.values())*100:.1f}%)",
        f"Phases completed: {sum(1 for r in results if r.success)}/{len(results)} ({sum(1 for r in results if r.success)/len(results)*100:.0f}%)",
        f"Status: {'SUCCESS ✅' if all(r.success for r in results) else 'PARTIAL ⚠️'}",
        "",
        "Output files:",
        f"  - {output_dir}/phase_report.json",
        f"  - {output_dir}/phase_report.md",
        f"  - {output_dir}/execution_log.txt",
        "",
        "=" * 80
    ])
    
    logger.save()
    logger.log(f"Generated: {output_dir / 'execution_log.txt'}")


async def main():
    """Main execution function."""
    # Setup
    output_dir = Path("artifacts/plan1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = ExecutionLogger(output_dir / "execution_log.txt")
    
    try:
        # Stage 1: CPP Ingestion
        cpp_document, cpp_duration, cpp_metrics = stage1_cpp_ingestion(logger)
        
        # Stage 2: CPP Adaptation
        preprocessed, adapter_duration = stage2_cpp_adaptation(cpp_document, logger)
        
        # Stage 3: Orchestrator Execution
        results, orchestrator_duration = await stage3_orchestrator_execution(preprocessed, logger)
        
        # Generate reports
        timings = {
            "cpp_ingestion": cpp_duration,
            "cpp_adaptation": adapter_duration,
            "orchestration": orchestrator_duration
        }
        generate_reports(results, cpp_metrics, timings, output_dir, logger)
        
        # Final output
        print("\n" + "=" * 80)
        print("✅ EXECUTION COMPLETE")
        print("=" * 80)
        print(f"Total time: {sum(timings.values()):.1f}s")
        print(f"Phases: {sum(1 for r in results if r.success)}/{len(results)}")
        print(f"\nOutput files:")
        print(f"  - artifacts/plan1/phase_report.json")
        print(f"  - artifacts/plan1/phase_report.md")
        print(f"  - artifacts/plan1/execution_log.txt")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.log(f"ERROR: {e}")
        logger.save()
        print(f"\n❌ EXECUTION FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
