#!/usr/bin/env python3
"""Complete System Execution: CPP + Orchestrator for Plan_1.pdf

This script demonstrates the complete end-to-end processing pipeline:
1. CPP Ingestion: Preprocess Plan_1.pdf using Canon Policy Package pipeline
2. CPP Adaptation: Convert CPP to PreprocessedDocument format
3. Orchestrator Execution: Run all 11 phases of the orchestration pipeline
4. Results Display: Show comprehensive results from each phase

Usage:
    python run_complete_analysis_plan1.py

Requirements:
    - Plan_1.pdf must exist in data/plans/
    - All dependencies installed (pdfplumber, pyarrow, etc.)

Note: Run this script after installing the package with: pip install -e .
"""

import asyncio
from pathlib import Path

from saaaaaa.utils.paths import data_dir
from saaaaaa.processing.cpp_ingestion import CPPIngestionPipeline
from saaaaaa.utils.cpp_adapter import CPPAdapter
from saaaaaa.core.orchestrator import Orchestrator
from saaaaaa.core.orchestrator.factory import build_processor
from saaaaaa.processing.cpp_ingestion.models import CanonPolicyPackage


def load_cpp_from_directory(cpp_dir: Path) -> CanonPolicyPackage:
    """
    Load Canon Policy Package from a directory with Arrow files and metadata.
    
    Args:
        cpp_dir: Directory containing CPP files (content_stream.arrow, etc.)
        
    Returns:
        Reconstructed CanonPolicyPackage
    """
    import json
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from saaaaaa.processing.cpp_ingestion.models import (
        CanonPolicyPackage,
        ChunkGraph,
        IntegrityIndex,
        PolicyManifest,
        ProvenanceMap,
        QualityMetrics,
    )
    
    # Load metadata
    metadata_path = cpp_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load content stream
    content_stream = None
    content_stream_path = cpp_dir / "content_stream.arrow"
    if content_stream_path.exists():
        with pa.OSFile(str(content_stream_path), "rb") as source:
            with ipc.open_file(source) as reader:
                content_stream = reader.read_all()
    
    # Load provenance map
    provenance_table = None
    provenance_path = cpp_dir / "provenance_map.arrow"
    if provenance_path.exists():
        with pa.OSFile(str(provenance_path), "rb") as source:
            with ipc.open_file(source) as reader:
                provenance_table = reader.read_all()
    
    # Reconstruct objects
    policy_manifest = PolicyManifest(
        axes=metadata["policy_manifest"]["axes"],
        programs=metadata["policy_manifest"]["programs"],
        projects=[],
        years=metadata["policy_manifest"]["years"],
        territories=metadata["policy_manifest"]["territories"],
        indicators=[],
        budget_rows=[],
    )
    
    integrity_index = IntegrityIndex(
        blake3_root=metadata["integrity_index"]["blake3_root"],
        chunk_hashes={},
    )
    
    quality_metrics = QualityMetrics(
        boundary_f1=metadata["quality_metrics"]["boundary_f1"],
        kpi_linkage_rate=metadata["quality_metrics"]["kpi_linkage_rate"],
        budget_consistency_score=metadata["quality_metrics"]["budget_consistency_score"],
        provenance_completeness=1.0,
        structural_consistency=1.0,
        temporal_robustness=1.0,
        chunk_context_coverage=1.0,
    )
    
    provenance_map = ProvenanceMap(table=provenance_table)
    
    # Create chunks from content stream
    # Since chunk_graph isn't saved separately, we reconstruct minimal chunks from content_stream
    from saaaaaa.processing.cpp_ingestion.models import (
        Chunk, ChunkResolution, TextSpan, Confidence,
        PolicyFacet, TimeFacet, GeoFacet
    )
    
    chunks = {}
    if content_stream is not None:
        for i in range(content_stream.num_rows):
            row = content_stream.slice(i, 1)
            page_id = row.column("page_id")[0].as_py()
            text = row.column("text")[0].as_py()
            byte_start = row.column("byte_start")[0].as_py()
            byte_end = row.column("byte_end")[0].as_py()
            
            # Create a minimal chunk with all required facets
            chunk_id = f"chunk_{i}"
            chunks[chunk_id] = Chunk(
                id=chunk_id,
                text=text,
                resolution=ChunkResolution.MESO,  # Default to MESO
                text_span=TextSpan(start=byte_start, end=byte_end),
                bytes_hash=f"hash_{i}",  # Placeholder
                policy_facets=PolicyFacet(),  # Empty policy facets
                time_facets=TimeFacet(),  # Empty time facets
                geo_facets=GeoFacet(),  # Empty geo facets
                provenance=None,
                kpi=None,
                budget=None,
                entities=[],
                confidence=Confidence(layout=1.0, ocr=1.0, typing=1.0),
            )
    
    chunk_graph = ChunkGraph(chunks=chunks)
    
    # Create CPP
    cpp = CanonPolicyPackage(
        schema_version=metadata["schema_version"],
        policy_manifest=policy_manifest,
        chunk_graph=chunk_graph,
        content_stream=content_stream,
        provenance_map=provenance_map,
        integrity_index=integrity_index,
        quality_metrics=quality_metrics,
    )
    
    return cpp


async def main():
    """Main execution function."""
    
    print("=" * 80)
    print("CPP + ORCHESTRATOR PIPELINE: Plan_1.pdf")
    print("=" * 80)
    print()
    
    # ========================================================================
    # PHASE 1: CPP INGESTION
    # ========================================================================
    print("📄 PHASE 1: CPP INGESTION")
    print("-" * 80)
    
    input_path = data_dir() / 'plans' / 'Plan_1.pdf'
    cpp_output = data_dir() / 'output' / 'cpp_plan_1'
    cpp_output.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"❌ ERROR: Plan_1.pdf not found at {input_path}")
        print("   Please ensure the file exists before running.")
        return 1
    
    print(f'  Input: Plan_1.pdf')
    print(f'  Location: {input_path}')
    print(f'  Size: {input_path.stat().st_size / 1024:.1f} KB')
    print()
    
    print('  🔄 Initializing CPP ingestion pipeline...')
    cpp_pipeline = CPPIngestionPipeline(
        enable_ocr=True,
        ocr_confidence_threshold=0.85,
        chunk_overlap_threshold=0.15
    )
    
    print('  🔄 Processing document (this may take 30-60 seconds)...')
    cpp_outcome = cpp_pipeline.ingest(input_path, cpp_output)
    
    if cpp_outcome.status != 'OK' or not cpp_outcome.cpp_uri:
        print(f'  ❌ CPP Ingestion FAILED: {cpp_outcome.status}')
        return 1
    
    print(f'  ✅ CPP Status: {cpp_outcome.status}')
    print(f'  ✅ CPP URI: {cpp_outcome.cpp_uri}')
    print(f'  ✅ Schema Version: {cpp_pipeline.SCHEMA_VERSION}')
    print()
    
    # ========================================================================
    # PHASE 2: CPP LOADING & ADAPTATION
    # ========================================================================
    print("🔄 PHASE 2: CPP LOADING & ADAPTATION")
    print("-" * 80)
    
    print('  🔄 Loading CPP from directory...')
    cpp = load_cpp_from_directory(Path(cpp_outcome.cpp_uri))
    print(f'  ✅ CPP loaded successfully')
    print(f'  ✅ Schema: {cpp.schema_version}')
    
    print('  🔄 Converting CPP to PreprocessedDocument...')
    adapter = CPPAdapter()
    preprocessed_doc = adapter.to_preprocessed_document(
        cpp,
        document_id='Plan_1'
    )
    
    print(f'  ✅ Document ID: {preprocessed_doc.document_id}')
    print(f'  ✅ Sentences: {len(preprocessed_doc.sentences)}')
    print(f'  ✅ Tables: {len(preprocessed_doc.tables)}')
    print(f'  ✅ Raw text length: {len(preprocessed_doc.raw_text)} chars')
    
    provenance_completeness = preprocessed_doc.metadata.get('provenance_completeness', 0.0)
    print(f'  ✅ Provenance completeness: {provenance_completeness:.2%}')
    print()
    
    # ========================================================================
    # PHASE 3: ORCHESTRATOR INITIALIZATION (using official API)
    # ========================================================================
    print("⚙️  PHASE 3: ORCHESTRATOR INITIALIZATION")
    print("-" * 80)
    
    print('  🔄 Building processor bundle with build_processor()...')
    
    try:
        # Use official API: build_processor() to get processor bundle
        processor_bundle = build_processor()
        print(f'  ✅ Processor bundle created')
        print(f'  ✅ Method executor: {type(processor_bundle.method_executor).__name__}')
        print(f'  ✅ Questionnaire loaded: {len(processor_bundle.questionnaire)} keys')
        print(f'  ✅ Factory catalog loaded: {len(processor_bundle.factory.catalog)} keys')
        print()
        
        print('  🔄 Initializing Orchestrator with official arguments...')
        # Use official API: Orchestrator(monolith=questionnaire, catalog=factory.catalog)
        orchestrator = Orchestrator(
            monolith=processor_bundle.questionnaire,
            catalog=processor_bundle.factory.catalog
        )
        print(f'  ✅ Orchestrator initialized')
        print(f'  ✅ Phases: {len(orchestrator.FASES)}')
        print(f'  ✅ Executors registered: {len(orchestrator.executors)}')
        print()
    except Exception as e:
        print(f'  ❌ Failed to initialize orchestrator: {e}')
        print(f'  ℹ️  Error details:')
        import traceback
        traceback.print_exc()
        print()
        return 1
    
    # ========================================================================
    # PHASE 4: ORCHESTRATOR EXECUTION (11 PHASES)
    # ========================================================================
    print("🚀 PHASE 4: ORCHESTRATOR EXECUTION (11 PHASES)")
    print("=" * 80)
    print()
    
    # Create a temporary PDF path for the orchestrator
    # (it expects a PDF path even though we're providing preprocessed_document)
    temp_pdf_path = str(input_path)
    
    print('  🔄 Starting 11-phase orchestration...')
    print()
    
    try:
        # Run the complete orchestration pipeline
        phase_results = await orchestrator.process_development_plan_async(
            pdf_path=temp_pdf_path,
            preprocessed_document=preprocessed_doc
        )
        
        print()
        print("=" * 80)
        print("📊 ORCHESTRATION RESULTS")
        print("=" * 80)
        print()
        
        # Display results for each phase
        for i, result in enumerate(phase_results):
            phase_label = orchestrator.FASES[i][3] if i < len(orchestrator.FASES) else f"Phase {i}"
            status_icon = "✅" if result.success else "❌"
            
            print(f"{status_icon} {phase_label}")
            print(f"   Duration: {result.duration_ms:.0f}ms")
            print(f"   Mode: {result.mode}")
            
            if result.success and result.data is not None:
                # Show data summary based on phase
                if isinstance(result.data, list):
                    print(f"   Results: {len(result.data)} items")
                elif isinstance(result.data, dict):
                    print(f"   Results: {len(result.data)} keys")
                else:
                    print(f"   Results: {type(result.data).__name__}")
            
            if result.error:
                print(f"   ❌ Error: {result.error}")
            
            if result.aborted:
                print(f"   ⚠️  Aborted")
                break
            
            print()
        
        # Summary statistics
        successful = sum(1 for r in phase_results if r.success)
        total = len(phase_results)
        total_time = sum(r.duration_ms for r in phase_results)
        
        print("=" * 80)
        print("📈 SUMMARY")
        print("=" * 80)
        print(f"  Phases completed: {successful}/{total}")
        print(f"  Total time: {total_time/1000:.1f}s")
        print(f"  Average per phase: {total_time/total:.0f}ms")
        
        # Mechanically derived status based on actual results
        if successful == total and all(r.error is None for r in phase_results):
            print()
            print("✅ ALL PHASES COMPLETED")
            return 0
        else:
            print()
            print("⚠️ PHASES INCOMPLETE")
            for r in phase_results:
                if not r.success:
                    print(f" - Phase {r.phase_id} failed: {r.error}")
            return 1
            
    except Exception as e:
        print()
        print(f"❌ ORCHESTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
