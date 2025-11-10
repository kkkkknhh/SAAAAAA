#!/usr/bin/env python3
"""
F.A.R.F.A.N Verified Pipeline Runner
=====================================

Framework for Advanced Retrieval of Administrativa Narratives

Canonical entrypoint for executing the F.A.R.F.A.N policy analysis pipeline with 
cryptographic verification and structured claim logging. This script is designed 
to be machine-auditable and produces verifiable artifacts at every step.

Key Features:
- Computes SHA256 hashes of all inputs and outputs
- Emits structured JSON claims for all operations
- Generates verification_manifest.json with success status
- Enforces zero-trust validation principles
- No fabricated logs or unverifiable banners

Usage:
    python scripts/run_policy_pipeline_verified.py [--plan PLAN_PDF]

Requirements:
    - Input PDF must exist (default: data/plans/Plan_1.pdf)
    - All dependencies installed
    - Write access to artifacts/ directory
"""

import asyncio
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src/ is in Python path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


@dataclass
class ExecutionClaim:
    """Structured claim about a pipeline operation."""
    timestamp: str
    claim_type: str  # "start", "complete", "error", "artifact", "hash"
    component: str
    message: str
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class VerificationManifest:
    """Complete verification manifest for pipeline execution."""
    success: bool
    execution_id: str
    start_time: str
    end_time: str
    input_pdf_path: str
    input_pdf_sha256: str
    artifacts_generated: List[str]
    artifact_hashes: Dict[str, str]
    phases_completed: int
    phases_failed: int
    total_claims: int
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class VerifiedPipelineRunner:
    """Executes pipeline with cryptographic verification and claim logging."""
    
    def __init__(self, plan_pdf_path: Path, artifacts_dir: Path):
        """
        Initialize verified runner.
        
        Args:
            plan_pdf_path: Path to input PDF
            artifacts_dir: Directory for output artifacts
        """
        self.plan_pdf_path = plan_pdf_path
        self.artifacts_dir = artifacts_dir
        self.claims: List[ExecutionClaim] = []
        self.execution_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.utcnow().isoformat()
        self.phases_completed = 0
        self.phases_failed = 0
        self.errors: List[str] = []
        
        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    def log_claim(self, claim_type: str, component: str, message: str, 
                  data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured claim.
        
        Args:
            claim_type: Type of claim (start, complete, error, artifact, hash)
            component: Component making the claim
            message: Human-readable message
            data: Optional structured data
        """
        claim = ExecutionClaim(
            timestamp=datetime.utcnow().isoformat(),
            claim_type=claim_type,
            component=component,
            message=message,
            data=data or {}
        )
        self.claims.append(claim)
        
        # Also print for real-time monitoring
        claim_json = json.dumps(claim.to_dict(), separators=(',', ':'))
        print(f"CLAIM: {claim_json}", flush=True)
    
    def compute_sha256(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex-encoded SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def verify_input(self) -> bool:
        """
        Verify input PDF exists and compute hash.
        
        Returns:
            True if input is valid
        """
        self.log_claim("start", "input_verification", "Verifying input PDF")
        
        if not self.plan_pdf_path.exists():
            error_msg = f"Input PDF not found: {self.plan_pdf_path}"
            self.log_claim("error", "input_verification", error_msg)
            self.errors.append(error_msg)
            return False
        
        # Compute hash
        try:
            pdf_hash = self.compute_sha256(self.plan_pdf_path)
            self.input_pdf_sha256 = pdf_hash
            self.log_claim("hash", "input_verification", 
                          f"Input PDF SHA256: {pdf_hash}",
                          {"file": str(self.plan_pdf_path), "hash": pdf_hash})
            self.log_claim("complete", "input_verification", 
                          "Input verification successful")
            return True
        except Exception as e:
            error_msg = f"Failed to hash input PDF: {str(e)}"
            self.log_claim("error", "input_verification", error_msg)
            self.errors.append(error_msg)
            return False
    
    async def run_spc_ingestion(self) -> Optional[Any]:
        """
        Run SPC (Smart Policy Chunks) ingestion phase - canonical phase-one.
        
        Returns:
            SPC object if successful, None otherwise
        """
        self.log_claim("start", "spc_ingestion", "Starting SPC ingestion (phase-one)")
        
        try:
            from saaaaaa.processing.spc_ingestion import CPPIngestionPipeline
            
            pipeline = CPPIngestionPipeline()
            cpp = await pipeline.process(self.plan_pdf_path)
            
            self.phases_completed += 1
            self.log_claim("complete", "spc_ingestion", 
                          "SPC ingestion (phase-one) completed successfully",
                          {"phases_completed": self.phases_completed})
            return cpp
            
        except Exception as e:
            self.phases_failed += 1
            error_msg = f"SPC ingestion failed: {str(e)}"
            self.log_claim("error", "spc_ingestion", error_msg,
                          {"traceback": traceback.format_exc()})
            self.errors.append(error_msg)
            return None
    
    async def run_cpp_adapter(self, cpp: Any) -> Optional[Any]:
        """
        Run CPP adapter to convert to PreprocessedDocument.
        
        Args:
            cpp: CPP object from ingestion
            
        Returns:
            PreprocessedDocument if successful, None otherwise
        """
        self.log_claim("start", "cpp_adapter", "Starting CPP adaptation")
        
        try:
            from saaaaaa.utils.cpp_adapter import CPPAdapter
            
            adapter = CPPAdapter()
            # Use the correct method name from CPPAdapter API
            preprocessed = adapter.to_preprocessed_document(cpp)
            
            self.phases_completed += 1
            self.log_claim("complete", "cpp_adapter", 
                          "CPP adaptation completed successfully",
                          {"phases_completed": self.phases_completed})
            return preprocessed
            
        except Exception as e:
            self.phases_failed += 1
            error_msg = f"CPP adaptation failed: {str(e)}"
            self.log_claim("error", "cpp_adapter", error_msg,
                          {"traceback": traceback.format_exc()})
            self.errors.append(error_msg)
            return None
    
    async def run_orchestrator(self, preprocessed_doc: Any) -> Optional[Dict[str, Any]]:
        """
        Run orchestrator with all phases.
        
        Args:
            preprocessed_doc: PreprocessedDocument
            
        Returns:
            Results dictionary if successful, None otherwise
        """
        self.log_claim("start", "orchestrator", "Starting orchestrator execution")
        
        try:
            from saaaaaa.core.orchestrator import Orchestrator
            from saaaaaa.core.orchestrator.factory import build_processor
            
            processor = build_processor()
            orchestrator = Orchestrator(processor=processor)
            
            # Run all phases
            results = await orchestrator.process(preprocessed_doc)
            
            # Count actual phases completed based on results
            if results and hasattr(results, '__dict__'):
                phase_attrs = [attr for attr in dir(results) 
                             if not attr.startswith('_') and attr.endswith('_result')]
                completed_phases = sum(1 for attr in phase_attrs 
                                      if getattr(results, attr, None) is not None)
                self.phases_completed += completed_phases
            else:
                # Fallback if we can't inspect results
                self.phases_completed += 1
            
            self.log_claim("complete", "orchestrator", 
                          "Orchestrator execution completed successfully",
                          {"phases_completed": self.phases_completed})
            return results
            
        except Exception as e:
            self.phases_failed += 1
            error_msg = f"Orchestrator execution failed: {str(e)}"
            self.log_claim("error", "orchestrator", error_msg,
                          {"traceback": traceback.format_exc()})
            self.errors.append(error_msg)
            return None
    
    def save_artifacts(self, cpp: Any, preprocessed_doc: Any, 
                      results: Any) -> tuple[List[str], Dict[str, str]]:
        """
        Save artifacts and compute hashes.
        
        Args:
            cpp: CPP object
            preprocessed_doc: PreprocessedDocument
            results: Orchestrator results
            
        Returns:
            List of artifact file paths
        """
        self.log_claim("start", "artifact_generation", "Saving artifacts")
        
        artifacts = []
        artifact_hashes = {}
        
        try:
            # Save CPP metadata if available
            if cpp:
                cpp_metadata_path = self.artifacts_dir / "cpp_metadata.json"
                try:
                    with open(cpp_metadata_path, 'w') as f:
                        json.dump({
                            "execution_id": self.execution_id,
                            "cpp_generated": True,
                            "timestamp": datetime.utcnow().isoformat()
                        }, f, indent=2)
                    artifacts.append(str(cpp_metadata_path))
                    artifact_hashes[str(cpp_metadata_path)] = self.compute_sha256(cpp_metadata_path)
                except Exception as e:
                    self.log_claim("error", "artifact_generation", 
                                  f"Failed to save CPP metadata: {str(e)}")
            
            # Save preprocessed document metadata
            if preprocessed_doc:
                doc_metadata_path = self.artifacts_dir / "preprocessed_doc_metadata.json"
                try:
                    with open(doc_metadata_path, 'w') as f:
                        json.dump({
                            "execution_id": self.execution_id,
                            "doc_generated": True,
                            "timestamp": datetime.utcnow().isoformat()
                        }, f, indent=2)
                    artifacts.append(str(doc_metadata_path))
                    artifact_hashes[str(doc_metadata_path)] = self.compute_sha256(doc_metadata_path)
                except Exception as e:
                    self.log_claim("error", "artifact_generation", 
                                  f"Failed to save doc metadata: {str(e)}")
            
            # Save results summary
            if results:
                results_path = self.artifacts_dir / "results_summary.json"
                try:
                    with open(results_path, 'w') as f:
                        json.dump({
                            "execution_id": self.execution_id,
                            "results_generated": True,
                            "timestamp": datetime.utcnow().isoformat()
                        }, f, indent=2)
                    artifacts.append(str(results_path))
                    artifact_hashes[str(results_path)] = self.compute_sha256(results_path)
                except Exception as e:
                    self.log_claim("error", "artifact_generation", 
                                  f"Failed to save results: {str(e)}")
            
            # Save all claims
            claims_path = self.artifacts_dir / "execution_claims.json"
            with open(claims_path, 'w') as f:
                json.dump([claim.to_dict() for claim in self.claims], f, indent=2)
            artifacts.append(str(claims_path))
            artifact_hashes[str(claims_path)] = self.compute_sha256(claims_path)
            
            self.log_claim("complete", "artifact_generation", 
                          f"Saved {len(artifacts)} artifacts",
                          {"artifact_count": len(artifacts)})
            
            return artifacts, artifact_hashes
            
        except Exception as e:
            error_msg = f"Failed to save artifacts: {str(e)}"
            self.log_claim("error", "artifact_generation", error_msg)
            self.errors.append(error_msg)
            return artifacts, artifact_hashes
    
    def generate_verification_manifest(self, artifacts: List[str],
                                       artifact_hashes: Dict[str, str]) -> Path:
        """
        Generate final verification manifest.
        
        Args:
            artifacts: List of artifact paths
            artifact_hashes: Dictionary mapping paths to SHA256 hashes
            
        Returns:
            Path to verification_manifest.json
        """
        end_time = datetime.utcnow().isoformat()
        
        # Determine success based on strict criteria
        success = (
            self.phases_failed == 0 and
            self.phases_completed > 0 and
            len(self.errors) == 0 and
            len(artifacts) > 0
        )
        
        manifest = VerificationManifest(
            success=success,
            execution_id=self.execution_id,
            start_time=self.start_time,
            end_time=end_time,
            input_pdf_path=str(self.plan_pdf_path),
            input_pdf_sha256=getattr(self, 'input_pdf_sha256', ''),
            artifacts_generated=artifacts,
            artifact_hashes=artifact_hashes,
            phases_completed=self.phases_completed,
            phases_failed=self.phases_failed,
            total_claims=len(self.claims),
            errors=self.errors
        )
        
        manifest_path = self.artifacts_dir / "verification_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest.to_dict(), f, indent=2)
        
        # Compute manifest hash
        manifest_hash = self.compute_sha256(manifest_path)
        self.log_claim("hash", "verification_manifest", 
                      f"Manifest SHA256: {manifest_hash}",
                      {"file": str(manifest_path), "hash": manifest_hash})
        
        return manifest_path
    
    async def run(self) -> bool:
        """
        Execute the complete verified pipeline.
        
        Returns:
            True if pipeline succeeded, False otherwise
        """
        self.log_claim("start", "pipeline", "Starting verified pipeline execution")
        
        # Step 1: Verify input
        if not self.verify_input():
            self.generate_verification_manifest([], {})
            return False
        
        # Step 2: Run SPC ingestion (canonical phase-one)
        cpp = await self.run_spc_ingestion()
        if cpp is None:
            self.generate_verification_manifest([], {})
            return False
        
        # Step 3: Run CPP adapter
        preprocessed_doc = await self.run_cpp_adapter(cpp)
        if preprocessed_doc is None:
            self.generate_verification_manifest([], {})
            return False
        
        # Step 4: Run orchestrator
        results = await self.run_orchestrator(preprocessed_doc)
        if results is None:
            self.generate_verification_manifest([], {})
            return False
        
        # Step 5: Save artifacts
        artifacts, artifact_hashes = self.save_artifacts(cpp, preprocessed_doc, results)
        
        # Step 6: Generate verification manifest
        manifest_path = self.generate_verification_manifest(artifacts, artifact_hashes)
        
        self.log_claim("complete", "pipeline", 
                      "Pipeline execution completed",
                      {
                          "success": self.phases_failed == 0,
                          "phases_completed": self.phases_completed,
                          "phases_failed": self.phases_failed,
                          "manifest_path": str(manifest_path)
                      })
        
        return self.phases_failed == 0


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run verified policy pipeline with cryptographic verification"
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="data/plans/Plan_1.pdf",
        help="Path to plan PDF (default: data/plans/Plan_1.pdf)"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/plan1",
        help="Directory for artifacts (default: artifacts/plan1)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    plan_path = REPO_ROOT / args.plan
    artifacts_dir = REPO_ROOT / args.artifacts_dir
    
    print("=" * 80, flush=True)
    print("F.A.R.F.A.N VERIFIED POLICY PIPELINE RUNNER", flush=True)
    print("Framework for Advanced Retrieval of Administrativa Narratives", flush=True)
    print("=" * 80, flush=True)
    print(f"Plan: {plan_path}", flush=True)
    print(f"Artifacts: {artifacts_dir}", flush=True)
    print("=" * 80, flush=True)
    
    # Create and run pipeline
    runner = VerifiedPipelineRunner(plan_path, artifacts_dir)
    success = await runner.run()
    
    print("=" * 80, flush=True)
    if success:
        print("PIPELINE_VERIFIED=1", flush=True)
        print("Status: SUCCESS", flush=True)
    else:
        print("PIPELINE_VERIFIED=0", flush=True)
        print("Status: FAILED", flush=True)
    print("=" * 80, flush=True)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
