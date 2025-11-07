"""Enhanced ExecutorConfig with embedded calibration - NO SEPARATE LAYER"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import hashlib
import json

@dataclass
class MethodConstraints:
    """Constraints for a specific method - embedded in config."""
    min_evidence_snippets: int = 3
    max_evidence_snippets: int = 20
    contradiction_tolerance: float = 0.1
    uncertainty_penalty: float = 0.25
    requires_provenance: bool = True

@dataclass(frozen=True)
class ExecutorConfig:
    """Immutable executor configuration with embedded calibration."""
    
    # Execution constraints
    timeout_s: float
    retry: int
    max_parallel: int
    
    # Memory and resource limits
    max_memory_mb: int
    max_output_size_mb: int
    
    # LLM parameters (used in executors)
    temperature: float = 0.7
    max_tokens: int = 2000
    seed: int = 42
    
    # Thresholds (used in scoring)
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "evidence_threshold": 0.7,
        "contradiction_threshold": 0.3,
        "confidence_threshold": 0.6,
    })
    
    # METHOD-SPECIFIC CONSTRAINTS (embedded, not separate)
    method_constraints: Dict[Tuple[str, str], MethodConstraints] = field(
        default_factory=dict
    )
    
    # Validation requirements
    require_calibration: bool = True
    require_provenance: bool = True
    require_manifest: bool = True
    
    # Failure modes
    fail_on_missing_calibration: bool = True
    fail_on_contradiction: bool = True
    fail_on_insufficient_evidence: bool = True
    
    def compute_hash(self) -> str:
        """Compute SHA256 of this configuration."""
        config_dict = {
            "timeout_s": self.timeout_s,
            "retry": self.retry,
            "max_parallel": self.max_parallel,
            "max_memory_mb": self.max_memory_mb,
            "max_output_size_mb": self.max_output_size_mb,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "thresholds": self.thresholds,
            "require_calibration": self.require_calibration,
            "require_provenance": self.require_provenance,
            "require_manifest": self.require_manifest,
        }
        json_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def get_method_constraints(
        self, 
        class_name: str, 
        method_name: str
    ) -> MethodConstraints:
        """Get constraints for a method, with strict validation."""
        key = (class_name, method_name)
        
        if self.require_calibration and key not in self.method_constraints:
            raise RuntimeError(
                f"No constraints configured for {class_name}.{method_name} "
                f"and require_calibration=True"
            )
        
        return self.method_constraints.get(
            key, 
            MethodConstraints()  # Default if not strict
        )
    
    @classmethod
    def from_env(cls) -> "ExecutorConfig":
        """Create config from environment variables."""
        import os
        return cls(
            timeout_s=float(os.getenv("EXECUTOR_TIMEOUT", "30")),
            retry=int(os.getenv("EXECUTOR_RETRY", "2")),
            max_parallel=int(os.getenv("EXECUTOR_PARALLEL", "1")),
            max_memory_mb=int(os.getenv("EXECUTOR_MAX_MEMORY_MB", "512")),
            max_output_size_mb=int(os.getenv("EXECUTOR_MAX_OUTPUT_MB", "10")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
            seed=int(os.getenv("LLM_SEED", "42")),
        )


# Conservative configuration with embedded constraints
CONSERVATIVE_CONFIG = ExecutorConfig(
    timeout_s=30.0,
    retry=2,
    max_parallel=1,
    max_memory_mb=512,
    max_output_size_mb=10,
    temperature=0.7,
    max_tokens=2000,
    seed=42,
    method_constraints={
        # Key scoring methods
        ("BayesianEvidenceScorer", "compute_evidence_score"): MethodConstraints(
            min_evidence_snippets=3,
            max_evidence_snippets=15,
            contradiction_tolerance=0.05,
            uncertainty_penalty=0.25,
            requires_provenance=True,
        ),
        ("PolicyContradictionDetector", "identify_contradictions"): MethodConstraints(
            min_evidence_snippets=2,
            max_evidence_snippets=20,
            contradiction_tolerance=0.0,
            uncertainty_penalty=0.4,
            requires_provenance=True,
        ),
        # Add more as needed WITHOUT creating separate registry
    },
    require_calibration=True,
    require_provenance=True,
    require_manifest=True,
    fail_on_missing_calibration=True,
    fail_on_contradiction=True,
    fail_on_insufficient_evidence=True,
)