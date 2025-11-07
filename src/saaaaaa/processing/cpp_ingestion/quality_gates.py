"""
Quality gates for CPP validation.

Validates quality metrics and enforces invariants.
"""

from typing import Any, Dict

from .models import CanonPolicyPackage


class QualityGates:
    """Quality validation gates."""
    
    # Thresholds
    BOUNDARY_F1_MIN = 0.85
    KPI_LINKAGE_MIN = 0.80
    BUDGET_CONSISTENCY_MIN = 0.95
    PROVENANCE_COMPLETENESS_MIN = 1.0
    STRUCTURAL_CONSISTENCY_MIN = 1.0
    CHUNK_OVERLAP_MAX = 0.15
    
    def validate(self, cpp: CanonPolicyPackage) -> Dict[str, Any]:
        """
        Validate CPP against quality gates.
        
        Args:
            cpp: Canon Policy Package to validate
            
        Returns:
            Dictionary with validation results
        """
        failures = []
        
        # Check provenance completeness
        if cpp.quality_metrics.provenance_completeness < self.PROVENANCE_COMPLETENESS_MIN:
            failures.append(
                f"Provenance completeness {cpp.quality_metrics.provenance_completeness} "
                f"< {self.PROVENANCE_COMPLETENESS_MIN}"
            )
        
        # Check structural consistency
        if cpp.quality_metrics.structural_consistency < self.STRUCTURAL_CONSISTENCY_MIN:
            failures.append(
                f"Structural consistency {cpp.quality_metrics.structural_consistency} "
                f"< {self.STRUCTURAL_CONSISTENCY_MIN}"
            )
        
        # Check KPI linkage rate
        if cpp.quality_metrics.kpi_linkage_rate < self.KPI_LINKAGE_MIN:
            failures.append(
                f"KPI linkage rate {cpp.quality_metrics.kpi_linkage_rate} "
                f"< {self.KPI_LINKAGE_MIN}"
            )
        
        # Check budget consistency
        if cpp.quality_metrics.budget_consistency_score < self.BUDGET_CONSISTENCY_MIN:
            failures.append(
                f"Budget consistency {cpp.quality_metrics.budget_consistency_score} "
                f"< {self.BUDGET_CONSISTENCY_MIN}"
            )
        
        # Check boundary quality
        if cpp.quality_metrics.boundary_f1 < self.BOUNDARY_F1_MIN:
            failures.append(
                f"Boundary F1 {cpp.quality_metrics.boundary_f1} "
                f"< {self.BOUNDARY_F1_MIN}"
            )
        
        return {
            "passed": len(failures) == 0,
            "failures": failures,
        }
