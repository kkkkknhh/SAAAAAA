#!/usr/bin/env python3
"""
Auto-Calibration Decision System

Deterministic, rule-based classifier to determine calibration requirements:
- REQUIRES_CALIBRATION
- NO_CALIBRATION_REQUIRED  
- FLAG_FOR_REVIEW

Input: Method from canonical catalog + usage metadata
Output: Calibration decision with explicit rationale

Uses canonical method catalog: config/canonical_method_catalog.json (1,996 methods)

Criteria:
- Role in critical pipelines
- Sensitivity of outputs
- Frequency of use
- Presence/complexity of configurable parameters
- Dependency relationships
- Methods that feed calibrated stages

NO fuzzy guessing. If inconclusive, hard-stop to FLAG_FOR_REVIEW.
"""

import json
import sys
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Import canonical enums
from saaaaaa.core.orchestrator.catalogo_completo_canonico import (
    MethodPriority,
    MethodComplexity,
)


class CalibrationDecision(Enum):
    """Calibration requirement decision"""
    REQUIRES_CALIBRATION = "REQUIRES_CALIBRATION"
    NO_CALIBRATION_REQUIRED = "NO_CALIBRATION_REQUIRED"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"


@dataclass
class CalibrationRationale:
    """Explanation for calibration decision"""
    decision: CalibrationDecision
    confidence: float  # 0.0 to 1.0
    reasons: List[str]
    risk_factors: List[str]
    recommendation: str


class CalibrationDecisionEngine:
    """
    Deterministic rule-based engine for calibration decisions.
    
    Decision Rules:
    
    1. REQUIRES_CALIBRATION if:
       - Method priority is CRITICAL or HIGH
       - Method is used in critical path
       - Method has HIGH complexity
       - Method has numeric_support or temporal_support requirements
       - Method is called frequently (>10 usages)
       - Method has configurable parameters
       
    2. NO_CALIBRATION_REQUIRED if:
       - Method priority is LOW
       - Method complexity is LOW
       - Method is simple utility function (__init__, getters, formatters)
       - Method has zero configurable parameters
       - Method never executes scoring/evidence logic
       
    3. FLAG_FOR_REVIEW if:
       - Insufficient usage data
       - Contradictory signals
       - New method not yet categorized
       - Method has unusual patterns
    """
    
    # Simple utility method patterns (auto NO_CALIBRATION_REQUIRED)
    UTILITY_PATTERNS = [
        '__init__',
        '__str__',
        '__repr__',
        '__eq__',
        '__hash__',
        'get_',
        'set_',
        '_format_',
        '_serialize_',
        '_deserialize_',
    ]
    
    # Critical method patterns (auto REQUIRES_CALIBRATION)
    CRITICAL_PATTERNS = [
        'score',
        'evaluate',
        'calculate',
        'compute',
        'analyze',
        'bayesian',
        'confidence',
        'evidence',
    ]
    
    def __init__(self, catalog_data=None, usage_intelligence_path: Optional[Path] = None):
        # Load canonical catalog if not provided
        if catalog_data is None:
            catalog_path = repo_root / "config" / "canonical_method_catalog.json"
            with open(catalog_path) as f:
                catalog_data = json.load(f)
        
        self.catalog_data = catalog_data
        self.catalog_methods = {
            (m['class_name'], m['method_name']): m 
            for m in catalog_data['methods']
            if m['class_name']
        }
        self.usage_data = {}
        
        if usage_intelligence_path and usage_intelligence_path.exists():
            with open(usage_intelligence_path, 'r') as f:
                data = json.load(f)
                self.usage_data = data.get('methods', {})
    
    def decide(self, class_name: str, method_name: str) -> CalibrationRationale:
        """
        Make calibration decision for a method.
        
        Returns CalibrationRationale with decision and explicit reasoning.
        """
        # Get method from catalog
        method = self.catalog_methods.get((class_name, method_name))
        
        if not method:
            return CalibrationRationale(
                decision=CalibrationDecision.FLAG_FOR_REVIEW,
                confidence=0.0,
                reasons=["Method not in canonical catalog"],
                risk_factors=["Unknown method"],
                recommendation="Add method to catalog before determining calibration requirements"
            )
        
        # Get usage data
        fqn = f"{class_name}.{method_name}"
        usage = self.usage_data.get(fqn, {})
        
        # Apply decision rules
        return self._apply_rules(method, usage)
    
    def _apply_rules(self, method: dict, usage: dict) -> CalibrationRationale:
        """Apply deterministic decision rules"""
        
        reasons = []
        risk_factors = []
        confidence = 1.0
        
        # Rule 1: Check for simple utility patterns
        if self._is_utility_method(method.method_name):
            return CalibrationRationale(
                decision=CalibrationDecision.NO_CALIBRATION_REQUIRED,
                confidence=0.95,
                reasons=[
                    f"Method '{method.method_name}' matches utility pattern",
                    "Simple getter/setter/formatter methods don't require calibration"
                ],
                risk_factors=[],
                recommendation="No calibration needed - this is a simple utility method"
            )
        
        # Rule 2: CRITICAL priority → REQUIRES_CALIBRATION
        if method.priority == MethodPriority.CRITICAL:
            reasons.append("Method has CRITICAL priority in catalog")
            return CalibrationRationale(
                decision=CalibrationDecision.REQUIRES_CALIBRATION,
                confidence=1.0,
                reasons=reasons + ["All CRITICAL methods must be calibrated"],
                risk_factors=["High impact on analysis quality"],
                recommendation="Create calibration profile with strict parameters"
            )
        
        # Rule 3: HIGH priority → REQUIRES_CALIBRATION
        if method.priority == MethodPriority.HIGH:
            reasons.append("Method has HIGH priority in catalog")
            return CalibrationRationale(
                decision=CalibrationDecision.REQUIRES_CALIBRATION,
                confidence=0.9,
                reasons=reasons + ["HIGH priority methods should be calibrated"],
                risk_factors=["Significant impact on analysis"],
                recommendation="Create calibration profile"
            )
        
        # Rule 4: HIGH complexity → REQUIRES_CALIBRATION
        if method.complexity == MethodComplexity.HIGH:
            reasons.append("Method has HIGH complexity")
            return CalibrationRationale(
                decision=CalibrationDecision.REQUIRES_CALIBRATION,
                confidence=0.85,
                reasons=reasons + ["Complex methods benefit from calibration"],
                risk_factors=["Complex logic may have hidden sensitivities"],
                recommendation="Create calibration profile to control complexity"
            )
        
        # Rule 5: Check execution requirements
        if method.execution_requirements.computational == "HIGH":
            reasons.append("High computational requirements")
        if method.execution_requirements.memory == "HIGH":
            reasons.append("High memory requirements")
        
        # Rule 6: Check method name patterns for criticality
        if self._is_critical_method(method.method_name):
            reasons.append(f"Method name '{method.method_name}' suggests critical analysis function")
            return CalibrationRationale(
                decision=CalibrationDecision.REQUIRES_CALIBRATION,
                confidence=0.8,
                reasons=reasons,
                risk_factors=["Analysis/scoring methods need calibration"],
                recommendation="Create calibration profile for scoring parameters"
            )
        
        # Rule 7: Check usage frequency
        total_usages = usage.get('total_usages', 0)
        if total_usages > 10:
            reasons.append(f"Method used {total_usages} times - high usage")
            return CalibrationRationale(
                decision=CalibrationDecision.REQUIRES_CALIBRATION,
                confidence=0.7,
                reasons=reasons + ["Frequently used methods should be calibrated"],
                risk_factors=["High usage means errors propagate widely"],
                recommendation="Calibrate to ensure consistent behavior across all uses"
            )
        
        # Rule 8: Check if already in calibration registry
        if usage.get('in_calibration_registry', False):
            reasons.append("Method already has calibration entry")
            return CalibrationRationale(
                decision=CalibrationDecision.REQUIRES_CALIBRATION,
                confidence=1.0,
                reasons=reasons + ["Existing calibration should be maintained"],
                risk_factors=[],
                recommendation="Keep existing calibration"
            )
        
        # Rule 9: LOW priority + LOW complexity → likely NO_CALIBRATION_REQUIRED
        if method.priority == MethodPriority.LOW and method.complexity == MethodComplexity.LOW:
            if total_usages < 5:
                reasons.append("LOW priority, LOW complexity, infrequent use")
                return CalibrationRationale(
                    decision=CalibrationDecision.NO_CALIBRATION_REQUIRED,
                    confidence=0.8,
                    reasons=reasons,
                    risk_factors=[],
                    recommendation="Simple, low-priority method doesn't need calibration"
                )
        
        # Rule 10: If we got here, need human review
        reasons.append("No clear decision from automated rules")
        if method.priority == MethodPriority.MEDIUM:
            reasons.append("MEDIUM priority requires case-by-case evaluation")
        if method.complexity == MethodComplexity.MEDIUM:
            reasons.append("MEDIUM complexity requires case-by-case evaluation")
        
        confidence = 0.5
        return CalibrationRationale(
            decision=CalibrationDecision.FLAG_FOR_REVIEW,
            confidence=confidence,
            reasons=reasons,
            risk_factors=["Unclear calibration requirements"],
            recommendation="Manual review required - insufficient signals for automated decision"
        )
    
    def _is_utility_method(self, method_name: str) -> bool:
        """Check if method matches utility pattern"""
        method_lower = method_name.lower()
        return any(pattern in method_lower for pattern in self.UTILITY_PATTERNS)
    
    def _is_critical_method(self, method_name: str) -> bool:
        """Check if method matches critical analysis pattern"""
        method_lower = method_name.lower()
        return any(pattern in method_lower for pattern in self.CRITICAL_PATTERNS)
    
    def batch_decide(self) -> Dict[str, CalibrationRationale]:
        """Run decisions for all catalogued methods"""
        results = {}
        
        for method in self.catalog.all_methods():
            decision = self.decide(method.class_name, method.method_name)
            fqn = f"{method.class_name}.{method.method_name}"
            results[fqn] = decision
        
        return results
    
    def generate_report(self, output_path: Path):
        """Generate calibration decisions report"""
        results = self.batch_decide()
        
        # Build method-keyed decisions (CORRECT STRUCTURE)
        method_decisions = {}
        
        # Build summary by category
        summary_by_category = {
            "REQUIRES_CALIBRATION": 0,
            "NO_CALIBRATION_REQUIRED": 0,
            "FLAG_FOR_REVIEW": 0
        }
        
        for fqn, rationale in results.items():
            # Store as method-keyed
            method_decisions[fqn] = {
                "decision": rationale.decision.value,
                "confidence": rationale.confidence,
                "reasons": rationale.reasons,
                "risk_factors": rationale.risk_factors,
                "recommendation": rationale.recommendation,
            }
            
            # Update summary
            summary_by_category[rationale.decision.value] += 1
        
        report = {
            "metadata": {
                "generated_at": "2025-11-09",
                "total_methods": len(results),
                "catalog_version": self.catalog.catalog_version,
            },
            "summary": {
                "requires_calibration": summary_by_category["REQUIRES_CALIBRATION"],
                "no_calibration_required": summary_by_category["NO_CALIBRATION_REQUIRED"],
                "flag_for_review": summary_by_category["FLAG_FOR_REVIEW"],
            },
            "decisions": method_decisions,  # METHOD-KEYED, not category-keyed
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Calibration decisions report written to: {output_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("AUTO-CALIBRATION DECISION SUMMARY")
        print("="*80)
        print(f"Total methods analyzed: {len(results)}")
        print(f"  - REQUIRES_CALIBRATION: {summary_by_category['REQUIRES_CALIBRATION']}")
        print(f"  - NO_CALIBRATION_REQUIRED: {summary_by_category['NO_CALIBRATION_REQUIRED']}")
        print(f"  - FLAG_FOR_REVIEW: {summary_by_category['FLAG_FOR_REVIEW']}")
        
        return report


def main():
    repo_root = Path(__file__).parent.parent
    usage_path = repo_root / "config" / "method_usage_intelligence.json"
    
    engine = CalibrationDecisionEngine(usage_intelligence_path=usage_path)
    
    output_path = repo_root / "config" / "calibration_decisions.json"
    report = engine.generate_report(output_path)
    
    print("\n✓ Auto-calibration decision system complete!")
    
    # Show some examples
    print("\n" + "="*80)
    print("SAMPLE DECISIONS")
    print("="*80)
    
    # Group by category for display
    by_category = {
        "REQUIRES_CALIBRATION": [],
        "NO_CALIBRATION_REQUIRED": [],
        "FLAG_FOR_REVIEW": []
    }
    
    for method_fqn, decision_data in report["decisions"].items():
        category = decision_data["decision"]
        by_category[category].append({
            "method": method_fqn,
            **decision_data
        })
    
    for category in ["REQUIRES_CALIBRATION", "NO_CALIBRATION_REQUIRED", "FLAG_FOR_REVIEW"]:
        items = by_category[category][:3]
        if items:
            print(f"\n{category}:")
            for item in items:
                print(f"  • {item['method']}")
                print(f"    Confidence: {item['confidence']:.2f}")
                print(f"    Reasons: {', '.join(item['reasons'][:2])}")


if __name__ == "__main__":
    main()
