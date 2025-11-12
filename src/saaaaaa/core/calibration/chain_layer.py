"""
Chain Layer (@chain) - Data Flow Validation.

This layer validates that method chains have correct data flow,
ensuring required inputs are available and type-compatible.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class ChainLayerEvaluator:
    """Evaluates chain integrity through data flow validation."""

    def __init__(self, method_signatures: Dict[str, Dict]):
        """
        Initialize with method signatures.

        Args:
            method_signatures: Dictionary mapping method IDs to their signatures
                             (required_inputs, optional_inputs, outputs)
        """
        self.signatures = method_signatures
        logger.info("chain_layer_initialized", extra={"method_count": len(method_signatures)})

    def evaluate(self, method_id: str, provided_inputs: List[str]) -> float:
        """
        Evaluate chain integrity for a single method.

        Args:
            method_id: The method to evaluate
            provided_inputs: List of available input names

        Returns:
            Score in {0.0, 0.3, 0.6, 0.8, 1.0}
        """
        # Check if method is in signatures (FIX: Return 0.0 not 0.1 for missing)
        if method_id not in self.signatures:
            logger.warning("undeclared_method", extra={"method_id": method_id})
            return 0.0  # FIXED: Was 0.1 in buggy version

        signature = self.signatures[method_id]
        required = set(signature.get("required_inputs", []))
        optional = set(signature.get("optional_inputs", []))
        provided = set(provided_inputs)

        # Check required inputs
        missing_required = required - provided
        if missing_required:
            logger.warning(
                "missing_required_inputs",
                extra={
                    "method_id": method_id,
                    "missing": list(missing_required),
                    "score": 0.0
                }
            )
            return 0.0

        # Check optional inputs coverage
        missing_optional = optional - provided
        optional_coverage = 1.0 - (len(missing_optional) / len(optional)) if optional else 1.0

        # Discrete mapping based on optional coverage
        if optional_coverage == 1.0:
            score = 1.0  # All inputs provided
        elif optional_coverage >= 0.75:
            score = 0.8  # Most optional inputs provided
        elif optional_coverage >= 0.5:
            score = 0.6  # Half of optional inputs provided
        else:
            score = 0.3  # Only required inputs provided

        logger.info(
            "chain_evaluated",
            extra={
                "method_id": method_id,
                "required_satisfied": len(required),
                "optional_coverage": optional_coverage,
                "score": score
            }
        )

        return score

    @classmethod
    def from_file(cls, signatures_path: Path) -> "ChainLayerEvaluator":
        """
        Load evaluator from method signatures JSON file.

        Args:
            signatures_path: Path to method_signatures.json

        Returns:
            Configured ChainLayerEvaluator
        """
        with open(signatures_path) as f:
            data = json.load(f)

        return cls(method_signatures=data["methods"])
