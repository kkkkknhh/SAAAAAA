"""
Congruence Layer (@C) - Ensemble Compatibility Evaluation.

This layer evaluates whether multiple methods in an ensemble are compatible,
checking scale, semantic overlap, and fusion validity.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class CongruenceLayerEvaluator:
    """Evaluates congruence of method ensembles."""

    def __init__(self, method_registry: Dict[str, Dict]):
        """
        Initialize with method registry.

        Args:
            method_registry: Dictionary mapping method IDs to their metadata
                           (output_range, semantic_tags, fusion_requirements)
        """
        self.registry = method_registry
        logger.info("congruence_layer_initialized", extra={"method_count": len(method_registry)})

    def evaluate(self, method_ids: List[str], subgraph_id: str,
                 fusion_rule: str = "weighted_average",
                 available_inputs: List[str] = None) -> float:
        """
        Evaluate congruence of method ensemble.

        Args:
            method_ids: List of methods in the ensemble
            subgraph_id: Identifier for the computation subgraph
            fusion_rule: Method for combining outputs
            available_inputs: Available inputs for fusion

        Returns:
            C_play = c_scale · c_sem · c_fusion (score in [0,1])
        """
        if available_inputs is None:
            available_inputs = []

        # FIXED: Check single method exists in registry before returning 1.0
        if len(method_ids) < 2:
            if len(method_ids) == 1 and method_ids[0] in self.registry:
                return 1.0  # Single method automatically congruent
            else:
                logger.warning("single_method_not_in_registry", extra={"method_id": method_ids[0] if method_ids else None})
                return 0.0  # FIXED: Don't assume 1.0 for unknown methods

        # Compute c_scale (range compatibility)
        c_scale = self._compute_scale_compatibility(method_ids)

        # Compute c_sem (semantic overlap)
        c_sem = self._compute_semantic_overlap(method_ids)

        # Compute c_fusion (fusion rule validity)
        c_fusion = self._compute_fusion_validity(method_ids, fusion_rule, available_inputs)

        # Final congruence score
        score = c_scale * c_sem * c_fusion

        logger.info(
            "congruence_evaluated",
            extra={
                "methods": method_ids,
                "subgraph": subgraph_id,
                "c_scale": c_scale,
                "c_sem": c_sem,
                "c_fusion": c_fusion,
                "score": score
            }
        )

        return score

    def _compute_scale_compatibility(self, method_ids: List[str]) -> float:
        """
        Check if all methods have compatible output ranges.

        FIXED: Check if ranges are within [0,1] instead of exact equality.
        """
        ranges = []
        for method_id in method_ids:
            if method_id not in self.registry:
                logger.warning("method_not_in_registry", extra={"method_id": method_id})
                return 0.0

            output_range = self.registry[method_id].get("output_range", [0.0, 1.0])
            # Normalize range type to avoid (0, 1) vs (0.0, 1.0) issues
            ranges.append((float(output_range[0]), float(output_range[1])))

        # FIXED: Check if all ranges are within [0,1] instead of exact equality
        if not all(r[0] >= 0.0 and r[1] <= 1.0 for r in ranges):
            logger.warning("incompatible_ranges", extra={"ranges": ranges})
            return 0.0

        # Check if all ranges are similar (difference < 0.1)
        min_lower = min(r[0] for r in ranges)
        max_upper = max(r[1] for r in ranges)

        if max_upper - min_lower > 0.5:
            return 0.5  # Partially compatible
        else:
            return 1.0  # Fully compatible

    def _compute_semantic_overlap(self, method_ids: List[str]) -> float:
        """Compute semantic overlap using Jaccard similarity of tags."""
        all_tags = []
        for method_id in method_ids:
            if method_id not in self.registry:
                return 0.0
            tags = set(self.registry[method_id].get("semantic_tags", []))
            all_tags.append(tags)

        if not all_tags:
            return 0.0

        # Compute pairwise Jaccard similarities
        similarities = []
        for i in range(len(all_tags)):
            for j in range(i + 1, len(all_tags)):
                intersection = len(all_tags[i] & all_tags[j])
                union = len(all_tags[i] | all_tags[j])
                if union > 0:
                    similarities.append(intersection / union)
                else:
                    similarities.append(0.0)

        if not similarities:
            return 1.0  # Single method case

        # Average Jaccard similarity
        return sum(similarities) / len(similarities)

    def _compute_fusion_validity(self, method_ids: List[str], fusion_rule: str,
                                  available_inputs: List[str]) -> float:
        """
        Check if fusion rule is valid given method requirements.

        FIXED: Add type-checking for fusion_requirements before calling .update()
        """
        required_fusion_inputs = set()

        for method_id in method_ids:
            if method_id not in self.registry:
                return 0.0

            fusion_reqs = self.registry[method_id].get("fusion_requirements", [])

            # FIXED: Ensure fusion_requirements is iterable
            if not isinstance(fusion_reqs, (list, set, tuple)):
                logger.error(
                    "invalid_fusion_requirements_type",
                    extra={"method_id": method_id, "type": type(fusion_reqs).__name__}
                )
                return 0.0

            required_fusion_inputs.update(fusion_reqs)

        # Check if all required fusion inputs are available
        missing = required_fusion_inputs - set(available_inputs)

        if missing:
            logger.warning("missing_fusion_inputs", extra={"missing": list(missing)})
            return 0.5  # Partially valid

        return 1.0  # Fully valid

    @classmethod
    def from_file(cls, registry_path: Path) -> "CongruenceLayerEvaluator":
        """
        Load evaluator from method registry JSON file.

        Args:
            registry_path: Path to method_registry.json

        Returns:
            Configured CongruenceLayerEvaluator
        """
        with open(registry_path) as f:
            data = json.load(f)

        return cls(method_registry=data["methods"])
