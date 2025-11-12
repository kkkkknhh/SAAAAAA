"""
Meta Layer (@m) - Governance and Transparency Evaluation.

This layer evaluates metadata quality including transparency,
governance compliance, and computational cost.
"""
import logging
from .config import MetaLayerConfig

logger = logging.getLogger(__name__)


class MetaLayerEvaluator:
    """Evaluates governance and transparency compliance."""

    def __init__(self, config: MetaLayerConfig):
        """
        Initialize with configuration.

        Args:
            config: MetaLayerConfig specifying weights and requirements
        """
        self.config = config
        logger.info("meta_layer_initialized", extra={"config": config.to_dict() if hasattr(config, 'to_dict') else str(config)})

    def evaluate(self, method_id: str, method_version: str, config_hash: str,
                 formula_exported: bool = False,
                 full_trace: bool = False,
                 logs_conform: bool = False,
                 signature_valid: bool = False,
                 execution_time_s: float = None) -> float:
        """
        Evaluate governance compliance.

        Returns:
            x_@m = w_transparency·m_transp + w_governance·m_gov + w_cost·m_cost
            (using weights from config)
        """
        # Compute transparency score
        m_transp = self._compute_transparency(
            formula_exported=formula_exported,
            full_trace=full_trace,
            logs_conform=logs_conform
        )

        # Compute governance score
        m_gov = self._compute_governance(
            method_version=method_version,
            config_hash=config_hash,
            signature_valid=signature_valid
        )

        # Compute cost score
        m_cost = self._compute_cost(execution_time_s)

        # Weighted combination
        score = (self.config.w_transparency * m_transp +
                self.config.w_governance * m_gov +
                self.config.w_cost * m_cost)

        logger.info(
            "meta_evaluated",
            extra={
                "method_id": method_id,
                "version": method_version,
                "m_transp": m_transp,
                "m_gov": m_gov,
                "m_cost": m_cost,
                "score": score
            }
        )

        return score

    def _compute_transparency(self, formula_exported: bool,
                             full_trace: bool,
                             logs_conform: bool) -> float:
        """
        Compute transparency score.

        Checks formula export, full trace availability, and log conformance.
        """
        checks_passed = 0
        checks_total = 0

        if self.config.require_formula_export:
            checks_total += 1
            if formula_exported:
                checks_passed += 1

        if self.config.require_full_trace:
            checks_total += 1
            if full_trace:
                checks_passed += 1

        if self.config.require_log_conformance:
            checks_total += 1
            if logs_conform:
                checks_passed += 1

        if checks_total == 0:
            return 1.0  # No requirements

        return checks_passed / checks_total

    def _compute_governance(self, method_version: str,
                           config_hash: str,
                           signature_valid: bool) -> float:
        """
        Compute governance score.

        FIXED: Only count components if required by config.
        FIXED: Don't arbitrarily reject version "1.0" - accept semantic versions.
        """
        checks_passed = 0
        checks_total = 0

        # Version check (FIXED: Accept any semantic version, not just != "1.0")
        if self.config.require_tagged_version:
            checks_total += 1
            # Accept versions matching semantic versioning pattern (v.X.Y or vX.Y.Z)
            if method_version and (method_version.count('.') >= 1):
                checks_passed += 1

        # Config hash check
        if self.config.require_config_hash_match:
            checks_total += 1
            if config_hash and len(config_hash) > 0:
                checks_passed += 1

        # Signature check (FIXED: Only check if required)
        if self.config.require_valid_signature:
            checks_total += 1
            if signature_valid:
                checks_passed += 1

        if checks_total == 0:
            return 1.0  # No requirements

        return checks_passed / checks_total

    def _compute_cost(self, execution_time_s: float = None) -> float:
        """
        Compute cost score based on execution time.

        Returns:
            1.0 if runtime <= threshold_fast
            0.5 if threshold_fast < runtime <= threshold_acceptable
            0.0 if runtime > threshold_acceptable or None (FIXED)
        """
        # FIXED: Return 0.0 when execution_time_s is None (not 0.5)
        if execution_time_s is None:
            return 0.0

        if execution_time_s <= self.config.threshold_fast:
            return 1.0
        elif execution_time_s <= self.config.threshold_acceptable:
            return 0.5
        else:
            return 0.0
