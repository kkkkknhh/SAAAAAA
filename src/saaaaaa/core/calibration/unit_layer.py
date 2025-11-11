"""
Unit Layer (@u) - STUB IMPLEMENTATION.

This evaluates PDT quality. Full implementation will compute S, M, I, P components.

TODO: Implement full S, M, I, P computation as specified in Part 2.
"""
import logging
from .config import UnitLayerConfig
from .pdt_structure import PDTStructure
from .data_structures import LayerID, LayerScore

logger = logging.getLogger(__name__)


class UnitLayerEvaluator:
    """
    Evaluates Unit Layer (@u) - PDT quality.
    
    STUB: Returns fixed score for now.
    TODO: Implement S, M, I, P component evaluation.
    """
    
    def __init__(self, config: UnitLayerConfig):
        self.config = config
    
    def evaluate(self, pdt: PDTStructure) -> LayerScore:
        """
        STUB: Evaluate PDT quality.
        
        TODO: Implement:
        - S: Structural compliance (block coverage, hierarchy, order)
        - M: Mandatory sections ratio
        - I: Indicator quality (structure, linkage, logic)
        - P: PPI completeness (presence, structure, consistency)
        - Hard gates enforcement
        - Anti-gaming penalties
        
        Returns:
            LayerScore for unit quality
        """
        logger.warning(
            "unit_layer_stub",
            extra={
                "total_tokens": pdt.total_tokens,
                "indicator_present": pdt.indicator_matrix_present,
                "ppi_present": pdt.ppi_matrix_present,
                "stub_score": 0.75
            }
        )
        
        # STUB: Return stub score from configuration
        stub_score = self.config.w_S + self.config.w_M + self.config.w_I + self.config.w_P  # = 1.0
        stub_score = stub_score * 0.75  # Scale to reasonable stub value
        
        return LayerScore(
            layer=LayerID.UNIT,
            score=stub_score,
            components={"S": 0.75, "M": 0.75, "I": 0.75, "P": 0.75},
            rationale=f"Unit layer (STUB IMPLEMENTATION - not data-driven, returns {stub_score:.2f})",
            metadata={"stub": True, "warning": "This is a placeholder implementation"}
        )
