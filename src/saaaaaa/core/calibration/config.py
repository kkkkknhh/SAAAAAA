"""
Calibration system configuration.

This module will be fully implemented in Part 2 of the implementation guide.
For now, it contains placeholder classes to allow data_structures.py to be tested.

COMPLETE SPECIFICATION PROVIDED IN PART 2.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UnitLayerConfig:
    """
    Configuration for Unit Layer (@u) - PDT quality evaluation.
    
    PLACEHOLDER: Full implementation in Part 2.
    """
    # Placeholder fields - will be replaced in Part 2
    min_structural_compliance: float = 0.5
    ppi_required: bool = True
    indicator_required: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.min_structural_compliance <= 1.0:
            raise ValueError(
                f"min_structural_compliance must be in [0.0, 1.0], "
                f"got {self.min_structural_compliance}"
            )


@dataclass(frozen=True)
class MetaLayerConfig:
    """
    Configuration for Meta Layer (@m) - Governance and observability.
    
    PLACEHOLDER: Full implementation in Part 2.
    """
    # Placeholder fields - will be replaced in Part 2
    weight_transparency: float = 0.5
    weight_governance: float = 0.4
    weight_cost: float = 0.1
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.weight_transparency + self.weight_governance + self.weight_cost
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Meta layer weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class ChoquetAggregationConfig:
    """
    Configuration for Choquet 2-Additive aggregation.
    
    PLACEHOLDER: Full implementation in Part 2.
    """
    # Placeholder fields - will be replaced in Part 2
    layer_weights: dict[str, float] = field(default_factory=dict)
    interaction_weights: dict[tuple[str, str], float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate normalization constraint."""
        if not self.layer_weights or not self.interaction_weights:
            return  # Allow empty for placeholder
        
        total_linear = sum(self.layer_weights.values())
        total_interaction = sum(self.interaction_weights.values())
        total = total_linear + total_interaction
        
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Choquet weights must sum to 1.0, got linear={total_linear}, "
                f"interaction={total_interaction}, total={total}"
            )


@dataclass(frozen=True)
class CalibrationSystemConfig:
    """
    Complete calibration system configuration.
    
    PLACEHOLDER: Full implementation in Part 2.
    """
    unit_layer: UnitLayerConfig = field(default_factory=UnitLayerConfig)
    meta_layer: MetaLayerConfig = field(default_factory=MetaLayerConfig)
    choquet: ChoquetAggregationConfig = field(default_factory=ChoquetAggregationConfig)
    random_seed: int = 42
    
    def compute_system_hash(self) -> str:
        """
        Compute deterministic hash of configuration.
        
        This ensures reproducibility: same config = same hash.
        """
        import hashlib
        import json
        
        # Convert to JSON-serializable dict
        config_dict = {
            "unit_layer": {
                "min_structural_compliance": self.unit_layer.min_structural_compliance,
                "ppi_required": self.unit_layer.ppi_required,
                "indicator_required": self.unit_layer.indicator_required,
            },
            "meta_layer": {
                "weight_transparency": self.meta_layer.weight_transparency,
                "weight_governance": self.meta_layer.weight_governance,
                "weight_cost": self.meta_layer.weight_cost,
            },
            "choquet": {
                "layer_weights": self.choquet.layer_weights,
                "interaction_weights": {
                    f"{k[0]}_{k[1]}": v 
                    for k, v in self.choquet.interaction_weights.items()
                },
            },
            "random_seed": self.random_seed,
        }
        
        # Compute SHA256 hash
        config_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()


# Default configuration instance
DEFAULT_CALIBRATION_CONFIG = CalibrationSystemConfig()
