"""
Executor protocols and contracts for rigorous type safety.

This module defines formal protocols that ensure executor components
adhere to strict contracts. All protocols are designed with maximum
responsibility and fabulous rigor! ✨

Version: 2.0.0
Status: Production-ready with Barbie seal of approval 💖
"""

from enum import Enum
from typing import Protocol, TypedDict, runtime_checkable
from dataclasses import dataclass


# ============================================================================
# CALIBRATION MODE ENUM (Issue #5 Fix)
# ============================================================================

class CalibrationMode(Enum):
    """Explicit calibration execution modes.

    This enum resolves Issue #5: Calibration Orchestrator Optional but Validation Mandatory.

    Modes:
        STRICT: Fail if any calibration missing or placeholder (production mode)
        LENIENT: Skip uncalibrated methods with warning (development mode)
        NONE: Ignore calibrations entirely (testing mode)
    """

    STRICT = "strict"
    LENIENT = "lenient"
    NONE = "none"


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for calibration execution behavior.

    This replaces the ambiguous optional CalibrationOrchestrator with
    explicit mode-based configuration.

    Attributes:
        mode: Calibration enforcement mode
        skip_threshold: Methods with calibration score below this are skipped (LENIENT mode)
        fail_on_placeholder: Raise error if placeholder calibration detected (STRICT mode)
    """

    mode: CalibrationMode = CalibrationMode.STRICT
    skip_threshold: float = 0.3
    fail_on_placeholder: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.skip_threshold <= 1.0:
            raise ValueError(f"skip_threshold must be in [0.0, 1.0], got {self.skip_threshold}")


# ============================================================================
# SIGNAL PACK PROTOCOL (Issue #12 Fix)
# ============================================================================

class SignalPattern(TypedDict):
    """Type definition for signal patterns."""
    pattern: str
    category: str
    confidence: float


@runtime_checkable
class SignalPackProtocol(Protocol):
    """Formal contract for signal packs.

    This protocol resolves Issue #12: Signal Usage Not Validated.
    All signal packs MUST implement this protocol for safe usage.

    Required Attributes:
        version: Signal pack version string
        policy_area: Policy area identifier
        patterns: List of signal patterns
        indicators: List of indicator strings
        regex: Dictionary of regex patterns
        verbs: List of action verbs
        entities: List of named entities
        thresholds: Dictionary of threshold values

    Required Methods:
        compute_hash(): Return deterministic hash of signal pack
        get_keys_used(): Return list of keys accessed during processing
    """

    # Required attributes
    version: str
    policy_area: str
    patterns: list[SignalPattern]
    indicators: list[str]
    regex: dict[str, str]
    verbs: list[str]
    entities: list[str]
    thresholds: dict[str, float]

    # Required methods
    def compute_hash(self) -> str:
        """Return SHA-256 hash of signal pack contents."""
        ...

    def get_keys_used(self) -> list[str]:
        """Return list of keys accessed during signal processing."""
        ...


def validate_signal_pack(signal_pack: object) -> tuple[bool, str]:
    """Validate that an object conforms to SignalPackProtocol.

    Args:
        signal_pack: Object to validate

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.

    Example:
        >>> valid, msg = validate_signal_pack(my_signal_pack)
        >>> if not valid:
        ...     logger.error(f"Invalid signal pack: {msg}")
    """
    if not isinstance(signal_pack, SignalPackProtocol):
        return False, "Object does not implement SignalPackProtocol"

    # Check required attributes
    required_attrs = ['version', 'policy_area', 'patterns', 'indicators',
                      'regex', 'verbs', 'entities', 'thresholds']

    for attr in required_attrs:
        if not hasattr(signal_pack, attr):
            return False, f"Missing required attribute: {attr}"

    # Check required methods
    required_methods = ['compute_hash', 'get_keys_used']
    for method in required_methods:
        if not hasattr(signal_pack, method):
            return False, f"Missing required method: {method}"
        if not callable(getattr(signal_pack, method)):
            return False, f"Attribute {method} is not callable"

    # Validate attribute types
    if not isinstance(signal_pack.version, str):
        return False, f"version must be str, got {type(signal_pack.version).__name__}"

    if not isinstance(signal_pack.policy_area, str):
        return False, f"policy_area must be str, got {type(signal_pack.policy_area).__name__}"

    if not isinstance(signal_pack.patterns, list):
        return False, f"patterns must be list, got {type(signal_pack.patterns).__name__}"

    if not isinstance(signal_pack.indicators, list):
        return False, f"indicators must be list, got {type(signal_pack.indicators).__name__}"

    if not isinstance(signal_pack.regex, dict):
        return False, f"regex must be dict, got {type(signal_pack.regex).__name__}"

    if not isinstance(signal_pack.verbs, list):
        return False, f"verbs must be list, got {type(signal_pack.verbs).__name__}"

    if not isinstance(signal_pack.entities, list):
        return False, f"entities must be list, got {type(signal_pack.entities).__name__}"

    if not isinstance(signal_pack.thresholds, dict):
        return False, f"thresholds must be dict, got {type(signal_pack.thresholds).__name__}"

    # Validate method calls work
    try:
        hash_result = signal_pack.compute_hash()
        if not isinstance(hash_result, str):
            return False, f"compute_hash() must return str, got {type(hash_result).__name__}"
    except Exception as e:
        return False, f"compute_hash() raised exception: {e}"

    try:
        keys_result = signal_pack.get_keys_used()
        if not isinstance(keys_result, list):
            return False, f"get_keys_used() must return list, got {type(keys_result).__name__}"
    except Exception as e:
        return False, f"get_keys_used() raised exception: {e}"

    return True, ""


# ============================================================================
# ADVANCED MODULE GATES (Issue #6 Fix)
# ============================================================================

@dataclass(frozen=True)
class AdvancedModuleGates:
    """Control which frontier modules are actively initialized.

    This resolves Issue #6: Undocumented Module Activation Conditions.
    Modules are only initialized when explicitly enabled AND activation
    conditions are met.

    Attributes:
        enable_quantum: Enable quantum execution optimizer (requires num_methods >= 3)
        enable_neuromorphic: Enable neuromorphic flow controller (requires data flow processing)
        enable_causal: Enable causal inference graph (requires 2+ questions)
        enable_info_theory: Enable information flow optimizer (requires data flow)
        enable_meta_learning: Enable meta-learning strategy selection
        enable_attention: Enable attention mechanism for method prioritization
        enable_topological: Enable topological analysis
        enable_category_theory: Enable category theory abstractions
        enable_probabilistic: Enable probabilistic programming

    Example:
        >>> gates = AdvancedModuleGates(
        ...     enable_quantum=True,
        ...     enable_neuromorphic=False,  # Disable for performance
        ...     enable_causal=True
        ... )
        >>> # Only quantum and causal modules will be initialized
    """

    enable_quantum: bool = True
    enable_neuromorphic: bool = True
    enable_causal: bool = True
    enable_info_theory: bool = True
    enable_meta_learning: bool = True
    enable_attention: bool = True
    enable_topological: bool = False  # Computationally expensive
    enable_category_theory: bool = False  # Theoretical only
    enable_probabilistic: bool = False  # Experimental

    def should_activate_quantum(self, num_methods: int) -> bool:
        """Check if quantum optimizer should activate.

        Activation condition: enable_quantum=True AND num_methods >= 3

        Args:
            num_methods: Number of methods in execution sequence

        Returns:
            True if quantum optimizer should be initialized
        """
        return self.enable_quantum and num_methods >= 3

    def should_activate_neuromorphic(self, has_data_flow: bool) -> bool:
        """Check if neuromorphic controller should activate.

        Activation condition: enable_neuromorphic=True AND has_data_flow=True

        Args:
            has_data_flow: Whether execution includes data flow processing

        Returns:
            True if neuromorphic controller should be initialized
        """
        return self.enable_neuromorphic and has_data_flow

    def should_activate_causal(self, num_questions: int) -> bool:
        """Check if causal inference should activate.

        Activation condition: enable_causal=True AND num_questions >= 2

        Args:
            num_questions: Number of questions in execution

        Returns:
            True if causal graph should be initialized
        """
        return self.enable_causal and num_questions >= 2

    def get_active_modules(self, num_methods: int, has_data_flow: bool, num_questions: int) -> list[str]:
        """Get list of modules that should be active given execution context.

        Args:
            num_methods: Number of methods in sequence
            has_data_flow: Whether execution includes data flow
            num_questions: Number of questions in execution

        Returns:
            List of module names that should be initialized

        Example:
            >>> gates = AdvancedModuleGates()
            >>> active = gates.get_active_modules(num_methods=5, has_data_flow=True, num_questions=1)
            >>> print(active)
            ['quantum', 'neuromorphic', 'info_theory', 'meta_learning', 'attention']
        """
        active = []

        if self.should_activate_quantum(num_methods):
            active.append('quantum')

        if self.should_activate_neuromorphic(has_data_flow):
            active.append('neuromorphic')

        if self.should_activate_causal(num_questions):
            active.append('causal')

        if self.enable_info_theory:
            active.append('info_theory')

        if self.enable_meta_learning:
            active.append('meta_learning')

        if self.enable_attention:
            active.append('attention')

        if self.enable_topological:
            active.append('topological')

        if self.enable_category_theory:
            active.append('category_theory')

        if self.enable_probabilistic:
            active.append('probabilistic')

        return active


# ============================================================================
# EXECUTION CONTEXT PROTOCOL
# ============================================================================

@runtime_checkable
class ExecutionContextProtocol(Protocol):
    """Protocol for execution context objects.

    This ensures compatibility between legacy dict-based contexts
    and new ImmutableExecutionContext from Intervention #4.
    """

    def get(self, key: str, default: object = None) -> object:
        """Get value from context."""
        ...

    def __getitem__(self, key: str) -> object:
        """Get value from context (dict-style)."""
        ...


__all__ = [
    'CalibrationMode',
    'CalibrationConfig',
    'SignalPattern',
    'SignalPackProtocol',
    'validate_signal_pack',
    'AdvancedModuleGates',
    'ExecutionContextProtocol',
]
