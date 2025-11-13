"""
Thread-safe deterministic random number generation.

This module provides thread-local deterministic RNG to resolve Issue #10:
Deterministic Context Not Thread-Safe. Replaces global random.seed()
with thread-isolated RNG instances.

Version: 2.0.0
Status: Production-ready with maximum responsibility! 💖
"""

import hashlib
import random
import threading
from contextlib import contextmanager
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class DeterministicRNG:
    """Thread-safe deterministic random number generator.

    This class resolves Issue #10 by providing thread-local RNG instances
    instead of modifying global random state. Each thread gets its own
    seeded RNG, preventing conflicts in concurrent execution.

    Attributes:
        np_rng: NumPy random generator (if NumPy available)
        python_rng: Python random.Random instance
        seed: The seed used to initialize this RNG
        thread_id: ID of thread this RNG belongs to

    Example:
        >>> with DeterministicRNG.for_context("policy_123", "corr_456") as rng:
        ...     value = rng.random()  # Thread-safe random value
        ...     array = rng.np_random.random((3, 3))  # NumPy arrays
    """

    _local = threading.local()

    def __init__(self, seed: int, thread_id: Optional[int] = None):
        """Initialize RNG with given seed.

        Args:
            seed: Seed value for deterministic generation
            thread_id: Optional thread ID for tracking
        """
        self.seed = seed
        self.thread_id = thread_id or threading.get_ident()

        # Initialize Python RNG
        self.python_rng = random.Random(seed)

        # Initialize NumPy RNG if available
        if HAS_NUMPY:
            self.np_rng = np.random.default_rng(seed)
        else:
            self.np_rng = None

    @classmethod
    def seed_from_context(
        cls,
        policy_unit_id: str,
        correlation_id: str,
        additional_entropy: Optional[str] = None
    ) -> 'DeterministicRNG':
        """Create seeded RNG for execution context.

        This method generates a deterministic seed from the execution context,
        including thread ID to ensure thread-local isolation.

        Args:
            policy_unit_id: Policy unit identifier
            correlation_id: Correlation/execution identifier
            additional_entropy: Optional additional entropy for seed

        Returns:
            DeterministicRNG instance seeded for this context

        Example:
            >>> rng = DeterministicRNG.seed_from_context("policy_123", "exec_456")
            >>> # Use rng.random() for deterministic random numbers
        """
        thread_id = threading.get_ident()

        # Build seed material from context
        components = [
            str(policy_unit_id),
            str(correlation_id),
            str(thread_id),  # Include thread ID for isolation
        ]

        if additional_entropy:
            components.append(str(additional_entropy))

        material = "|".join(components)

        # Generate deterministic seed from SHA-256 hash
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        base_seed = int.from_bytes(digest[:4], byteorder="big")

        # Create RNG instance
        rng = cls(base_seed, thread_id=thread_id)

        # Store in thread-local storage
        cls._local.rng = rng

        return rng

    @classmethod
    def get_current(cls) -> Optional['DeterministicRNG']:
        """Get current thread's RNG instance.

        Returns:
            DeterministicRNG if one exists for this thread, None otherwise

        Example:
            >>> rng = DeterministicRNG.get_current()
            >>> if rng:
            ...     value = rng.random()
        """
        return getattr(cls._local, 'rng', None)

    @classmethod
    @contextmanager
    def for_context(
        cls,
        policy_unit_id: str,
        correlation_id: str,
        additional_entropy: Optional[str] = None
    ):
        """Context manager for deterministic execution.

        This is the recommended way to use DeterministicRNG. It ensures
        proper setup and cleanup of thread-local RNG.

        Args:
            policy_unit_id: Policy unit identifier
            correlation_id: Correlation identifier
            additional_entropy: Optional additional entropy

        Yields:
            DeterministicRNG instance for use in context

        Example:
            >>> with DeterministicRNG.for_context("policy_123", "exec_456") as rng:
            ...     # All random operations use rng
            ...     result = rng.choice([1, 2, 3])
            ...     # RNG automatically cleaned up after block
        """
        previous = getattr(cls._local, 'rng', None)
        rng = cls.seed_from_context(policy_unit_id, correlation_id, additional_entropy)

        try:
            yield rng
        finally:
            # Restore previous thread-local RNG to support nested contexts
            if previous is None:
                if hasattr(cls._local, 'rng'):
                    delattr(cls._local, 'rng')
            else:
                cls._local.rng = previous

    # ========================================================================
    # RANDOM GENERATION METHODS (Python stdlib compatibility)
    # ========================================================================

    def random(self) -> float:
        """Generate random float in [0.0, 1.0).

        Returns:
            Random float

        Example:
            >>> rng = DeterministicRNG(42)
            >>> value = rng.random()  # Deterministic based on seed 42
        """
        return self.python_rng.random()

    def randint(self, a: int, b: int) -> int:
        """Generate random integer in [a, b].

        Args:
            a: Lower bound (inclusive)
            b: Upper bound (inclusive)

        Returns:
            Random integer
        """
        return self.python_rng.randint(a, b)

    def choice(self, seq):
        """Choose random element from non-empty sequence.

        Args:
            seq: Sequence to choose from

        Returns:
            Random element from sequence
        """
        return self.python_rng.choice(seq)

    def shuffle(self, x):
        """Shuffle sequence x in place.

        Args:
            x: Mutable sequence to shuffle
        """
        return self.python_rng.shuffle(x)

    def sample(self, population, k):
        """Choose k unique random elements from population.

        Args:
            population: Sequence to sample from
            k: Number of elements to choose

        Returns:
            List of k unique elements
        """
        return self.python_rng.sample(population, k)

    def uniform(self, a: float, b: float) -> float:
        """Generate random float in [a, b] or [b, a].

        Args:
            a: Lower/upper bound
            b: Upper/lower bound

        Returns:
            Random float
        """
        return self.python_rng.uniform(a, b)

    def gauss(self, mu: float, sigma: float) -> float:
        """Generate random float from Gaussian distribution.

        Args:
            mu: Mean
            sigma: Standard deviation

        Returns:
            Random float from N(mu, sigma^2)
        """
        return self.python_rng.gauss(mu, sigma)

    # ========================================================================
    # NUMPY COMPATIBILITY (if available)
    # ========================================================================

    @property
    def np_random(self):
        """Access NumPy random generator.

        Returns:
            NumPy Generator instance if NumPy available, None otherwise

        Raises:
            RuntimeError: If NumPy not available

        Example:
            >>> rng = DeterministicRNG(42)
            >>> if rng.has_numpy:
            ...     array = rng.np_random.random((3, 3))
        """
        if not HAS_NUMPY:
            raise RuntimeError("NumPy not available. Install numpy to use np_random.")
        return self.np_rng

    @property
    def has_numpy(self) -> bool:
        """Check if NumPy is available.

        Returns:
            True if NumPy installed and RNG initialized
        """
        return HAS_NUMPY and self.np_rng is not None

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def get_state(self) -> dict:
        """Get current RNG state for serialization.

        Returns:
            Dictionary with seed and thread_id

        Example:
            >>> rng = DeterministicRNG(42)
            >>> state = rng.get_state()
            >>> # Later: restore from state
            >>> new_rng = DeterministicRNG(**state)
        """
        return {
            'seed': self.seed,
            'thread_id': self.thread_id,
        }

    def __repr__(self) -> str:
        """String representation of RNG."""
        return f"DeterministicRNG(seed={self.seed}, thread_id={self.thread_id})"


# ============================================================================
# LEGACY COMPATIBILITY WRAPPER
# ============================================================================

@contextmanager
def deterministic_legacy(policy_unit_id: Optional[str], correlation_id: Optional[str]):
    """Legacy compatibility wrapper for old deterministic() context manager.

    This provides backward compatibility with code using the old
    deterministic() function that modified global random state.

    Args:
        policy_unit_id: Policy unit identifier
        correlation_id: Correlation identifier

    Yields:
        Dictionary with 'np' and 'python' seeds (legacy format)

    Example:
        >>> with deterministic_legacy("policy_123", "exec_456") as seeds:
        ...     # seeds.np and seeds.python available for legacy code
        ...     pass
    """
    # Create RNG
    with DeterministicRNG.for_context(
        str(policy_unit_id or "default"),
        str(correlation_id or "default")
    ) as rng:
        # Create legacy seeds object
        class LegacySeeds:
            def __init__(self, np_seed: int, python_seed: int):
                self.np = np_seed
                self.python = python_seed

        seeds = LegacySeeds(np_seed=rng.seed, python_seed=rng.seed + 1)

        yield seeds


__all__ = [
    'DeterministicRNG',
    'deterministic_legacy',
    'HAS_NUMPY',
]
