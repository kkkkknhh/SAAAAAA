"""Cross-Cut Signal Channel: questionnaire.monolith → orchestrator.

This module implements the strategic signal propagation system that continuously
irrigates patterns, indicators, regex, verbs, entities, and thresholds into the
answer-generation process.

Architecture:
- SignalPack: Typed, versioned signal payload
- SignalRegistry: In-memory LRU cache with TTL
- SignalClient: Circuit-breaker enabled HTTP client
- Signal-aware execution integration

Design Principles:
- Deterministic signal application
- Graceful degradation on signal unavailability
- Full traceability of signal usage
- Observability via metrics and structured logging
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import blake3
import structlog
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


logger = structlog.get_logger(__name__)


PolicyArea = Literal["fiscal", "salud", "ambiente", "energía", "transporte"]


class SignalPack(BaseModel):
    """
    Versioned strategic signal payload for policy-aware execution.
    
    Contains curated patterns, indicators, and thresholds specific to a policy area.
    All packs carry fingerprints for drift detection and validation windows.
    
    Attributes:
        version: Semantic version string (e.g., "1.0.0")
        policy_area: Policy domain this pack targets
        patterns: Text patterns for narrative detection
        indicators: Key performance indicators for scoring
        regex: Regular expressions for structured extraction
        verbs: Action verbs for policy intent detection
        entities: Named entities relevant to policy area
        thresholds: Named thresholds for scoring/filtering
        ttl_s: Time-to-live in seconds for cache management
        source_fingerprint: BLAKE3 hash of source content
        valid_from: ISO timestamp when signal becomes valid
        valid_to: ISO timestamp when signal expires
        metadata: Optional additional metadata
    """
    
    version: str = Field(
        description="Semantic version string (e.g., '1.0.0')"
    )
    policy_area: PolicyArea = Field(
        description="Policy domain this pack targets"
    )
    patterns: list[str] = Field(
        default_factory=list,
        description="Text patterns for narrative detection"
    )
    indicators: list[str] = Field(
        default_factory=list,
        description="Key performance indicators for scoring"
    )
    regex: list[str] = Field(
        default_factory=list,
        description="Regular expressions for structured extraction"
    )
    verbs: list[str] = Field(
        default_factory=list,
        description="Action verbs for policy intent detection"
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Named entities relevant to policy area"
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Named thresholds for scoring/filtering"
    )
    ttl_s: int = Field(
        default=3600,
        ge=0,
        description="Time-to-live in seconds for cache management"
    )
    source_fingerprint: str = Field(
        default="",
        description="BLAKE3 hash of source content"
    )
    valid_from: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when signal becomes valid"
    )
    valid_to: str = Field(
        default="",
        description="ISO timestamp when signal expires"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional additional metadata"
    )
    
    model_config = {
        "frozen": True,
        "extra": "forbid",
    }
    
    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError(f"Version must be in format 'X.Y.Z', got '{v}'")
        for part in parts:
            if not part.isdigit():
                raise ValueError(f"Version parts must be numeric, got '{v}'")
        return v
    
    @field_validator("thresholds")
    @classmethod
    def validate_thresholds(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate threshold values are in valid range."""
        for key, value in v.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Threshold '{key}' must be in range [0.0, 1.0], got {value}"
                )
        return v
    
    def compute_hash(self) -> str:
        """
        Compute deterministic BLAKE3 hash of signal pack content.
        
        Returns:
            Hex string of BLAKE3 hash
        """
        content_json = self.model_dump_json(
            exclude={"source_fingerprint", "valid_from", "valid_to", "metadata"},
            indent=None,
            sort_keys=True,
        )
        return blake3.blake3(content_json.encode("utf-8")).hexdigest()
    
    def is_valid(self, now: datetime | None = None) -> bool:
        """
        Check if signal pack is currently valid.
        
        Args:
            now: Current time (defaults to utcnow)
            
        Returns:
            True if signal is within validity window
        """
        if now is None:
            now = datetime.now(timezone.utc)
        
        valid_from_dt = datetime.fromisoformat(self.valid_from.replace("Z", "+00:00"))
        if now < valid_from_dt:
            return False
        
        if self.valid_to:
            valid_to_dt = datetime.fromisoformat(self.valid_to.replace("Z", "+00:00"))
            if now > valid_to_dt:
                return False
        
        return True
    
    def get_keys_used(self) -> list[str]:
        """
        Get list of signal keys that have non-empty values.
        
        Returns:
            List of key names with content
        """
        keys = []
        if self.patterns:
            keys.append("patterns")
        if self.indicators:
            keys.append("indicators")
        if self.regex:
            keys.append("regex")
        if self.verbs:
            keys.append("verbs")
        if self.entities:
            keys.append("entities")
        if self.thresholds:
            keys.append("thresholds")
        return keys


@dataclass
class CacheEntry:
    """Entry in the signal registry cache."""
    signal_pack: SignalPack
    inserted_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class SignalRegistry:
    """
    In-memory LRU cache for signal packs with TTL management.
    
    Features:
    - LRU eviction when capacity exceeded
    - TTL-based expiration
    - Access tracking for observability
    - Thread-safe operations (single-process)
    
    Attributes:
        max_size: Maximum number of cached signal packs
        default_ttl_s: Default TTL for cached entries
    """
    
    def __init__(self, max_size: int = 100, default_ttl_s: int = 3600):
        """
        Initialize signal registry.
        
        Args:
            max_size: Maximum cache size
            default_ttl_s: Default TTL in seconds
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl_s = default_ttl_s
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(
            "signal_registry_initialized",
            max_size=max_size,
            default_ttl_s=default_ttl_s,
        )
    
    def put(self, policy_area: str, signal_pack: SignalPack) -> None:
        """
        Store signal pack in registry.
        
        Args:
            policy_area: Policy area key
            signal_pack: Signal pack to store
        """
        now = time.time()
        
        # Remove expired entries before insertion
        self._evict_expired()
        
        # LRU eviction if at capacity
        if len(self._cache) >= self._max_size and policy_area not in self._cache:
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key)
            self._evictions += 1
            logger.debug("signal_registry_evicted_lru", key=oldest_key)
        
        # Insert or update
        entry = CacheEntry(signal_pack=signal_pack, inserted_at=now)
        self._cache[policy_area] = entry
        self._cache.move_to_end(policy_area)  # Mark as most recently used
        
        logger.info(
            "signal_registry_put",
            policy_area=policy_area,
            version=signal_pack.version,
            hash=signal_pack.compute_hash()[:16],
        )
    
    def get(self, policy_area: str) -> SignalPack | None:
        """
        Retrieve signal pack from registry.
        
        Args:
            policy_area: Policy area key
            
        Returns:
            Signal pack if found and valid, None otherwise
        """
        now = time.time()
        
        entry = self._cache.get(policy_area)
        if entry is None:
            self._misses += 1
            logger.debug("signal_registry_miss", policy_area=policy_area)
            return None
        
        # Check TTL expiration
        ttl = entry.signal_pack.ttl_s or self._default_ttl_s
        if now - entry.inserted_at > ttl:
            # Expired, remove from cache
            self._cache.pop(policy_area)
            self._misses += 1
            logger.debug(
                "signal_registry_expired",
                policy_area=policy_area,
                age_s=now - entry.inserted_at,
            )
            return None
        
        # Check validity window
        if not entry.signal_pack.is_valid():
            self._cache.pop(policy_area)
            self._misses += 1
            logger.debug("signal_registry_invalid", policy_area=policy_area)
            return None
        
        # Valid hit
        entry.access_count += 1
        entry.last_accessed = now
        self._cache.move_to_end(policy_area)  # Mark as most recently used
        self._hits += 1
        
        logger.debug(
            "signal_registry_hit",
            policy_area=policy_area,
            access_count=entry.access_count,
        )
        
        return entry.signal_pack
    
    def _evict_expired(self) -> None:
        """Remove expired entries from cache."""
        now = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            ttl = entry.signal_pack.ttl_s or self._default_ttl_s
            if now - entry.inserted_at > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._cache.pop(key)
            self._evictions += 1
        
        if expired_keys:
            logger.debug("signal_registry_evicted_expired", count=len(expired_keys))
    
    def get_metrics(self) -> dict[str, Any]:
        """
        Get registry metrics for observability.
        
        Returns:
            Dict with metrics:
            - hit_rate: Cache hit rate [0.0, 1.0]
            - size: Current cache size
            - capacity: Maximum cache size
            - hits: Total cache hits
            - misses: Total cache misses
            - evictions: Total evictions
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        # Compute staleness stats
        now = time.time()
        staleness_values = []
        for entry in self._cache.values():
            staleness_values.append(now - entry.inserted_at)
        
        avg_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0.0
        max_staleness = max(staleness_values) if staleness_values else 0.0
        
        return {
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "capacity": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "staleness_avg_s": avg_staleness,
            "staleness_max_s": max_staleness,
        }
    
    def clear(self) -> None:
        """Clear all entries from registry."""
        self._cache.clear()
        logger.info("signal_registry_cleared")


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class SignalClient:
    """
    HTTP client for fetching signal packs with circuit breaker and retry logic.
    
    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker for fault isolation
    - Structured logging
    - Graceful degradation
    
    Note: This is a stub implementation. In production, this would use
    an actual HTTP client (httpx, requests) to fetch from FastAPI endpoints.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        max_retries: int = 3,
        timeout_s: float = 10.0,
        circuit_breaker_threshold: int = 5,
    ):
        """
        Initialize signal client.
        
        Args:
            base_url: Base URL for signal service
            max_retries: Maximum retry attempts
            timeout_s: Request timeout in seconds
            circuit_breaker_threshold: Failures before circuit opens
        """
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._circuit_breaker_threshold = circuit_breaker_threshold
        
        self._failure_count = 0
        self._circuit_open = False
        self._last_failure_time = 0.0
        
        logger.info(
            "signal_client_initialized",
            base_url=base_url,
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ConnectionError),
    )
    def fetch_signal_pack(self, policy_area: str) -> SignalPack | None:
        """
        Fetch signal pack from remote service.
        
        Args:
            policy_area: Policy area to fetch
            
        Returns:
            SignalPack if successful, None on failure
            
        Raises:
            CircuitBreakerError: If circuit breaker is open
        """
        # Check circuit breaker
        if self._circuit_open:
            now = time.time()
            # Allow retry after 60 seconds
            if now - self._last_failure_time < 60.0:
                logger.warning(
                    "signal_client_circuit_open",
                    policy_area=policy_area,
                )
                raise CircuitBreakerError("Circuit breaker is open")
            else:
                # Try to close circuit
                self._circuit_open = False
                self._failure_count = 0
                logger.info("signal_client_circuit_closed")
        
        try:
            # TODO: Actual HTTP request implementation
            # import httpx
            # response = httpx.get(
            #     f"{self._base_url}/signals/{policy_area}",
            #     timeout=self._timeout_s,
            # )
            # response.raise_for_status()
            # data = response.json()
            # return SignalPack(**data)
            
            # Stub: Return None to indicate service unavailable
            logger.warning(
                "signal_client_stub_implementation",
                policy_area=policy_area,
                message="TODO: Implement actual HTTP client",
            )
            self._record_failure()
            return None
            
        except Exception as e:
            logger.error(
                "signal_client_fetch_failed",
                policy_area=policy_area,
                error=str(e),
            )
            self._record_failure()
            return None
    
    def _record_failure(self) -> None:
        """Record a failure and potentially open circuit."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self._circuit_breaker_threshold:
            self._circuit_open = True
            logger.warning(
                "signal_client_circuit_opened",
                failure_count=self._failure_count,
            )


@dataclass
class SignalUsageMetadata:
    """
    Metadata about signal usage in an execution.
    
    Attributes:
        version: Signal pack version used
        policy_area: Policy area of signals
        hash: Content hash of signal pack
        keys_used: List of signal keys actually used
        timestamp_utc: ISO timestamp of usage
    """
    
    version: str
    policy_area: str
    hash: str
    keys_used: list[str]
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "policy_area": self.policy_area,
            "hash": self.hash,
            "keys_used": self.keys_used,
            "timestamp_utc": self.timestamp_utc,
        }


def create_default_signal_pack(policy_area: PolicyArea) -> SignalPack:
    """
    Create default signal pack for a policy area (conservative mode).
    
    Args:
        policy_area: Policy area
        
    Returns:
        SignalPack with conservative defaults
    """
    return SignalPack(
        version="0.0.0",
        policy_area=policy_area,
        patterns=[],
        indicators=[],
        regex=[],
        verbs=[],
        entities=[],
        thresholds={
            "min_confidence": 0.9,
            "min_evidence": 0.8,
        },
        ttl_s=0,  # No expiration for defaults
        source_fingerprint="default",
        metadata={"mode": "conservative_fallback"},
    )
