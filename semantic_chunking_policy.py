"""
Semantic Chunking Policy - Strategic High-Level Wiring Stub.

This file exists for backward compatibility and strategic wiring validation.
The actual implementation is in src/saaaaaa/processing/semantic_chunking_policy.py

Note: Wildcard import is intentional here for backward compatibility.
This stub re-exports all symbols from the actual implementation.
"""

# Import from actual implementation for backward compatibility
# Wildcard import is intentional to maintain backward compatibility
try:
    from saaaaaa.processing.semantic_chunking_policy import *  # noqa: F401, F403
except ImportError:
    # Graceful fallback if src module is not available
    pass
