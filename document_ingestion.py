"""
Document Ingestion - Strategic High-Level Wiring Stub.

This file exists for backward compatibility and strategic wiring validation.
The actual implementation is in src/saaaaaa/processing/document_ingestion.py

Note: Wildcard import is intentional here for backward compatibility.
This stub re-exports all symbols from the actual implementation.
"""

# Import from actual implementation for backward compatibility
# Wildcard import is intentional to maintain backward compatibility
try:
    from saaaaaa.processing.document_ingestion import *  # noqa: F401, F403
except ImportError:
    # Graceful fallback if src module is not available
    pass
