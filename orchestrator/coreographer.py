"""Compatibility shim exposing the legacy coreographer API.

NOTE: 'coreographer' is a typo that was preserved for backward compatibility.
The correct spelling is 'choreographer'. This shim redirects to the real
implementation at src/saaaaaa/core/orchestrator/choreographer.py.

New code should import directly from saaaaaa.core.orchestrator.choreographer.
"""
from saaaaaa.core.orchestrator.choreographer import *  # noqa: F401,F403
