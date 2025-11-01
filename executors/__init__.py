"""Compatibility wrapper exposing orchestrator executors."""
from saaaaaa.core.orchestrator import executors as _executors

__all__ = getattr(_executors, "__all__", [])

for name in dir(_executors):
    if name.startswith("_"):
        continue
    globals()[name] = getattr(_executors, name)
