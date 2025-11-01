"""Compatibility wrapper for QMCM hooks."""
from saaaaaa.utils.qmcm_hooks import (  # noqa: F401
    QMCMRecorder,
    get_global_recorder,
    qmcm_record,
)

# Provide backward-compatible alias
record_qmcm_call = qmcm_record

__all__ = [
    "QMCMRecorder",
    "get_global_recorder",
    "qmcm_record",
    "record_qmcm_call",
]
