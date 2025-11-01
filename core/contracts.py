"""Typed contracts shared across the orchestration boundary."""
from __future__ import annotations

from typing import Literal, Sequence, TypedDict


class IndustrialInput(TypedDict):
    """Input payload expected by the industrial policy processor."""

    questions: Sequence[str]
    locale: Literal["es", "en"]


class IndustrialOutput(TypedDict):
    """Output payload returned by the industrial policy processor."""

    decisions: Sequence[str]
    warnings: Sequence[str]
