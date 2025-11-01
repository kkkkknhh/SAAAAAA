"""Factory helpers for orchestrator components."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.contracts import IndustrialInput

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from policy_processor import IndustrialPolicyProcessor


def _load_questionnaire(path: Path) -> Any:
    """Load questionnaire data from *path* with UTF-8 encoding."""
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def build_processor(
    path: str | Path = "questionnaire_monolith.json",
    locale: str = "es",
) -> "IndustrialPolicyProcessor":
    """Build an :class:`IndustrialPolicyProcessor` initialised with questionnaire data."""
    questionnaire_path = Path(path)
    data = _load_questionnaire(questionnaire_path)
    questions = data.get("industrial", [])
    industrial_input: IndustrialInput = {"questions": list(questions), "locale": locale}
    from policy_processor import IndustrialPolicyProcessor

    processor = IndustrialPolicyProcessor()
    if hasattr(processor, "bootstrap_from_contract"):
        processor.bootstrap_from_contract(industrial_input)  # type: ignore[attr-defined]
    return processor
