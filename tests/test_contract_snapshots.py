"""Snapshot tests that guard contract schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from saaaaaa.utils import core_contracts


SNAPSHOT_PATH = Path(__file__).parent / "data" / "contract_snapshots.json"


def _format_type(annotation: object) -> str:
    text = repr(annotation)
    return text.replace("typing.", "")


def _collect_contracts() -> Dict[str, Dict[str, str]]:
    members: Dict[str, Dict[str, str]] = {}
    for name in dir(core_contracts):
        if not name.endswith("Contract"):
            continue
        obj = getattr(core_contracts, name)
        annotations = getattr(obj, "__annotations__", None)
        if not isinstance(annotations, dict):
            continue
        members[name] = {
            field: _format_type(annotation)
            for field, annotation in sorted(annotations.items())
        }
    return dict(sorted(members.items()))


def test_contract_snapshots_are_stable() -> None:
    assert SNAPSHOT_PATH.exists(), (
        "Contract snapshot missing. Run the governance tests to regenerate "
        "or update tests/data/contract_snapshots.json."
    )

    current = _collect_contracts()
    stored = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert current == stored, (
        "Core contract schema changed. Update tests/data/contract_snapshots.json "
        "after stakeholder approval."
    )
