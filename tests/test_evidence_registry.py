"""Tests for the append-only evidence registry."""
import json
from pathlib import Path

import pytest

from evidence_registry import EvidenceRegistry


def test_append_builds_hash_chain(tmp_path: Path) -> None:
    registry = EvidenceRegistry(storage_path=tmp_path / "ledger.json", auto_load=False)

    first = registry.append("ReportAssemblyProducer.produce_micro_answer", ["evidence-a"], {"score": 0.9})
    second = registry.append("ReportAssemblyProducer.produce_meso_cluster", ["evidence-b"], {"score": 0.8})

    assert first.index == 0
    assert second.index == 1
    assert second.previous_hash == first.entry_hash
    assert first.entry_hash != second.entry_hash
    registry.verify()


def test_registry_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    registry = EvidenceRegistry(storage_path=path, auto_load=False)
    registry.append("method_a", ["e1"], None)
    registry.append("method_b", ["e2"], {"meta": "yes"})
    registry.save()

    reloaded = EvidenceRegistry(storage_path=path, auto_load=True)
    assert len(reloaded.records) == 2
    assert reloaded.records[0].method_name == "method_a"
    assert reloaded.records[1].metadata["meta"] == "yes"
    assert reloaded.verify()


def test_registry_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    registry = EvidenceRegistry(storage_path=path, auto_load=False)
    registry.append("method_a", ["e1"], None)
    registry.save()

    # Tamper with evidence payload directly
    content = json.loads(path.read_text())
    content[0]["evidence"] = ["tampered"]
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ValueError):
        EvidenceRegistry(storage_path=path, auto_load=True)
