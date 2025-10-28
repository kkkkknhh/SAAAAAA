"""Tests for the JSON contract loader utility."""
from pathlib import Path

from json_contract_loader import JSONContractLoader


def test_load_report_assembly_contracts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    loader = JSONContractLoader(base_path=repo_root)
    report = loader.load_directory("schemas/report_assembly")

    assert report.is_successful, report.summary()
    assert len(report.documents) == 4

    sample_key = next(iter(sorted(report.documents.keys())))
    document = report.documents[sample_key]
    assert document.checksum
    assert len(document.checksum) == 64
    assert isinstance(document.payload, dict)


def test_invalid_contract_reports_error(tmp_path: Path) -> None:
    invalid_dir = tmp_path / "contracts"
    invalid_dir.mkdir()
    malformed = invalid_dir / "broken.json"
    malformed.write_text("{" , encoding="utf-8")

    loader = JSONContractLoader(base_path=invalid_dir)
    report = loader.load_directory(Path("."))

    assert not report.is_successful
    assert any("broken.json" in error for error in report.errors)


def test_non_object_contract_fails(tmp_path: Path) -> None:
    invalid_dir = tmp_path / "contracts"
    invalid_dir.mkdir()
    not_object = invalid_dir / "array.json"
    not_object.write_text("[1, 2, 3]", encoding="utf-8")

    loader = JSONContractLoader(base_path=invalid_dir)
    report = loader.load(["array.json"])

    assert not report.is_successful
    assert "Contract document must be a JSON object" in report.errors[0]
