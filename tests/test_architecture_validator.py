"""Tests for the architecture validation utilities."""

from pathlib import Path

import pytest

from validation import validate_architecture


@pytest.fixture(scope="module")
def architecture_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent.parent
    return (
        root / "policy_analysis_architecture.json",
        root / "COMPLETE_METHOD_CLASS_MAP.json",
    )


def test_architecture_methods_are_resolvable(architecture_paths: tuple[Path, Path]) -> None:
    spec_path, inventory_path = architecture_paths
    result = validate_architecture(spec_path, inventory_path)

    assert result.coverage > 0, "Coverage should be greater than zero."
    assert not result.missing_methods, (
        "Every method referenced in the architecture must exist in the codebase.",
    )
