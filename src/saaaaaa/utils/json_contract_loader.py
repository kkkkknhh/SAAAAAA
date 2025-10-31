"""Utility helpers to load and validate JSON contract documents."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Union


PathLike = Union[str, Path]


def _canonical_dump(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ContractDocument:
    """Materialized JSON contract with checksum information."""

    path: Path
    payload: Dict[str, object]
    checksum: str


@dataclass
class ContractLoadReport:
    """Result of attempting to load multiple contract documents."""

    documents: Dict[str, ContractDocument]
    errors: List[str]

    @property
    def is_successful(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        parts = [f"contracts={len(self.documents)}"]
        if self.errors:
            parts.append(f"errors={len(self.errors)}")
        return ", ".join(parts)


class JSONContractLoader:
    """Load JSON contract files and compute integrity metadata."""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).resolve().parent

    def load(self, paths: Iterable[PathLike]) -> ContractLoadReport:
        documents: Dict[str, ContractDocument] = {}
        errors: List[str] = []
        for raw in paths:
            path = self._resolve_path(raw)
            try:
                payload = self._read_payload(path)
            except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{path}: {exc}")
                continue

            checksum = hashlib.sha256(_canonical_dump(payload).encode("utf-8")).hexdigest()
            documents[str(path)] = ContractDocument(path=path, payload=payload, checksum=checksum)
        return ContractLoadReport(documents=documents, errors=errors)

    def load_directory(self, relative_directory: PathLike, pattern: str = "*.json") -> ContractLoadReport:
        directory = self._resolve_path(relative_directory)
        if not directory.exists():
            return ContractLoadReport(documents={}, errors=[f"Directory not found: {directory}"])
        if not directory.is_dir():
            return ContractLoadReport(documents={}, errors=[f"Not a directory: {directory}"])

        paths = sorted(directory.glob(pattern))
        return self.load(paths)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, raw: PathLike) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = self.base_path / path
        return path

    @staticmethod
    def _read_payload(path: Path) -> Dict[str, object]:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Contract document must be a JSON object")
        return data


__all__ = [
    "ContractDocument",
    "ContractLoadReport",
    "JSONContractLoader",
]
