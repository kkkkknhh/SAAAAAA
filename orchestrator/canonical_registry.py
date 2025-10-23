"""Canonical method registry for orchestrator dispatch.

This module builds a dictionary mapping fully-qualified method names declared
across orchestrator metadata to the real Python callables that implement them.
It eagerly imports producer modules at process start to ensure that any missing
methods are surfaced immediately, keeping orchestration failures fail-fast.
"""
from __future__ import annotations

import json
import re
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Iterable, Set

__all__ = ["CANONICAL_METHODS"]

# Pattern for canonical references such as ``IndustrialPolicyProcessor.process``.
_CANONICAL_SYMBOL_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b")

# Metadata sources that declare canonical method usage across the pipeline.
_METADATA_SOURCES = (
    "question_component_map.json",
    "execution_mapping.yaml",
    "policy_analysis_architecture.yaml",
    "policy_analysis_architecture.json",
)

# File describing where classes live in the codebase.
_CLASS_METHOD_MAP = "COMPLETE_METHOD_CLASS_MAP.json"

_FAILED_IMPORT = object()


class CanonicalRegistryError(ImportError):
    """Raised when canonical symbols cannot be resolved to real callables."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_class_module_map(path: Path) -> Dict[str, str]:
    """Load the map of class names to module names from the inventory JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files", {})
    mapping: Dict[str, str] = {}

    for file_name, info in files.items():
        module_name = Path(file_name).stem
        for class_name in info.get("classes", {}):
            mapping[class_name] = module_name

    return mapping


def _extract_canonical_symbols(paths: Iterable[Path]) -> Set[str]:
    """Extract fully-qualified canonical symbols from metadata files."""
    symbols: Set[str] = set()

    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        matches = _CANONICAL_SYMBOL_PATTERN.findall(text)
        symbols.update(matches)

    return symbols


def _import_module(module_name: str):
    """Import a module while converting ``SystemExit`` into ``ImportError``."""
    try:
        return import_module(module_name)
    except SystemExit as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            f"Module '{module_name}' exited during import: {exc}"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            f"Module '{module_name}' could not be imported: {exc}"
        ) from exc


def _resolve_symbol(
    symbol: str,
    module_name: str,
    module_cache: Dict[str, object],
) -> Callable[..., object]:
    """Resolve a canonical symbol to a callable."""
    module = module_cache.get(module_name)
    if module is _FAILED_IMPORT:
        raise ImportError(f"Module '{module_name}' failed during a previous import")
    if module is None:
        module = _import_module(module_name)
        module_cache[module_name] = module

    # Retrieve class or container object first.
    class_name, method_name = symbol.split(".", 1)
    container = getattr(module, class_name, None)
    if container is None:
        raise AttributeError(
            f"Module '{module_name}' does not define '{class_name}'"
        )

    target = getattr(container, method_name, None)
    if target is None or not callable(target):
        raise AttributeError(
            f"'{symbol}' is not a callable attribute on '{module_name}.{class_name}'"
        )

    return target


def _build_canonical_registry() -> Dict[str, Callable[..., object]]:
    root = _project_root()
    metadata_paths = [root / name for name in _METADATA_SOURCES]
    class_map_path = root / _CLASS_METHOD_MAP

    class_module_map = _load_class_module_map(class_map_path)
    canonical_symbols = sorted(_extract_canonical_symbols(metadata_paths))

    module_cache: Dict[str, object] = {}
    registry: Dict[str, Callable[..., object]] = {}
    unresolved: Dict[str, str] = {}

    for symbol in canonical_symbols:
        class_name, _ = symbol.split(".", 1)
        module_name = class_module_map.get(class_name)
        if not module_name:
            unresolved[symbol] = "unknown_class"
            continue

        try:
            registry[symbol] = _resolve_symbol(symbol, module_name, module_cache)
        except ImportError as exc:
            module_cache[module_name] = _FAILED_IMPORT
            unresolved[symbol] = f"module_import_failed: {exc}"
        except AttributeError as exc:
            unresolved[symbol] = str(exc)

    if unresolved:
        errors = "\n".join(f"- {symbol}: {reason}" for symbol, reason in sorted(unresolved.items()))
        raise CanonicalRegistryError(
            "Failed to resolve canonical methods:\n" + errors
        )

    return registry


CANONICAL_METHODS: Dict[str, Callable[..., object]] = _build_canonical_registry()
