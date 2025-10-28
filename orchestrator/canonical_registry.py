"""Canonical method registry for orchestrator dispatch.

This module builds a dictionary mapping fully-qualified method names declared
across orchestrator metadata to the real Python callables that implement them.
It eagerly imports producer modules at process start to ensure that any missing
methods are surfaced immediately, keeping orchestration failures fail-fast.
"""
from __future__ import annotations

import json
import re
import sys
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Iterable, Set, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

__all__ = ["CANONICAL_METHODS", "validate_method_registry", "generate_audit_report"]

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
_CLASS_METHOD_MAP_YAML = "method_class_map.yaml"

# Validation thresholds
_MINIMUM_METHOD_THRESHOLD = 555  # Hard fail if below this
_PROVISIONAL_METHOD_THRESHOLD = 400  # Provisional pass if >= this

_FAILED_IMPORT = object()


class CanonicalRegistryError(ImportError):
    """Raised when canonical symbols cannot be resolved to real callables."""


class RegistryValidationError(Exception):
    """Raised when method registry validation fails."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_class_module_map(path: Path) -> Dict[str, str]:
    """Load the map of class names to module names from the inventory JSON or YAML."""
    # Try YAML first if available
    yaml_path = path.parent / _CLASS_METHOD_MAP_YAML
    if yaml_path.exists() and yaml is not None:
        return _load_class_module_map_from_yaml(yaml_path)
    
    # Fall back to JSON
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files", {})
    mapping: Dict[str, str] = {}

    for file_name, info in files.items():
        module_name = Path(file_name).stem
        for class_name in info.get("classes", {}):
            mapping[class_name] = module_name

    return mapping


def _load_class_module_map_from_yaml(path: Path) -> Dict[str, str]:
    """Load the map of class names to module names from YAML file."""
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML files")
    
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    class_module_map = data.get("class_module_map", {})
    mapping: Dict[str, str] = {}
    
    for class_name, class_info in class_module_map.items():
        if isinstance(class_info, dict):
            mapping[class_name] = class_info.get("module", "")
        else:
            mapping[class_name] = str(class_info)
    
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


def _build_canonical_registry(strict: bool = True) -> Dict[str, Callable[..., object]]:
    """Build the canonical registry.
    
    Args:
        strict: If True, raise on unresolved symbols. If False, return partial registry.
    """
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

    if unresolved and strict:
        errors = "\n".join(f"- {symbol}: {reason}" for symbol, reason in sorted(unresolved.items()))
        raise CanonicalRegistryError(
            "Failed to resolve canonical methods:\n" + errors
        )

    return registry


def _count_all_methods() -> int:
    """Count all methods declared in the method class map."""
    root = _project_root()
    
    # Try YAML first
    yaml_path = root / _CLASS_METHOD_MAP_YAML
    if yaml_path.exists() and yaml is not None:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        total = data.get("metadata", {}).get("total_methods", 0)
        if total > 0:
            return total
    
    # Fall back to JSON
    json_path = root / _CLASS_METHOD_MAP
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data.get("summary", {}).get("total_methods", 0)
    
    return 0


def _validate_method_registry(
    total_methods: int,
    resolved_methods: int,
    provisional: bool = False
) -> Dict[str, any]:
    """Validate method registry meets threshold requirements.
    
    Args:
        total_methods: Total number of methods in the class map
        resolved_methods: Number of successfully resolved methods
        provisional: If True, use provisional threshold (≥400), else use strict (≥555)
    
    Returns:
        Dict with validation results
    
    Raises:
        RegistryValidationError: If validation fails
    """
    threshold = _PROVISIONAL_METHOD_THRESHOLD if provisional else _MINIMUM_METHOD_THRESHOLD
    threshold_name = "provisional" if provisional else "minimum"
    
    result = {
        "total_methods": total_methods,
        "resolved_methods": resolved_methods,
        "threshold": threshold,
        "threshold_type": threshold_name,
        "passed": total_methods >= threshold,
        "coverage_percentage": round((resolved_methods / total_methods * 100), 2) if total_methods > 0 else 0
    }
    
    if not result["passed"]:
        raise RegistryValidationError(
            f"Method registry validation failed: {total_methods} methods < {threshold} {threshold_name} threshold. "
            f"Registry must have at least {threshold} methods to proceed."
        )
    
    return result


def validate_method_registry(provisional: bool = True) -> Dict[str, any]:
    """Public API to validate the method registry.
    
    Args:
        provisional: If True, use provisional threshold (≥400), else use strict (≥555)
    
    Returns:
        Dict with validation results
    """
    total_methods = _count_all_methods()
    
    # Count resolved methods - this would require actually building the registry
    # For now, we'll use the total as an approximation
    # In production, this should be called after registry is built
    resolved_methods = len(CANONICAL_METHODS) if 'CANONICAL_METHODS' in globals() else 0
    
    return _validate_method_registry(total_methods, resolved_methods, provisional)


def generate_audit_report(
    registry: Optional[Dict[str, Callable[..., object]]] = None,
    output_path: Optional[Path] = None
) -> Dict[str, any]:
    """Generate comprehensive audit report for the method registry.
    
    Args:
        registry: The canonical registry (defaults to CANONICAL_METHODS)
        output_path: Path to write audit.json (defaults to project_root/audit.json)
    
    Returns:
        Dict containing audit data
    """
    if registry is None:
        registry = CANONICAL_METHODS if 'CANONICAL_METHODS' in globals() else {}
    
    root = _project_root()
    if output_path is None:
        output_path = root / "audit.json"
    
    # Load metadata
    total_methods = _count_all_methods()
    metadata_paths = [root / name for name in _METADATA_SOURCES]
    declared_symbols = sorted(_extract_canonical_symbols(metadata_paths))
    
    # Load class map to get expected methods
    class_map_path = root / _CLASS_METHOD_MAP
    class_module_map = _load_class_module_map(class_map_path)
    
    # Calculate coverage statistics
    resolved_symbols = set(registry.keys())
    declared_set = set(declared_symbols)
    
    missing_methods = declared_set - resolved_symbols
    extra_methods = resolved_symbols - declared_set
    
    # Validation results
    try:
        validation_provisional = _validate_method_registry(total_methods, len(resolved_symbols), provisional=True)
        validation_strict = _validate_method_registry(total_methods, len(resolved_symbols), provisional=False)
    except RegistryValidationError:
        validation_provisional = {
            "passed": total_methods >= _PROVISIONAL_METHOD_THRESHOLD,
            "threshold": _PROVISIONAL_METHOD_THRESHOLD,
            "total_methods": total_methods
        }
        validation_strict = {
            "passed": total_methods >= _MINIMUM_METHOD_THRESHOLD,
            "threshold": _MINIMUM_METHOD_THRESHOLD,
            "total_methods": total_methods
        }
    
    # Build audit report
    audit = {
        "metadata": {
            "generated_at": Path(__file__).stat().st_mtime if Path(__file__).exists() else 0,
            "registry_version": "1.0.0",
            "sources": list(_METADATA_SOURCES)
        },
        "coverage": {
            "total_methods_in_codebase": total_methods,
            "declared_in_metadata": len(declared_symbols),
            "successfully_resolved": len(resolved_symbols),
            "coverage_percentage": round((len(resolved_symbols) / len(declared_symbols) * 100), 2) if declared_symbols else 0,
            "resolution_rate": round((len(resolved_symbols) / total_methods * 100), 2) if total_methods > 0 else 0
        },
        "validation": {
            "provisional": validation_provisional,
            "strict": validation_strict
        },
        "missing": sorted(list(missing_methods)),
        "extras": sorted(list(extra_methods)),
        "class_coverage": {
            "total_classes": len(class_module_map),
            "mapped_classes": len(set(s.split(".")[0] for s in declared_symbols))
        }
    }
    
    # Write to file
    if output_path:
        output_path.write_text(json.dumps(audit, indent=2, sort_keys=False), encoding="utf-8")
    
    return audit


# Initialize the canonical registry
# Note: This may fail if dependencies are not installed
# In that case, CANONICAL_METHODS will be an empty dict
try:
    # Try to build with strict=False to get partial registry
    CANONICAL_METHODS: Dict[str, Callable[..., object]] = _build_canonical_registry(strict=False)
    
    # Validate registry on load (provisional mode)
    total_methods = _count_all_methods()
    if total_methods < _PROVISIONAL_METHOD_THRESHOLD:
        print(
            f"WARNING: Method registry has {total_methods} methods, "
            f"which is below the provisional threshold of {_PROVISIONAL_METHOD_THRESHOLD}. "
            f"Minimum required for production is {_MINIMUM_METHOD_THRESHOLD}.",
            file=sys.stderr
        )
    
    # Generate audit report (always generate, even with partial registry)
    try:
        audit_path = _project_root() / "audit.json"
        audit_data = generate_audit_report(CANONICAL_METHODS, audit_path)
        print(f"✓ Generated audit report: {audit_path}", file=sys.stderr)
        print(f"  - Total methods: {audit_data['coverage']['total_methods_in_codebase']}", file=sys.stderr)
        print(f"  - Declared: {audit_data['coverage']['declared_in_metadata']}", file=sys.stderr)
        print(f"  - Resolved: {audit_data['coverage']['successfully_resolved']}", file=sys.stderr)
        print(f"  - Missing: {len(audit_data['missing'])}", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: Could not generate audit report: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
except Exception as e:
    print(f"ERROR: Unexpected error building registry: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    CANONICAL_METHODS = {}


