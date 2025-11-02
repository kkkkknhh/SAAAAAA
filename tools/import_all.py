"""Import every module in key packages to surface hidden errors."""
from __future__ import annotations

import importlib
import pkgutil
import traceback
from collections.abc import Iterable, Iterator, Sequence

PKG_PREFIXES: Sequence[str] = ("core.", "executors.", "orchestrator.")


def _iter_modules(prefix: str, errors: list[tuple[str, BaseException, str]]) -> Iterator[str]:
    module_name = prefix[:-1]
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive logging
        errors.append((module_name, exc, traceback.format_exc()))
        return
    if hasattr(module, "__path__"):
        for _, name, _ in pkgutil.walk_packages(module.__path__, prefix=prefix):
            yield name


def collect_modules(prefixes: Iterable[str], errors: list[tuple[str, BaseException, str]]) -> list[str]:
    modules = set()
    for prefix in prefixes:
        for name in _iter_modules(prefix, errors):
            modules.add(name)
    return sorted(modules)


def main() -> None:
    errors: list[tuple[str, BaseException, str]] = []
    modules = collect_modules(PKG_PREFIXES, errors)
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - enumerating failures
            errors.append((module_name, exc, traceback.format_exc()))
    if errors:
        print("=== IMPORT ERRORS ===")
        for idx, (name, error, tb) in enumerate(errors, start=1):
            print(f"[{idx}] {name}: {error}\n{tb}")
        raise SystemExit(1)
    print("All modules imported cleanly.")


if __name__ == "__main__":  # pragma: no cover
    main()
