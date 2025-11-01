.PHONY: verify

verify:
@python -m compileall -q core orchestrator executors
@python tools/scan_core_purity.py
@lint-imports --config contracts/importlinter.ini
@ruff check .
@mypy . --strict
@pycycle core orchestrator executors
@python tools/import_all.py
@pytest -q -ra
@coverage run -m pytest >/dev/null 2>&1 || true; coverage report -m || true
