.PHONY: install setup verify clean

# Install dependencies and setup the package for development
install: setup

# Setup the development environment (dependencies + editable package install)
setup:
	@echo "Installing Python dependencies..."
	@pip install -r requirements.txt
	@echo "Installing package in editable mode..."
	@pip install -e .
	@echo "✓ Setup complete! Package installed and ready to use."

# Run all verification checks
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

# Clean build artifacts and cache files
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned"
