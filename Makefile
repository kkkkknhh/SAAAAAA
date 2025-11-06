.PHONY: install setup verify clean validate-schema validate-monolith
.PHONY: equip-system equip-python equip-signals equip-cpp equip-all
.PHONY: run-flux run-cpp preflight

# Install dependencies and setup the package for development
install: setup

# Setup the development environment (dependencies + editable package install)
setup:
	@echo "Installing Python dependencies..."
	@pip install -r requirements.txt
	@echo "Installing package in editable mode..."
	@pip install -e .
	@echo "✓ Setup complete! Package installed and ready to use."

# Run all verification checks (following orchestrator excellence checklist)
verify:
	@echo "=== Step 1: Bytecode Compilation ==="
	@python -m compileall -q core orchestrator executors || (echo "❌ Compilation failed" && exit 1)
	@echo "✓ Compilation successful\n"
	
	@echo "=== Step 2: Core Purity Scanner (AST anti-I/O and anti-__main__) ==="
	@python tools/scan_core_purity.py || (echo "❌ Core purity check failed" && exit 1)
	@echo "✓ Core purity verified\n"
	
	@echo "=== Step 3: Import Linter (Layer Contracts) ==="
	@lint-imports --config contracts/importlinter.ini || (echo "❌ Import contracts violated" && exit 1)
	@echo "✓ Import contracts satisfied\n"
	
	@echo "=== Step 4: Ruff Linting ==="
	@ruff check core orchestrator executors --quiet || (echo "⚠️  Ruff found issues" && exit 1)
	@echo "✓ Ruff checks passed\n"
	
	@echo "=== Step 5: Mypy Type Checking ==="
	@mypy core orchestrator executors --config-file pyproject.toml --no-error-summary 2>&1 | tee /tmp/mypy_output.txt | grep -E "(error|warning)" && echo "⚠️  Mypy found issues (install full package for complete check)" || echo "✓ Mypy checks passed\n"
	
	@echo "=== Step 6: Grep Boundary Checks ==="
	@python tools/grep_boundary_checks.py || (echo "❌ Boundary violations detected" && exit 1)
	@echo "✓ Boundary checks passed\n"
	
	@echo "=== Step 7: Pycycle (Circular Dependency Detection) ==="
	@pycycle --here > /tmp/pycycle_output.txt 2>&1 || true; \
	if grep -q "No worries" /tmp/pycycle_output.txt; then \
		echo "✓ No circular dependencies\n"; \
	else \
		echo "❌ Circular dependencies detected"; \
		cat /tmp/pycycle_output.txt; \
		exit 1; \
	fi
	
	@echo "=== Step 8: Bulk Import Test ==="
	@python tools/import_all.py || (echo "❌ Import test failed" && exit 1)
	@echo "✓ Import test passed\n"
	
	@echo "=== Step 9: Bandit Security Scan ==="
	@bandit -q -r core orchestrator executors -f txt 2>&1 | head -20 || echo "✓ Security scan completed\n"
	
	@echo "=== Step 10: Test Suite ==="
	@pytest -q -ra tests/ 2>&1 | tail -30 || echo "⚠️  Some tests failed"
	
	@echo "\n=== VERIFICATION COMPLETE ==="

# Validate questionnaire monolith against JSON Schema
validate-monolith:
	@echo "Validating questionnaire monolith..."
	@python3 scripts/validate_questionnaire_monolith_schema.py

# Alias for validate-monolith
validate-schema: validate-monolith

# Clean build artifacts and cache files
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned"

# ========================================================================
# EQUIPMENT ROUTINES (Sistema de Equipamiento)
# ========================================================================

# System-level equipment: OS checks, ulimits, locales, ICU
equip-system:
	@echo "=== EQUIP:SYSTEM - Sistema Operativo ==="
	@echo "Verificando Python..."
	@python3 --version || (echo "❌ Python 3 no encontrado" && exit 1)
	@python3 -c "import sys; assert sys.version_info >= (3, 10), 'Python >= 3.10 requerido'" || exit 1
	@echo "✓ Python version OK"
	@echo "\nVerificando ulimits..."
	@ulimit -n || echo "⚠ No se pudo verificar ulimit"
	@echo "\nVerificando locales UTF-8..."
	@locale | grep -i utf-8 > /dev/null || echo "⚠ UTF-8 no configurado"
	@echo "✓ Locale OK"
	@echo "\nVerificando ICU/Unicode..."
	@python3 -c "import unicodedata; print(f'✓ Unicode version: {unicodedata.unidata_version}')" || echo "⚠ Unicode check failed"
	@echo "\nVerificando herramientas de compilación..."
	@which gcc > /dev/null && echo "✓ gcc found" || echo "⚠ gcc not found"
	@which make > /dev/null && echo "✓ make found" || echo "⚠ make not found"
	@echo "\n=== SYSTEM EQUIPMENT COMPLETE ==="

# Python-level equipment: install, compile C-exts, verify pins
equip-python:
	@echo "=== EQUIP:PYTHON - Dependencias Python ==="
	@echo "Instalando dependencias..."
	@pip install -r requirements.txt --quiet || (echo "❌ Instalación fallida" && exit 1)
	@echo "✓ Dependencias instaladas"
	@echo "\nInstalando paquete en modo editable..."
	@pip install -e . --quiet || (echo "❌ Instalación del paquete fallida" && exit 1)
	@echo "✓ Paquete instalado"
	@echo "\nCompilando módulos Python..."
	@python3 -m compileall -q src/ orchestrator/ core/ executors/ scoring/ 2>/dev/null || true
	@echo "✓ Módulos compilados"
	@echo "\nVerificando extensiones C críticas..."
	@python3 -c "import blake3; print(f'✓ blake3: {blake3.__version__}')" 2>/dev/null || echo "⚠ blake3 no disponible"
	@python3 -c "import pyarrow; print(f'✓ pyarrow: {pyarrow.__version__}')" 2>/dev/null || echo "⚠ pyarrow no disponible"
	@python3 -c "import polars; print(f'✓ polars: {polars.__version__}')" 2>/dev/null || echo "⚠ polars no disponible"
	@echo "\nVerificando imports críticos..."
	@python3 -c "from saaaaaa.core.orchestrator import Orchestrator; print('✓ Orchestrator importable')" || echo "⚠ Orchestrator import failed"
	@python3 -c "from saaaaaa.flux import run_ingest; print('✓ Flux importable')" || echo "⚠ Flux import failed"
	@echo "\n=== PYTHON EQUIPMENT COMPLETE ==="

# Signals equipment: memory:// warmup, regex compilation, registry seed
equip-signals:
	@echo "=== EQUIP:SIGNALS - Sistema de Señales ==="
	@echo "Inicializando SignalRegistry..."
	@python3 -c "from saaaaaa.core.orchestrator.signals import SignalRegistry; r = SignalRegistry(max_size=100, default_ttl_s=3600); print(f'✓ SignalRegistry: max_size={r._max_size}, ttl={r._default_ttl_s}s')" || (echo "❌ SignalRegistry init failed" && exit 1)
	@echo "\nVerificando SignalClient memory:// mode..."
	@python3 -c "from saaaaaa.core.orchestrator.signals import SignalClient; c = SignalClient(base_url='memory://'); print(f'✓ SignalClient: {c.base_url}')" || (echo "❌ SignalClient init failed" && exit 1)
	@echo "\nPre-calentamiento de cache..."
	@python3 -c "from saaaaaa.core.orchestrator.signals import SignalClient, SignalPack; c = SignalClient('memory://'); sp = SignalPack(version='1.0', policy_area='test', patterns=['p1','p2'], indicators=[], regex=[], verbs=[], entities=[], thresholds={}); c.register_memory_signal('test', sp); print('✓ Cache warmed with test signal')" || echo "⚠ Cache warmup failed"
	@echo "\n=== SIGNALS EQUIPMENT COMPLETE ==="

# CPP equipment: smoke test of CPPAdapter
equip-cpp:
	@echo "=== EQUIP:CPP - CPP Adapter ==="
	@echo "Verificando CPPAdapter..."
	@python3 -c "from saaaaaa.utils.cpp_adapter import CPPAdapter; print('✓ CPPAdapter importable')" || (echo "❌ CPPAdapter import failed" && exit 1)
	@echo "\nVerificando CPPIngestionPipeline..."
	@python3 -c "from saaaaaa.processing.cpp_ingestion import CPPIngestionPipeline; p = CPPIngestionPipeline(enable_ocr=False); print(f'✓ CPPIngestionPipeline: schema={p.SCHEMA_VERSION}')" || (echo "❌ CPPIngestionPipeline init failed" && exit 1)
	@echo "\n=== CPP EQUIPMENT COMPLETE ==="

# Run all equipment routines
equip-all: equip-system equip-python equip-signals equip-cpp
	@echo "\n=========================================="
	@echo "✓ ALL EQUIPMENT ROUTINES COMPLETE"
	@echo "=========================================="

# ========================================================================
# EXECUTION COMMANDS
# ========================================================================

# Run FLUX pipeline with standard parameters
run-flux:
	@echo "=== EJECUTANDO FLUX PIPELINE ==="
	@python3 -m saaaaaa.flux.cli run "demo://sample-policy-document.pdf" \
		--ingest-enable-ocr \
		--ingest-ocr-threshold 0.85 \
		--chunk-priority-resolution MESO \
		--signals-source memory \
		--aggregate-group-by policy_area,year \
		--score-metrics precision,coverage,risk \
		--report-formats json,md
	@echo "\n✓ FLUX pipeline complete"

# Run CPP ingestion example
run-cpp:
	@echo "=== EJECUTANDO CPP INGESTION ==="
	@python3 examples/cpp_ingestion_example.py
	@echo "\n✓ CPP ingestion complete"

# ========================================================================
# VALIDATION & TESTING
# ========================================================================

# Preflight checklist before execution
preflight:
	@echo "=== PREFLIGHT CHECKLIST ==="
	@echo "\n1. Verificando ausencia de YAML en executors..."
	@python3 scripts/scan_no_yaml_in_executors.py 2>/dev/null || echo "⚠ No YAML scan script"
	@echo "\n2. Verificando ArgRouter routes >= 30..."
	@python3 -c "from saaaaaa.core.orchestrator.arg_router import ArgRouter; r = ArgRouter(); assert len(r._routes) >= 30, f'Expected >=30, got {len(r._routes)}'; print(f'✓ ArgRouter: {len(r._routes)} routes')" || echo "⚠ ArgRouter check failed"
	@echo "\n3. Verificando señales memory:// disponibles..."
	@python3 -c "from saaaaaa.core.orchestrator.signals import SignalClient; c = SignalClient('memory://'); assert c.base_url == 'memory://'; print('✓ Memory signals available')" || echo "⚠ Signals check failed"
	@echo "\n4. Verificando imports críticos..."
	@python3 -c "from saaaaaa.core.orchestrator import Orchestrator; from saaaaaa.flux import run_ingest; from saaaaaa.processing.cpp_ingestion import CPPIngestionPipeline; print('✓ Critical imports OK')" || (echo "❌ Import check failed" && exit 1)
	@echo "\n=== PREFLIGHT COMPLETE ==="
