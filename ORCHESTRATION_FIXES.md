# ORCHESTRATION FIXES - ACTION ITEMS

This document provides specific, actionable fixes for all issues identified in the orchestration audit.

---

## CRITICAL SECURITY FIXES (DO THESE FIRST)

### ISSUE 1: Missing permissions blocks in workflows
**Files:** `d2_concurrence.yml`, `data-contracts.yml`, `static-analysis.yml`

**Fix:**
```yaml
# Add at the top of each workflow (after name:)
permissions:
  contents: read
```

**Why:** Default workflow permissions are write-all, which violates principle of least privilege.

---

### ISSUE 2: Hardcoded secrets in atroz_quickstart.sh
**File:** `atroz_quickstart.sh` lines 118-130

**Current (INSECURE):**
```bash
ATROZ_API_SECRET=dev-secret-key-change-in-production
ATROZ_JWT_SECRET=dev-jwt-secret-change-in-production
```

**Fix:**
```bash
# Generate random secrets
if [ ! -f ".env" ]; then
    print_info "Creating .env file with generated secrets..."
    
    # Check if openssl is available (preferred)
    if command -v openssl &> /dev/null; then
        # Generate 64 hex characters (32 bytes of entropy)
        API_SECRET=$(openssl rand -hex 32)
        JWT_SECRET=$(openssl rand -hex 32)
    else
        # Fallback to /dev/urandom if openssl unavailable
        # Generate 64 hex characters (32 bytes of entropy)
        API_SECRET=$(head -c 32 /dev/urandom | xxd -p -c 32)
        JWT_SECRET=$(head -c 32 /dev/urandom | xxd -p -c 32)
    fi
    
    cat > .env << EOF
# AtroZ Dashboard Configuration
ATROZ_API_PORT=$PORT
ATROZ_API_SECRET=$API_SECRET
ATROZ_JWT_SECRET=$JWT_SECRET
# ... rest of config
EOF
    
    print_warn "Generated new secrets - save these in a secure location!"
    print_warn "For production, use proper secret management (AWS Secrets Manager, HashiCorp Vault, etc.)"
fi
```

---

### ISSUE 3: Environment variable injection vulnerability
**File:** `atroz_quickstart.sh` line 136

**Current (VULNERABLE):**
```bash
export $(cat .env | grep -v '^#' | xargs)
```

**Why vulnerable:** Command injection possible via specially crafted .env values

**Fix:**
```bash
# Safe method to load environment
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi
```

---

### ISSUE 4: Unpinned dependencies in pyproject.toml
**File:** `pyproject.toml` lines 11-15

**Current (RISKY):**
```toml
dependencies = [
    "hypothesis>=6",
    "pandas>=2",
    "pydantic>=1.10",
    "scipy>=1.11",
    "scikit-learn>=1.4",
]
```

**Fix:**
```toml
dependencies = [
    "hypothesis>=6.0.0,<7.0.0",
    "pandas>=2.0.0,<3.0.0",
    "pydantic>=1.10.0,<2.0.0",
    "scipy>=1.11.0,<2.0.0",
    "scikit-learn>=1.4.0,<2.0.0",
]
```

---

### ISSUE 5: Outdated GitHub Actions versions
**Files:** `d2_concurrence.yml`, `strategic-wiring.yml`

**Current:**
```yaml
uses: actions/checkout@v3
uses: actions/upload-artifact@v3
```

**Fix:**
```yaml
uses: actions/checkout@v4
uses: actions/upload-artifact@v4
```

**Script to fix all:**
```bash
# Update specific GitHub Actions to v4 (be precise to avoid unintended changes)
find .github/workflows -name "*.yml" -exec sed -i 's/actions\/checkout@v3/actions\/checkout@v4/g' {} \;
find .github/workflows -name "*.yml" -exec sed -i 's/actions\/upload-artifact@v3/actions\/upload-artifact@v4/g' {} \;
find .github/workflows -name "*.yml" -exec sed -i 's/actions\/download-artifact@v3/actions\/download-artifact@v4/g' {} \;
find .github/workflows -name "*.yml" -exec sed -i 's/actions\/setup-python@v3/actions\/setup-python@v5/g' {} \;
find .github/workflows -name "*.yml" -exec sed -i 's/actions\/setup-python@v4/actions\/setup-python@v5/g' {} \;
find .github/workflows -name "*.yml" -exec sed -i 's/actions\/cache@v2/actions\/cache@v3/g' {} \;
```

---

### ISSUE 6: Dangerous continue-on-error usage
**Files:** Multiple workflows

**Current:**
```yaml
- name: Run validation
  run: python validate.py
  continue-on-error: true
```

**Fix - Option 1 (Preferred):** Remove continue-on-error, let it fail
```yaml
- name: Run validation
  run: python validate.py
```

**Fix - Option 2 (If failures are truly optional):** Document why
```yaml
- name: Run validation (non-blocking during transition period)
  run: python validate.py
  continue-on-error: true
  # TODO: Make this blocking after 2025-12-01
```

---

### ISSUE 7: Missing .importlinter config file
**File:** `.github/workflows/governance-pipeline.yml` creates it on-the-fly

**Fix:** Create `.importlinter` in repository root:
```ini
[importlinter]
root_package = saaaaaa
include_external_packages = False

[importlinter:contract:1]
name = Core modules should not depend on API
type = forbidden
source_modules =
    saaaaaa.core
forbidden_modules =
    saaaaaa.api

[importlinter:contract:2]
name = Core modules should not depend on infrastructure
type = forbidden
source_modules =
    saaaaaa.core
forbidden_modules =
    saaaaaa.infrastructure

[importlinter:contract:3]
name = Processing depends only on core
type = layers
layers =
    saaaaaa.core
    saaaaaa.processing

[importlinter:contract:4]
name = Analysis depends on processing and core
type = layers
layers =
    saaaaaa.core
    saaaaaa.processing
    saaaaaa.analysis
```

Then remove lines 80-118 from `governance-pipeline.yml`

---

### ISSUE 8: Python version inconsistency
**Files:** Multiple workflows and pyproject.toml

**Decision needed:** Choose one version (3.10 or 3.11)

**Recommended:** Python 3.10 (most common in codebase)

**Fix all workflows:**
```bash
find .github/workflows -name "*.yml" -exec sed -i "s/python-version: '3.11'/python-version: '3.10'/g" {} \;
```

**Fix pyproject.toml:**
```toml
[tool.pyright]
pythonVersion = "3.10"

[tool.mypy]
python_version = "3.10"
```

---

## HIGH PRIORITY FIXES

### ISSUE 9: Missing timeouts on all jobs

**Add to every job:**
```yaml
jobs:
  my-job:
    runs-on: ubuntu-latest
    timeout-minutes: 20  # Adjust per job
```

**Recommended timeouts:**
- Linting jobs: 10 minutes
- Type checking: 15 minutes
- Test jobs: 20 minutes
- Full validation: 30 minutes
- Complete pipeline: 45 minutes

**Script to add timeouts:**
```python
#!/usr/bin/env python3
import yaml
import sys
from pathlib import Path

TIMEOUTS = {
    'boundary-enforcement.yml': 20,
    'd2_concurrence.yml': 20,
    'data-contracts.yml': 15,
    'data_contract_validation.yml': 20,
    'governance-pipeline.yml': 45,
    'static-analysis.yml': 15,
    'strategic-wiring.yml': 20,
    'type-safety.yml': 25,
}

for workflow_file, timeout in TIMEOUTS.items():
    path = Path(f'.github/workflows/{workflow_file}')
    if not path.exists():
        continue
    
    with open(path) as f:
        data = yaml.safe_load(f)
    
    for job_name, job_config in data.get('jobs', {}).items():
        if 'timeout-minutes' not in job_config:
            job_config['timeout-minutes'] = timeout
    
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f'✓ Added timeouts to {workflow_file}')
```

---

### ISSUE 10: Missing pip caching

**Add to every workflow after Python setup:**
```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.10'
    cache: 'pip'  # Add this line
    cache-dependency-path: |
      requirements.txt
      requirements_atroz.txt
```

**Alternative with actions/cache:**
```yaml
- name: Cache pip packages
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt', '**/pyproject.toml') }}
    restore-keys: |
      ${{ runner.os }}-pip-

- name: Install dependencies
  run: pip install -r requirements.txt
```

---

### ISSUE 11: Inline scripts should be external files

**Example:** `.github/workflows/type-safety.yml` lines 59-89

**Current:**
```yaml
- name: Verify no **kwargs in public APIs
  run: |
    python -c "
    import re
    # ... 30 lines of Python ...
    "
```

**Fix:** Create `scripts/check_kwargs.py`:
```python
#!/usr/bin/env python3
"""Check for **kwargs usage in public APIs."""
import re
import sys
from pathlib import Path

def check_kwargs(files):
    issues = []
    for file in files:
        try:
            with open(file) as f:
                content = f.read()
                matches = re.findall(
                    r'^\s*def\s+([a-z_][a-z0-9_]*)\s*\([^)]*\*\*kwargs',
                    content,
                    re.MULTILINE
                )
                if matches:
                    issues.append(f'{file}: {matches}')
        except FileNotFoundError:
            pass
    
    if issues:
        print('WARNING: Found **kwargs in public APIs:')
        for issue in issues:
            print(f'  - {issue}')
        return 1
    else:
        print('✓ No **kwargs found in core public APIs')
        return 0

if __name__ == '__main__':
    files = [
        'contracts.py',
        'orchestrator.py',
        'document_ingestion.py',
        'embedding_policy.py',
    ]
    sys.exit(check_kwargs(files))
```

**Updated workflow:**
```yaml
- name: Verify no **kwargs in public APIs
  run: python scripts/check_kwargs.py
```

---

### ISSUE 12: Add secret scanning workflow

**Create:** `.github/workflows/security-scan.yml`
```yaml
name: Security Scanning

permissions:
  contents: read

on:
  push:
    branches: [main, develop]
  pull_request:
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  secret-scan:
    name: Scan for Secrets
    runs-on: ubuntu-latest
    timeout-minutes: 10
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for secret scanning
      
      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Upload scan results
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: gitleaks-report
          path: gitleaks-report.json

  dependency-scan:
    name: Scan Dependencies
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install safety pip-audit
      
      - name: Run Safety check
        run: safety check --json --output safety-report.json
        continue-on-error: true
      
      - name: Run pip-audit
        run: pip-audit --format json --output pip-audit-report.json
        continue-on-error: true
      
      - name: Upload scan results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: dependency-scan-results
          path: |
            safety-report.json
            pip-audit-report.json
```

---

### ISSUE 13: Enable Dependabot

**Create:** `.github/dependabot.yml`
```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 10
    reviewers:
      - "maintainer-team"
    labels:
      - "dependencies"
      - "python"
    commit-message:
      prefix: "deps"
      include: "scope"
    
  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 5
    reviewers:
      - "maintainer-team"
    labels:
      - "dependencies"
      - "github-actions"
    commit-message:
      prefix: "ci"
```

---

### ISSUE 14: Port conflict detection

**File:** `atroz_quickstart.sh`

**Add before starting services:**
```bash
# Check if ports are available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

print_info "Checking port availability..."
if ! check_port $PORT; then
    print_error "Port $PORT is already in use"
    print_info "Please set ATROZ_API_PORT to a different port or stop the service using port $PORT"
    exit 1
fi

if ! check_port $STATIC_PORT; then
    print_error "Port $STATIC_PORT is already in use"
    print_info "Please set ATROZ_STATIC_PORT to a different port or stop the service using port $STATIC_PORT"
    exit 1
fi
print_info "✓ Ports $PORT and $STATIC_PORT are available"
```

---

### ISSUE 15: Split large workflows into smaller jobs

**Example:** `governance-pipeline.yml` has 9 steps in one job

**Current structure:**
```yaml
jobs:
  governance-checks:  # One big job with 9 steps
```

**Better structure:**
```yaml
jobs:
  compile:
    name: Compile Python Files
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      # ... compile step only
  
  ast-scan:
    name: AST Scanner
    runs-on: ubuntu-latest
    timeout-minutes: 10
    needs: compile
    steps:
      # ... AST scan only
  
  import-linter:
    name: Import Contracts
    runs-on: ubuntu-latest
    timeout-minutes: 10
    needs: compile
    steps:
      # ... import linter only
  
  # ... etc for each validation
  
  summary:
    name: Validation Summary
    runs-on: ubuntu-latest
    needs: [compile, ast-scan, import-linter, ...]
    if: always()
    steps:
      - name: Generate summary
        run: echo "All validations complete"
```

**Benefits:**
- Parallel execution (faster)
- Better failure isolation
- Clearer logs
- Easier to rerun specific checks

---

### ISSUE 16: Fix temp file race condition

**File:** `scripts/validate_contracts_local.sh` line 33

**Current (VULNERABLE):**
```bash
eval "$command" > /tmp/check_output.log 2>&1
```

**Fix:**
```bash
# At top of script
TEMP_LOG=$(mktemp)
trap "rm -f $TEMP_LOG" EXIT

# In run_check function
eval "$command" > "$TEMP_LOG" 2>&1
```

---

## MEDIUM PRIORITY FIXES

### ISSUE 17: Add workflow documentation

**Create:** `.github/workflows/README.md`
```markdown
# GitHub Workflows Documentation

## Overview
This directory contains all CI/CD workflows for the SAAAAAA project.

## Workflows

### Core Validation Workflows

#### boundary-enforcement.yml
- **Purpose**: Enforce core module boundaries, prevent __main__ blocks
- **Triggers**: PR, push to main/develop/copilot/**
- **Duration**: ~5 minutes
- **Key checks**:
  - No __main__ blocks in core modules
  - AST boundary scanning
  - Contract definitions validation

#### governance-pipeline.yml
- **Purpose**: Complete governance validation pipeline
- **Triggers**: PR, push, manual
- **Duration**: ~15 minutes
- **Key checks**:
  1. Compile all Python files
  2. AST scanner (anti-I/O, anti-__main__)
  3. Import-linter (layer contracts)
  4. Ruff (lint/bugs)
  5. Mypy --strict
  6. Pycycle (circular dependencies)
  7. Bulk import test
  8. Pytest
  9. Coverage ≥80%

[... continue for each workflow ...]

## Troubleshooting

### Workflow fails with "command not found"
Check that all required tools are installed in setup step.

### Workflow times out
Check if timeout-minutes is set appropriately for the job.

[... etc ...]
```

---

### ISSUE 18: Add configuration schemas

**Create:** `config/schemas/execution_mapping.schema.json`
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Execution Mapping Configuration",
  "type": "object",
  "required": ["version", "metadata", "modules", "dimensions"],
  "properties": {
    "version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+\\.\\d+$"
    },
    "metadata": {
      "type": "object",
      "required": ["generated", "standard", "policy"],
      "properties": {
        "generated": {"type": "string", "format": "date"},
        "standard": {"type": "string"},
        "policy": {"type": "string"},
        "hermetic": {"type": "boolean"}
      }
    },
    "modules": {
      "type": "object",
      "patternProperties": {
        "^[a-z_]+$": {
          "type": "object",
          "required": ["file", "class", "methods", "provides"],
          "properties": {
            "file": {"type": "string", "pattern": "\\.py$"},
            "class": {"type": "string"},
            "methods": {
              "type": "array",
              "items": {"type": "string"}
            },
            "provides": {
              "type": "array",
              "items": {"type": "string"}
            }
          }
        }
      }
    }
  }
}
```

**Add validation workflow step:**
```yaml
- name: Validate execution mapping
  run: |
    python -m jsonschema -i config/execution_mapping.yaml config/schemas/execution_mapping.schema.json
```

---

### ISSUE 19: Standardize error handling

**Create policy document:** `docs/ERROR_HANDLING_POLICY.md`
```markdown
# Error Handling Policy for CI/CD

## General Principles
1. Fail fast: Don't continue if critical step fails
2. Explicit is better than implicit: Document why failures are allowed
3. Consistent patterns: Use same approach across all workflows

## Patterns

### Hard Failure (Default)
Use when step is critical:
```yaml
- name: Critical validation
  run: python validate.py
```

### Soft Failure (Documented)
Use only when truly optional, with comment explaining why:
```yaml
- name: Optional check (will be required after 2025-12-01)
  run: python optional_check.py
  continue-on-error: true
  # TODO: Remove continue-on-error after transition period
```

### Conditional Failure
Use when failure should depend on context:
```yaml
- name: Check with threshold
  id: check
  run: python check.py --threshold 80
  continue-on-error: true

- name: Evaluate result
  if: steps.check.outcome == 'failure'
  run: |
    echo "Check failed, but continuing for now"
    # Log to monitoring system
```

## Anti-Patterns
❌ **DON'T**: `command || echo "warning"`
❌ **DON'T**: `command || true`
❌ **DON'T**: Redirect errors to /dev/null
❌ **DON'T**: Use `continue-on-error` without comment
```

---

## ADDITIONAL ENHANCEMENTS

### Enhancement 1: Pre-commit hook improvements

**Update:** `.pre-commit-config.yaml`
```yaml
repos:
  # Standard hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key

  # Secret detection
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

  # Shell script validation
  - repo: https://github.com/shellcheck-py/shellcheck-py
    rev: v0.9.0.6
    hooks:
      - id: shellcheck

  # YAML validation
  - repo: https://github.com/adrienverge/yamllint
    rev: v1.33.0
    hooks:
      - id: yamllint
        args: ['-d', '{extends: default, rules: {line-length: {max: 120}}}']

  # Existing hooks...
  - repo: local
    hooks:
      # ... existing local hooks

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.0  # Updated version
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0  # Updated version
    hooks:
      - id: mypy
        args: [--strict, --config-file=pyproject.toml]
        additional_dependencies: [types-all]
```

---

### Enhancement 2: Makefile improvements

**Update:** `Makefile`
```makefile
.PHONY: help verify clean install test lint format type-check

# Configuration
PYTHON := python3
PYTEST := pytest
MYPY := mypy
RUFF := ruff
COVERAGE := coverage

# Directories
SRC_DIR := src
TEST_DIR := tests
CORE_DIRS := core orchestrator executors

# Help target
help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation
install:  ## Install dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev:  ## Install development dependencies
	$(MAKE) install
	$(PYTHON) -m pip install pytest mypy ruff coverage pre-commit

# Cleaning
clean:  ## Clean build artifacts and cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/

# Testing
test:  ## Run tests
	$(PYTEST) $(TEST_DIR) -v

test-coverage:  ## Run tests with coverage
	$(COVERAGE) run -m pytest $(TEST_DIR)
	$(COVERAGE) report -m
	$(COVERAGE) html

# Linting and formatting
lint:  ## Run linter
	$(RUFF) check $(SRC_DIR)

format:  ## Format code
	$(RUFF) format $(SRC_DIR)

format-check:  ## Check code formatting
	$(RUFF) format --check $(SRC_DIR)

# Type checking
type-check:  ## Run type checker
	$(MYPY) $(SRC_DIR) --strict

# Comprehensive verification
verify: clean  ## Run all verifications
	@echo "=== Compiling Python files ==="
	$(PYTHON) -m compileall -q $(CORE_DIRS)
	
	@echo "=== Scanning core purity ==="
	$(PYTHON) tools/scan_core_purity.py
	
	@echo "=== Checking import contracts ==="
	lint-imports --config contracts/importlinter.ini
	
	@echo "=== Running linter ==="
	$(MAKE) lint
	
	@echo "=== Type checking ==="
	$(MAKE) type-check
	
	@echo "=== Checking circular dependencies ==="
	pycycle $(CORE_DIRS)
	
	@echo "=== Testing imports ==="
	$(PYTHON) tools/import_all.py
	
	@echo "=== Running tests ==="
	$(MAKE) test
	
	@echo "=== Checking coverage ==="
	$(MAKE) test-coverage
	
	@echo "✓ All verifications passed!"

# CI target
ci: verify  ## Run CI verification (alias for verify)

.DEFAULT_GOAL := help
```

---

### Enhancement 3: Add health checks

**Create:** `scripts/health_check.sh`
```bash
#!/bin/bash
# Health check script for services

set -euo pipefail

check_service() {
    local name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    echo -n "Checking $name..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo " ✓ healthy"
            return 0
        fi
        
        sleep 1
        attempt=$((attempt + 1))
    done
    
    echo " ✗ unhealthy after ${max_attempts}s"
    return 1
}

# Check API server
check_service "API Server" "http://localhost:${ATROZ_API_PORT:-5000}/api/v1/health"

# Check static server
check_service "Static Server" "http://localhost:${ATROZ_STATIC_PORT:-8000}/"

echo "All services healthy!"
```

**Use in workflow:**
```yaml
- name: Start services
  run: ./atroz_quickstart.sh dev &

- name: Wait for services
  run: ./scripts/health_check.sh
```

---

## IMPLEMENTATION CHECKLIST

Use this checklist to track implementation of fixes:

### Critical Security Fixes
- [ ] Add permissions blocks to all workflows
- [ ] Fix hardcoded secrets in atroz_quickstart.sh
- [ ] Fix env injection vulnerability
- [ ] Pin dependencies in pyproject.toml
- [ ] Update all GitHub Actions to v4
- [ ] Remove dangerous continue-on-error usage
- [ ] Add .importlinter config to repo
- [ ] Standardize Python version

### High Priority Fixes
- [ ] Add timeouts to all workflow jobs
- [ ] Implement pip caching
- [ ] Extract inline scripts to files
- [ ] Add secret scanning workflow
- [ ] Enable Dependabot
- [ ] Add port conflict detection
- [ ] Split large workflows
- [ ] Fix temp file race condition

### Medium Priority Fixes
- [ ] Add workflow documentation
- [ ] Add configuration schemas
- [ ] Standardize error handling
- [ ] Add health checks
- [ ] Improve Makefile
- [ ] Update pre-commit hooks
- [ ] Add monitoring and metrics

### Testing
- [ ] Test all workflow changes
- [ ] Test shell script fixes
- [ ] Verify no breaking changes
- [ ] Update documentation

---

## TESTING YOUR FIXES

### Test Workflows Locally
```bash
# Install act (GitHub Actions local runner)
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Test a workflow
act -W .github/workflows/boundary-enforcement.yml

# Test with specific event
act pull_request -W .github/workflows/governance-pipeline.yml
```

### Test Shell Scripts
```bash
# Install shellcheck
sudo apt-get install shellcheck  # Linux
brew install shellcheck  # macOS

# Check scripts
shellcheck atroz_quickstart.sh
shellcheck scripts/*.sh

# Test scripts in dry-run mode (add this option to scripts)
./atroz_quickstart.sh --dry-run
```

### Test Configuration Changes
```bash
# Validate YAML
yamllint .github/workflows/*.yml
yamllint config/*.yaml

# Validate JSON schemas
python -m jsonschema -i config/execution_mapping.yaml schemas/execution_mapping.schema.json
```

---

## ROLLOUT STRATEGY

### Phase 1: Critical Security (Week 1)
1. Create feature branch
2. Apply all critical fixes
3. Test thoroughly
4. Get security review
5. Merge to main

### Phase 2: High Priority (Week 2-3)
1. Apply high priority fixes incrementally
2. Test each change
3. Monitor CI performance
4. Adjust timeouts if needed

### Phase 3: Medium Priority (Week 4-6)
1. Add documentation
2. Implement schemas
3. Standardize patterns
4. Add monitoring

### Phase 4: Continuous Improvement (Ongoing)
1. Monitor metrics
2. Optimize performance
3. Update as needed
