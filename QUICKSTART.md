# Quick Start Guide - Build Hygiene

## TL;DR - For Developers

This repository now follows strict build hygiene practices. Here's what you need to know:

## Setup (First Time)

```bash
# 1. Ensure you have Python 3.11.x
python --version  # Should show 3.11.x

# 2. Copy environment template
cp .env.example .env

# 3. Install in editable mode
pip install -e .

# 4. Verify setup
python3 tools/validation/validate_build_hygiene.py
```

## Key Rules

### ✓ Python Version
- **Use Python 3.11.x only** (specified in `.python-version`)
- Don't change the version without team discussion

### ✓ Dependencies
- **Never use wildcards** (`*`) or open ranges (`>=`, `~=`) in requirements.txt
- Always pin to exact versions: `package==1.2.3`
- Update both `requirements.txt` AND `constraints.txt`

### ✓ Running Code
```bash
# Good - use -m flag
python -m saaaaaa.core.module_name

# Good - after pip install -e .
from saaaaaa.core import module_name

# Bad - don't manipulate PYTHONPATH
export PYTHONPATH=/path/to/src  # ❌ NO!
```

### ✓ Configuration
- Configuration lives in `orchestrator/settings.py`
- **Only orchestrator code** should import settings
- Core modules should receive config as parameters

```python
# In orchestrator - OK ✓
from orchestrator.settings import settings

# In core modules - NO ✗
# Don't import settings here, use dependency injection
```

### ✓ Package Structure
- Every directory must have `__init__.py`
- Before creating new packages, ensure they follow the structure

## Directory Layout

```
SAAAAAA/
├── orchestrator/      # Orchestration (reads settings)
├── executors/         # Execution engines  
├── contracts/         # API contracts
├── core/             # (in src/saaaaaa/core/) Core logic
├── tests/            # Tests
├── tools/            # Utilities
└── examples/         # Examples
```

## Common Tasks

### Adding a New Dependency
```bash
# 1. Add to requirements.txt with exact version
echo "new-package==1.2.3" >> requirements.txt

# 2. Add to constraints.txt
echo "new-package==1.2.3" >> constraints.txt

# 3. Install
pip install -r requirements.txt -c constraints.txt

# 4. Validate
python3 tools/validation/validate_build_hygiene.py
```

### Creating a New Package
```bash
# 1. Create directory
mkdir -p new_package

# 2. Add __init__.py
touch new_package/__init__.py

# 3. Validate
python3 tools/validation/validate_build_hygiene.py
```

### Running Tests
```bash
# After pip install -e .
pytest tests/

# Specific test file
pytest tests/test_something.py

# With coverage
pytest --cov=src/saaaaaa tests/
```

## Validation

Check if your setup is correct:
```bash
python3 tools/validation/validate_build_hygiene.py
```

All checks should pass ✓

## Need Help?

- Read `BUILD_HYGIENE.md` for comprehensive guide
- Read `CHECKLIST_SUMMARY.md` for implementation details
- Run validation script to identify issues

## Pre-commit Checklist

Before committing:
- [ ] All dependencies pinned to exact versions
- [ ] New packages have `__init__.py`
- [ ] No PYTHONPATH hacks
- [ ] Config changes only in orchestrator/
- [ ] Validation script passes

## Questions?

**Q: Why Python 3.11 specifically?**
A: For consistency and to use specific language features. All type checkers are configured for 3.11.

**Q: Why pin exact versions?**
A: Reproducibility. Everyone gets the same dependencies, no surprises.

**Q: Can I use `pip install` directly?**
A: Use `pip install -r requirements.txt -c constraints.txt` to ensure constraints are applied.

**Q: Why not use Poetry/Pipenv?**
A: This project uses traditional requirements.txt + constraints.txt for maximum compatibility.
