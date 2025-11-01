# Post-Reorganization Steps

After the repository reorganization, some manual steps are needed to update imports and references.

## Automatic Import Updates

A script has been provided to automatically update import statements in your code:

```bash
# Test what would be changed (dry run)
python scripts/update_imports.py --dry-run tests examples scripts

# Actually update the files
python scripts/update_imports.py tests examples scripts
```

This script will:
- Update all import statements to use the new `saaaaaa.*` package structure
- Update file path references (e.g., `inventory.json` → `config/inventory.json`)
- Create backup files (`.bak`) before making changes

## Manual Steps Required

### 1. Install Package in Development Mode

After reorganization, install the package:

```bash
pip install -e .
```

This makes the `saaaaaa` package importable throughout your code.

### 2. Update Custom Scripts

If you have custom scripts not in the repository, update their imports:

**Before:**
```python
from aggregation import DataAggregator
from orchestrator.core import Orchestrator
```

**After:**
```python
from saaaaaa.processing.aggregation import DataAggregator
from saaaaaa.core.orchestrator.core import Orchestrator
```

### 3. Update Configuration Files

If you have external configuration files that reference file paths, update them:
- `inventory.json` → `config/inventory.json`
- `questionnaire_monolith.json` → `data/questionnaire_monolith.json`
- `schemas/` → `config/schemas/`

### 4. Update IDE/Editor Settings

Update your IDE project settings:
- **Source root:** Set to `src/`
- **Python path:** Should include `src/`
- **Test discovery:** Update to point to `tests/` directory

### 5. Update CI/CD Pipelines

If you have CI/CD configurations, update them:
- Install command: `pip install -e .`
- Test paths: `tests/`
- Coverage paths: `src/saaaaaa/`

## Verification Steps

### Test Imports
```python
# In Python REPL
from saaaaaa.core import ORCHESTRATOR_MONILITH
from saaaaaa.processing import document_ingestion
from saaaaaa.analysis import bayesian_multilevel_system
print("✓ All imports successful!")
```

### Run Tests
```bash
# Run all tests
pytest tests/

# Or run specific test
pytest tests/test_orchestrator_integration.py
```

### Check Coverage
```bash
pytest --cov=src/saaaaaa tests/
```

## Common Issues

### Issue: ModuleNotFoundError: No module named 'saaaaaa'
**Solution:** Run `pip install -e .` from the repository root

### Issue: Old imports still failing
**Solution:** Run `python scripts/update_imports.py tests examples scripts` without `--dry-run`

### Issue: FileNotFoundError for config/data files
**Solution:** Update file paths in your code:
```python
# Old
config_path = 'inventory.json'

# New
config_path = 'config/inventory.json'
```

### Issue: Tests can't find modules
**Solution:** Ensure you've installed the package with `pip install -e .`

## Rollback Instructions

If you need to rollback (not recommended):
```bash
git log  # Find the commit hash before reorganization
git revert <commit-hash>
```

## Benefits Realized

After completing these steps, you'll enjoy:

1. ✅ **Standard Python package structure** - follows PEP 420
2. ✅ **Better IDE support** - autocomplete and navigation work better
3. ✅ **Clear organization** - modules grouped by functionality
4. ✅ **Easier maintenance** - logical structure for adding new code
5. ✅ **Professional appearance** - matches industry standards
6. ✅ **Better testing** - clear separation of code and tests
7. ✅ **Improved documentation** - centralized in `docs/`

## Need Help?

See these documents:
- `docs/REPOSITORY_STRUCTURE.md` - Detailed structure documentation
- `docs/MIGRATION_GUIDE.md` - Complete import migration guide
- `README.md` - Updated project README

## Progress Checklist

Track your migration progress:

- [ ] Run `pip install -e .`
- [ ] Run `python scripts/update_imports.py --dry-run tests examples scripts`
- [ ] Review proposed changes
- [ ] Run `python scripts/update_imports.py tests examples scripts` (without --dry-run)
- [ ] Test imports in Python REPL
- [ ] Run test suite: `pytest tests/`
- [ ] Update any custom scripts or tools
- [ ] Update CI/CD configuration
- [ ] Update IDE/editor settings
- [ ] Verify all functionality works
- [ ] Remove `.bak` backup files once satisfied

Once complete, the reorganization is finished and you're ready to work with the new structure!
