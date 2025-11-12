# QUESTIONNAIRE INTEGRITY PROTOCOL

## Executive Summary

This document defines the **QUESTIONNAIRE DETERMINISM ENFORCEMENT PROTOCOL** that ensures 100% verifiable, immutable, and hash-verified access to the questionnaire monolith across the entire SAAAAAA system.

**Status**: ✅ ENFORCED  
**Authority**: SIN_CARRETA Compliance Directive  
**Version**: 1.0.0  
**Last Updated**: 2025-11-12

---

## The Law: Five Non-Negotiable Rules

### Rule 1: Single Load Point
**`factory.load_questionnaire()` is the ONLY way to load questionnaire data.**

```python
# ✅ CORRECT
from saaaaaa.core.orchestrator.factory import load_questionnaire
q = load_questionnaire()

# ❌ FORBIDDEN
with open('data/questionnaire_monolith.json') as f:
    q = json.load(f)
```

### Rule 2: Immutable Data Structures
**All questionnaire data uses `MappingProxyType` or `tuple` - no mutable containers.**

```python
# ✅ Data is immutable
q.data  # MappingProxyType[str, Any]
q.micro_questions  # tuple[MappingProxyType, ...]

# ❌ These will raise TypeError
q.data['new_key'] = 'value'
q.micro_questions[0]['field'] = 'value'
```

### Rule 3: Hash Verification
**Every load verifies SHA256 == EXPECTED_QUESTIONNAIRE_HASH**

Current Expected Hash:
```
f4a48932f6a3c408e65589680de334d54e69d4a43adb787bb91571788a91feb8
```

If questionnaire changes legitimately, update `EXPECTED_QUESTIONNAIRE_HASH` in:
```
src/saaaaaa/core/orchestrator/factory.py
```

### Rule 4: Structure Validation
**300 questions with exact schema or FAIL**

```python
assert q.question_count == 300
assert q.version == "1.0.0"
assert all(isinstance(mq, MappingProxyType) for mq in q.micro_questions)
```

### Rule 5: No Direct File Access
**`data/questionnaire_monolith.json` is NEVER read directly outside factory.py**

CI will fail if any Python file (except factory.py) contains:
- `questionnaire_monolith.json` in code (not comments/docstrings)
- Direct `json.load()` of questionnaire file
- Direct `Path().read_text()` of questionnaire file

---

## Architecture

### CanonicalQuestionnaire Dataclass

```python
@dataclass(frozen=True)
class CanonicalQuestionnaire:
    """Immutable, validated questionnaire with hash verification."""
    
    data: MappingProxyType[str, Any]           # Complete questionnaire data
    sha256: str                                 # SHA256 of raw file
    micro_questions: tuple[MappingProxyType, ...] # All 300 questions
    question_count: int                         # Always 300
    version: str                                # e.g., "1.0.0"
    schema_version: str                         # e.g., "1.1.0"
    
    def __post_init__(self) -> None:
        """Validates count, hash, and immutability on construction."""
```

### Loading Flow

```
┌─────────────────────────────────────────────────────────┐
│                   Application Code                       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  load_questionnaire()  │
         │   (factory.py)         │
         └────────────┬───────────┘
                      │
                      ├─► Read raw file bytes
                      ├─► Compute SHA256 hash
                      ├─► Parse JSON with OrderedDict
                      ├─► validate_questionnaire_structure()
                      ├─► _deep_freeze() for immutability
                      └─► Return CanonicalQuestionnaire
                      
         ┌────────────────────────┐
         │ CanonicalQuestionnaire │
         │   (frozen dataclass)   │
         └────────────────────────┘
                      │
                      ├─► Verify question_count == 300
                      ├─► Verify sha256 == EXPECTED_HASH
                      ├─► Verify all data is MappingProxyType
                      └─► Log initialization
```

---

## Usage Guide

### Basic Usage

```python
from saaaaaa.core.orchestrator.factory import load_questionnaire, CanonicalQuestionnaire

# Load questionnaire
q: CanonicalQuestionnaire = load_questionnaire()

# Access data (immutable)
version = q.version
question_count = q.question_count
all_questions = q.micro_questions

# Access nested data
dimensions = q.data['blocks'].get('dimensions', [])
policy_areas = q.data['blocks'].get('policy_areas', [])

# Access specific question
first_question = q.micro_questions[0]
question_id = first_question['question_id']
question_text = first_question.get('question_text', '')
```

### Using with Provider (Recommended)

```python
from saaaaaa.core.orchestrator import get_questionnaire_provider

provider = get_questionnaire_provider()

# Auto-loads canonical questionnaire if not already loaded
canonical_q = provider.get_canonical()

# Or set data explicitly
from saaaaaa.core.orchestrator.factory import load_questionnaire
provider.set_data(load_questionnaire())
```

### Legacy Compatibility

```python
# DEPRECATED: Returns mutable dict for backward compatibility
from saaaaaa.core.orchestrator.factory import load_questionnaire_monolith

data = load_questionnaire_monolith()  # Logs deprecation warning
# This internally calls load_questionnaire() and converts to dict
```

---

## Validation Rules

### Question Structure Validation

Every question must have:
- `question_id`: str (format: "Q001" to "Q300")
- `question_global`: int (range: 1-300, unique)
- `base_slot`: str (format: "D[1-6]-Q[1-5]")
- No duplicates in question_id or question_global

### Top-Level Structure

Required keys:
- `version`: str
- `schema_version`: str
- `blocks`: dict
  - `micro_questions`: list[dict] (length: 300)
  - `meso_questions`: list[dict] (optional)
  - `macro_question`: dict (optional)

---

## CI/CD Enforcement

### GitHub Actions Workflow

File: `.github/workflows/questionnaire-integrity.yml`

Four verification jobs run on every push/PR:

1. **verify-questionnaire-hash**: Compares file hash with expected hash
2. **verify-import-discipline**: Scans for direct file access violations
3. **test-canonical-loader**: Tests load_questionnaire() functionality
4. **verify-immutability**: Tests MappingProxyType enforcement

### Manual Verification

```bash
# Verify hash
sha256sum data/questionnaire_monolith.json
# Should output: f4a48932f6a3c408e65589680de334d54e69d4a43adb787bb91571788a91feb8

# Check for violations
grep -r "questionnaire_monolith.json" src/ --include="*.py" \
  | grep -v "src/saaaaaa/core/orchestrator/factory.py"
# Should output nothing

# Test canonical loader
python -c "
from saaaaaa.core.orchestrator.factory import load_questionnaire
q = load_questionnaire()
assert q.question_count == 300
assert q.sha256 == 'f4a48932f6a3c408e65589680de334d54e69d4a43adb787bb91571788a91feb8'
print('✅ QUESTIONNAIRE_VALIDATED')
"
```

---

## Changing the Questionnaire

### Legitimate Changes

If you need to modify `data/questionnaire_monolith.json`:

1. **Make your changes** to the JSON file
2. **Compute new hash**:
   ```bash
   sha256sum data/questionnaire_monolith.json
   ```
3. **Update expected hash** in `src/saaaaaa/core/orchestrator/factory.py`:
   ```python
   EXPECTED_QUESTIONNAIRE_HASH: Final[str] = "NEW_HASH_HERE"
   ```
4. **Update question count** if changed:
   ```python
   EXPECTED_QUESTION_COUNT_CANONICAL: Final[int] = NEW_COUNT
   ```
5. **Commit both files together**:
   ```bash
   git add data/questionnaire_monolith.json
   git add src/saaaaaa/core/orchestrator/factory.py
   git commit -m "feat: Update questionnaire to version X.Y.Z"
   ```

### CI Will Verify

- Hash matches between file and constant
- No direct access violations
- All validation rules pass
- Immutability is preserved

---

## Violation Detection

### Build Failures

**If CI fails with "QUESTIONNAIRE HASH MISMATCH":**
- Someone modified the file without updating the hash
- Either revert the file or update the hash following "Changing the Questionnaire"

**If CI fails with "DIRECT QUESTIONNAIRE ACCESS DETECTED":**
- A file is accessing questionnaire_monolith.json directly
- Refactor to use `load_questionnaire()` instead

**If CI fails with "Question count mismatch":**
- The file has wrong number of questions
- Either fix the file or update `EXPECTED_QUESTION_COUNT_CANONICAL`

---

## Migration Guide

### For Existing Code Using Dict

**Before:**
```python
from saaaaaa.core.orchestrator.factory import load_questionnaire_monolith

data = load_questionnaire_monolith()  # Returns mutable dict
questions = data['blocks']['micro_questions']
questions[0]['new_field'] = 'value'  # Works but dangerous
```

**After:**
```python
from saaaaaa.core.orchestrator.factory import load_questionnaire, CanonicalQuestionnaire

q: CanonicalQuestionnaire = load_questionnaire()
questions = q.micro_questions  # Immutable tuple
# questions[0]['new_field'] = 'value'  # TypeError - can't modify

# If you need to work with the data, access it immutably:
question_data = dict(questions[0])  # Make a mutable copy if needed
```

### For Code Using Provider

**Before:**
```python
provider = get_questionnaire_provider()
data = provider.get_data()  # Returns dict or None
```

**After:**
```python
provider = get_questionnaire_provider()
q = provider.get_canonical()  # Returns CanonicalQuestionnaire, auto-loads if needed
```

---

## Security Considerations

### Why Hash Verification?

1. **Tamper Detection**: Any unauthorized modification is immediately detected
2. **Reproducibility**: Same file always produces same hash
3. **CI Enforcement**: Automated verification prevents silent corruption
4. **Audit Trail**: Git history shows exactly when/why hash changed

### Why Immutability?

1. **Thread Safety**: No race conditions from concurrent modifications
2. **Predictability**: Data can't change during execution
3. **Debugging**: Easier to reason about program state
4. **Correctness**: Prevents accidental mutations

### Why Single Load Point?

1. **Centralized Validation**: All loads go through same validation
2. **Consistency**: Same data structure everywhere
3. **Observability**: Single place to log/monitor
4. **Maintainability**: Changes only need to happen in one place

---

## Troubleshooting

### "cannot import name 'CanonicalQuestionnaire'"

**Solution**: Update your import:
```python
from saaaaaa.core.orchestrator.factory import CanonicalQuestionnaire
```

### "TypeError: 'mappingproxy' object does not support item assignment"

**Expected**: This means immutability is working correctly.

**Solution**: Don't modify questionnaire data. If you need a mutable copy:
```python
mutable_copy = dict(q.data)
```

### "Questionnaire hash mismatch" warning in logs

**Cause**: File was modified without updating expected hash.

**Solution**: Follow "Changing the Questionnaire" section above.

---

## Contact & Support

For questions about questionnaire integrity:
1. Review this document
2. Check CI logs for specific violations
3. Verify you're using `load_questionnaire()` not direct file access
4. Ensure you're not trying to modify immutable data

For architectural questions, consult:
- `src/saaaaaa/core/orchestrator/factory.py` - Implementation
- `.github/workflows/questionnaire-integrity.yml` - CI checks
- `SIN_CARRETA_COMPLIANCE.md` - Compliance directive

---

## Appendix: Constants Reference

### File Locations

```python
QUESTIONNAIRE_PATH = Path("data/questionnaire_monolith.json")
FACTORY_MODULE = "src/saaaaaa/core/orchestrator/factory.py"
CI_WORKFLOW = ".github/workflows/questionnaire-integrity.yml"
```

### Expected Values

```python
EXPECTED_QUESTIONNAIRE_HASH = "f4a48932f6a3c408e65589680de334d54e69d4a43adb787bb91571788a91feb8"
EXPECTED_QUESTION_COUNT_CANONICAL = 300
QUESTIONNAIRE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.1.0"
```

### Type Signatures

```python
load_questionnaire(path: Path | None = None) -> CanonicalQuestionnaire
load_questionnaire_monolith(path: Path | None = None) -> dict[str, Any]  # DEPRECATED
validate_questionnaire_structure(data: dict[str, object]) -> None
```

---

**END OF QUESTIONNAIRE INTEGRITY PROTOCOL**

*This document is authoritative. Any code violating these rules MUST be rejected in code review and WILL fail CI.*
