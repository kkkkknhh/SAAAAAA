# GitHub Actions Workflows Documentation

This directory contains CI/CD workflows for the SAAAAAA project, implementing comprehensive quality gates, security scanning, and validation checks.

## Workflow Overview

### Security Workflows

#### `secret-scanning.yml`
**Purpose**: Automated credential detection and security scanning  
**Triggers**: Push to main/develop, pull requests, daily scheduled run  
**Key Features**:
- TruffleHog secret scanning with verified secrets only
- Pattern matching for hardcoded credentials
- AWS key detection
- Private key detection
- Configuration file validation

**Timeout**: 15 minutes

---

### Code Quality Workflows

#### `governance-pipeline.yml`
**Purpose**: CI/CD governance pipeline with 9-step validation  
**Triggers**: Push, pull requests, manual dispatch  
**Permissions**: Read-only access to repository contents  
**Key Steps**:
1. Compile - Validate Python syntax with `compileall`
2. AST Scanner - Check for `__main__` blocks and I/O in core modules
3. Import-linter - Enforce layer contracts
4. Ruff - Lint and bug detection
5. MyPy - Strict type checking
6. Pycycle - Circular dependency detection
7. Bulk import - Test all module imports
8. Pytest - Run test suite
9. Coverage - 80% threshold for orchestrator & contracts

**Timeout**: 30 minutes  
**Python Version**: 3.10  
**Caching**: pip dependencies

---

#### `type-safety.yml`
**Purpose**: Strict type checking and contract validation  
**Triggers**: Push, pull requests, manual dispatch  
**Permissions**: Read-only  
**Jobs**:

1. **type-check** (15 min)
   - Ruff linting
   - MyPy strict type checking
   - Pyright strict mode
   - **kwargs detection in public APIs
   
2. **contract-tests** (20 min)
   - Contract test execution
   - Property-based tests with Hypothesis
   - Contract test coverage verification
   
3. **lint-validation** (10 min)
   - typing.Any usage detection
   - Frozen dataclass verification
   
4. **validation-summary** (5 min)
   - Aggregate results summary

**Python Version**: 3.10  
**Tools Used**: `tools/type_safety_checks.py`

---

#### `static-analysis.yml`
**Purpose**: Static analysis with Pyright and MyPy  
**Triggers**: Push, pull requests  
**Timeout**: 15 minutes  
**Python Version**: 3.10  
**Tools**: Pyright (strict), MyPy

---

### Module-Specific Workflows

#### `boundary-enforcement.yml`
**Purpose**: Core module boundary enforcement  
**Triggers**: Push, pull requests  
**Permissions**: Read-only  
**Jobs**:

1. **boundary-check** (10 min)
   - Scan for `__main__` blocks in core modules
   - Enhanced boundary scanner on analysis modules
   - Verify contract definitions exist
   
2. **contract-runtime-tests** (10 min)
   - Runtime contract validation tests
   - Contract coverage verification
   
3. **boundary-enforcement-tests** (10 min)
   - Boundary enforcement tests
   - Semantic chunking tests

**Python Version**: 3.10

---

#### `d2_concurrence.yml`
**Purpose**: D2 Method Concurrence validation  
**Triggers**: Push/PR on specific paths (policy_processor.py, orchestrator, etc.)  
**Permissions**: Read-only  
**Jobs**:

1. **validate-d2-strict** (15 min)
   - Strict D2 validation
   - Report generation
   
2. **validate-d2-non-strict** (15 min)
   - 95% threshold validation
   - Non-strict mode report
   
3. **test-d2-orchestrator** (10 min)
   - D2 orchestrator tests

**Python Version**: 3.10

---

#### `strategic-wiring.yml`
**Purpose**: Strategic high-level wiring validation  
**Triggers**: Push, pull requests, manual dispatch  
**Timeout**: 15 minutes  
**Python Version**: 3.10  
**Configuration**: `config/strategic_files.txt`  
**Tools**: `tools/validate_strategic_files.py`  
**Validation Steps**:
- Python syntax validation
- Strategic wiring unit tests
- Integration validation
- Provenance tracking verification
- Documentation validation

---

### Data Contract Workflows

#### `data-contracts.yml`
**Purpose**: Data contract validation  
**Triggers**: Push to main, pull requests  
**Timeout**: 10 minutes  
**Python Version**: 3.10  
**Validations**:
- Schema validation
- Cross-reference integrity
- Questionnaire linting
- Rubric linting
- Scoring parity validation
- Deterministic artifact generation

---

#### `data_contract_validation.yml`
**Purpose**: Data contract and schema validation  
**Triggers**: Push, pull requests, manual dispatch  
**Permissions**: Read-only  
**Jobs**:

1. **validate-weights** (10 min) - Aggregation weight validation
2. **validate-schema** (10 min) - Monolith schema validation
3. **test-validation-models** (10 min) - Validation model tests
4. **test-recommendation-engine** (10 min) - Recommendation engine tests
5. **integration-check** (5 min) - Integration validation

**Python Version**: 3.10

---

## Configuration Files

### Strategic Files Configuration
- **Location**: `config/strategic_files.txt`
- **Purpose**: Centralized list of strategic files requiring validation
- **Format**: One file path per line, comments start with `#`

### Import Layer Contracts
- **Location**: `.importlinter`
- **Purpose**: Define module dependency contracts
- **Enforced Rules**:
  - Core modules cannot depend on API
  - Core modules cannot depend on infrastructure
  - Processing depends only on core
  - Analysis depends on processing and core

---

## Security Best Practices

All workflows implement the following security measures:

1. **Least Privilege**: `permissions: contents: read` by default
2. **Timeout Protection**: All jobs have `timeout-minutes` set
3. **Pinned Dependencies**: pyproject.toml uses version ranges with upper bounds
4. **Action Versions**: Using @v4/@v5 for GitHub actions (no @v3)
5. **Pip Caching**: Enabled for faster builds and reduced bandwidth
6. **Secret Scanning**: Automated credential detection
7. **No Dynamic Config**: .importlinter is version-controlled, not generated

---

## Performance Optimizations

1. **Pip Caching**: All Python setup steps use `cache: 'pip'`
2. **Timeouts**: Appropriate timeouts prevent hung jobs
3. **Conditional Steps**: Some validations skip if files don't exist
4. **Parallel Jobs**: Independent validations run in parallel

---

## Artifact Retention

- **Pipeline Logs**: 30 days
- **Coverage Reports**: 30 days
- **Security Reports**: 90 days
- **Validation Reports**: 30 days

---

## Python Version Policy

**Standard Version**: Python 3.10  
**Rationale**: Consistency across all workflows for reproducible builds  
**Configuration**: Set in `pyproject.toml` and all workflow files

---

## Tool Scripts

### `tools/type_safety_checks.py`
Consolidated type safety validation scripts:
- `kwargs` - Check for **kwargs in public APIs
- `coverage` - Verify contract test coverage
- `any` - Check for typing.Any usage
- `frozen` - Verify frozen dataclasses

### `tools/validate_strategic_files.py`
Strategic file validation helper:
- `syntax` - Validate Python syntax
- `provenance` - Check provenance tracking
- `all` - Run all validations

---

## Troubleshooting

### Common Issues

**Issue**: Job timeout  
**Solution**: Check the `timeout-minutes` setting and ensure it's appropriate for the job

**Issue**: Missing dependencies  
**Solution**: Verify requirements files exist and pip cache is working

**Issue**: Permission denied  
**Solution**: Check workflow permissions block, should be `contents: read` minimum

**Issue**: Secret scanning false positives  
**Solution**: Review patterns in `secret-scanning.yml`, adjust if needed

---

## Contributing

When adding new workflows:

1. Set explicit `permissions` block
2. Add `timeout-minutes` to all jobs
3. Enable pip caching with `cache: 'pip'`
4. Use Python 3.10 for consistency
5. Document the workflow in this README
6. Follow security best practices

---

## Maintenance

### Regular Updates

- **Monthly**: Review and update GitHub Action versions
- **Quarterly**: Review security scanning patterns
- **As Needed**: Update Python version across all workflows
- **As Needed**: Update dependency version constraints

### Monitoring

Monitor workflow runs for:
- Unexpected timeouts
- Cache miss rates
- Secret scanning alerts
- Coverage drops below thresholds

---

## Contact

For questions about workflows or CI/CD configuration, please open an issue or contact the development team.
