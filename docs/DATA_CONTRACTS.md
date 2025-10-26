# Data Contracts Hardening

This document captures the operational rules for the questionnaire and rubric configuration files after the v3.0.0 migration to atomic identifiers.

## Versioning and Compatibility

- Every configuration file declares a semantic `version` and a 64-character `content_hash` derived from the canonical JSON payload.
- `rubric_scoring.json` specifies `requires_questionnaire_version` and validation fails if it diverges from `questionnaire.json`.
- Historical schemas are frozen under `schemas/` and any change requires a new semantic version alongside updated compatibility metadata.

## Evidence Taxonomy

`questionnaire.json` enumerates the verification sources in `metadata.sources_of_verification`. All questions reference that taxonomy through `required_evidence_keys`, while the rubric enforces the same keys at matrix and policy-area levels. Unknown keys trigger validation failures.

## Determinism, NA Rules, and Penalties

- Modalities that rely on semantic similarity (`TYPE_F`) must include a `determinism` block containing the seed contract: `{ "seed_required": true, "seed_source": "seed_factory_v1" }`.
- `rubric_scoring.json` centralises NA behaviour in `na_rules`. Missing modality or policy-area entries prevent the rubric from loading.
- Compliance penalties (`contradictory_info`, `missing_indicator`, `OOD_flag`) are declared once with explicit weights.

## Validation Commands

```bash
# Full schema, weight, compatibility, and checksum validation
python schema_validator.py

# Cross-file integrity (Q→PA→CL mapping, weight sums, modality matrix)
python tools/integrity/check_cross_refs.py

# Strict linting with duplicate-key detection and schema enforcement
python tools/lint/json_lint.py questionnaire.json --schema schemas/questionnaire.schema.json
python tools/lint/json_lint.py rubric_scoring.json --schema schemas/rubric_scoring.schema.json
```

Validation is automatically executed on import (`orchestrator/__init__.py`), by pre-commit hooks, and in CI (`.github/workflows/data-contracts.yml`).

## Deterministic Artifacts

The CI pipeline generates two runs of deterministic artifacts and compares their SHA-256 hashes:

```bash
python tools/integrity/dump_artifacts.py artifacts/run1
python tools/integrity/dump_artifacts.py artifacts/run2
diff artifacts/run1/deterministic_snapshot.json artifacts/run2/deterministic_snapshot.json
```

Any divergence indicates a non-deterministic path in the scoring workflow.

## Migrations

Use the canonical migration script to translate legacy P#/D#/Q# structures into PAxx/DIMxx/Qxxx identifiers, normalise weights, and update hashes:

```bash
python tools/migrations/migrate_ids_v1_to_v2.py \
    --questionnaire questionnaire.json \
    --rubric rubric_scoring.json \
    --execution-mapping execution_mapping.yaml \
    --write
```

Running the script rewrites the target files with sorted keys, injects `content_hash`, refreshes `config/metadata_checksums.json`, and enforces the new evidence taxonomy.

## Tests

Property-based regression tests covering invalid references, weight violations, missing modalities, absent NA rules, and determinism gaps live under `tests/data/test_questionnaire_and_rubric.py`.

## Release Management

After merging into `main`, create an immutable tag named `data-contracts-vX.Y.Z`, regenerate `content_hash` values, and refresh `config/metadata_checksums.json`. Any subsequent change must bump the patch component (e.g., `v3.0.1`) and re-run the validation suite before tagging.
