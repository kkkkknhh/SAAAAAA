# Deprecated YAML Parameter Files

**Date**: 2025-11-13
**Status**: DEPRECATED - DO NOT USE

## Summary

These YAML files contained method-specific parameters that have been **consolidated into the authoritative JSON configuration**: `config/method_parameters.json`

## Deprecated Files

| File | Purpose | Replacement |
|------|---------|-------------|
| `OperationalizationAuditor_v3.0_COMPLETO.yaml` | Auditor thresholds/weights | `config/method_parameters.json` → `OperationalizationAuditor` section |
| `trazabilidad_cohrencia.yaml` | Traceability parameters | `config/method_parameters.json` → `SEMANTIC_ALIGNMENT` section |
| `causalextractor.yaml` | Causal extraction config | `config/method_parameters.json` → `CausalExtractor` section |
| `causal_exctractor.yaml` | Duplicate (typo) | Removed (duplicate of above) |
| `derek_beach_cdaf_config.yaml` | Derek Beach parameters | `config/method_parameters.json` → `BayesianMechanismInference` section |

## Migration Path

**OLD CODE (Deprecated)**:
```python
import yaml

# Don't do this anymore!
with open("OperationalizationAuditor_v3.0_COMPLETO.yaml") as f:
    config = yaml.safe_load(f)
    threshold = config['thresholds']['APROBADO']['min_score']
```

**NEW CODE (Correct)**:
```python
from saaaaaa.config import method_parameters

# Use this instead
threshold = method_parameters.get_parameter(
    "OperationalizationAuditor",
    "thresholds.APROBADO.min_score"
)
```

## Why Deprecated?

1. **Scattered Configuration**: Parameters were spread across 5+ YAML files
2. **Duplication**: Same parameters defined in multiple places
3. **No Validation**: YAML files had no schema enforcement
4. **Poor Discoverability**: Hard to find which file contained which parameter
5. **No Epistemological Documentation**: Parameter values lacked justification

## New System Benefits

✅ **Single Source of Truth**: `config/method_parameters.json`
✅ **Full Documentation**: Every parameter has epistemological justification
✅ **Schema Validation**: Programmatic validation of all parameters
✅ **Type Safety**: Parameters categorized as threshold/weight/constant/lexicon
✅ **Academic Grounding**: References to supporting literature

## Calibration YAMLs (NOT Deprecated)

These files are **still active** - they contain training data, not configuration:
- `VFARFAN_D1Q1_COMPLETE_10_AREAS.yaml` - Evidence patterns for calibration
- `catalogo_principal.yaml` - Catalog index for calibration system

**Do NOT deprecate these** - they are training inputs, not method parameters.

## Rollback (If Needed)

If you need to temporarily revert to YAML configuration:
```bash
cp .deprecated_yaml_parameters/*.yaml ./
# Then restart your application
```

However, this is **strongly discouraged** as it breaks the new parameter system.

## Questions?

See:
- `docs/PARAMETERIZATION_STRATEGY.md` - Full migration strategy
- `PARAMETER_ANALYSIS_SUMMARY.md` - Parameter documentation
- `src/saaaaaa/config/method_parameters.py` - Parameter loader API

---

*Deprecated: 2025-11-13*
*Migration Complete*
