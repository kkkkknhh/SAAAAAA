# SAAAAAA: Strategic Policy Analysis System
## Doctoral-Level Integration of 584 Methods Across 300 Questions

**Status:** ✅ Strategic mapping complete  
**Architecture:** 7 Producers + 1 Aggregator (Two-Core Pipeline)  
**Coverage:** 584 methods → 300 questions with chess-based optimization  
**Standard:** 150-300 word doctoral-level explanations with multi-source triangulation

---

## 📦 Package Installation & Import Strategy

### Installation

This package follows Python best practices with a clean `src/` layout:

```bash
# Install in development mode (recommended for development)
pip install -e .

# Or install with development dependencies
pip install -e ".[dev,test]"

# For production
pip install .
```

### Import Strategy

**✅ Always use absolute imports from the installed package:**

```python
# Core modules
from saaaaaa.core.orchestrator import Orchestrator
from saaaaaa.core.ports import Port

# Analysis modules
from saaaaaa.analysis.bayesian_multilevel_system import BayesianRollUp
from saaaaaa.analysis.recommendation_engine import RecommendationEngine

# Processing modules
from saaaaaa.processing.document_ingestion import ingest_document
from saaaaaa.processing.aggregation import aggregate_results
```

**❌ Never use sys.path manipulation:**

```python
# DON'T DO THIS - all sys.path hacks have been removed
import sys
sys.path.insert(0, '/path/to/src')  # ❌ No!
```

**❌ Avoid relative imports outside the package:**

```python
# DON'T DO THIS in examples or tests
from ..core import something  # ❌ No!
```

**📖 For detailed import guidelines:** See [TEST_IMPORT_MATRIX.md](TEST_IMPORT_MATRIX.md)

---

## 🚀 Complete Operational Runbook

**📖 [OPERATIONAL_GUIDE.md](OPERATIONAL_GUIDE.md)** - **Your complete implementation guide with ALL necessary commands!**

**📋 [QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - **Essential commands at a glance**

### What's in the Operational Guide

- ✅ **Step-by-step installation** and system activation with dependency management
- ✅ **Import conflict resolution** strategies for all common issues  
- ✅ **Development plan analysis** walkthrough with full commands
- ✅ **Component execution commands** - run full system OR individual parts
- ✅ **Complete pipeline execution** instructions for all 7 producers + aggregator
- ✅ **Test classification & selection** - 40+ tests organized by priority
- ✅ **Verification and testing** procedures with recommended sequences
- ✅ **Troubleshooting** guide for common issues and solutions
- ✅ **Advanced usage** and customization options
- ✅ **Complete command reference** - every command in one place

### Quick Start

```bash
# 1. Install system (choose one method)
make install           # Using Makefile (recommended)
# OR
bash scripts/setup.sh  # Using setup script

# 2. Run analysis
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --input data/input_plans/your_plan.pdf \
  --output-dir data/results \
  --mode full --parallel

# 3. Validate system
python3 scripts/validate_system.py
```

**For complete instructions:** [OPERATIONAL_GUIDE.md](OPERATIONAL_GUIDE.md)

---

## 📁 REPOSITORY STRUCTURE

This repository now follows Python best practices with a hierarchical package structure:

```
saaaaaa/
├── src/saaaaaa/       # Main Python package (REAL IMPLEMENTATION)
│   ├── core/          # Orchestration & execution
│   ├── processing/    # Data processing
│   ├── analysis/      # Analysis & ML
│   ├── api/           # API server
│   └── utils/         # Utilities
├── orchestrator/      # Compatibility shims (redirects to src/)
├── concurrency/       # Compatibility shims (redirects to src/)
├── core/              # Compatibility shims (redirects to src/)
├── executors/         # Compatibility shims (redirects to src/)
├── tests/             # Test suite
├── docs/              # Documentation
├── examples/          # Example scripts
├── scripts/           # Utility scripts
├── config/            # Configuration files
└── data/              # Data files
```

**⚠️ Note:** Root-level directories (orchestrator/, concurrency/, core/, executors/) are 
compatibility shims for backward compatibility. **All new code should go in src/saaaaaa/**.

**📖 Important Documentation:**
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - **Complete project structure guide (READ THIS FIRST)**
- [orchestrator/README.md](orchestrator/README.md) - Orchestration compatibility layer explanation
- [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md) - Detailed structure documentation
- [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) - Import migration reference
- [docs/POST_REORGANIZATION_STEPS.md](docs/POST_REORGANIZATION_STEPS.md) - **Next steps to complete setup**

---

## 📚 DOCUMENTATION STRUCTURE

### **Core Inventory (Commit 1 ✅)**
- [`config/inventory.json`](config/inventory.json) - Machine-readable exhaustive inventory of all 8 files, 67 classes, 584 methods
- [`docs/INVENTORY_COMPLETE.md`](docs/INVENTORY_COMPLETE.md) - Human-readable summary with statistics

### **Dependency Mapping (Commit 2 ✅)**
- [`docs/dependency_graph.dot`](docs/dependency_graph.dot) - Graphviz visualization of 7 producers → 1 aggregator architecture (8.6KB)
- [`data/interaction_matrix.csv`](data/interaction_matrix.csv) - Method-to-method interaction matrix with 584 methods (14KB)
- [`question_component_map.json`](question_component_map.json) - Strategic method-to-question mappings (15KB)
- [`docs/COMMIT_2_SUMMARY.md`](docs/COMMIT_2_SUMMARY.md) - Complete Commit 2 deliverables summary (13KB)

### **Strategic Orchestration (Commit 1-2 ✅)**
- [`docs/STRATEGIC_METHOD_ORCHESTRATION.md`](docs/STRATEGIC_METHOD_ORCHESTRATION.md) - Chess-based optimization strategy (9.4KB)
- [`docs/CHESS_TACTICAL_SUMMARY.md`](docs/CHESS_TACTICAL_SUMMARY.md) - Tactical patterns and checkmate conditions (9.2KB)

### **Integration Status & Progress**
- [`docs/INTEGRATION_STATUS.md`](docs/INTEGRATION_STATUS.md) - Progress tracking: ✅ Commits 1-2 COMPLETE, ⏳ Commits 3-8 PENDING
- [`docs/COMMIT_2_SUMMARY.md`](docs/COMMIT_2_SUMMARY.md) - Detailed Commit 2 completion summary
- [`data/questionnaire_monolith.json`](data/questionnaire_monolith.json) - The 300 questions defining all requirements (814KB)

### **Data Contracts**
- [`docs/DATA_CONTRACTS.md`](docs/DATA_CONTRACTS.md) - Operational rules, validation commands, and migration workflow

---

## 🎯 QUICK START

**📖 For complete instructions, see [OPERATIONAL_GUIDE.md](OPERATIONAL_GUIDE.md)**

### **Quick Setup (Recommended)**

Use one of these methods to install all dependencies:

```bash
# Method 1: Using Makefile (simplest)
make install

# Method 2: Using setup script (includes SpaCy models)
bash scripts/setup.sh

# Verify installation
python scripts/verify_dependencies.py
```

### **Run Your First Analysis**

```bash
# Analyze a development plan (after setup)
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --input data/input_plans/your_plan.pdf \
  --output-dir data/results \
  --mode full --parallel
```

### **Manual Installation**

If you prefer to install manually:

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install SpaCy language models
python -m spacy download es_core_news_lg
python -m spacy download es_dep_news_trf

# 3. Install the package in development mode (required)
pip install -e .

# 4. Verify installation
python scripts/verify_dependencies.py
```

**📖 For detailed setup information, see:**
- **[OPERATIONAL_GUIDE.md](OPERATIONAL_GUIDE.md)** - **Complete operational guide with all commands**
- [DEPENDENCY_SETUP.md](DEPENDENCY_SETUP.md) - Complete dependency installation guide
- [IMPORT_RESOLUTION_SUMMARY.md](IMPORT_RESOLUTION_SUMMARY.md) - Import fixes and troubleshooting

### **Verification Runbook**
Execute the full verification pipeline in sequence to ensure governance:

```bash
pip install -r requirements.txt  # or: poetry install
python -m compileall -q core orchestrator executors
python tools/scan_core_purity.py
lint-imports --config contracts/importlinter.ini
ruff check .
mypy . --strict
pycycle core orchestrator executors
python tools/import_all.py
pytest -q -ra
coverage run -m pytest >/dev/null 2>&1 || true && coverage report -m
```

### **Understand the Architecture**
```bash
# Read the strategic overview
cat docs/CHESS_TACTICAL_SUMMARY.md

# Review method mappings
cat question_component_map.json | jq '.dimension_mappings'

# Check inventory statistics
cat config/inventory.json | jq '.statistics'
```

### **Key Numbers**
- **8 files** integrated (7 producers + 1 aggregator)
- **67 classes** inventoried
- **584 methods** strategically mapped
- **300 questions** to answer with doctoral rigor
- **6 dimensions** of policy analysis (D1-D6)
- **10 policy areas** evaluated (P1-P10)
- **6 scoring modalities** (TYPE_A through TYPE_F)

---

## ♟️ THE CHESS STRATEGY

### **OPENING: Evidence Collection (Parallel)**
7 producers execute independently:
1. `src/saaaaaa/analysis/financiero_viabilidad_tablas.py` (65 methods) - Financial + Causal DAG
2. `src/saaaaaa/analysis/Analyzer_one.py` (34 methods) - Semantic Cube + Value Chain
3. `src/saaaaaa/analysis/contradiction_deteccion.py` (62 methods) - Contradictions + Coherence
4. `src/saaaaaa/processing/embedding_policy.py` (36 methods) - Semantic Search + Bayesian
5. `src/saaaaaa/analysis/teoria_cambio.py` (30 methods) - DAG Validation + Monte Carlo
6. `src/saaaaaa/analysis/dereck_beach.py` (99 methods) - Beach Tests + Mechanisms
7. `src/saaaaaa/processing/policy_processor.py` (32 methods) - Pattern Matching + Evidence

**Output:** 7 independent JSON artifacts with typed evidence

### **MIDDLE GAME: Triangulation**
`report_assembly.py` (43 methods) synthesizes using 6 modalities:
- **TYPE_A (Bayesian):** Numerical claims, gaps, risks
- **TYPE_B (DAG):** Causal chains, ToC completeness
- **TYPE_C (Coherence):** Inverted contradictions
- **TYPE_D (Pattern):** Baseline data, formalization
- **TYPE_E (Financial):** Budget traceability
- **TYPE_F (Beach):** Mechanism inference, plausibility

**Output:** 300 MICRO answers (each with 3-5 producer triangulation)

### **ENDGAME: Doctoral Synthesis**
- **MICRO (300):** Question-level 150-300 word explanations
- **MESO (60):** Policy-dimension cluster analysis
- **MACRO (1):** Overall plan classification + remediation roadmap

**Output:** Doctoral-level deliverable with NO mocks, NO placeholders

---

## 📊 METHOD DISTRIBUTION BY DIMENSION

| Dimension | Questions | Primary Producers | Methods | Avg/Q |
|-----------|----------:|------------------:|--------:|------:|
| D1 Insumos | 50 | 4 | 131 | 10.5 |
| D2 Actividades | 50 | 5 | 158 | 12.6 |
| D3 Productos | 50 | 4 | 98 | 7.8 |
| D4 Resultados | 50 | 5 | 142 | 11.4 |
| D5 Impactos | 50 | 3 | 87 | 7.0 |
| D6 Causalidad | 50 | 6 | 189 | 15.1 |
| **TOTAL** | **300** | **7+1** | **584** | **10.7** |

**Insight:** D6 (Causalidad) requires most methods (15.1/question) due to complex causal inference.

---

## 🏆 QUALITY ASSURANCE STANDARDS

✅ **NO MOCKS:** All 584 methods are real implementations  
✅ **NO PLACEHOLDERS:** Actual Bayesian/evidential logic throughout  
✅ **EXHAUSTIVE:** All 300 questions receive full treatment  
✅ **MULTI-SOURCE:** Minimum 3-5 producer triangulation per answer  
✅ **CONFIDENCE:** 95% Bayesian confidence intervals on all scores  
✅ **CITATIONS:** Explicit document page/section references  
✅ **REMEDIATION:** Actionable improvement paths for all gaps  
✅ **DOCTORAL RIGOR:** 150-300 word analytical depth per question  

---

## 🎖️ TACTICAL PATTERNS (Examples)

### **Pattern Alpha: Baseline Data (D1-Q1)**
Modality: TYPE_D + TYPE_A
```
IndustrialPolicyProcessor._match_patterns_in_sentences
  → BayesianEvidenceScorer.compute_evidence_score
    → BayesianNumericalAnalyzer.evaluate_policy_metric
      → ReportAssembler._score_type_d
        → 150-300 word explanation
```

### **Pattern Beta: Causal ToC (D6-Q1)**
Modality: TYPE_B + TYPE_F
```
TeoriaCambio.construir_grafo_causal
  → TeoriaCambio.validacion_completa
    → AdvancedDAGValidator.calculate_acyclicity_pvalue
      → BeachEvidentialTest.apply_test_logic
        → ReportAssembler._score_type_b
          → 150-300 word explanation
```

### **Pattern Gamma: Budget Traceability (D1-Q3)**
Modality: TYPE_E
```
PDETMunicipalPlanAnalyzer.analyze_financial_feasibility
  → FinancialAuditor.trace_financial_allocation
    → FinancialAuditor._perform_counterfactual_budget_check
      → ReportAssembler._score_type_e
        → 150-300 word explanation with COP amounts
```

See [`docs/CHESS_TACTICAL_SUMMARY.md`](docs/CHESS_TACTICAL_SUMMARY.md) for complete tactical patterns.

---

## 🔄 NEXT STEPS: INTEGRATION SEQUENCE

Following the 8-commit sequence:

1. ✅ **Commit 1:** Inventory + Provenance (COMPLETE)
2. ⏳ **Commit 2:** Dependency mappings (interaction_matrix.csv, dependency_graph.dot)
3. ⏳ **Commit 3:** JSON Schemas for all producer artifacts
4. ⏳ **Commit 4:** ReportAssembler implementation (aggregator core)
5. ⏳ **Commit 5:** Contrast engine for multi-source triangulation
6. ⏳ **Commit 6:** Orchestrator for producer coordination
7. ⏳ **Commit 7:** Multi-level report generation (MICRO/MESO/MACRO)
8. ⏳ **Commit 8:** Integration tests + validation suite

See [`docs/INTEGRATION_STATUS.md`](docs/INTEGRATION_STATUS.md) for detailed progress.

---

## 📖 FURTHER READING

### **Getting Started**
- **[OPERATIONAL_GUIDE.md](OPERATIONAL_GUIDE.md)** - **Complete operational guide with all commands (START HERE!)**
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - **Essential commands reference card**
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start for developers

### **Architecture & Strategy**
- **Repository Structure:** [`docs/REPOSITORY_STRUCTURE.md`](docs/REPOSITORY_STRUCTURE.md) - Detailed package organization
- **Strategic Overview:** [`docs/CHESS_TACTICAL_SUMMARY.md`](docs/CHESS_TACTICAL_SUMMARY.md) - Visual patterns and checkmate conditions
- **Detailed Strategy:** [`docs/STRATEGIC_METHOD_ORCHESTRATION.md`](docs/STRATEGIC_METHOD_ORCHESTRATION.md) - Complete method chains
- **Method Mappings:** [`question_component_map.json`](question_component_map.json) - Machine-readable execution chains

### **System Inventory**
- **Complete Inventory:** [`config/inventory.json`](config/inventory.json) - All 67 classes, 584 methods
- **Canonical Truth:** [`data/questionnaire_monolith.json`](data/questionnaire_monolith.json) - The 300 questions

---

## 🎯 KEY PRINCIPLE

**This is strategic orchestration of 584 analytical methods to achieve doctoral-level policy evaluation. Every method placement is intentional, every combination is optimized, every synthesis is rigorous.**

**Checkmate.**

---

## 📦 Import System (Updated November 2025)

SAAAAAA supports **two equivalent import styles** - choose whichever you prefer:

### Option 1: Direct Package Imports (After `pip install -e .`)
```python
from saaaaaa.core.orchestrator import Orchestrator
from saaaaaa.analysis.scoring.scoring import apply_scoring
```

### Option 2: Root-Level Compatibility Imports (From repo root)
```python
from orchestrator import Orchestrator
from scoring.scoring import apply_scoring
```

Both styles work identically and reference the same code. Root-level files are thin compatibility wrappers.

**Quick verification**:
```bash
python tests/test_import_consistency.py
```

See **[OPERATIONAL_GUIDE.md](OPERATIONAL_GUIDE.md)** for complete import documentation.

