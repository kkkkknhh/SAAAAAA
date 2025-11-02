# SAAAAAA System - Complete Operational Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation & Setup](#installation--setup)
4. [System Activation](#system-activation)
5. [Import Conflict Resolution](#import-conflict-resolution)
6. [Development Plan Analysis](#development-plan-analysis)
7. [Running the Full Pipeline](#running-the-full-pipeline)
8. [Component Execution Commands](#component-execution-commands)
9. [Verification & Testing](#verification--testing)
10. [Test Classification & Selection](#test-classification--selection)
11. [Common Operations](#common-operations)
12. [Troubleshooting](#troubleshooting)
13. [Advanced Usage](#advanced-usage)
14. [Command Reference](#command-reference)

---

## Overview

**SAAAAAA** is a Strategic Policy Analysis System that integrates 584 analytical methods across 300 policy evaluation questions. The system uses a chess-based orchestration strategy with 7 producer modules and 1 aggregator to provide doctoral-level policy analysis.

### Key Components

- **7 Producer Modules**: Independent parallel analysis engines
- **1 Aggregator Module**: Synthesizes multi-source evidence
- **584 Methods**: Real implementations (no mocks or placeholders)
- **300 Questions**: Comprehensive policy evaluation framework
- **6 Dimensions**: D1-D6 covering inputs through causality
- **10 Policy Areas**: P1-P10 evaluation domains

---

## System Requirements

### Required Software

- **Python**: 3.10 or higher (3.11 recommended)
- **pip**: Latest version
- **Git**: For repository management
- **Minimum RAM**: 8GB (16GB recommended for large analyses)
- **Disk Space**: 5GB minimum for dependencies and models

### Operating Systems

- Linux (Ubuntu 20.04+, Debian 11+)
- macOS (10.15+)
- Windows 10/11 (via WSL2 recommended)

---

## Installation & Setup

### Quick Installation (Recommended)

The fastest way to get started is using the automated setup script:

```bash
# Clone the repository
git clone https://github.com/kkkkknhh/SAAAAAA.git
cd SAAAAAA

# Run automated setup
bash scripts/setup.sh
```

This script will:
1. Install all Python dependencies from `requirements.txt`
2. Download required SpaCy language models (es_core_news_lg, es_dep_news_trf)
3. Verify the installation

### Manual Installation

For more control over the installation process:

#### Step 1: Clone the Repository

```bash
git clone https://github.com/kkkkknhh/SAAAAAA.git
cd SAAAAAA
```

#### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Step 3: Install Python Dependencies

```bash
# Install all required packages with pinned versions (RECOMMENDED)
pip install -r requirements.txt

# Or with constraints for stricter version control
pip install -r requirements.txt -c constraints.txt
```

**Note**: The project has 82+ dependencies defined in `requirements.txt`. While `pyproject.toml` exists for packaging metadata, **always use `requirements.txt` for installation** to ensure all dependencies are installed with correct versions.

#### Step 4: Install SpaCy Language Models

The system requires Spanish language models for NLP tasks:

```bash
# Download large Spanish core model
python3 -m spacy download es_core_news_lg

# Download transformer-based Spanish dependency model
python3 -m spacy download es_dep_news_trf
```

#### Step 5: Install Package in Development Mode

This makes the `saaaaaa` package importable throughout your code:

```bash
pip install -e .
```

#### Step 6: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration (optional)
nano .env
```

#### Step 7: Verify Installation

```bash
# Run verification script
python3 scripts/verify_dependencies.py

# Expected output: All dependencies verified successfully
```

---

## System Activation

### Complete System Activation Sequence

Follow these commands in order to fully activate the SAAAAAA system:

#### 1. Environment Preparation

```bash
# Navigate to project directory
cd /path/to/SAAAAAA

# Activate virtual environment (if using one)
source venv/bin/activate

# Verify Python version
python3 --version  # Should be 3.10 or higher
```

#### 2. Dependency Verification

```bash
# Verify all dependencies are installed
python3 scripts/verify_dependencies.py

# Check SpaCy models
python3 -c "import spacy; nlp = spacy.load('es_core_news_lg'); print('✓ SpaCy models loaded')"
```

#### 3. System Compilation

```bash
# Compile Python modules to check for syntax errors
python3 -m compileall -q src/saaaaaa

# Expected output: No errors
```

#### 4. Import Validation

```bash
# Validate all imports are correct
python3 scripts/validate_imports.py

# Test core imports
python3 -c "
from saaaaaa.core import ORCHESTRATOR_MONILITH
from saaaaaa.processing import document_ingestion
from saaaaaa.analysis import bayesian_multilevel_system
print('✓ Core modules imported successfully')
"
```

#### 5. System Registry Validation

```bash
# Validate the class registry
python3 scripts/validate_registry.py

# Verify strategic wiring
python3 scripts/validate_strategic_wiring.py
```

#### 6. Configuration Check

```bash
# Verify configuration files exist
ls -la config/inventory.json
ls -la config/schemas/
ls -la data/questionnaire_monolith.json

# Validate configuration structure
python3 scripts/validate_schema.py
```

### Verification of Successful Activation

Run this comprehensive check:

```bash
# Full system validation
python3 scripts/validate_system.py

# Expected output: All system components validated ✓
```

---

## Import Conflict Resolution

### Common Import Issues and Solutions

The SAAAAAA system has undergone repository reorganization. All core code now lives in `src/saaaaaa/` but backward compatibility shims exist at the root level.

#### Understanding the Import Structure

**New Structure (Preferred)**:
```python
# Core orchestration
from saaaaaa.core.orchestrator import Orchestrator, MethodExecutor
from saaaaaa.core.orchestrator.ORCHESTRATOR_MONILITH import OrchestrationEngine

# Analysis modules (7 producers)
from saaaaaa.analysis.financiero_viabilidad_tablas import PDETMunicipalPlanAnalyzer
from saaaaaa.analysis.Analyzer_one import SemanticAnalyzer, PerformanceAnalyzer
from saaaaaa.analysis.contradiction_deteccion import PolicyContradictionDetector
from saaaaaa.analysis.embedding_policy import BayesianNumericalAnalyzer
from saaaaaa.analysis.teoria_cambio import TeoriaCambio, AdvancedDAGValidator
from saaaaaa.analysis.dereck_beach import CDAFFramework, BeachEvidentialTest
from saaaaaa.analysis.bayesian_multilevel_system import BayesianMultilevelScorer

# Processing modules
from saaaaaa.processing.document_ingestion import DocumentIngestionEngine
from saaaaaa.processing.policy_processor import IndustrialPolicyProcessor
from saaaaaa.processing.embedding_policy import PolicyAnalysisEmbedder

# Utilities
from saaaaaa.utils.contracts import ProducerContract, ScoringModality
from saaaaaa.utils.validation.schema_validator import SchemaValidator
```

**Legacy Structure (DEPRECATED - DO NOT USE)**:
```python
# DEPRECATED: These imports are maintained for backward compatibility only
# Use the new structure above instead
from orchestrator.core import Orchestrator  # DEPRECATED
from concurrency import TaskExecutor  # DEPRECATED
```

**⚠️ Important**: The legacy `orchestrator` module is deprecated. All new code and examples should use the new structure with `saaaaaa.core.orchestrator` instead.

#### Resolving Import Conflicts

##### Issue 1: ModuleNotFoundError for Internal Modules

**Problem**: `ModuleNotFoundError: No module named 'recommendation_engine'`

**Cause**: Root-level modules not being found after reorganization

**Solution**:
```bash
# Option 1: Install package in development mode (recommended)
pip install -e .

# Option 2: Update imports to use new structure
python3 scripts/update_imports.py src/ tests/ examples/

# Option 3: Add PYTHONPATH temporarily
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"
```

##### Issue 2: Circular Import Dependencies

**Problem**: Circular imports between orchestrator and producer modules

**Cause**: Direct imports creating dependency cycles

**Solution**:
```python
# Use TYPE_CHECKING to avoid runtime circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saaaaaa.analysis.teoria_cambio import TeoriaCambio
else:
    TeoriaCambio = None  # type: ignore

# Or use lazy imports
def get_teoria_cambio():
    from saaaaaa.analysis.teoria_cambio import TeoriaCambio
    return TeoriaCambio
```

##### Issue 3: Missing SpaCy Language Models

**Problem**: `OSError: [E050] Can't find model 'es_core_news_lg'`

**Cause**: SpaCy language models not downloaded

**Solution**:
```bash
# Download Spanish language models
python3 -m spacy download es_core_news_lg
python3 -m spacy download es_dep_news_trf

# Verify installation
python3 -c "import spacy; nlp = spacy.load('es_core_news_lg'); print('✓ Models loaded')"
```

##### Issue 4: Import Linting Violations

**Problem**: `lint-imports` reports boundary violations

**Cause**: Imports crossing architectural boundaries

**Solution**:
```bash
# Check import boundaries
lint-imports --config contracts/importlinter.ini

# Fix violations by updating import paths
# Example: core/ should not import from orchestrator/
# Instead, both should import from src/saaaaaa/core/
```

### Import Best Practices

1. **Always use `src/saaaaaa/` imports** for new code
2. **Install in development mode**: `pip install -e .`
3. **Check imports before committing**: `python3 scripts/validate_imports.py`
4. **Avoid circular dependencies**: Use TYPE_CHECKING or lazy imports
5. **Update legacy imports**: `python3 scripts/update_imports.py <directory>`

### Verifying Import Health

```bash
# Complete import verification sequence
python3 -m compileall -q src/saaaaaa
python3 scripts/validate_imports.py
lint-imports --config contracts/importlinter.ini
python3 -c "from saaaaaa.core.orchestrator import Orchestrator; print('✓ Imports OK')"
```

---

## Development Plan Analysis

### Analyzing Your First Development Plan

This section guides you through analyzing a municipal development plan using SAAAAAA.

#### Step 1: Prepare Your Development Plan Document

```bash
# Create data directory for input documents
mkdir -p data/input_plans

# Place your PDF document
# Example: copy your plan to data/input_plans/plan_municipal_2024.pdf
```

Supported formats:
- PDF (`.pdf`)
- Text (`.txt`)
- JSON (`.json`)

#### Step 2: Document Ingestion

```bash
# Run document ingestion
python3 -m saaaaaa.processing.document_ingestion \
  --input data/input_plans/plan_municipal_2024.pdf \
  --output data/processed/plan_parsed.json

# This extracts text, tables, and metadata from your document
```

#### Step 3: Policy Processing

```bash
# Process the policy document
python3 -m saaaaaa.processing.policy_processor \
  --input data/processed/plan_parsed.json \
  --output data/processed/policy_analysis.json

# This identifies patterns, baseline data, and evidence
```

#### Step 4: Run Producer Modules (Parallel Analysis)

Execute all 7 producer modules to analyze different aspects:

```bash
# Producer 1: Financial Viability & Causal DAG
python3 -m saaaaaa.analysis.financiero_viabilidad_tablas \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_1_financial.json

# Producer 2: Semantic Cube & Value Chain
python3 -m saaaaaa.analysis.Analyzer_one \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_2_semantic.json

# Producer 3: Contradictions & Coherence
python3 -m saaaaaa.analysis.contradiction_deteccion \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_3_contradictions.json

# Producer 4: Semantic Search & Bayesian
python3 -m saaaaaa.processing.embedding_policy \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_4_embedding.json

# Producer 5: DAG Validation & Monte Carlo
python3 -m saaaaaa.analysis.teoria_cambio \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_5_toc.json

# Producer 6: Beach Tests & Mechanisms
python3 -m saaaaaa.analysis.dereck_beach \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_6_beach.json

# Producer 7: Pattern Matching & Evidence
python3 -m saaaaaa.processing.policy_processor \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_7_patterns.json \
  --mode evidence
```

#### Step 5: Aggregate Results

```bash
# Run the aggregator to synthesize all producer outputs
python3 -m saaaaaa.processing.aggregation \
  --producer-dir data/producers \
  --output data/aggregated/report_assembly.json

# This creates the triangulated evidence synthesis
```

#### Step 6: Generate Multi-Level Reports

```bash
# Generate MICRO level (300 question-level explanations)
python3 -m saaaaaa.core.report_generator \
  --input data/aggregated/report_assembly.json \
  --output data/reports/micro_report.json \
  --level micro

# Generate MESO level (60 policy-dimension clusters)
python3 -m saaaaaa.core.report_generator \
  --input data/aggregated/report_assembly.json \
  --output data/reports/meso_report.json \
  --level meso

# Generate MACRO level (overall classification + remediation)
python3 -m saaaaaa.core.report_generator \
  --input data/aggregated/report_assembly.json \
  --output data/reports/macro_report.json \
  --level macro
```

### Quick Analysis with Orchestrator

For a fully automated analysis:

```bash
# Run the complete orchestration pipeline
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --input data/input_plans/plan_municipal_2024.pdf \
  --output-dir data/analysis_results \
  --mode full

# This executes all steps automatically:
# 1. Document ingestion
# 2. Policy processing
# 3. All 7 producers in parallel
# 4. Aggregation
# 5. Multi-level report generation
```

---

## Running the Full Pipeline

### End-to-End Pipeline Execution

#### Option 1: Using the Main Orchestrator (Recommended)

```bash
# Complete automated pipeline
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --input data/input_plans/your_plan.pdf \
  --output-dir data/results/$(date +%Y%m%d_%H%M%S) \
  --mode full \
  --parallel \
  --verbose

# Flags explanation:
# --mode full: Run complete analysis (ingestion → reports)
# --parallel: Execute producers in parallel (faster)
# --verbose: Show detailed progress
```

#### Option 2: Step-by-Step Execution

For more control or debugging:

```bash
# 1. Document Ingestion
python3 -m saaaaaa.processing.document_ingestion \
  --input data/input_plans/plan.pdf \
  --output data/stage1_ingestion.json

# 2. Policy Processing
python3 -m saaaaaa.processing.policy_processor \
  --input data/stage1_ingestion.json \
  --output data/stage2_policy.json

# 3. Execute Producers (can be run in parallel)
bash scripts/run_all_producers.sh \
  --input data/stage2_policy.json \
  --output-dir data/producers

# 4. Aggregation
python3 -m saaaaaa.processing.aggregation \
  --producer-dir data/producers \
  --output data/stage4_aggregated.json

# 5. Report Generation
bash scripts/generate_all_reports.sh \
  --input data/stage4_aggregated.json \
  --output-dir data/reports
```

#### Option 3: Using the Choreographer

For complex workflows with dependencies:

```bash
# Execute with choreographer
python3 -m saaaaaa.core.choreographer \
  --config config/workflow_config.yaml \
  --input data/input_plans/plan.pdf \
  --output-dir data/results

# The choreographer manages:
# - Task dependencies
# - Parallel execution
# - Error handling
# - Resource management
```

### Pipeline Output Structure

After execution, you'll have:

```
data/results/YYYYMMDD_HHMMSS/
├── 01_ingestion/
│   └── document_parsed.json
├── 02_processing/
│   └── policy_analysis.json
├── 03_producers/
│   ├── producer_1_financial.json
│   ├── producer_2_semantic.json
│   ├── producer_3_contradictions.json
│   ├── producer_4_embedding.json
│   ├── producer_5_toc.json
│   ├── producer_6_beach.json
│   └── producer_7_patterns.json
├── 04_aggregation/
│   └── report_assembly.json
└── 05_reports/
    ├── micro_report.json       # 300 question-level analyses
    ├── meso_report.json        # 60 cluster analyses
    ├── macro_report.json       # Overall classification
    └── executive_summary.pdf   # Human-readable report
```

---

## Component Execution Commands

### Individual Producer Execution

Execute individual producer modules for targeted analysis or debugging:

#### Producer 1: Financial Viability & Causal DAG
```bash
# Standalone execution
python3 -m saaaaaa.analysis.financiero_viabilidad_tablas \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_1_financial.json \
  --verbose

# With specific analysis modes
python3 -m saaaaaa.analysis.financiero_viabilidad_tablas \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_1_financial.json \
  --mode financial_audit \
  --enable-dag-analysis
```

**Key Methods**: `analyze_financial_feasibility`, `trace_financial_allocation`, `build_causal_dag`

#### Producer 2: Semantic Cube & Value Chain
```bash
# Semantic analysis
python3 -m saaaaaa.analysis.Analyzer_one \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_2_semantic.json \
  --enable-value-chain

# With ontology mapping
python3 -m saaaaaa.analysis.Analyzer_one \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_2_semantic.json \
  --use-municipal-ontology
```

**Key Methods**: `build_semantic_cube`, `analyze_value_chain`, `extract_semantic_relations`

#### Producer 3: Contradictions & Coherence
```bash
# Contradiction detection
python3 -m saaaaaa.analysis.contradiction_deteccion \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_3_contradictions.json \
  --coherence-threshold 0.85

# With temporal logic verification
python3 -m saaaaaa.analysis.contradiction_deteccion \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_3_contradictions.json \
  --enable-temporal-logic
```

**Key Methods**: `detect_contradictions`, `verify_temporal_coherence`, `calculate_coherence_score`

#### Producer 4: Semantic Search & Bayesian Embedding
```bash
# Embedding-based analysis
python3 -m saaaaaa.processing.embedding_policy \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_4_embedding.json \
  --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# With Bayesian scoring
python3 -m saaaaaa.processing.embedding_policy \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_4_embedding.json \
  --enable-bayesian-scoring \
  --confidence-level 0.95
```

**Key Methods**: `compute_embeddings`, `semantic_search`, `bayesian_evidence_score`

#### Producer 5: Theory of Change & DAG Validation
```bash
# ToC construction
python3 -m saaaaaa.analysis.teoria_cambio \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_5_toc.json \
  --validate-dag

# With Monte Carlo simulation
python3 -m saaaaaa.analysis.teoria_cambio \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_5_toc.json \
  --monte-carlo-runs 10000 \
  --simulate-interventions
```

**Key Methods**: `construir_grafo_causal`, `validacion_completa`, `monte_carlo_simulation`

#### Producer 6: Beach Evidential Tests
```bash
# Beach test execution
python3 -m saaaaaa.analysis.dereck_beach \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_6_beach.json \
  --test-types straw_in_the_wind,hoop,smoking_gun,doubly_decisive

# With mechanism inference
python3 -m saaaaaa.analysis.dereck_beach \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_6_beach.json \
  --infer-mechanisms \
  --bayesian-updating
```

**Key Methods**: `apply_beach_tests`, `infer_causal_mechanisms`, `assess_test_strength`

#### Producer 7: Pattern Matching & Evidence Processing
```bash
# Pattern-based analysis
python3 -m saaaaaa.processing.policy_processor \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_7_patterns.json \
  --mode evidence \
  --extract-baseline-data

# With advanced pattern matching
python3 -m saaaaaa.processing.policy_processor \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_7_patterns.json \
  --pattern-library config/patterns.json \
  --fuzzy-matching
```

**Key Methods**: `match_patterns_in_sentences`, `extract_evidence`, `formalize_baseline_data`

### Core System Components

#### Document Ingestion
```bash
# PDF ingestion
python3 -m saaaaaa.processing.document_ingestion \
  --input data/input_plans/plan.pdf \
  --output data/processed/document_parsed.json \
  --extract-tables \
  --detect-language

# Multiple document formats
python3 -m saaaaaa.processing.document_ingestion \
  --input data/input_plans/ \
  --output data/processed/batch_ingestion.json \
  --formats pdf,txt,docx \
  --parallel
```

#### Policy Processing
```bash
# Basic processing
python3 -m saaaaaa.processing.policy_processor \
  --input data/processed/document_parsed.json \
  --output data/processed/policy_analysis.json

# Advanced processing with all features
python3 -m saaaaaa.processing.policy_processor \
  --input data/processed/document_parsed.json \
  --output data/processed/policy_analysis.json \
  --enable-semantic-chunking \
  --extract-metadata \
  --validate-structure
```

#### Aggregation
```bash
# Standard aggregation
python3 -m saaaaaa.processing.aggregation \
  --producer-dir data/producers \
  --output data/aggregated/report_assembly.json

# With triangulation settings
python3 -m saaaaaa.processing.aggregation \
  --producer-dir data/producers \
  --output data/aggregated/report_assembly.json \
  --min-sources 3 \
  --confidence-threshold 0.90 \
  --enable-cross-validation
```

#### Report Generation
```bash
# Generate all report levels
python3 -m saaaaaa.core.report_generator \
  --input data/aggregated/report_assembly.json \
  --output-dir data/reports \
  --levels micro,meso,macro

# Generate specific level with options
python3 -m saaaaaa.core.report_generator \
  --input data/aggregated/report_assembly.json \
  --output data/reports/micro_detailed.json \
  --level micro \
  --word-count 250 \
  --include-citations \
  --format json,pdf
```

### Parallel Execution of Producers

```bash
# Execute all producers in parallel (recommended)
bash scripts/run_all_producers.sh \
  --input data/processed/policy_analysis.json \
  --output-dir data/producers \
  --parallel \
  --workers 7

# Execute specific producers
bash scripts/run_all_producers.sh \
  --input data/processed/policy_analysis.json \
  --output-dir data/producers \
  --producers 1,2,3,4 \
  --parallel
```

### Orchestrator Modes

```bash
# Full orchestration (all steps)
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --input data/input_plans/plan.pdf \
  --output-dir data/results \
  --mode full \
  --parallel

# Partial orchestration (specific phases)
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --input data/processed/policy_analysis.json \
  --output-dir data/results \
  --mode producers_only \
  --parallel

# Debug mode (verbose logging)
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --input data/input_plans/plan.pdf \
  --output-dir data/results \
  --mode full \
  --debug \
  --log-level DEBUG \
  --save-intermediates
```

### Utility Scripts

#### Validation Scripts
```bash
# Validate entire system
python3 scripts/validate_system.py

# Validate specific components
python3 scripts/validate_d1_orchestration.py        # Orchestration layer
python3 scripts/validate_d2_concurrence.py          # Concurrency module
python3 scripts/validate_strategic_wiring.py        # Method wiring
python3 scripts/validate_registry.py                # Class registry
python3 scripts/validate_schema.py                  # Data schemas
python3 scripts/validate_imports.py                 # Import structure

# Validate monolith integrity
python3 scripts/validate_monolith.py
```

#### Build and Generate Scripts
```bash
# Build monolith
python3 scripts/build_monolith.py --output data/monolith.json

# Generate inventory
python3 scripts/generate_inventory.py --output config/inventory.json

# Generate all reports
bash scripts/generate_all_reports.sh \
  --input data/aggregated/report_assembly.json \
  --output-dir data/reports
```

#### Recommendation Engine
```bash
# Generate recommendations CLI
python3 scripts/recommendation_cli.py \
  --analysis data/reports/macro_report.json \
  --level MACRO \
  --output data/recommendations.json

# Interactive mode
python3 scripts/recommendation_cli.py --interactive
```

---

## Verification & Testing

### Pre-Execution Verification

Before running analysis, verify the system:

```bash
# 1. Dependency check
python3 scripts/verify_dependencies.py

# 2. System integrity
python3 scripts/validate_system.py

# 3. Import verification
python3 scripts/validate_imports.py

# 4. Schema validation
python3 scripts/validate_schema.py

# 5. Complete verification pipeline
bash scripts/validate_contracts_local.sh
```

### Running Tests

#### Unit Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_orchestrator.py -v

# Run tests with coverage
pytest tests/ --cov=src/saaaaaa --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

#### Integration Tests

```bash
# Run integration tests
pytest tests/integration/ -v

# Test orchestrator integration
python3 scripts/validate_d1_orchestration.py

# Test concurrency module
python3 scripts/validate_d2_concurrence.py
```

#### Contract Tests

```bash
# Validate data contracts
python3 scripts/validate_contracts_local.sh

# Validate signatures
python3 scripts/signature_ci_check.py

# Validate registry
python3 scripts/validate_registry.py
```

#### End-to-End Tests

```bash
# Run complete pipeline test
python3 scripts/verify_system_complete.py

# Validate strategic wiring
python3 scripts/validate_strategic_wiring.py

# Test with sample data
pytest tests/test_e2e_pipeline.py -v -s
```

### Quality Assurance Checks

```bash
# Code quality
ruff check .

# Type checking
mypy src/saaaaaa --strict

# Import linting
lint-imports --config contracts/importlinter.ini

# Circular dependency check
pycycle src/saaaaaa
```

### Validation Runbook

Execute the complete verification sequence:

```bash
# Full validation sequence (from README)
pip install -r requirements.txt
python -m compileall -q src/saaaaaa
python tools/scan_core_purity.py
lint-imports --config contracts/importlinter.ini
ruff check .
mypy src/saaaaaa --strict
pycycle src/saaaaaa
python tools/import_all.py
pytest -q -ra
coverage run -m pytest
coverage report -m
```

---

## Test Classification & Selection

### Test Suite Organization

The SAAAAAA test suite contains 40+ test files organized by category and purpose. Understanding which tests to run is critical for efficient development and debugging.

### Test Categories

#### 1. Core/Orchestrator Tests (Critical - Always Run Before Deployment)

**Purpose**: Validate the orchestration engine and core execution flow

```bash
# Golden path tests (MUST PASS)
pytest tests/test_orchestrator_golden.py -v

# Smoke tests (MUST PASS)
pytest tests/test_smoke_orchestrator.py -v

# Integration tests
pytest tests/test_orchestrator_integration.py -v

# Fix validation tests
pytest tests/test_orchestrator_fixes.py -v
```

**When to Run**:
- Before every commit to orchestrator code
- Before deployments
- After import structure changes
- Daily in CI/CD

**Expected Behavior**: All tests should PASS. These verify core system functionality.

#### 2. Integration Tests (Run After Component Changes)

**Purpose**: Validate multi-component interactions and end-to-end flows

```bash
# Gold standard integration tests
pytest tests/test_gold_canario_integration.py -v

# Macro reporting integration
pytest tests/test_gold_canario_macro_reporting.py -v

# Meso reporting integration
pytest tests/test_gold_canario_meso_reporting.py -v

# Micro analysis tests
pytest tests/test_gold_canario_micro_bayesian.py -v
pytest tests/test_gold_canario_micro_provenance.py -v
pytest tests/test_gold_canario_micro_stress.py -v

# Known failure tests (document integration issues)
pytest tests/test_integration_failures.py -v --continue-on-collection-errors
```

**When to Run**:
- After changes to any producer module
- After aggregation logic changes
- Before major releases
- Weekly in integration testing

**Expected Behavior**: Gold standard tests should PASS. Known failure tests may have expected failures documented.

#### 3. Contract & Validation Tests (Run Before Commits)

**Purpose**: Ensure data contracts, type signatures, and schemas are correct

```bash
# Comprehensive contract tests (PRIORITY)
pytest tests/test_contracts_comprehensive.py -v

# Runtime contract validation
pytest tests/test_contract_runtime.py -v

# Schema validation
pytest tests/test_schema_validation.py -v

# Signature validation
pytest tests/test_signature_validation.py -v

# Defensive signatures
pytest tests/test_defensive_signatures.py -v

# Aggregation validation
pytest tests/test_aggregation_validation.py -v

# Embedding policy contracts
pytest tests/test_embedding_policy_contracts.py -v

# Contract snapshots
pytest tests/test_contract_snapshots.py -v
```

**When to Run**:
- Before every commit
- After changing function signatures
- After modifying data schemas
- After adding new producers

**Expected Behavior**: All should PASS. Contract violations indicate breaking changes.

#### 4. Property-Based & Fuzzing Tests (Run Nightly)

**Purpose**: Discover edge cases through randomized testing

```bash
# Property-based testing with Hypothesis
pytest tests/test_property_based.py -v --hypothesis-seed=random

# With specific seed for reproducibility
pytest tests/test_property_based.py -v --hypothesis-seed=12345
```

**When to Run**:
- Nightly automated runs
- Before major releases
- When investigating mysterious bugs

**Expected Behavior**: Should discover and report any edge cases.

#### 5. Operational Tests (Run in Staging/Production)

**Purpose**: Validate system behavior under operational conditions

```bash
# Boot checks
pytest tests/operational/test_boot_checks.py -v

# Synthetic traffic tests
pytest tests/operational/test_synthetic_traffic.py -v
```

**When to Run**:
- During deployment validation
- In staging environments
- For production readiness checks

#### 6. Component-Specific Tests

```bash
# Concurrency module
pytest tests/test_concurrency.py -v

# Scoring module
pytest tests/test_scoring.py -v

# Aggregation
pytest tests/test_aggregation.py -v

# Boundaries
pytest tests/test_boundaries.py -v

# Infrastructure
pytest tests/test_infrastructure.py -v
```

**When to Run**: After changes to specific components

#### 7. Regression Tests (Run Always)

```bash
# Semantic chunking regression
pytest tests/test_regression_semantic_chunking.py -v

# Score normalization fix
pytest tests/test_score_normalization_fix.py -v

# Runtime error fixes
pytest tests/test_runtime_error_fixes.py -v
```

**When to Run**:
- Every test run
- These prevent previously fixed bugs from returning

### Recommended Test Sequences

#### Quick Validation (Pre-Commit) - 2-5 minutes
```bash
# Essential tests before committing
pytest tests/test_orchestrator_golden.py \
       tests/test_smoke_orchestrator.py \
       tests/test_contracts_comprehensive.py \
       -v --tb=short
```

#### Standard Validation (Pre-Push) - 10-15 minutes
```bash
# Comprehensive validation before pushing
pytest tests/test_orchestrator_*.py \
       tests/test_contracts*.py \
       tests/test_schema_validation.py \
       tests/test_signature_validation.py \
       -v
```

#### Full Validation (Pre-Release) - 30-60 minutes
```bash
# Complete test suite
pytest tests/ -v --cov=src/saaaaaa --cov-report=html
```

#### Targeted Component Testing
```bash
# When working on specific producers
pytest tests/ -k "financial" -v  # Financial module tests
pytest tests/ -k "bayesian" -v   # Bayesian analysis tests
pytest tests/ -k "beach" -v      # Beach evidential tests
pytest tests/ -k "toc" -v        # Theory of Change tests
```

### Test Priority Matrix

| Test Category | Priority | Frequency | Must Pass | Time |
|---------------|----------|-----------|-----------|------|
| Orchestrator Golden | **CRITICAL** | Every commit | ✅ YES | 1 min |
| Smoke Tests | **CRITICAL** | Every commit | ✅ YES | 30 sec |
| Contract Comprehensive | **HIGH** | Every commit | ✅ YES | 2 min |
| Integration Gold | **HIGH** | Daily | ✅ YES | 5 min |
| Signature Validation | **MEDIUM** | Before push | ✅ YES | 1 min |
| Property-Based | **MEDIUM** | Nightly | ⚠️ Explore | 10 min |
| Operational | **LOW** | Pre-deploy | ✅ YES | 2 min |

### Test Selection Guide

**Before Making Changes**:
```bash
# Establish baseline - these should all pass
pytest tests/test_orchestrator_golden.py tests/test_smoke_orchestrator.py -v
```

**During Development**:
```bash
# Run relevant component tests frequently
pytest tests/ -k "<your_component>" -v --tb=short
```

**Before Committing**:
```bash
# Quick validation
pytest tests/test_orchestrator_golden.py \
       tests/test_contracts_comprehensive.py \
       tests/test_smoke_orchestrator.py -v
```

**Before Pushing**:
```bash
# Standard validation
pytest tests/test_orchestrator*.py tests/test_contracts*.py -v
```

**Before Deploying**:
```bash
# Full validation
pytest tests/ -v --cov=src/saaaaaa
```

### Handling Test Failures

#### Expected Failures
Some tests document known issues:
- `test_integration_failures.py`: Documents integration challenges
- Check test docstrings for expected failure conditions

#### Debugging Failed Tests
```bash
# Run with verbose output and stop on first failure
pytest tests/test_failing.py -vv -x

# Run with debugger on failure
pytest tests/test_failing.py --pdb

# Run with maximum output
pytest tests/test_failing.py -vv --tb=long --capture=no
```

### Test Data Location
```
tests/data/                          # Test data files
  ├── test_questionnaire_and_rubric.py
  └── sample_*.json
examples/                            # Integration test data
  ├── all_data_sample.json
  ├── cluster_data_sample.json
  └── macro_data_sample.json
```

---

## Common Operations

### Working with the API Server

The system includes a REST API for integration:

```bash
# Start API server (development mode)
python3 -m saaaaaa.api.api_server --dev

# Start API server (production mode)
gunicorn --worker-class gevent \
  --workers 4 \
  --bind 0.0.0.0:5000 \
  saaaaaa.api.api_server:app

# Test API health
curl http://localhost:5000/api/v1/health

# Submit analysis via API
curl -X POST http://localhost:5000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"document_path": "data/input_plans/plan.pdf"}'
```

### Using the AtroZ Dashboard

For visual analysis and monitoring:

```bash
# Quick start with AtroZ dashboard
bash atroz_quickstart.sh dev

# This starts:
# - API server on port 5000
# - Dashboard on port 8000

# Access dashboard
open http://localhost:8000  # Opens in browser

# Stop AtroZ dashboard
bash stop_atroz.sh
```

### Batch Processing

Process multiple plans:

```bash
# Process all PDFs in a directory
for pdf in data/input_plans/*.pdf; do
  echo "Processing $pdf..."
  python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
    --input "$pdf" \
    --output-dir "data/results/$(basename $pdf .pdf)" \
    --mode full
done
```

### Exporting Results

```bash
# Convert JSON reports to PDF
python3 -m saaaaaa.utils.export_pdf \
  --input data/reports/macro_report.json \
  --output reports/executive_summary.pdf

# Generate CSV summary
python3 -m saaaaaa.utils.export_csv \
  --input data/reports/micro_report.json \
  --output reports/question_scores.csv

# Create visualization
python3 -m saaaaaa.utils.visualize \
  --input data/reports/meso_report.json \
  --output reports/cluster_analysis.html
```

### Monitoring and Logging

```bash
# View orchestrator logs
tail -f logs/orchestrator.log

# View API logs
tail -f logs/api_server.log

# View all logs
tail -f logs/*.log

# Search for errors
grep -r "ERROR" logs/

# Monitor system resources during analysis
htop  # or top
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: ModuleNotFoundError: No module named 'saaaaaa'

**Cause**: Package not installed in development mode

**Solution**:
```bash
pip install -e .
```

#### Issue 2: SpaCy model not found

**Cause**: Language models not downloaded

**Solution**:
```bash
python3 -m spacy download es_core_news_lg
python3 -m spacy download es_dep_news_trf
```

#### Issue 3: Import errors after reorganization

**Cause**: Old import statements

**Solution**:
```bash
# Update imports automatically
python scripts/update_imports.py tests examples scripts

# Or manually update to new structure:
# from orchestrator.core import X → from saaaaaa.core.orchestrator.core import X
```

#### Issue 4: FileNotFoundError for config files

**Cause**: Incorrect file paths after reorganization

**Solution**:
```bash
# Update paths:
# inventory.json → config/inventory.json
# questionnaire_monolith.json → data/questionnaire_monolith.json
# schemas/ → config/schemas/
```

#### Issue 5: Memory errors during analysis

**Cause**: Large documents or insufficient RAM

**Solution**:
```bash
# Process in chunks
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --input plan.pdf \
  --chunk-size 1000 \
  --output-dir results

# Or increase memory limits
export PYTHONMAXMEMORY=8192
```

#### Issue 6: Slow producer execution

**Cause**: Sequential processing

**Solution**:
```bash
# Enable parallel processing
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --parallel \
  --workers 7 \
  --input plan.pdf
```

#### Issue 7: API server won't start

**Cause**: Port already in use

**Solution**:
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use different port
python3 -m saaaaaa.api.api_server --port 5001
```

### Getting Help

1. **Check Documentation**:
   - [README.md](README.md) - Project overview
   - [QUICKSTART.md](QUICKSTART.md) - Quick start guide
   - [BUILD_HYGIENE.md](BUILD_HYGIENE.md) - Development practices
   - [DEPENDENCY_SETUP.md](DEPENDENCY_SETUP.md) - Dependency guide

2. **Validate System**:
   ```bash
   python3 scripts/validate_system.py
   ```

3. **Check Logs**:
   ```bash
   tail -f logs/*.log
   ```

4. **Run Diagnostics**:
   ```bash
   python3 scripts/bootstrap_validate.py
   ```

---

## Advanced Usage

### Custom Producer Development

Add your own analysis module:

```python
# src/saaaaaa/analysis/my_custom_producer.py
from typing import Dict, Any
from saaaaaa.utils.contracts import ProducerContract

class MyCustomProducer(ProducerContract):
    """Custom producer for specialized analysis."""
    
    def analyze(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform custom analysis."""
        # Your analysis logic here
        return {
            "producer_id": "custom_producer_8",
            "evidence": [...],
            "scores": {...}
        }
```

Register your producer:

```python
# config/producer_registry.py
from saaaaaa.analysis.my_custom_producer import MyCustomProducer

PRODUCERS = [
    # ... existing producers
    MyCustomProducer,
]
```

### Extending the Question Set

Add custom questions beyond the base 300:

```python
# config/custom_questions.json
{
  "questions": [
    {
      "id": "D7-Q1",
      "dimension": "D7_CustomDimension",
      "text": "Your custom question?",
      "producers": ["producer_1", "producer_8"],
      "modalities": ["TYPE_A", "TYPE_G"]
    }
  ]
}
```

### Custom Scoring Modalities

Define new scoring types:

```python
# src/saaaaaa/analysis/custom_scoring.py
from saaaaaa.utils.contracts import ScoringModality

class TypeGScoring(ScoringModality):
    """Custom scoring modality."""
    
    def compute_score(self, evidence: List[Dict]) -> float:
        # Your scoring logic
        return score
```

### Integration with External Systems

#### Webhook Integration

```python
# Send results to external system
python3 -m saaaaaa.integrations.webhook \
  --url https://your-system.com/webhook \
  --results data/reports/macro_report.json
```

#### Database Export

```python
# Export to PostgreSQL
python3 -m saaaaaa.integrations.database \
  --db-url postgresql://user:pass@localhost/saaaaaa \
  --results data/reports/
```

### Performance Optimization

#### Caching

```bash
# Enable result caching
export SAAAAAA_CACHE_ENABLED=true
export SAAAAAA_CACHE_DIR=cache/

python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --cache \
  --input plan.pdf
```

#### Distributed Processing

```bash
# Run producers on different machines
# Machine 1: Producers 1-3
python3 scripts/run_distributed_producers.py \
  --producers 1,2,3 \
  --redis-url redis://central-server:6379

# Machine 2: Producers 4-7
python3 scripts/run_distributed_producers.py \
  --producers 4,5,6,7 \
  --redis-url redis://central-server:6379

# Central server: Aggregation
python3 scripts/run_distributed_aggregator.py \
  --redis-url redis://localhost:6379
```

---

## Command Reference

### Complete Command Index

This comprehensive reference lists all key commands for the SAAAAAA system, organized by function.

#### Installation & Setup Commands
```bash
# Initial setup
git clone https://github.com/kkkkknhh/SAAAAAA.git
cd SAAAAAA
bash scripts/setup.sh

# Manual setup
pip install -r requirements.txt
python3 -m spacy download es_core_news_lg
python3 -m spacy download es_dep_news_trf
pip install -e .

# Verify installation
python3 scripts/verify_dependencies.py
python3 -c "from saaaaaa.core.orchestrator import Orchestrator; print('✓ OK')"
```

#### System Validation Commands
```bash
# Quick validation
python3 scripts/validate_system.py

# Component validation
python3 scripts/validate_imports.py
python3 scripts/validate_registry.py
python3 scripts/validate_schema.py
python3 scripts/validate_strategic_wiring.py
python3 scripts/validate_d1_orchestration.py
python3 scripts/validate_d2_concurrence.py

# Contract validation
bash scripts/validate_contracts_local.sh
python3 scripts/signature_ci_check.py

# Complete validation sequence
python3 -m compileall -q src/saaaaaa
lint-imports --config contracts/importlinter.ini
ruff check .
mypy src/saaaaaa --strict
pycycle src/saaaaaa
pytest -q -ra
coverage run -m pytest && coverage report -m
```

#### Orchestrator Execution Commands
```bash
# Full pipeline (recommended)
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --input data/input_plans/plan.pdf \
  --output-dir data/results \
  --mode full \
  --parallel

# Debug mode
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --input data/input_plans/plan.pdf \
  --output-dir data/results \
  --mode full \
  --debug \
  --log-level DEBUG

# Producers only
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
  --input data/processed/policy_analysis.json \
  --output-dir data/results \
  --mode producers_only \
  --parallel
```

#### Individual Producer Commands
```bash
# Producer 1: Financial & DAG
python3 -m saaaaaa.analysis.financiero_viabilidad_tablas \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_1.json

# Producer 2: Semantic Analysis
python3 -m saaaaaa.analysis.Analyzer_one \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_2.json

# Producer 3: Contradictions
python3 -m saaaaaa.analysis.contradiction_deteccion \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_3.json

# Producer 4: Embeddings
python3 -m saaaaaa.processing.embedding_policy \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_4.json

# Producer 5: Theory of Change
python3 -m saaaaaa.analysis.teoria_cambio \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_5.json

# Producer 6: Beach Tests
python3 -m saaaaaa.analysis.dereck_beach \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_6.json

# Producer 7: Pattern Matching
python3 -m saaaaaa.processing.policy_processor \
  --input data/processed/policy_analysis.json \
  --output data/producers/producer_7.json \
  --mode evidence
```

#### Processing Commands
```bash
# Document ingestion
python3 -m saaaaaa.processing.document_ingestion \
  --input data/input_plans/plan.pdf \
  --output data/processed/document_parsed.json

# Policy processing
python3 -m saaaaaa.processing.policy_processor \
  --input data/processed/document_parsed.json \
  --output data/processed/policy_analysis.json

# Aggregation
python3 -m saaaaaa.processing.aggregation \
  --producer-dir data/producers \
  --output data/aggregated/report_assembly.json
```

#### Report Generation Commands
```bash
# All report levels
python3 -m saaaaaa.core.report_generator \
  --input data/aggregated/report_assembly.json \
  --output-dir data/reports \
  --levels micro,meso,macro

# MICRO level (300 questions)
python3 -m saaaaaa.core.report_generator \
  --input data/aggregated/report_assembly.json \
  --output data/reports/micro_report.json \
  --level micro

# MESO level (60 clusters)
python3 -m saaaaaa.core.report_generator \
  --input data/aggregated/report_assembly.json \
  --output data/reports/meso_report.json \
  --level meso

# MACRO level (overall)
python3 -m saaaaaa.core.report_generator \
  --input data/aggregated/report_assembly.json \
  --output data/reports/macro_report.json \
  --level macro
```

#### Testing Commands
```bash
# Quick pre-commit tests
pytest tests/test_orchestrator_golden.py \
       tests/test_smoke_orchestrator.py \
       tests/test_contracts_comprehensive.py -v

# Standard pre-push tests
pytest tests/test_orchestrator*.py \
       tests/test_contracts*.py -v

# Full test suite
pytest tests/ -v --cov=src/saaaaaa --cov-report=html

# Specific test categories
pytest tests/test_gold_canario_*.py -v              # Integration tests
pytest tests/test_contracts*.py -v                   # Contract tests
pytest tests/operational/ -v                         # Operational tests
pytest tests/test_property_based.py -v              # Property-based tests

# With debugging
pytest tests/test_name.py -vv --pdb
pytest tests/test_name.py -vv --tb=long --capture=no
```

#### API & Dashboard Commands
```bash
# Start API server (development)
python3 -m saaaaaa.api.api_server --dev

# Start API server (production)
gunicorn --worker-class gevent \
  --workers 4 \
  --bind 0.0.0.0:5000 \
  saaaaaa.api.api_server:app

# Start AtroZ dashboard
bash atroz_quickstart.sh dev

# Test API health
curl http://localhost:5000/api/v1/health

# Submit analysis via API
curl -X POST http://localhost:5000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"document_path": "data/input_plans/plan.pdf"}'
```

#### Utility & Helper Commands
```bash
# Update imports
python3 scripts/update_imports.py src/ tests/ examples/

# Generate inventory
python3 scripts/generate_inventory.py --output config/inventory.json

# Build monolith
python3 scripts/build_monolith.py --output data/monolith.json

# Recommendation CLI
python3 scripts/recommendation_cli.py \
  --analysis data/reports/macro_report.json \
  --level MACRO

# Verify system complete
python3 scripts/verify_system_complete.py
python3 scripts/verify_complete_implementation.py

# Count producer methods
python3 scripts/count_producer_methods.py
```

#### Batch Processing Commands
```bash
# Process multiple PDFs
for pdf in data/input_plans/*.pdf; do
  python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH \
    --input "$pdf" \
    --output-dir "data/results/$(basename $pdf .pdf)" \
    --mode full --parallel
done

# Run all producers in parallel
bash scripts/run_all_producers.sh \
  --input data/processed/policy_analysis.json \
  --output-dir data/producers \
  --parallel

# Generate all reports
bash scripts/generate_all_reports.sh \
  --input data/aggregated/report_assembly.json \
  --output-dir data/reports
```

#### Import Troubleshooting Commands
```bash
# Fix import structure
pip install -e .
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"

# Update legacy imports
python3 scripts/update_imports.py src/ tests/

# Validate import structure
python3 scripts/validate_imports.py
lint-imports --config contracts/importlinter.ini
python3 -m compileall -q src/saaaaaa
```

#### Monitoring & Logging Commands
```bash
# View logs
tail -f logs/orchestrator.log
tail -f logs/api_server.log
tail -f logs/*.log

# Search for errors
grep -r "ERROR" logs/

# Monitor resources
htop  # or top

# Check disk usage
du -sh data/*
```

#### Code Quality Commands
```bash
# Linting
ruff check .
ruff check . --fix

# Type checking
mypy src/saaaaaa --strict
mypy src/saaaaaa --strict --show-error-codes

# Import analysis
lint-imports --config contracts/importlinter.ini
pycycle src/saaaaaa

# Format code
black src/ tests/
```

### Common Command Sequences

#### Complete Analysis Workflow
```bash
# 1. Setup
pip install -e .

# 2. Ingest document
python3 -m saaaaaa.processing.document_ingestion \
  --input data/input_plans/plan.pdf \
  --output data/processed/parsed.json

# 3. Process policy
python3 -m saaaaaa.processing.policy_processor \
  --input data/processed/parsed.json \
  --output data/processed/policy.json

# 4. Run all producers
bash scripts/run_all_producers.sh \
  --input data/processed/policy.json \
  --output-dir data/producers --parallel

# 5. Aggregate
python3 -m saaaaaa.processing.aggregation \
  --producer-dir data/producers \
  --output data/aggregated/assembly.json

# 6. Generate reports
python3 -m saaaaaa.core.report_generator \
  --input data/aggregated/assembly.json \
  --output-dir data/reports \
  --levels micro,meso,macro
```

#### Development Workflow
```bash
# 1. Make code changes
# ...

# 2. Quick validation
pytest tests/test_orchestrator_golden.py -v

# 3. Full validation
python3 scripts/validate_system.py
pytest tests/ -v

# 4. Code quality
ruff check . --fix
mypy src/saaaaaa --strict

# 5. Commit
git add .
git commit -m "Description"
```

---

## Appendix

### Quick Reference Commands

#### System Setup
```bash
bash scripts/setup.sh                    # Automated setup
pip install -e .                         # Install package
python3 scripts/verify_dependencies.py   # Verify installation
```

#### Analysis
```bash
# Quick analysis
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH --input plan.pdf

# Full pipeline
python3 -m saaaaaa.core.ORCHESTRATOR_MONILITH --input plan.pdf --mode full --parallel
```

#### Testing
```bash
pytest tests/ -v                         # Run tests
pytest --cov=src/saaaaaa tests/          # With coverage
python3 scripts/validate_system.py       # System validation
```

#### API
```bash
python3 -m saaaaaa.api.api_server --dev  # Start API
bash atroz_quickstart.sh dev             # Start dashboard
```

### File Structure Reference

```
SAAAAAA/
├── src/saaaaaa/          # Main package
│   ├── analysis/         # Producer modules
│   ├── processing/       # Data processing
│   ├── core/             # Orchestration
│   ├── api/              # REST API
│   └── utils/            # Utilities
├── config/               # Configuration
│   ├── inventory.json
│   └── schemas/
├── data/                 # Data files
│   ├── questionnaire_monolith.json
│   └── input_plans/
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── docs/                 # Documentation
└── examples/             # Example usage
```

### Key Documentation Files

- **README.md**: Project overview and architecture
- **QUICKSTART.md**: Quick start for developers
- **BUILD_HYGIENE.md**: Development best practices
- **DEPENDENCY_SETUP.md**: Dependency installation guide
- **PROJECT_STRUCTURE.md**: Repository structure
- **docs/CHESS_TACTICAL_SUMMARY.md**: Chess-based strategy
- **docs/INTEGRATION_STATUS.md**: Implementation progress

---

## Summary

This operational guide provides complete instructions for:

1. ✅ **Installing** the SAAAAAA system with all dependencies
2. ✅ **Activating** the system and verifying all components
3. ✅ **Analyzing** your first development plan
4. ✅ **Running** the full analysis pipeline
5. ✅ **Testing** and validating the system
6. ✅ **Troubleshooting** common issues
7. ✅ **Extending** the system for custom needs

For additional support:
- Review the [documentation files](#key-documentation-files)
- Run system diagnostics: `python3 scripts/validate_system.py`
- Check logs in `logs/` directory

**The system is now ready for doctoral-level policy analysis. Checkmate.**
