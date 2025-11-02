# SAAAAAA System - Complete Operational Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation & Setup](#installation--setup)
4. [System Activation](#system-activation)
5. [Development Plan Analysis](#development-plan-analysis)
6. [Running the Full Pipeline](#running-the-full-pipeline)
7. [Verification & Testing](#verification--testing)
8. [Common Operations](#common-operations)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

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
# Install all required packages with pinned versions
pip install -r requirements.txt

# Or with constraints for stricter version control
pip install -r requirements.txt -c constraints.txt
```

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
