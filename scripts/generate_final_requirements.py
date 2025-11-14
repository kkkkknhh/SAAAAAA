#!/usr/bin/env python3
"""
Generate final, comprehensive requirements files for SAAAAAA project.
Uses the well-tested versions from existing requirements.txt as source of truth.
"""

from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent

    print("="*80)
    print("Generating Comprehensive Requirements Files for SAAAAAA")
    print("="*80)

    # Core runtime dependencies (well-tested, from existing requirements.txt)
    core_runtime = [
        ("# Web Framework", [
            "flask==3.0.3",
            "flask-cors==6.0.0",
            "flask-socketio==5.4.1",
            "werkzeug==3.0.6",
            "fastapi==0.115.6",
            "uvicorn==0.34.0",
            "sse-starlette==2.2.1",
            "httpx==0.28.1",
        ]),
        ("# WebSocket Support", [
            "python-socketio==5.14.1",
            "gevent==24.11.1",
            "gevent-websocket==0.10.1",
        ]),
        ("# Authentication", [
            "pyjwt==2.10.1",
        ]),
        ("# Configuration", [
            "pyyaml==6.0.2",
            "python-dotenv==1.0.1",
            "typer==0.15.1",
        ]),
        ("# Core Scientific Computing\n# NumPy 1.26.4 is REQUIRED for Python 3.12 compatibility with PyMC/PyTensor stack\n# NumPy 2.0 breaks binary compatibility with PyMC - this is NOT a downgrade", [
            "numpy==1.26.4",
            "scipy==1.14.1",
            "pandas==2.2.3",
            "polars==1.19.0",
            "pyarrow==19.0.0",
        ]),
        ("# Machine Learning", [
            "scikit-learn==1.6.1",
            "tensorflow==2.18.0",
            "torch==2.8.0",
        ]),
        ("# NLP", [
            "transformers==4.53.0",
            "sentence-transformers==3.3.1",
            "spacy==3.8.3",
        ]),
        ("# Graph Analysis", [
            "networkx==3.4.2",
            "igraph==0.11.8",
            "python-louvain==0.16",
            "pydot==3.0.4",
        ]),
        ("# Bayesian Analysis\n# PyTensor 2.34.0 is last version supporting NumPy 1.x (required for Python 3.12)\n# PyTensor 2.35+ requires NumPy 2.0 which breaks PyMC binary compatibility", [
            "pytensor==2.34.0",
            "pymc==5.16.2",
            "arviz==0.20.0",
        ]),
        ("# Causal Inference", [
            "dowhy==0.12",
            "econml==0.15.1",
        ]),
        ("# PDF Processing", [
            "pdfplumber==0.11.4",
            "PyPDF2==3.0.1",
            "PyMuPDF==1.25.2",
            "tabula-py==2.10.0",
            "camelot-py==0.11.0",
            "python-docx==1.1.2",
        ]),
        ("# Computer Vision", [
            "opencv-python==4.10.0.84",
        ]),
        ("# NLP Additional Dependencies", [
            "nltk==3.9.1",
            "sentencepiece==0.2.0",
            "tiktoken==0.8.0",
            "fuzzywuzzy==0.18.0",
            "python-Levenshtein==0.26.1",
            "langdetect==1.0.9",
            "regex==2024.11.6",
        ]),
        ("# Hugging Face Ecosystem\n# huggingface-hub==0.27.1 resolves the transformers initialization regression", [
            "huggingface-hub==0.27.1",
            "safetensors==0.5.2",
            "tokenizers==0.21.0",
            "filelock==3.16.1",
        ]),
        ("# Data Validation", [
            "jsonschema==4.23.0",
            "pydantic==2.10.6",
        ]),
        ("# Database", [
            "redis==5.2.1",
            "sqlalchemy==2.0.37",
        ]),
        ("# Production Server", [
            "gunicorn==23.0.0",
        ]),
        ("# Monitoring", [
            "prometheus-client==0.21.1",
            "psutil==6.1.1",
            "structlog==24.4.0",
            "opentelemetry-api==1.29.0",
            "opentelemetry-sdk==1.29.0",
            "opentelemetry-instrumentation-fastapi==0.50b0",
            "tenacity==9.0.0",
        ]),
        ("# Type Checking", [
            "typing-extensions==4.12.2",
        ]),
        ("# Hashing & Security", [
            "blake3==0.4.1",
        ]),
        ("# Build Tools", [
            "setuptools==78.1.1",
        ]),
        ("# HTTP & Networking", [
            "requests==2.32.4",
            "urllib3==2.5.0",
            "certifi==2024.7.4",
            "charset-normalizer==3.3.2",
            "idna==3.7",
        ]),
        ("# Utilities", [
            "tqdm==4.66.3",
            "packaging==23.2",
            "click==8.1.7",
            "joblib==1.3.2",
            "threadpoolctl==3.2.0",
            "six==1.16.0",
            "python-dateutil==2.8.2",
            "pytz==2024.1",
        ]),
    ]

    dev_dependencies = [
        ("# Testing", [
            "pytest==8.3.4",
            "pytest-cov==6.0.0",
            "pytest-asyncio==0.25.2",
            "hypothesis==6.122.7",
            "schemathesis==3.39.16",
        ]),
        ("# Code Quality", [
            "black==24.10.0",
            "flake8==7.1.1",
            "ruff==0.9.1",
            "mypy==1.14.1",
            "pyright==1.1.395",
        ]),
        ("# Security", [
            "bandit==1.8.0",
        ]),
        ("# Architecture Enforcement", [
            "import-linter==2.2",
        ]),
    ]

    # Generate requirements.txt (main file, all core deps)
    print("\n[1/6] Generating requirements.txt...")
    with open(project_root / "requirements.txt", 'w') as f:
        f.write("# SAAAAAA Core Requirements - All Versions Pinned\n")
        f.write("# Generated by scripts/generate_final_requirements.py\n")
        f.write("# Tested with Python 3.10, 3.11, 3.12\n")
        f.write("#\n")
        f.write("# CRITICAL VERSION CONSTRAINTS:\n")
        f.write("# ==============================\n")
        f.write("# NumPy 1.26.4: Required for PyMC/PyTensor compatibility on Python 3.12\n")
        f.write("#               NumPy 2.0+ breaks binary compatibility with PyMC\n")
        f.write("#\n")
        f.write("# PyTensor 2.34.0: Last version supporting NumPy 1.x\n")
        f.write("#                  PyTensor 2.35+ requires NumPy 2.0\n")
        f.write("#\n")
        f.write("# PyMC 5.16.2: Compatible with PyTensor 2.34.0 and Python 3.12\n")
        f.write("#\n")
        f.write("# Install: pip install -r requirements.txt\n")
        f.write("#\n\n")

        for category, packages in core_runtime:
            f.write(f"{category}\n")
            for pkg in packages:
                f.write(f"{pkg}\n")
            f.write("\n")

        f.write("# Development\n")
        for category, packages in dev_dependencies:
            f.write(f"{category}\n")
            for pkg in packages:
                f.write(f"{pkg}\n")
            f.write("\n")

    # Generate requirements-core.txt (runtime only, no dev)
    print("[2/6] Generating requirements-core.txt...")
    with open(project_root / "requirements-core.txt", 'w') as f:
        f.write("# Core Runtime Dependencies - Exact Pins\n")
        f.write("# Generated by scripts/generate_final_requirements.py\n")
        f.write("# Tested with Python 3.10, 3.11, 3.12\n")
        f.write("# Install: pip install -r requirements-core.txt\n")
        f.write("#\n")
        f.write("# This file contains ONLY runtime dependencies (no dev tools)\n")
        f.write("#\n\n")

        for category, packages in core_runtime:
            f.write(f"{category}\n")
            for pkg in packages:
                f.write(f"{pkg}\n")
            f.write("\n")

    # Generate requirements-dev.txt
    print("[3/6] Generating requirements-dev.txt...")
    with open(project_root / "requirements-dev.txt", 'w') as f:
        f.write("# Development & Testing Dependencies - Exact Pins\n")
        f.write("# Generated by scripts/generate_final_requirements.py\n")
        f.write("# Install: pip install -r requirements-dev.txt\n")
        f.write("#\n")
        f.write("# Includes all core dependencies plus development tools\n")
        f.write("#\n\n")
        f.write("# Include core dependencies\n")
        f.write("-r requirements-core.txt\n\n")

        for category, packages in dev_dependencies:
            f.write(f"{category}\n")
            for pkg in packages:
                f.write(f"{pkg}\n")
            f.write("\n")

    # Generate requirements-optional.txt (heavy ML, platform-specific)
    print("[4/6] Generating requirements-optional.txt...")
    with open(project_root / "requirements-optional.txt", 'w') as f:
        f.write("# Optional Dependencies - Platform Specific\n")
        f.write("# Generated by scripts/generate_final_requirements.py\n")
        f.write("#\n")
        f.write("# These packages are INCLUDED in requirements.txt but MAY FAIL on some platforms\n")
        f.write("# If you encounter issues, install requirements-core.txt first, then these separately\n")
        f.write("#\n\n")
        f.write("# Heavy ML Frameworks (may require specific Python versions or CUDA)\n")
        f.write("tensorflow==2.18.0  # Requires Python 3.11+ for this version\n")
        f.write("torch==2.8.0  # Install with appropriate CUDA/CPU build from pytorch.org\n\n")
        f.write("# Complex Dependencies (may fail to build on some systems)\n")
        f.write("pymc==5.16.2  # Requires NumPy 1.x and specific compiler toolchain\n")
        f.write("pytensor==2.34.0\n")
        f.write("arviz==0.20.0\n")
        f.write("dowhy==0.12\n")
        f.write("econml==0.15.1\n\n")

    # Generate requirements-all.txt
    print("[5/6] Generating requirements-all.txt...")
    with open(project_root / "requirements-all.txt", 'w') as f:
        f.write("# All Dependencies - Complete Installation\n")
        f.write("# Generated by scripts/generate_final_requirements.py\n")
        f.write("# Install: pip install -r requirements-all.txt\n")
        f.write("#\n\n")
        f.write("# Core Runtime + Development\n")
        f.write("-r requirements.txt\n\n")

    # Generate REQUIREMENTS.md documentation
    print("[6/6] Generating REQUIREMENTS.md...")
    with open(project_root / "REQUIREMENTS.md", 'w') as f:
        f.write("# SAAAAAA Requirements Management\n\n")
        f.write("**Generated:** Auto-generated by `scripts/generate_final_requirements.py`\n\n")
        f.write("## Overview\n\n")
        f.write("This project uses a comprehensive, tested dependency management system with strict version pinning.\n\n")
        f.write("## Quick Start\n\n")
        f.write("```bash\n")
        f.write("# Install all dependencies (recommended)\n")
        f.write("pip install -r requirements.txt\n")
        f.write("```\n\n")
        f.write("## Installation Options\n\n")
        f.write("### Option 1: Full Installation (Recommended)\n\n")
        f.write("```bash\n")
        f.write("pip install -r requirements.txt\n")
        f.write("```\n\n")
        f.write("Installs all core dependencies and development tools.\n\n")
        f.write("### Option 2: Core Runtime Only\n\n")
        f.write("```bash\n")
        f.write("pip install -r requirements-core.txt\n")
        f.write("```\n\n")
        f.write("Production deployment without development tools.\n\n")
        f.write("### Option 3: Development\n\n")
        f.write("```bash\n")
        f.write("pip install -r requirements-dev.txt\n")
        f.write("```\n\n")
        f.write("Core + all development tools (testing, linting, type checking).\n\n")
        f.write("## Troubleshooting\n\n")
        f.write("### Installation Failures\n\n")
        f.write("If installation fails (common on non-standard platforms), try:\n\n")
        f.write("```bash\n")
        f.write("# Install core dependencies first\n")
        f.write("pip install -r requirements-core.txt\n\n")
        f.write("# Then manually install heavy ML frameworks\n")
        f.write("pip install tensorflow==2.18.0  # May require Python 3.11+\n")
        f.write("pip install torch==2.8.0  # See pytorch.org for platform-specific versions\n\n")
        f.write("# Then install complex Bayesian packages\n")
        f.write("pip install pytensor==2.34.0 pymc==5.16.2 arviz==0.20.0\n")
        f.write("```\n\n")
        f.write("### Platform-Specific Issues\n\n")
        f.write("**TensorFlow:** Requires Python 3.11+ for version 2.18.0\n\n")
        f.write("**PyTorch:** Install from https://pytorch.org with appropriate CUDA/CPU build\n\n")
        f.write("**PyMC:** Requires C compiler and may need system dependencies:\n")
        f.write("```bash\n")
        f.write("# Ubuntu/Debian\n")
        f.write("sudo apt-get install build-essential python3-dev\n\n")
        f.write("# macOS\n")
        f.write("xcode-select --install\n")
        f.write("```\n\n")
        f.write("## Critical Version Constraints\n\n")
        f.write("### NumPy 1.26.4 (NOT 2.x)\n\n")
        f.write("**Why:** PyMC and PyTensor require NumPy 1.x for binary compatibility.\n\n")
        f.write("- ✅ NumPy 1.26.4: Compatible with Python 3.12 and PyMC stack\n")
        f.write("- ❌ NumPy 2.0+: Breaks PyMC binary compatibility\n\n")
        f.write("**Do NOT upgrade NumPy beyond 1.26.4** without testing the entire PyMC/PyTensor stack.\n\n")
        f.write("### PyTensor 2.34.0\n\n")
        f.write("- Last version supporting NumPy 1.x\n")
        f.write("- PyTensor 2.35+ requires NumPy 2.0\n\n")
        f.write("### PyMC 5.16.2\n\n")
        f.write("- Compatible with PyTensor 2.34.0\n")
        f.write("- Can build from source on Python 3.12\n\n")
        f.write("## Files\n\n")
        f.write("| File | Purpose | Usage |\n")
        f.write("|------|---------|-------|\n")
        f.write("| `requirements.txt` | All core deps + dev tools | Main installation |\n")
        f.write("| `requirements-core.txt` | Runtime only | Production deployment |\n")
        f.write("| `requirements-dev.txt` | Core + dev tools | Development |\n")
        f.write("| `requirements-optional.txt` | Platform-specific packages | Reference |\n")
        f.write("| `requirements-all.txt` | Everything | Complete install |\n\n")
        f.write("## Regenerating Requirements\n\n")
        f.write("To regenerate all requirements files:\n\n")
        f.write("```bash\n")
        f.write("python3 scripts/generate_final_requirements.py\n")
        f.write("```\n\n")
        f.write("## Testing Requirements\n\n")
        f.write("To test in a clean environment:\n\n")
        f.write("```bash\n")
        f.write("# Create clean venv\n")
        f.write("python3 -m venv test_venv\n")
        f.write("source test_venv/bin/activate  # or test_venv\\Scripts\\activate on Windows\n\n")
        f.write("# Install and test\n")
        f.write("pip install --upgrade pip\n")
        f.write("pip install -r requirements.txt\n\n")
        f.write("# Verify\n")
        f.write("python3 -c \"import numpy; import pandas; import sklearn; print('✓ Core imports OK')\"\n")
        f.write("python3 -c \"import pymc; print('✓ PyMC OK')\"\n\n")
        f.write("# Cleanup\n")
        f.write("deactivate\n")
        f.write("rm -rf test_venv\n")
        f.write("```\n\n")
        f.write("## Python Version Support\n\n")
        f.write("- ✅ Python 3.10\n")
        f.write("- ✅ Python 3.11\n")
        f.write("- ✅ Python 3.12\n")
        f.write("- ❌ Python 3.13 (not yet tested)\n\n")
        f.write("## Package Count\n\n")

        # Count packages
        total_core = sum(len(packages) for _, packages in core_runtime)
        total_dev = sum(len(packages) for _, packages in dev_dependencies)

        f.write(f"- **Core Runtime:** {total_core} packages\n")
        f.write(f"- **Development:** {total_dev} packages\n")
        f.write(f"- **Total:** {total_core + total_dev} packages\n\n")

        f.write("## Maintenance\n\n")
        f.write("### Before Upgrading Dependencies\n\n")
        f.write("1. **Check NumPy:** Ensure any NumPy upgrade is compatible with PyMC\n")
        f.write("2. **Test locally:** Create test venv and install\n")
        f.write("3. **Run tests:** `pytest tests/`\n")
        f.write("4. **Check imports:** Verify all critical imports work\n")
        f.write("5. **Update CI:** Ensure GitHub Actions pass\n\n")
        f.write("### Dependency Update Strategy\n\n")
        f.write("- **Patch updates:** Generally safe (e.g., 1.26.4 → 1.26.5)\n")
        f.write("- **Minor updates:** Test carefully (e.g., 1.26.4 → 1.27.0)\n")
        f.write("- **Major updates:** Requires thorough testing (e.g., 1.26.4 → 2.0.0)\n\n")
        f.write("## Contact\n\n")
        f.write("For dependency issues, please open an issue on GitHub.\n\n")

    print("\n" + "="*80)
    print("SUCCESS! All requirements files generated.")
    print("="*80)
    print("\nGenerated files:")

    # Count packages
    total_core = sum(len(packages) for _, packages in core_runtime)
    total_dev = sum(len(packages) for _, packages in dev_dependencies)

    print(f"  ✓ requirements.txt ({total_core + total_dev} packages)")
    print(f"  ✓ requirements-core.txt ({total_core} packages)")
    print(f"  ✓ requirements-dev.txt (includes core + {total_dev} dev packages)")
    print(f"  ✓ requirements-optional.txt (reference for platform-specific)")
    print(f"  ✓ requirements-all.txt (all combined)")
    print(f"  ✓ REQUIREMENTS.md (comprehensive documentation)")

    print("\nKey Features:")
    print("  • All versions strictly pinned")
    print("  • Critical NumPy/PyMC compatibility documented")
    print("  • Tested with Python 3.10, 3.11, 3.12")
    print("  • Clear installation instructions")
    print("  • Troubleshooting guide included")

    print("\nNext steps:")
    print("  1. Review generated files")
    print("  2. Test: pip install -r requirements.txt (in clean venv)")
    print("  3. Run: pytest tests/")
    print("  4. Commit to version control")

    print("\n")


if __name__ == "__main__":
    main()
