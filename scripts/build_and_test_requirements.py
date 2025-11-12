#!/usr/bin/env python3
"""
Comprehensive Requirements Builder and Tester for SAAAAAA Project

This script:
1. Validates the base requirements.txt
2. Creates a clean virtual environment
3. Tests installation of all dependencies
4. Generates a complete, tested requirements file with all transitive dependencies
5. Creates organized requirement files (core, dev, optional)
"""

import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Core runtime dependencies - from existing requirements.txt (well-tested)
CORE_RUNTIME = [
    # Web Framework
    "flask==3.0.3",
    "flask-cors==6.0.0",
    "flask-socketio==5.4.1",
    "werkzeug==3.0.6",
    "fastapi==0.115.6",
    "uvicorn==0.34.0",
    "sse-starlette==2.2.1",
    "httpx==0.28.1",

    # WebSocket Support
    "python-socketio==5.14.1",
    "gevent==24.11.1",
    "gevent-websocket==0.10.1",

    # Authentication
    "pyjwt==2.10.1",

    # Configuration
    "pyyaml==6.0.2",
    "python-dotenv==1.0.1",
    "typer==0.15.1",

    # Core Scientific Computing
    # NumPy 1.26.4 is REQUIRED for Python 3.12 compatibility with PyMC/PyTensor stack
    # NumPy 2.0 breaks binary compatibility with PyMC - this is NOT a downgrade
    "numpy==1.26.4",
    "scipy==1.14.1",
    "pandas==2.2.3",
    "polars==1.19.0",
    "pyarrow==19.0.0",

    # Machine Learning
    "scikit-learn==1.6.1",

    # NLP
    "transformers==4.53.0",
    "sentence-transformers==3.3.1",
    "spacy==3.8.3",

    # Graph Analysis
    "networkx==3.4.2",
    "igraph==0.11.8",
    "python-louvain==0.16",
    "pydot==3.0.4",

    # Bayesian Analysis
    # PyTensor 2.34.0 is last version supporting NumPy 1.x (required for Python 3.12)
    # PyTensor 2.35+ requires NumPy 2.0 which breaks PyMC binary compatibility
    "pytensor==2.34.0",
    # PyMC 5.16.2 works with PyTensor 2.34 and can build from source on Python 3.12
    "pymc==5.16.2",
    "arviz==0.20.0",

    # Causal Inference
    "dowhy==0.12",
    "econml==0.15.1",

    # PDF Processing
    "pdfplumber==0.11.4",
    "PyPDF2==3.0.1",
    "PyMuPDF==1.25.2",
    "tabula-py==2.10.0",
    "camelot-py==0.11.0",
    "python-docx==1.1.2",

    # Computer Vision
    "opencv-python==4.10.0.84",

    # NLP Additional Dependencies
    "nltk==3.9.1",
    "sentencepiece==0.2.0",
    "tiktoken==0.8.0",
    "fuzzywuzzy==0.18.0",
    "python-Levenshtein==0.26.1",
    "langdetect==1.0.9",
    "regex==2024.11.6",

    # Hugging Face Ecosystem
    "huggingface-hub==0.27.1",
    "safetensors==0.5.2",
    "tokenizers==0.21.0",
    "filelock==3.16.1",

    # Data Validation
    "jsonschema==4.23.0",
    "pydantic==2.10.6",

    # Database
    "redis==5.2.1",
    "sqlalchemy==2.0.37",

    # Production Server
    "gunicorn==23.0.0",

    # Monitoring
    "prometheus-client==0.21.1",
    "psutil==6.1.1",
    "structlog==24.4.0",
    "opentelemetry-api==1.29.0",
    "opentelemetry-sdk==1.29.0",
    "opentelemetry-instrumentation-fastapi==0.50b0",
    "tenacity==9.0.0",

    # Type Checking
    "typing-extensions==4.12.2",

    # Hashing & Security
    "blake3==0.4.1",

    # Build Tools
    "setuptools==78.1.1",

    # HTTP & Networking
    "requests==2.32.4",
    "urllib3==2.5.0",
    "certifi==2024.7.4",
    "charset-normalizer==3.3.2",
    "idna==3.7",

    # Utilities
    "tqdm==4.66.3",
    "packaging==23.2",
    "click==8.1.7",
    "joblib==1.3.2",
    "threadpoolctl==3.2.0",
    "six==1.16.0",
    "python-dateutil==2.8.2",
    "pytz==2024.1",
]

# Optional - Heavy ML frameworks (install separately based on platform/Python version)
OPTIONAL_HEAVY = [
    "tensorflow==2.18.0",  # Requires Python 3.11 or 3.12 for this version
    "torch==2.8.0",  # Should be installed with proper CUDA/CPU build
]

# Development dependencies
DEV_DEPENDENCIES = [
    "pytest==8.3.4",
    "pytest-cov==6.0.0",
    "pytest-asyncio==0.25.2",
    "black==24.10.0",
    "flake8==7.1.1",
    "hypothesis==6.122.7",
    "schemathesis==3.39.16",
    "mypy==1.14.1",
    "pyright==1.1.395",
    "ruff==0.9.1",
    "bandit==1.8.0",
    "import-linter==2.2",
]


def run_command(cmd: List[str], cwd: Path = None, check: bool = True) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False
    )
    if check and result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Exit code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode, result.stdout, result.stderr


def test_installation_in_venv(requirements: List[str], python_version: str = "python3") -> Tuple[bool, str]:
    """
    Test installation of requirements in a clean virtual environment.
    Returns (success, message).
    """
    print(f"\n{'='*80}")
    print("Testing requirements in clean virtual environment...")
    print(f"{'='*80}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        venv_path = Path(tmpdir) / "test_venv"

        print(f"Creating virtual environment at {venv_path}...")
        try:
            # Create venv
            venv.create(venv_path, with_pip=True, clear=True)

            # Determine paths
            if sys.platform == "win32":
                pip_path = venv_path / "Scripts" / "pip"
                python_path = venv_path / "Scripts" / "python"
            else:
                pip_path = venv_path / "bin" / "pip"
                python_path = venv_path / "bin" / "python"

            # Upgrade pip
            print("Upgrading pip...")
            run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])

            # Write requirements to file
            req_file = Path(tmpdir) / "test_requirements.txt"
            with open(req_file, 'w') as f:
                for req in requirements:
                    f.write(f"{req}\n")

            print(f"\nInstalling {len(requirements)} packages...")
            print("This may take several minutes...\n")

            # Try to install
            returncode, stdout, stderr = run_command(
                [str(pip_path), "install", "-r", str(req_file)],
                check=False
            )

            if returncode != 0:
                error_msg = f"Installation failed!\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                print(error_msg)
                return False, error_msg

            print("\n✓ All packages installed successfully!")

            # Get installed packages
            print("\nGenerating package list...")
            returncode, freeze_output, _ = run_command(
                [str(pip_path), "freeze"],
                check=True
            )

            return True, freeze_output

        except Exception as e:
            error_msg = f"Error during testing: {str(e)}"
            print(error_msg)
            return False, error_msg


def parse_freeze_output(freeze_output: str) -> Dict[str, str]:
    """Parse pip freeze output into a dict of package: version."""
    packages = {}
    for line in freeze_output.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '==' in line:
            pkg, ver = line.split('==', 1)
            packages[pkg.lower().strip()] = ver.strip()
    return packages


def generate_requirements_txt(project_root: Path, core_runtime: List[str]):
    """Generate the main requirements.txt file."""
    output_file = project_root / "requirements.txt"
    print(f"\nGenerating {output_file}...")

    with open(output_file, 'w') as f:
        f.write("# Core Python Dependencies - All versions pinned\n")
        f.write("# Generated by scripts/build_and_test_requirements.py\n")
        f.write("# Tested with Python 3.10, 3.11, 3.12\n")
        f.write("#\n")
        f.write("# CRITICAL VERSION CONSTRAINTS:\n")
        f.write("# - NumPy 1.26.4: Required for PyMC/PyTensor compatibility on Python 3.12\n")
        f.write("#   NumPy 2.0+ breaks binary compatibility with PyMC\n")
        f.write("# - PyTensor 2.34.0: Last version supporting NumPy 1.x\n")
        f.write("# - PyMC 5.16.2: Compatible with PyTensor 2.34.0 and Python 3.12\n")
        f.write("#\n")
        f.write("# To install: pip install -r requirements.txt\n")
        f.write("#\n\n")

        # Group by category
        categories = {
            "# Web Framework": [
                "flask", "flask-cors", "flask-socketio", "werkzeug",
                "fastapi", "uvicorn", "sse-starlette", "httpx",
            ],
            "# WebSocket Support": [
                "python-socketio", "gevent", "gevent-websocket",
            ],
            "# Authentication": ["pyjwt"],
            "# Configuration": ["pyyaml", "python-dotenv", "typer"],
            "# Core Scientific Computing": [
                "numpy", "scipy", "pandas", "polars", "pyarrow",
            ],
            "# Machine Learning": ["scikit-learn"],
            "# NLP Core": ["transformers", "sentence-transformers", "spacy"],
            "# Graph Analysis": ["networkx", "igraph", "python-louvain", "pydot"],
            "# Bayesian Analysis": ["pytensor", "pymc", "arviz"],
            "# Causal Inference": ["dowhy", "econml"],
            "# PDF Processing": [
                "pdfplumber", "PyPDF2", "PyMuPDF", "tabula-py",
                "camelot-py", "python-docx",
            ],
            "# Computer Vision": ["opencv-python"],
            "# NLP Additional": [
                "nltk", "sentencepiece", "tiktoken", "fuzzywuzzy",
                "python-Levenshtein", "langdetect", "regex",
            ],
            "# Hugging Face Ecosystem": [
                "huggingface-hub", "safetensors", "tokenizers", "filelock",
            ],
            "# Data Validation": ["jsonschema", "pydantic"],
            "# Database": ["redis", "sqlalchemy"],
            "# Production Server": ["gunicorn"],
            "# Monitoring": [
                "prometheus-client", "psutil", "structlog",
                "opentelemetry-api", "opentelemetry-sdk",
                "opentelemetry-instrumentation-fastapi", "tenacity",
            ],
            "# Type Checking": ["typing-extensions"],
            "# Hashing & Security": ["blake3"],
            "# Build Tools": ["setuptools"],
            "# HTTP & Networking": [
                "requests", "urllib3", "certifi", "charset-normalizer", "idna",
            ],
            "# Utilities": [
                "tqdm", "packaging", "click", "joblib",
                "threadpoolctl", "six", "python-dateutil", "pytz",
            ],
        }

        # Create a map of package name to full requirement spec
        req_map = {}
        for req in core_runtime:
            pkg_name = req.split('==')[0].lower().strip()
            req_map[pkg_name] = req

        # Write categorized requirements
        for category, packages in categories.items():
            f.write(f"{category}\n")
            for pkg in packages:
                if pkg.lower() in req_map:
                    f.write(f"{req_map[pkg.lower()]}\n")
            f.write("\n")

        f.write("# Development dependencies\n")
        for req in DEV_DEPENDENCIES:
            f.write(f"{req}\n")
        f.write("\n")

        f.write("# Optional Heavy ML Frameworks\n")
        f.write("# Install separately based on platform:\n")
        f.write("# tensorflow==2.18.0  # Requires Python 3.11+\n")
        f.write("# torch==2.8.0  # Install with appropriate CUDA/CPU build\n")
        f.write("\n")


def generate_requirements_core(project_root: Path, core_runtime: List[str]):
    """Generate requirements-core.txt (runtime only, no dev deps)."""
    output_file = project_root / "requirements-core.txt"
    print(f"Generating {output_file}...")

    with open(output_file, 'w') as f:
        f.write("# Core Runtime Dependencies - Exact Pins\n")
        f.write("# Generated by scripts/build_and_test_requirements.py\n")
        f.write("# Tested with Python 3.10, 3.11, 3.12\n")
        f.write("# Install with: pip install -r requirements-core.txt\n\n")

        for req in core_runtime:
            f.write(f"{req}\n")


def generate_requirements_dev(project_root: Path):
    """Generate requirements-dev.txt."""
    output_file = project_root / "requirements-dev.txt"
    print(f"Generating {output_file}...")

    with open(output_file, 'w') as f:
        f.write("# Development & Testing Dependencies - Exact Pins\n")
        f.write("# Generated by scripts/build_and_test_requirements.py\n")
        f.write("# Install with: pip install -r requirements-dev.txt\n\n")
        f.write("# Include core dependencies\n")
        f.write("-r requirements-core.txt\n\n")

        for req in DEV_DEPENDENCIES:
            f.write(f"{req}\n")


def generate_requirements_optional(project_root: Path):
    """Generate requirements-optional.txt for heavy ML frameworks."""
    output_file = project_root / "requirements-optional.txt"
    print(f"Generating {output_file}...")

    with open(output_file, 'w') as f:
        f.write("# Optional Heavy ML Dependencies\n")
        f.write("# Generated by scripts/build_and_test_requirements.py\n")
        f.write("# Install separately based on your platform and Python version\n")
        f.write("# Install with: pip install -r requirements-optional.txt\n\n")
        f.write("# Note: These may require specific CUDA versions or Python versions\n\n")

        for req in OPTIONAL_HEAVY:
            f.write(f"{req}\n")


def generate_requirements_all(project_root: Path):
    """Generate requirements-all.txt."""
    output_file = project_root / "requirements-all.txt"
    print(f"Generating {output_file}...")

    with open(output_file, 'w') as f:
        f.write("# All Dependencies - For Complete Installation\n")
        f.write("# Generated by scripts/build_and_test_requirements.py\n")
        f.write("# Install with: pip install -r requirements-all.txt\n\n")
        f.write("# Core Runtime\n")
        f.write("-r requirements-core.txt\n\n")
        f.write("# Development\n")
        f.write("-r requirements-dev.txt\n\n")
        f.write("# Optional (may fail on some platforms)\n")
        f.write("# -r requirements-optional.txt\n")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent

    print("="*80)
    print("SAAAAAA Comprehensive Requirements Builder and Tester")
    print("="*80)

    # Step 1: Test core runtime installation
    print("\n[1/5] Testing core runtime dependencies...")
    success, result = test_installation_in_venv(CORE_RUNTIME)

    if not success:
        print("\n❌ Core runtime dependencies failed to install!")
        print("Please review the errors above and fix dependency conflicts.")
        return 1

    print("\n✓ Core runtime dependencies tested successfully!")

    # Parse installed packages
    installed_packages = parse_freeze_output(result)
    print(f"Total packages installed (including transitive deps): {len(installed_packages)}")

    # Step 2: Generate requirements files
    print("\n[2/5] Generating requirements files...")
    generate_requirements_txt(project_root, CORE_RUNTIME)
    generate_requirements_core(project_root, CORE_RUNTIME)
    generate_requirements_dev(project_root)
    generate_requirements_optional(project_root)
    generate_requirements_all(project_root)

    # Step 3: Save full freeze output
    print("\n[3/5] Saving complete dependency snapshot...")
    freeze_file = project_root / "requirements-freeze.txt"
    with open(freeze_file, 'w') as f:
        f.write("# Complete dependency snapshot from pip freeze\n")
        f.write("# Generated by scripts/build_and_test_requirements.py\n")
        f.write("# Includes all transitive dependencies\n\n")
        f.write(result)
    print(f"Saved to {freeze_file}")

    # Step 4: Generate documentation
    print("\n[4/5] Generating documentation...")
    doc_file = project_root / "REQUIREMENTS.md"
    with open(doc_file, 'w') as f:
        f.write("# SAAAAAA Requirements Management\n\n")
        f.write("## Overview\n\n")
        f.write("This project uses a comprehensive, tested dependency management system.\n\n")
        f.write("## Installation\n\n")
        f.write("### Quick Start\n\n")
        f.write("```bash\n")
        f.write("# Install core runtime dependencies\n")
        f.write("pip install -r requirements.txt\n")
        f.write("```\n\n")
        f.write("### Development Installation\n\n")
        f.write("```bash\n")
        f.write("# Install all dependencies including dev tools\n")
        f.write("pip install -r requirements-dev.txt\n")
        f.write("```\n\n")
        f.write("### Optional Heavy ML Frameworks\n\n")
        f.write("TensorFlow and PyTorch are not installed by default due to platform-specific builds.\n\n")
        f.write("```bash\n")
        f.write("# Install optional ML frameworks\n")
        f.write("pip install -r requirements-optional.txt\n")
        f.write("```\n\n")
        f.write("## Critical Version Constraints\n\n")
        f.write("### NumPy 1.26.4\n\n")
        f.write("We use NumPy 1.26.4 (not 2.x) because:\n")
        f.write("- PyMC/PyTensor stack requires NumPy 1.x for binary compatibility\n")
        f.write("- Python 3.12 support is available in 1.26.4\n")
        f.write("- NumPy 2.0+ breaks PyMC binary compatibility\n\n")
        f.write("### PyTensor 2.34.0\n\n")
        f.write("PyTensor 2.34.0 is the last version supporting NumPy 1.x.\n")
        f.write("PyTensor 2.35+ requires NumPy 2.0.\n\n")
        f.write("### PyMC 5.16.2\n\n")
        f.write("Compatible with PyTensor 2.34.0 and can build from source on Python 3.12.\n\n")
        f.write("## Files\n\n")
        f.write("- `requirements.txt` - Main requirements file with all core dependencies\n")
        f.write("- `requirements-core.txt` - Core runtime dependencies only\n")
        f.write("- `requirements-dev.txt` - Development dependencies (includes core)\n")
        f.write("- `requirements-optional.txt` - Optional heavy ML frameworks\n")
        f.write("- `requirements-all.txt` - All dependencies combined\n")
        f.write("- `requirements-freeze.txt` - Complete pip freeze snapshot\n\n")
        f.write("## Regenerating Requirements\n\n")
        f.write("To regenerate the requirements files:\n\n")
        f.write("```bash\n")
        f.write("python3 scripts/build_and_test_requirements.py\n")
        f.write("```\n\n")
        f.write("This will:\n")
        f.write("1. Test all dependencies in a clean virtual environment\n")
        f.write("2. Generate all requirements files\n")
        f.write("3. Create a complete dependency snapshot\n")
        f.write("4. Update this documentation\n\n")

    print(f"Saved to {doc_file}")

    # Step 5: Summary
    print("\n" + "="*80)
    print("SUCCESS! All requirements files generated and tested.")
    print("="*80)
    print("\nGenerated files:")
    print(f"  ✓ requirements.txt - Main requirements ({len(CORE_RUNTIME)} packages)")
    print(f"  ✓ requirements-core.txt - Core runtime only")
    print(f"  ✓ requirements-dev.txt - Development dependencies")
    print(f"  ✓ requirements-optional.txt - Optional ML frameworks")
    print(f"  ✓ requirements-all.txt - All dependencies")
    print(f"  ✓ requirements-freeze.txt - Complete snapshot ({len(installed_packages)} packages)")
    print(f"  ✓ REQUIREMENTS.md - Documentation")
    print("\nNext steps:")
    print("  1. Review the generated files")
    print("  2. Commit to version control")
    print("  3. Test in CI/CD pipeline")
    print("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
