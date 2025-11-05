"""
Setup configuration for SAAAAAA package.

SAAAAAA is a Strategic Policy Analysis System that integrates 584 analytical
methods across 300 policy evaluation questions using a chess-based orchestration
strategy with 7 producer modules and 1 aggregator.

Installation:
    pip install -e .

Usage:
    python3 -m saaaaaa.core.orchestrator --input plan.pdf --mode full

For more information, see README.md and OPERATIONAL_GUIDE.md
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

# Read requirements from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
install_requires = []
if requirements_file.exists():
    with open(requirements_file, encoding="utf-8") as f:
        install_requires = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="saaaaaa",
    version="0.1.0",
    description="Strategic Policy Analysis System - Doctoral-level integration of 584 methods across 300 questions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SAAAAAA Development Team",
    url="https://github.com/kkkkknhh/SAAAAAA",
    project_urls={
        "Bug Tracker": "https://github.com/kkkkknhh/SAAAAAA/issues",
        "Documentation": "https://github.com/kkkkknhh/SAAAAAA#readme",
        "Source Code": "https://github.com/kkkkknhh/SAAAAAA",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=24.3.0",
            "flake8>=7.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
            "hypothesis>=6.92.2",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "saaaaaa=saaaaaa.core.orchestrator:main",
            "saaaaaa-api=saaaaaa.api.api_server:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    keywords=[
        "policy analysis",
        "bayesian inference",
        "causal inference",
        "natural language processing",
        "machine learning",
        "municipal planning",
        "development plans",
        "evidential reasoning",
        "theory of change",
        "semantic analysis",
    ],
    include_package_data=True,
    zip_safe=False,
)
