"""
Setup configuration for SAAAAAA package.
Allows installation with: pip install -e .
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read requirements from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
install_requires = []
if requirements_file.exists():
    with open(requirements_file, encoding="utf-8") as f:
        install_requires = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

setup(
    name="saaaaaa",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires="~=3.11.0",
    install_requires=install_requires,
)
