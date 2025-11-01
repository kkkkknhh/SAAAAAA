"""
Setup configuration for SAAAAAA package.
Allows installation with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="saaaaaa",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires="~=3.11.0",
    install_requires=[
        # Read from requirements.txt
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
)
