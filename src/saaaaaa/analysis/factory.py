"""
Factory Layer for Analysis Module I/O Operations

This module provides centralized I/O operations for the analysis package,
implementing a clean separation between I/O and business logic following
the Ports and Adapters (Hexagonal Architecture) pattern.

All file I/O for the analysis package should be handled through this factory.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import spacy
except ImportError:
    spacy = None

logger = logging.getLogger(__name__)


# ============================================================================
# JSON I/O OPERATIONS
# ============================================================================

def load_json(file_path: str | Path) -> dict[str, Any]:
    """
    Load JSON data from file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dict containing the loaded JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded JSON from {file_path}")
    return data


def save_json(data: dict[str, Any], file_path: str | Path, indent: int = 2) -> None:
    """
    Save data to JSON file with formatted output.

    Args:
        data: Dictionary to save
        file_path: Path to output JSON file
        indent: Indentation level for formatting
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    logger.info(f"Saved JSON to {file_path}")


# ============================================================================
# YAML I/O OPERATIONS
# ============================================================================

def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """
    Load YAML data from file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dict containing the loaded YAML data

    Raises:
        ImportError: If PyYAML is not installed
        FileNotFoundError: If file doesn't exist
    """
    if yaml is None:
        raise ImportError("PyYAML is required for YAML operations. Install with: pip install pyyaml")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    logger.info(f"Loaded YAML from {file_path}")
    return data


# ============================================================================
# TEXT FILE I/O OPERATIONS
# ============================================================================

def read_text_file(file_path: str | Path) -> str:
    """
    Read text file with UTF-8 encoding.

    Args:
        file_path: Path to text file

    Returns:
        String content of the file

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    logger.debug(f"Read {len(content)} characters from {file_path}")
    return content


def write_text_file(content: str, file_path: str | Path) -> None:
    """
    Write text content to file with UTF-8 encoding.

    Args:
        content: Text content to write
        file_path: Path to output file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Written {len(content)} characters to {file_path}")


# ============================================================================
# CSV I/O OPERATIONS
# ============================================================================

def write_csv(rows: list[list[Any]], file_path: str | Path, headers: list[str] = None) -> None:
    """
    Write data to CSV file.

    Args:
        rows: List of rows to write
        file_path: Path to output CSV file
        headers: Optional list of column headers
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if headers:
            writer.writerow(headers)

        writer.writerows(rows)

    logger.info(f"Written {len(rows)} rows to CSV {file_path}")


# ============================================================================
# PDF OPERATIONS
# ============================================================================

def open_pdf_with_fitz(file_path: str | Path):
    """
    Open a PDF file using PyMuPDF (fitz).

    Args:
        file_path: Path to PDF file

    Returns:
        fitz.Document object

    Raises:
        ImportError: If PyMuPDF is not installed
        FileNotFoundError: If file doesn't exist
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    return fitz.open(file_path)


def open_pdf_with_pdfplumber(file_path: str | Path):
    """
    Open a PDF file using pdfplumber.

    Args:
        file_path: Path to PDF file

    Returns:
        pdfplumber.PDF object

    Raises:
        ImportError: If pdfplumber is not installed
        FileNotFoundError: If file doesn't exist
    """
    if pdfplumber is None:
        raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    return pdfplumber.open(file_path)


# ============================================================================
# SPACY MODEL LOADING
# ============================================================================

def load_spacy_model(model_name: str):
    """
    Load a spaCy language model.

    Args:
        model_name: Name of the spaCy model to load

    Returns:
        Loaded spaCy Language object

    Raises:
        ImportError: If spaCy is not installed
        OSError: If model is not found
    """
    if spacy is None:
        raise ImportError("spaCy is required. Install with: pip install spacy")

    try:
        nlp = spacy.load(model_name)
        logger.info(f"Loaded spaCy model: {model_name}")
        return nlp
    except OSError:
        logger.error(f"spaCy model '{model_name}' not found. Download with: python -m spacy download {model_name}")
        raise
