#!/usr/bin/env python3
"""
Comprehensive Dependency Scanner for SAAAAAA Project
Scans all Python files to extract actual import statements and map them to PyPI packages.
"""

import ast
import sys
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List

# Map of import names to PyPI package names (when they differ)
IMPORT_TO_PACKAGE = {
    'cv2': 'opencv-python',
    'yaml': 'pyyaml',
    'dotenv': 'python-dotenv',
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'bs4': 'beautifulsoup4',
    'jose': 'python-jose',
    'docx': 'python-docx',
    'fitz': 'PyMuPDF',
    'camelot': 'camelot-py',
    'pytensor': 'pytensor',
    'pymc': 'pymc',
    'louvain': 'python-louvain',
    'redis': 'redis',
    'sqlalchemy': 'sqlalchemy',
}

# Standard library modules (Python 3.11) - should NOT be in requirements
STDLIB_MODULES = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore',
    'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'bisect', 'builtins',
    'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs',
    'codeop', 'collections', 'colorsys', 'compileall', 'concurrent', 'configparser',
    'contextlib', 'contextvars', 'copy', 'copyreg', 'crypt', 'csv', 'ctypes',
    'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis',
    'distutils', 'doctest', 'email', 'encodings', 'enum', 'errno', 'faulthandler',
    'fcntl', 'filecmp', 'fileinput', 'fnmatch', 'fractions', 'ftplib', 'functools',
    'gc', 'getopt', 'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip',
    'hashlib', 'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib', 'imghdr',
    'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools', 'json',
    'keyword', 'lib2to3', 'linecache', 'locale', 'logging', 'lzma', 'mailbox',
    'mailcap', 'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder', 'multiprocessing',
    'netrc', 'nis', 'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev',
    'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform',
    'plistlib', 'poplib', 'posix', 'posixpath', 'pprint', 'profile', 'pstats',
    'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri', 'random',
    're', 'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched',
    'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal',
    'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd',
    'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct',
    'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny',
    'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap', 'threading',
    'time', 'timeit', 'tkinter', 'token', 'tokenize', 'tomllib', 'trace', 'traceback',
    'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types', 'typing', 'unicodedata',
    'unittest', 'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref',
    'webbrowser', 'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc',
    'zipapp', 'zipfile', 'zipimport', 'zlib', '_thread',
}


class ImportScanner(ast.NodeVisitor):
    """AST visitor to extract all import statements."""

    def __init__(self):
        self.imports: Set[str] = set()
        self.import_locations: Dict[str, List[str]] = defaultdict(list)

    def visit_Import(self, node):
        """Handle 'import x' statements."""
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            self.imports.add(module_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Handle 'from x import y' statements."""
        if node.module:
            module_name = node.module.split('.')[0]
            self.imports.add(module_name)
        self.generic_visit(node)


def scan_file(file_path: Path) -> Set[str]:
    """Scan a single Python file for imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        scanner = ImportScanner()
        scanner.visit(tree)
        return scanner.imports
    except (SyntaxError, UnicodeDecodeError, PermissionError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return set()


def scan_directory(root_dir: Path) -> Dict[str, Set[Path]]:
    """
    Scan all Python files in a directory tree.
    Returns a dict mapping package names to files that import them.
    """
    package_to_files: Dict[str, Set[Path]] = defaultdict(set)

    # Exclude certain directories
    exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', '.venv',
                    'env', '.env', 'build', 'dist', '*.egg-info', 'node_modules'}

    python_files = []
    for py_file in root_dir.rglob('*.py'):
        # Check if file is in an excluded directory
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue
        python_files.append(py_file)

    print(f"Scanning {len(python_files)} Python files...")

    for py_file in python_files:
        imports = scan_file(py_file)
        for imp in imports:
            # Skip local imports (they start with the project name or are relative)
            if imp.startswith('saaaaaa') or imp.startswith('.'):
                continue

            # Skip standard library
            if imp in STDLIB_MODULES:
                continue

            # Map to PyPI package name
            package_name = IMPORT_TO_PACKAGE.get(imp, imp)
            package_to_files[package_name].add(py_file)

    return package_to_files


def categorize_dependencies(packages: Set[str]) -> Dict[str, List[str]]:
    """Categorize packages into core, dev, optional, etc."""
    categories = {
        'core': [],
        'web': [],
        'ml': [],
        'nlp': [],
        'pdf': [],
        'bayesian': [],
        'dev': [],
        'monitoring': [],
    }

    # Core data processing
    core_packages = {
        'numpy', 'scipy', 'pandas', 'polars', 'pyarrow', 'joblib', 'threadpoolctl'
    }

    # Web frameworks
    web_packages = {
        'flask', 'fastapi', 'uvicorn', 'werkzeug', 'httpx', 'requests', 'urllib3',
        'flask-cors', 'flask-socketio', 'python-socketio', 'gevent', 'gevent-websocket',
        'sse-starlette', 'gunicorn', 'pyjwt'
    }

    # Machine Learning
    ml_packages = {
        'scikit-learn', 'tensorflow', 'torch', 'keras', 'tf-keras'
    }

    # NLP
    nlp_packages = {
        'transformers', 'sentence-transformers', 'spacy', 'nltk', 'sentencepiece',
        'tiktoken', 'fuzzywuzzy', 'python-Levenshtein', 'langdetect', 'regex',
        'huggingface-hub', 'safetensors', 'tokenizers'
    }

    # PDF Processing
    pdf_packages = {
        'pdfplumber', 'PyPDF2', 'PyMuPDF', 'tabula-py', 'camelot-py', 'python-docx',
        'opencv-python'
    }

    # Bayesian
    bayesian_packages = {
        'pytensor', 'pymc', 'arviz', 'dowhy', 'econml'
    }

    # Development
    dev_packages = {
        'pytest', 'pytest-cov', 'pytest-asyncio', 'black', 'flake8', 'mypy',
        'pyright', 'ruff', 'hypothesis', 'bandit', 'schemathesis', 'import-linter',
        'pycycle'
    }

    # Monitoring
    monitoring_packages = {
        'prometheus-client', 'psutil', 'structlog', 'opentelemetry-api',
        'opentelemetry-sdk', 'opentelemetry-instrumentation-fastapi', 'tenacity'
    }

    for pkg in sorted(packages):
        if pkg in core_packages:
            categories['core'].append(pkg)
        elif pkg in web_packages:
            categories['web'].append(pkg)
        elif pkg in ml_packages:
            categories['ml'].append(pkg)
        elif pkg in nlp_packages:
            categories['nlp'].append(pkg)
        elif pkg in pdf_packages:
            categories['pdf'].append(pkg)
        elif pkg in bayesian_packages:
            categories['bayesian'].append(pkg)
        elif pkg in dev_packages:
            categories['dev'].append(pkg)
        elif pkg in monitoring_packages:
            categories['monitoring'].append(pkg)
        else:
            # Uncategorized - add to core by default
            categories['core'].append(pkg)

    return categories


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent

    print(f"Scanning project at: {project_root}")
    print("=" * 80)

    # Scan all Python files
    package_to_files = scan_directory(project_root)

    print(f"\nFound {len(package_to_files)} unique third-party packages")
    print("=" * 80)

    # Categorize packages
    categories = categorize_dependencies(set(package_to_files.keys()))

    # Print results
    for category, packages in categories.items():
        if packages:
            print(f"\n{category.upper()} ({len(packages)} packages):")
            for pkg in sorted(packages):
                file_count = len(package_to_files[pkg])
                print(f"  - {pkg} (used in {file_count} files)")

    # Print all packages in a simple list
    print("\n" + "=" * 80)
    print("ALL PACKAGES (for requirements.txt):")
    print("=" * 80)
    all_packages = sorted(package_to_files.keys())
    for pkg in all_packages:
        print(pkg)

    return 0


if __name__ == '__main__':
    sys.exit(main())
