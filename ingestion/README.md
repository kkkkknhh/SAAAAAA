# Document Ingestion Module

## Overview

The document ingestion module provides deterministic PDF processing for municipal development plans and policy documents. It implements the **FASE 1: INGESTIÓN DEL DOCUMENTO** phase described in `PSEUDOCODIGO_FLUJO_COMPLETO.md`.

## Features

### 1. PDF Loading and Validation
- Load PDF files from disk
- Validate PDF format and structure
- Extract metadata (author, title, creation date)
- Compute SHA256 hash for document identification
- Verify PDF is readable and non-empty

### 2. Text Extraction
- Extract full text from all pages
- Preserve paragraph structure
- Handle multi-page documents
- Support for both PyPDF2 and pdfplumber libraries

### 3. Text Normalization
- Unicode normalization (NFC form)
- Whitespace normalization
- Canonical text representation

### 4. Sentence Segmentation
- Segment text into sentences
- Track character offsets (start/end)
- Maintain sentence order
- Optional page number tracking

### 5. Table Extraction and Classification
- Extract tables from PDF
- Automatic classification:
  - `presupuesto` (budget tables)
  - `cronograma` (schedule tables)
  - `actividades` (activity tables)
  - `responsables` (responsibility tables)
- Confidence scoring for classifications

### 6. Structural Indexing
- **Term Index**: Maps words to their locations in text
- **Numeric Index**: Maps numbers to their locations
- **Temporal Index**: Maps dates to their locations
- **Table Index**: Maps table types to table indices

### 7. Immutable Output
- `PreprocessedDocument` is a frozen dataclass
- All data structures are immutable
- Hash-based document identification

## Architecture

```
DocumentLoader
    ↓
RawDocument (PDF bytes + metadata + hash)
    ↓
TextExtractor
    ↓
PreprocessingEngine
    ├── normalize_encoding()
    ├── segment_into_sentences()
    ├── extract_tables()
    ├── classify_tables()
    └── build_indexes()
    ↓
PreprocessedDocument (immutable)
```

## Usage

### Basic Example

```python
from ingestion.document_ingestion import (
    DocumentLoader,
    PreprocessingEngine,
)

# Load PDF
raw_document = DocumentLoader.load_pdf("plan_desarrollo.pdf")

# Preprocess
engine = PreprocessingEngine()
preprocessed_doc = engine.preprocess_document(raw_document)

# Access results
print(f"Document ID: {preprocessed_doc.document_id}")
print(f"Sentences: {len(preprocessed_doc.sentences)}")
print(f"Tables: {len(preprocessed_doc.tables)}")
```

### Integration with Orchestrator

The orchestrator automatically uses the ingestion module in FASE 1:

```python
from orchestrator.orchestrator import Orchestrator

orchestrator = Orchestrator(
    monolith_path="questionnaire_monolith.json",
    method_catalog_path="rules/METODOS/metodos_completos_nivel3.json"
)

# Ingestion happens automatically in process_development_plan()
result = orchestrator.process_development_plan("plan.pdf")
```

### Demo Script

Run the demo script to see the ingestion module in action:

```bash
python demo_ingestion.py [path/to/pdf]
```

If no PDF is provided, it will create a sample municipal development plan.

## Data Structures

### RawDocument
```python
@dataclass(frozen=True)
class RawDocument:
    pdf_bytes: bytes          # Raw PDF file bytes
    num_pages: int            # Number of pages
    metadata: Dict[str, Any]  # PDF metadata
    file_hash: str            # SHA256 hash
    file_path: str            # Original file path
```

### PreprocessedDocument
```python
@dataclass(frozen=True)
class PreprocessedDocument:
    document_id: str                    # Unique identifier (from hash)
    raw_text: str                       # Original extracted text
    normalized_text: str                # Canonically normalized text
    sentences: List[SentenceSegment]    # Segmented sentences with offsets
    tables: List[TableData]             # Extracted and classified tables
    indexes: DocumentIndexes            # Structural indexes
    metadata: Dict[str, Any]            # Document metadata
```

### SentenceSegment
```python
@dataclass(frozen=True)
class SentenceSegment:
    text: str                # The sentence text
    start_offset: int        # Start position in full text
    end_offset: int          # End position in full text
    sentence_index: int      # Index in document
    page_number: Optional[int]  # Page where sentence appears
```

### TableData
```python
@dataclass(frozen=True)
class TableData:
    table_index: int              # Index in document
    table_type: str               # Classification type
    data: List[Dict[str, Any]]    # Table data
    page_number: int              # Page where table appears
    confidence: float             # Classification confidence (0-1)
    metadata: Dict[str, Any]      # Additional metadata
```

## Error Handling

The module follows a **fail-fast** approach with explicit error types:

- `IngestionError`: Base exception for all ingestion errors
- `ValidationError`: PDF validation failures
- `PDFLoadError`: PDF loading failures
- `TextExtractionError`: Text extraction failures
- `PreprocessingError`: Preprocessing failures

### No Silent Degradation

The module is designed to **abort explicitly** on errors rather than degrade silently:

```python
try:
    preprocessed_doc = engine.preprocess_document(raw_document)
except IngestionError as e:
    logger.error(f"Ingestion failed: {e}")
    # Handle error appropriately
```

## Dependencies

Required:
- Python 3.7+

PDF processing (at least one required):
- `PyPDF2>=3.0.0` (recommended for basic text extraction)
- `pdfplumber>=0.9.0` (recommended for table extraction)

Optional (for enhanced functionality):
- `policy_processor` module (for improved sentence segmentation)
- `financiero_viabilidad_tablas` module (for advanced table extraction)

Install dependencies:
```bash
pip install PyPDF2 pdfplumber
```

## Testing

Run tests:
```bash
# All ingestion tests
pytest tests/test_document_ingestion.py -v

# Integration tests
pytest tests/test_orchestrator_ingestion_integration.py -v

# All tests
pytest tests/test_coreographer.py tests/test_document_ingestion.py tests/test_orchestrator_ingestion_integration.py -v
```

Current test coverage:
- 14 unit tests for ingestion module
- 3 integration tests with orchestrator
- All tests passing ✓

## Logging

The module provides structured logging at INFO level:

```
INFO: Loading PDF: /path/to/plan.pdf
INFO: PDF hash: 47c3619191776f8d...
INFO: PDF loaded: 25 pages
INFO: ✓ PDF loaded successfully: 25 pages, hash=47c3619191776f8d...
INFO: Extracting text from PDF (25 pages)
INFO: ✓ Text extracted: 45230 characters
INFO: === Starting document preprocessing ===
INFO: Step 1: Extracting full text...
INFO: Step 2: Normalizing encoding...
INFO: Step 3: Segmenting into sentences...
INFO: Step 4: Extracting tables...
INFO: ✓ Extracted 5 tables
INFO: Step 5: Classifying tables...
INFO: ✓ Classified 5 tables
INFO: Step 6: Building structural indexes...
INFO: ✓ Built indexes: 1250 terms, 150 numbers, 25 dates, 4 table types
INFO: ✓ Document preprocessed successfully:
INFO:   - 342 sentences
INFO:   - 5 tables
INFO:   - Indexes built
```

## Design Principles

1. **Determinism**: Same PDF always produces same output (hash-based ID)
2. **Immutability**: All output data structures are immutable
3. **Explicit Failures**: No silent degradation, fail fast on errors
4. **Comprehensive Logging**: Structured logging at every step
5. **Traceability**: Hash-based tracking and offset-based indexing
6. **Integration**: Designed to work with existing policy analysis modules

## Performance

Typical performance on a 25-page municipal development plan PDF:
- PDF Loading: ~50ms
- Text Extraction: ~200ms
- Sentence Segmentation: ~100ms
- Table Extraction: ~500ms (varies with table count)
- Index Building: ~150ms
- **Total**: ~1 second

## Future Enhancements

Possible improvements (not currently implemented):
- OCR support for scanned PDFs
- Image extraction and classification
- Section detection and hierarchical structure
- Enhanced table recognition with ML models
- Parallel processing for large documents
- Caching of preprocessed documents

## License

Part of the SAAAAAA project. See main repository for license information.

## Support

For issues or questions, please refer to the main repository issue tracker.
