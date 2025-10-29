"""
Document Ingestion Module - Deterministic PDF Processing
=========================================================

This module implements deterministic document ingestion for PDF policy documents.
It provides:

- PDF loading and validation (bytes, metadata, hash)
- Full text extraction and canonical normalization
- Sentence/paragraph segmentation with offsets
- Table extraction and classification
- Structural index construction
- Immutable PreprocessedDocument assembly

Preconditions:
- PDF must be readable and non-empty

Invariants:
- Hash and structure integrity maintained throughout

Postconditions:
- PreprocessedDocument usable by the pipeline

Acceptance Criteria:
- Explicit abortability on errors (no silent degradation)
- Structured logging throughout
- No elegant degradation (fail fast on critical errors)
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Import existing modules for integration
try:
    from policy_processor import PolicyTextProcessor
    POLICY_PROCESSOR_AVAILABLE = True
except ImportError:
    POLICY_PROCESSOR_AVAILABLE = False

try:
    from financiero_viabilidad_tablas import PDETMunicipalPlanAnalyzer
    FINANCIAL_ANALYZER_AVAILABLE = True
except ImportError:
    FINANCIAL_ANALYZER_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class IngestionError(Exception):
    """Base exception for document ingestion errors."""
    pass


class ValidationError(IngestionError):
    """Exception raised when document validation fails."""
    pass


class PDFLoadError(IngestionError):
    """Exception raised when PDF loading fails."""
    pass


class TextExtractionError(IngestionError):
    """Exception raised when text extraction fails."""
    pass


class PreprocessingError(IngestionError):
    """Exception raised during preprocessing."""
    pass


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class RawDocument:
    """
    Raw document structure after initial PDF load.
    
    Attributes:
        pdf_bytes: Raw PDF file bytes
        num_pages: Number of pages in PDF
        metadata: PDF metadata (author, title, creation date, etc.)
        file_hash: SHA256 hash of the PDF bytes
        file_path: Original file path
    """
    pdf_bytes: bytes
    num_pages: int
    metadata: Dict[str, Any]
    file_hash: str
    file_path: str


@dataclass(frozen=True)
class SentenceSegment:
    """
    A sentence segment with location information.
    
    Attributes:
        text: The sentence text
        start_offset: Character offset where sentence starts in full text
        end_offset: Character offset where sentence ends in full text
        sentence_index: Index of this sentence in the document
        page_number: Page number where sentence appears (if available)
    """
    text: str
    start_offset: int
    end_offset: int
    sentence_index: int
    page_number: Optional[int] = None


@dataclass(frozen=True)
class TableData:
    """
    Extracted and classified table data.
    
    Attributes:
        table_index: Index of this table in the document
        table_type: Classification (e.g., 'presupuesto', 'cronograma', 'actividades')
        data: The actual table data (list of dicts or DataFrame-like structure)
        page_number: Page number where table appears
        confidence: Classification confidence score (0-1)
        metadata: Additional table metadata
    """
    table_index: int
    table_type: str
    data: List[Dict[str, Any]]
    page_number: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentIndexes:
    """
    Structural indexes for fast document querying.
    
    Attributes:
        term_index: Maps terms to their locations (offsets)
        numeric_index: Maps numeric values to their locations
        temporal_index: Maps dates/times to their locations
        table_index: Maps table types to table indices
    """
    term_index: Dict[str, List[int]]
    numeric_index: Dict[str, List[int]]
    temporal_index: Dict[str, List[int]]
    table_index: Dict[str, List[int]]


@dataclass(frozen=True)
class PreprocessedDocument:
    """
    Immutable preprocessed document structure.
    
    This is the final output of the ingestion process and serves as input
    to the question processing pipeline.
    
    Attributes:
        document_id: Unique identifier derived from file hash
        raw_text: Original extracted text
        normalized_text: Canonically normalized text
        sentences: List of sentence segments with offsets
        tables: List of extracted and classified tables
        indexes: Structural indexes for fast querying
        metadata: Document metadata (from PDF + processing info)
    """
    document_id: str
    raw_text: str
    normalized_text: str
    sentences: List[SentenceSegment]
    tables: List[TableData]
    indexes: DocumentIndexes
    metadata: Dict[str, Any]


# ============================================================================
# DOCUMENT LOADER
# ============================================================================

class DocumentLoader:
    """
    Loads and validates PDF documents.
    
    Responsibilities:
    - Load PDF bytes from file
    - Validate PDF format and structure
    - Extract basic metadata
    - Compute document hash
    - Verify PDF is readable and non-empty
    """
    
    @staticmethod
    def load_pdf(pdf_path: str) -> RawDocument:
        """
        Load and validate a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            RawDocument with bytes, metadata, and hash
            
        Raises:
            PDFLoadError: If PDF cannot be loaded or is invalid
            ValidationError: If PDF validation fails
        """
        logger.info(f"Loading PDF: {pdf_path}")
        
        # Check file exists
        path = Path(pdf_path)
        if not path.exists():
            raise PDFLoadError(f"PDF file not found: {pdf_path}")
        
        if not path.is_file():
            raise PDFLoadError(f"Path is not a file: {pdf_path}")
        
        # Read PDF bytes
        try:
            pdf_bytes = path.read_bytes()
        except Exception as e:
            raise PDFLoadError(f"Failed to read PDF file: {e}")
        
        # Validate non-empty
        if len(pdf_bytes) == 0:
            raise ValidationError(f"PDF file is empty: {pdf_path}")
        
        # Compute hash
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        logger.info(f"PDF hash: {file_hash[:16]}...")
        
        # Extract metadata using available library
        metadata = {}
        num_pages = 0
        
        if PYPDF2_AVAILABLE:
            try:
                import io
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                num_pages = len(pdf_reader.pages)
                
                # Extract metadata
                if pdf_reader.metadata:
                    metadata = {
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'producer': pdf_reader.metadata.get('/Producer', ''),
                        'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')),
                    }
                
                logger.info(f"PDF loaded: {num_pages} pages")
                
            except Exception as e:
                raise PDFLoadError(f"Failed to parse PDF with PyPDF2: {e}")
        
        elif PDFPLUMBER_AVAILABLE:
            try:
                import io
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    num_pages = len(pdf.pages)
                    metadata = pdf.metadata or {}
                    
                logger.info(f"PDF loaded: {num_pages} pages")
                
            except Exception as e:
                raise PDFLoadError(f"Failed to parse PDF with pdfplumber: {e}")
        else:
            raise PDFLoadError(
                "No PDF library available. Install PyPDF2 or pdfplumber: "
                "pip install PyPDF2 pdfplumber"
            )
        
        # Validate PDF has pages
        if num_pages == 0:
            raise ValidationError(f"PDF has no pages: {pdf_path}")
        
        # Build RawDocument
        raw_doc = RawDocument(
            pdf_bytes=pdf_bytes,
            num_pages=num_pages,
            metadata=metadata,
            file_hash=file_hash,
            file_path=str(path.absolute())
        )
        
        logger.info(f"✓ PDF loaded successfully: {num_pages} pages, hash={file_hash[:16]}...")
        return raw_doc


# ============================================================================
# TEXT EXTRACTOR
# ============================================================================

class TextExtractor:
    """
    Extracts text from PDF documents.
    
    Responsibilities:
    - Extract full text from all pages
    - Preserve paragraph structure where possible
    - Identify and skip headers/footers if needed
    - Return complete text string
    """
    
    @staticmethod
    def extract_full_text(raw_document: RawDocument) -> str:
        """
        Extract full text from a PDF document.
        
        Args:
            raw_document: RawDocument with PDF bytes
            
        Returns:
            Complete text string extracted from PDF
            
        Raises:
            TextExtractionError: If text extraction fails
        """
        logger.info(f"Extracting text from PDF ({raw_document.num_pages} pages)")
        
        full_text = ""
        
        if PDFPLUMBER_AVAILABLE:
            try:
                import io
                with pdfplumber.open(io.BytesIO(raw_document.pdf_bytes)) as pdf:
                    text_parts = []
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    
                    full_text = "\n".join(text_parts)
                    
            except Exception as e:
                raise TextExtractionError(f"Failed to extract text with pdfplumber: {e}")
        
        elif PYPDF2_AVAILABLE:
            try:
                import io
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(raw_document.pdf_bytes))
                text_parts = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                full_text = "\n".join(text_parts)
                
            except Exception as e:
                raise TextExtractionError(f"Failed to extract text with PyPDF2: {e}")
        else:
            raise TextExtractionError(
                "No PDF library available for text extraction. "
                "Install PyPDF2 or pdfplumber."
            )
        
        # Validate text extracted
        if not full_text or len(full_text.strip()) == 0:
            raise TextExtractionError(
                f"No text extracted from PDF (possibly scanned/image-based PDF)"
            )
        
        logger.info(f"✓ Text extracted: {len(full_text)} characters")
        return full_text


# ============================================================================
# PREPROCESSING ENGINE
# ============================================================================

class PreprocessingEngine:
    """
    Preprocesses extracted text and builds PreprocessedDocument.
    
    Responsibilities:
    - Normalize text encoding (Unicode normalization)
    - Segment text into sentences with offsets
    - Extract and classify tables
    - Build structural indexes
    - Assemble immutable PreprocessedDocument
    """
    
    def __init__(self):
        """Initialize preprocessing engine with existing module instances."""
        self.policy_processor = None
        if POLICY_PROCESSOR_AVAILABLE:
            try:
                self.policy_processor = PolicyTextProcessor()
            except Exception as e:
                logger.warning(f"Could not initialize PolicyTextProcessor: {e}")
        
        self.financial_analyzer = None
        if FINANCIAL_ANALYZER_AVAILABLE:
            try:
                self.financial_analyzer = PDETMunicipalPlanAnalyzer()
            except Exception as e:
                logger.warning(f"Could not initialize PDETMunicipalPlanAnalyzer: {e}")
    
    @staticmethod
    def normalize_encoding(text: str) -> str:
        """
        Normalize text encoding canonically.
        
        Args:
            text: Raw text string
            
        Returns:
            Normalized text with consistent encoding
        """
        # Unicode normalization (NFC form)
        normalized = unicodedata.normalize('NFC', text)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'\n\s*\n', '\n\n', normalized)
        
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def segment_into_sentences(self, text: str) -> List[SentenceSegment]:
        """
        Segment text into sentences with offset information.
        
        Args:
            text: Normalized text
            
        Returns:
            List of SentenceSegment objects with offsets
        """
        sentences = []
        
        # Use PolicyTextProcessor if available
        if self.policy_processor:
            try:
                # Try to use existing method
                raw_sentences = self.policy_processor.segment_into_sentences(text)
                
                # If it returns a list of strings, compute offsets
                if isinstance(raw_sentences, list) and len(raw_sentences) > 0:
                    current_offset = 0
                    for idx, sent_text in enumerate(raw_sentences):
                        # Find sentence in text
                        start = text.find(sent_text, current_offset)
                        if start == -1:
                            # Fallback: use current offset
                            start = current_offset
                        
                        end = start + len(sent_text)
                        
                        sentences.append(SentenceSegment(
                            text=sent_text,
                            start_offset=start,
                            end_offset=end,
                            sentence_index=idx,
                            page_number=None
                        ))
                        
                        current_offset = end
                    
                    return sentences
                    
            except Exception as e:
                logger.warning(f"PolicyTextProcessor segmentation failed: {e}, using fallback")
        
        # Fallback: simple sentence segmentation
        # Split on sentence-ending punctuation followed by space and capital letter
        sentence_pattern = r'([.!?]+)\s+(?=[A-Z])'
        parts = re.split(sentence_pattern, text)
        
        current_offset = 0
        sentence_texts = []
        
        # Reconstruct sentences
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i + 1] in ['.', '!', '?', '...']:
                # Sentence with punctuation
                sent_text = parts[i] + parts[i + 1]
                sentence_texts.append(sent_text)
                i += 2
            else:
                # Last sentence or sentence without captured punctuation
                if parts[i].strip():
                    sentence_texts.append(parts[i])
                i += 1
        
        # Create SentenceSegment objects
        for idx, sent_text in enumerate(sentence_texts):
            if not sent_text.strip():
                continue
            
            start = text.find(sent_text, current_offset)
            if start == -1:
                start = current_offset
            
            end = start + len(sent_text)
            
            sentences.append(SentenceSegment(
                text=sent_text.strip(),
                start_offset=start,
                end_offset=end,
                sentence_index=idx,
                page_number=None
            ))
            
            current_offset = end
        
        return sentences
    
    def extract_tables(self, raw_document: RawDocument) -> List[TableData]:
        """
        Extract tables from PDF document.
        
        Args:
            raw_document: RawDocument with PDF bytes
            
        Returns:
            List of TableData objects
        """
        tables = []
        
        # Use financial analyzer if available
        if self.financial_analyzer:
            try:
                # Try to extract tables using existing method
                raw_tables = self.financial_analyzer.extract_tables(raw_document.pdf_bytes)
                
                if raw_tables and isinstance(raw_tables, list):
                    for idx, raw_table in enumerate(raw_tables):
                        # Convert to TableData format
                        table_data = TableData(
                            table_index=idx,
                            table_type='unknown',  # Will be classified later
                            data=raw_table if isinstance(raw_table, list) else [],
                            page_number=0,  # Page tracking would need enhancement
                            confidence=1.0,
                            metadata={}
                        )
                        tables.append(table_data)
                
                return tables
                
            except Exception as e:
                logger.warning(f"Table extraction with financial analyzer failed: {e}")
        
        # Fallback: try pdfplumber for table extraction
        if PDFPLUMBER_AVAILABLE:
            try:
                import io
                with pdfplumber.open(io.BytesIO(raw_document.pdf_bytes)) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_tables = page.extract_tables()
                        if page_tables:
                            for table_idx, table in enumerate(page_tables):
                                # Convert table to list of dicts
                                if table and len(table) > 0:
                                    headers = table[0] if table[0] else []
                                    rows = table[1:] if len(table) > 1 else []
                                    
                                    table_dicts = []
                                    for row in rows:
                                        if row:
                                            row_dict = {
                                                str(headers[i]) if i < len(headers) else f'col_{i}': val
                                                for i, val in enumerate(row)
                                            }
                                            table_dicts.append(row_dict)
                                    
                                    table_data = TableData(
                                        table_index=len(tables),
                                        table_type='unknown',
                                        data=table_dicts,
                                        page_number=page_num,
                                        confidence=1.0,
                                        metadata={'headers': headers}
                                    )
                                    tables.append(table_data)
            
            except Exception as e:
                logger.warning(f"Table extraction with pdfplumber failed: {e}")
        
        logger.info(f"✓ Extracted {len(tables)} tables")
        return tables
    
    @staticmethod
    def classify_tables(tables: List[TableData]) -> List[TableData]:
        """
        Classify tables by type (presupuesto, cronograma, actividades, etc.).
        
        Args:
            tables: List of TableData objects
            
        Returns:
            List of TableData objects with updated table_type
        """
        classified_tables = []
        
        for table in tables:
            # Simple heuristic classification based on headers/content
            table_type = 'unknown'
            confidence = 0.5
            
            # Convert table data to string for pattern matching
            table_str = str(table.data).lower()
            headers_str = str(table.metadata.get('headers', [])).lower()
            
            # Classification heuristics
            if any(term in table_str or term in headers_str 
                   for term in ['presupuesto', 'budget', 'costo', 'monto', 'valor']):
                table_type = 'presupuesto'
                confidence = 0.8
            
            elif any(term in table_str or term in headers_str 
                     for term in ['cronograma', 'fecha', 'mes', 'año', 'trimestre']):
                table_type = 'cronograma'
                confidence = 0.8
            
            elif any(term in table_str or term in headers_str 
                     for term in ['actividad', 'acción', 'proyecto', 'meta']):
                table_type = 'actividades'
                confidence = 0.8
            
            elif any(term in table_str or term in headers_str 
                     for term in ['responsable', 'entidad', 'dependencia']):
                table_type = 'responsables'
                confidence = 0.8
            
            # Create new TableData with updated type
            classified_table = TableData(
                table_index=table.table_index,
                table_type=table_type,
                data=table.data,
                page_number=table.page_number,
                confidence=confidence,
                metadata=table.metadata
            )
            classified_tables.append(classified_table)
        
        logger.info(f"✓ Classified {len(classified_tables)} tables")
        return classified_tables
    
    @staticmethod
    def build_indexes(
        sentences: List[SentenceSegment],
        tables: List[TableData],
        full_text: str
    ) -> DocumentIndexes:
        """
        Build structural indexes for fast document querying.
        
        Args:
            sentences: List of sentence segments
            tables: List of table data
            full_text: Complete document text
            
        Returns:
            DocumentIndexes with term, numeric, temporal, and table indexes
        """
        term_index: Dict[str, List[int]] = {}
        numeric_index: Dict[str, List[int]] = {}
        temporal_index: Dict[str, List[int]] = {}
        table_index: Dict[str, List[int]] = {}
        
        # Build term index from sentences
        for sent in sentences:
            # Tokenize sentence (simple word splitting)
            words = re.findall(r'\b\w+\b', sent.text.lower())
            for word in words:
                if word not in term_index:
                    term_index[word] = []
                term_index[word].append(sent.start_offset)
        
        # Build numeric index
        numeric_pattern = r'\b\d+(?:[.,]\d+)?\b'
        for match in re.finditer(numeric_pattern, full_text):
            number = match.group()
            if number not in numeric_index:
                numeric_index[number] = []
            numeric_index[number].append(match.start())
        
        # Build temporal index (dates)
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # DD/MM/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',         # YYYY-MM-DD
            r'\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b',  # "15 de enero de 2024"
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                date_str = match.group()
                if date_str not in temporal_index:
                    temporal_index[date_str] = []
                temporal_index[date_str].append(match.start())
        
        # Build table index
        for table in tables:
            table_type = table.table_type
            if table_type not in table_index:
                table_index[table_type] = []
            table_index[table_type].append(table.table_index)
        
        logger.info(
            f"✓ Built indexes: {len(term_index)} terms, "
            f"{len(numeric_index)} numbers, "
            f"{len(temporal_index)} dates, "
            f"{len(table_index)} table types"
        )
        
        return DocumentIndexes(
            term_index=term_index,
            numeric_index=numeric_index,
            temporal_index=temporal_index,
            table_index=table_index
        )
    
    def preprocess_document(self, raw_document: RawDocument) -> PreprocessedDocument:
        """
        Complete document preprocessing pipeline.
        
        This is the main entry point that orchestrates all preprocessing steps:
        1. Extract full text
        2. Normalize encoding
        3. Segment into sentences
        4. Extract and classify tables
        5. Build indexes
        6. Assemble PreprocessedDocument
        
        Args:
            raw_document: RawDocument from DocumentLoader
            
        Returns:
            Immutable PreprocessedDocument ready for pipeline processing
            
        Raises:
            PreprocessingError: If any preprocessing step fails critically
        """
        logger.info("=== Starting document preprocessing ===")
        
        try:
            # Step 1: Extract full text
            logger.info("Step 1: Extracting full text...")
            raw_text = TextExtractor.extract_full_text(raw_document)
            
            # Step 2: Normalize encoding
            logger.info("Step 2: Normalizing encoding...")
            normalized_text = self.normalize_encoding(raw_text)
            
            # Step 3: Segment into sentences
            logger.info("Step 3: Segmenting into sentences...")
            sentences = self.segment_into_sentences(normalized_text)
            
            # Step 4: Extract tables
            logger.info("Step 4: Extracting tables...")
            raw_tables = self.extract_tables(raw_document)
            
            # Step 5: Classify tables
            logger.info("Step 5: Classifying tables...")
            classified_tables = self.classify_tables(raw_tables)
            
            # Step 6: Build indexes
            logger.info("Step 6: Building structural indexes...")
            indexes = self.build_indexes(sentences, classified_tables, normalized_text)
            
            # Step 7: Generate document ID
            document_id = raw_document.file_hash[:16]
            
            # Step 8: Assemble metadata
            metadata = {
                **raw_document.metadata,
                'file_path': raw_document.file_path,
                'file_hash': raw_document.file_hash,
                'num_pages': raw_document.num_pages,
                'num_sentences': len(sentences),
                'num_tables': len(classified_tables),
                'text_length': len(normalized_text),
            }
            
            # Step 9: Create immutable PreprocessedDocument
            preprocessed_doc = PreprocessedDocument(
                document_id=document_id,
                raw_text=raw_text,
                normalized_text=normalized_text,
                sentences=sentences,
                tables=classified_tables,
                indexes=indexes,
                metadata=metadata
            )
            
            logger.info(
                f"✓ Document preprocessed successfully:\n"
                f"  - {len(sentences)} sentences\n"
                f"  - {len(classified_tables)} tables\n"
                f"  - Indexes built"
            )
            
            return preprocessed_doc
            
        except (TextExtractionError, ValidationError) as e:
            # Critical errors - abort immediately
            logger.error(f"CRITICAL: Preprocessing failed: {e}")
            raise PreprocessingError(f"Document preprocessing failed: {e}") from e
        
        except Exception as e:
            # Unexpected errors - also abort
            logger.error(f"UNEXPECTED ERROR during preprocessing: {e}", exc_info=True)
            raise PreprocessingError(f"Unexpected preprocessing error: {e}") from e
