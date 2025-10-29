"""
Tests for Document Ingestion Module
====================================

These tests verify the document ingestion pipeline including:
- PDF loading and validation
- Text extraction
- Sentence segmentation
- Table extraction and classification
- Index building
- PreprocessedDocument assembly
"""

import os
import tempfile
import unittest
from pathlib import Path

# Try to import reportlab for PDF creation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from ingestion.document_ingestion import (
    DocumentLoader,
    TextExtractor,
    PreprocessingEngine,
    PreprocessedDocument,
    RawDocument,
    IngestionError,
    ValidationError,
    PDFLoadError,
    TextExtractionError,
    PreprocessingError,
)


class TestDocumentLoader(unittest.TestCase):
    """Test DocumentLoader class."""
    
    @classmethod
    def setUpClass(cls):
        """Create a test PDF file."""
        if not REPORTLAB_AVAILABLE:
            cls.test_pdf_path = None
            return
        
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_pdf_path = os.path.join(cls.temp_dir, "test_document.pdf")
        
        # Create a simple test PDF
        c = canvas.Canvas(cls.test_pdf_path, pagesize=letter)
        c.setTitle("Test Municipal Development Plan")
        c.setAuthor("Test Author")
        
        # Add some text
        c.drawString(100, 750, "Plan de Desarrollo Municipal 2024-2028")
        c.drawString(100, 720, "")
        c.drawString(100, 690, "Introducción")
        c.drawString(100, 660, "Este es un plan de desarrollo municipal de prueba.")
        c.drawString(100, 630, "Contiene varias secciones y objetivos estratégicos.")
        
        # Add second page
        c.showPage()
        c.drawString(100, 750, "Objetivos Estratégicos")
        c.drawString(100, 720, "1. Mejorar la infraestructura vial.")
        c.drawString(100, 690, "2. Fortalecer la educación pública.")
        c.drawString(100, 660, "3. Promover el desarrollo económico local.")
        
        c.save()
        
        cls.empty_pdf_path = os.path.join(cls.temp_dir, "empty.pdf")
        Path(cls.empty_pdf_path).touch()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if REPORTLAB_AVAILABLE and hasattr(cls, 'temp_dir'):
            import shutil
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_load_valid_pdf(self):
        """Test loading a valid PDF document."""
        if not REPORTLAB_AVAILABLE or not self.test_pdf_path:
            self.skipTest("reportlab not available")
        
        raw_doc = DocumentLoader.load_pdf(self.test_pdf_path)
        
        self.assertIsInstance(raw_doc, RawDocument)
        self.assertIsInstance(raw_doc.pdf_bytes, bytes)
        self.assertGreater(len(raw_doc.pdf_bytes), 0)
        self.assertEqual(raw_doc.num_pages, 2)
        self.assertIsInstance(raw_doc.metadata, dict)
        self.assertIsInstance(raw_doc.file_hash, str)
        self.assertEqual(len(raw_doc.file_hash), 64)  # SHA256 hex digest
    
    def test_load_nonexistent_pdf(self):
        """Test loading a non-existent PDF."""
        with self.assertRaises(PDFLoadError):
            DocumentLoader.load_pdf("/nonexistent/file.pdf")
    
    def test_load_empty_pdf(self):
        """Test loading an empty PDF file."""
        if not REPORTLAB_AVAILABLE or not self.empty_pdf_path:
            self.skipTest("reportlab not available")
        
        with self.assertRaises(ValidationError):
            DocumentLoader.load_pdf(self.empty_pdf_path)
    
    def test_pdf_hash_consistency(self):
        """Test that PDF hash is consistent."""
        if not REPORTLAB_AVAILABLE or not self.test_pdf_path:
            self.skipTest("reportlab not available")
        
        raw_doc1 = DocumentLoader.load_pdf(self.test_pdf_path)
        raw_doc2 = DocumentLoader.load_pdf(self.test_pdf_path)
        
        self.assertEqual(raw_doc1.file_hash, raw_doc2.file_hash)


class TestTextExtractor(unittest.TestCase):
    """Test TextExtractor class."""
    
    @classmethod
    def setUpClass(cls):
        """Create test PDF."""
        if not REPORTLAB_AVAILABLE:
            cls.test_pdf_path = None
            return
        
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_pdf_path = os.path.join(cls.temp_dir, "test_text.pdf")
        
        c = canvas.Canvas(cls.test_pdf_path, pagesize=letter)
        c.drawString(100, 750, "Primera línea de texto.")
        c.drawString(100, 720, "Segunda línea de texto.")
        c.save()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if REPORTLAB_AVAILABLE and hasattr(cls, 'temp_dir'):
            import shutil
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_extract_text_from_pdf(self):
        """Test extracting text from PDF."""
        if not REPORTLAB_AVAILABLE or not self.test_pdf_path:
            self.skipTest("reportlab not available")
        
        raw_doc = DocumentLoader.load_pdf(self.test_pdf_path)
        text = TextExtractor.extract_full_text(raw_doc)
        
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        # Check that some expected text is present
        self.assertIn("línea", text.lower())


class TestPreprocessingEngine(unittest.TestCase):
    """Test PreprocessingEngine class."""
    
    def test_normalize_encoding(self):
        """Test text encoding normalization."""
        text = "Test   text\n\n\nwith  multiple    spaces\n\nand lines."
        normalized = PreprocessingEngine.normalize_encoding(text)
        
        self.assertIsInstance(normalized, str)
        # Should normalize multiple spaces
        self.assertNotIn("   ", normalized)
        # Should preserve some structure
        self.assertIn("Test", normalized)
    
    def test_segment_into_sentences(self):
        """Test sentence segmentation."""
        engine = PreprocessingEngine()
        text = "Primera oración. Segunda oración. Tercera oración."
        
        sentences = engine.segment_into_sentences(text)
        
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 0)
        
        # Check sentence structure
        for sent in sentences:
            self.assertIsInstance(sent.text, str)
            self.assertIsInstance(sent.start_offset, int)
            self.assertIsInstance(sent.end_offset, int)
            self.assertGreaterEqual(sent.end_offset, sent.start_offset)
    
    def test_classify_tables_presupuesto(self):
        """Test table classification for budget tables."""
        from ingestion.document_ingestion import TableData
        
        table = TableData(
            table_index=0,
            table_type='unknown',
            data=[
                {'concepto': 'Infraestructura', 'presupuesto': '1000000'},
                {'concepto': 'Educación', 'presupuesto': '500000'}
            ],
            page_number=1,
            confidence=1.0,
            metadata={'headers': ['concepto', 'presupuesto']}
        )
        
        classified = PreprocessingEngine.classify_tables([table])
        
        self.assertEqual(len(classified), 1)
        self.assertEqual(classified[0].table_type, 'presupuesto')
        self.assertGreater(classified[0].confidence, 0.5)
    
    def test_classify_tables_cronograma(self):
        """Test table classification for schedule tables."""
        from ingestion.document_ingestion import TableData
        
        table = TableData(
            table_index=0,
            table_type='unknown',
            data=[
                {'actividad': 'Fase 1', 'fecha': '2024-01-01'},
                {'actividad': 'Fase 2', 'fecha': '2024-06-01'}
            ],
            page_number=1,
            confidence=1.0,
            metadata={'headers': ['actividad', 'fecha']}
        )
        
        classified = PreprocessingEngine.classify_tables([table])
        
        self.assertEqual(len(classified), 1)
        self.assertEqual(classified[0].table_type, 'cronograma')
    
    def test_build_indexes(self):
        """Test building structural indexes."""
        from ingestion.document_ingestion import SentenceSegment
        
        sentences = [
            SentenceSegment(
                text="El presupuesto es de 1000000 pesos.",
                start_offset=0,
                end_offset=36,
                sentence_index=0
            ),
            SentenceSegment(
                text="La fecha límite es 2024-12-31.",
                start_offset=37,
                end_offset=68,
                sentence_index=1
            )
        ]
        
        full_text = "El presupuesto es de 1000000 pesos. La fecha límite es 2024-12-31."
        
        indexes = PreprocessingEngine.build_indexes(sentences, [], full_text)
        
        self.assertIsInstance(indexes.term_index, dict)
        self.assertIsInstance(indexes.numeric_index, dict)
        self.assertIsInstance(indexes.temporal_index, dict)
        self.assertIsInstance(indexes.table_index, dict)
        
        # Check that some terms are indexed
        self.assertIn('presupuesto', indexes.term_index)
        
        # Check numeric index
        self.assertGreater(len(indexes.numeric_index), 0)
        
        # Check temporal index
        self.assertGreater(len(indexes.temporal_index), 0)


class TestPreprocessedDocument(unittest.TestCase):
    """Test PreprocessedDocument assembly."""
    
    @classmethod
    def setUpClass(cls):
        """Create test PDF."""
        if not REPORTLAB_AVAILABLE:
            cls.test_pdf_path = None
            return
        
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_pdf_path = os.path.join(cls.temp_dir, "test_full.pdf")
        
        c = canvas.Canvas(cls.test_pdf_path, pagesize=letter)
        c.setTitle("Plan Municipal de Desarrollo")
        
        # Add content
        c.drawString(100, 750, "Plan de Desarrollo Municipal 2024-2028")
        c.drawString(100, 720, "")
        c.drawString(100, 690, "Objetivos Estratégicos")
        c.drawString(100, 660, "1. Mejorar infraestructura. Construir 50 km de vías.")
        c.drawString(100, 630, "2. Fortalecer educación. Beneficiar 10000 estudiantes.")
        
        c.save()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if REPORTLAB_AVAILABLE and hasattr(cls, 'temp_dir'):
            import shutil
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        if not REPORTLAB_AVAILABLE or not self.test_pdf_path:
            self.skipTest("reportlab not available")
        
        # Load PDF
        raw_doc = DocumentLoader.load_pdf(self.test_pdf_path)
        
        # Preprocess
        engine = PreprocessingEngine()
        preprocessed_doc = engine.preprocess_document(raw_doc)
        
        # Verify result
        self.assertIsInstance(preprocessed_doc, PreprocessedDocument)
        self.assertIsInstance(preprocessed_doc.document_id, str)
        self.assertGreater(len(preprocessed_doc.document_id), 0)
        
        self.assertIsInstance(preprocessed_doc.raw_text, str)
        self.assertGreater(len(preprocessed_doc.raw_text), 0)
        
        self.assertIsInstance(preprocessed_doc.normalized_text, str)
        self.assertGreater(len(preprocessed_doc.normalized_text), 0)
        
        self.assertIsInstance(preprocessed_doc.sentences, list)
        self.assertGreater(len(preprocessed_doc.sentences), 0)
        
        self.assertIsInstance(preprocessed_doc.tables, list)
        # Tables may be empty if none detected
        
        self.assertIsInstance(preprocessed_doc.indexes.term_index, dict)
        self.assertIsInstance(preprocessed_doc.indexes.numeric_index, dict)
        self.assertIsInstance(preprocessed_doc.indexes.temporal_index, dict)
        self.assertIsInstance(preprocessed_doc.indexes.table_index, dict)
        
        self.assertIsInstance(preprocessed_doc.metadata, dict)
        self.assertIn('file_path', preprocessed_doc.metadata)
        self.assertIn('file_hash', preprocessed_doc.metadata)
        self.assertIn('num_pages', preprocessed_doc.metadata)
        self.assertIn('num_sentences', preprocessed_doc.metadata)
    
    def test_preprocessed_document_immutability(self):
        """Test that PreprocessedDocument is immutable."""
        if not REPORTLAB_AVAILABLE or not self.test_pdf_path:
            self.skipTest("reportlab not available")
        
        raw_doc = DocumentLoader.load_pdf(self.test_pdf_path)
        engine = PreprocessingEngine()
        preprocessed_doc = engine.preprocess_document(raw_doc)
        
        # Try to modify - should raise AttributeError (frozen dataclass)
        with self.assertRaises(AttributeError):
            preprocessed_doc.document_id = "new_id"


class TestErrorHandling(unittest.TestCase):
    """Test error handling and explicit abortability."""
    
    def test_pdf_load_error_on_invalid_file(self):
        """Test that invalid PDF raises PDFLoadError."""
        # Create a text file, not a PDF
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            f.write("This is not a PDF file")
            temp_path = f.name
        
        try:
            with self.assertRaises((PDFLoadError, ValidationError)):
                DocumentLoader.load_pdf(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_preprocessing_error_propagation(self):
        """Test that preprocessing errors propagate correctly."""
        # This tests that errors in the pipeline are not silently caught
        
        # Create an invalid RawDocument
        from ingestion.document_ingestion import RawDocument
        
        invalid_doc = RawDocument(
            pdf_bytes=b'',  # Empty bytes
            num_pages=0,
            metadata={},
            file_hash='test',
            file_path='/invalid/path'
        )
        
        engine = PreprocessingEngine()
        
        # Should raise an error, not silently degrade
        with self.assertRaises((TextExtractionError, PreprocessingError)):
            engine.preprocess_document(invalid_doc)


if __name__ == '__main__':
    unittest.main()
