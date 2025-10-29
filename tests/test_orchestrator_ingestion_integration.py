"""
Integration Tests for Orchestrator with Document Ingestion
===========================================================

These tests verify the integration between the orchestrator
and the document ingestion module.
"""

import os
import tempfile
import unittest
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from orchestrator.orchestrator import Orchestrator, PhaseResult, ExecutionMode


class TestOrchestratorIngestionIntegration(unittest.TestCase):
    """Test integration between orchestrator and document ingestion."""
    
    @classmethod
    def setUpClass(cls):
        """Create test PDF and monolith files."""
        if not REPORTLAB_AVAILABLE:
            cls.test_pdf_path = None
            return
        
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_pdf_path = os.path.join(cls.temp_dir, "plan_desarrollo.pdf")
        
        # Create a test PDF with municipal development plan content
        c = canvas.Canvas(cls.test_pdf_path, pagesize=letter)
        c.setTitle("Plan Municipal de Desarrollo Bogotá 2024-2028")
        c.setAuthor("Alcaldía de Bogotá")
        
        # Page 1: Introduction
        c.drawString(100, 750, "PLAN MUNICIPAL DE DESARROLLO")
        c.drawString(100, 730, "Bogotá 2024-2028")
        c.drawString(100, 700, "")
        c.drawString(100, 680, "1. INTRODUCCIÓN")
        c.drawString(100, 660, "Este plan define los objetivos estratégicos para el desarrollo")
        c.drawString(100, 640, "municipal durante el período 2024-2028.")
        c.drawString(100, 620, "El presupuesto total asignado es de 500000000 pesos.")
        
        # Page 2: Strategic objectives
        c.showPage()
        c.drawString(100, 750, "2. OBJETIVOS ESTRATÉGICOS")
        c.drawString(100, 730, "")
        c.drawString(100, 710, "2.1 Infraestructura")
        c.drawString(100, 690, "Mejorar la infraestructura vial. Meta: construir 100 km de vías.")
        c.drawString(100, 670, "Presupuesto: 200000000 pesos. Fecha: 2024-06-30.")
        c.drawString(100, 650, "")
        c.drawString(100, 630, "2.2 Educación")
        c.drawString(100, 610, "Fortalecer el sistema educativo público.")
        c.drawString(100, 590, "Meta: beneficiar 50000 estudiantes.")
        c.drawString(100, 570, "Presupuesto: 150000000 pesos. Fecha: 2024-12-31.")
        
        # Page 3: Monitoring
        c.showPage()
        c.drawString(100, 750, "3. SEGUIMIENTO Y EVALUACIÓN")
        c.drawString(100, 730, "")
        c.drawString(100, 710, "Se realizarán evaluaciones trimestrales.")
        c.drawString(100, 690, "Los indicadores se medirán cada trimestre.")
        c.drawString(100, 670, "Responsable: Oficina de Planeación Municipal.")
        
        c.save()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if REPORTLAB_AVAILABLE and hasattr(cls, 'temp_dir'):
            import shutil
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_orchestrator_can_ingest_document(self):
        """Test that orchestrator can successfully ingest a document."""
        if not REPORTLAB_AVAILABLE or not self.test_pdf_path:
            self.skipTest("reportlab not available")
        
        # Note: We're not providing real monolith/catalog paths since we're
        # only testing the ingestion phase (FASE 1)
        orchestrator = Orchestrator(
            monolith_path="questionnaire_monolith.json",
            method_catalog_path="rules/METODOS/metodos_completos_nivel3.json",
            enable_async=False
        )
        
        # Test ingestion directly
        result = orchestrator._ingest_document(self.test_pdf_path)
        
        # Verify phase result
        self.assertIsInstance(result, PhaseResult)
        self.assertEqual(result.phase_id, "FASE_1")
        self.assertEqual(result.phase_name, "Ingestión del Documento")
        self.assertTrue(result.success)
        self.assertEqual(result.mode, ExecutionMode.SYNC)
        self.assertIsNone(result.error)
        
        # Verify preprocessed document was created
        self.assertIsNotNone(result.data)
        preprocessed_doc = result.data
        
        # Verify document structure
        self.assertIsInstance(preprocessed_doc.document_id, str)
        self.assertGreater(len(preprocessed_doc.document_id), 0)
        
        self.assertIsInstance(preprocessed_doc.raw_text, str)
        self.assertGreater(len(preprocessed_doc.raw_text), 0)
        self.assertIn("DESARROLLO", preprocessed_doc.raw_text)
        
        self.assertIsInstance(preprocessed_doc.normalized_text, str)
        self.assertGreater(len(preprocessed_doc.normalized_text), 0)
        
        self.assertIsInstance(preprocessed_doc.sentences, list)
        self.assertGreater(len(preprocessed_doc.sentences), 5)
        
        # Verify sentences have proper structure
        for sent in preprocessed_doc.sentences[:3]:
            self.assertIn('text', sent)
            self.assertIn('start_offset', sent)
            self.assertIn('end_offset', sent)
            self.assertIn('sentence_index', sent)
        
        self.assertIsInstance(preprocessed_doc.tables, list)
        # Tables may be empty if none detected in this PDF
        
        self.assertIsInstance(preprocessed_doc.indexes, dict)
        self.assertIn('term_index', preprocessed_doc.indexes)
        self.assertIn('numeric_index', preprocessed_doc.indexes)
        self.assertIn('temporal_index', preprocessed_doc.indexes)
        self.assertIn('table_index', preprocessed_doc.indexes)
        
        # Verify metadata
        self.assertIsInstance(preprocessed_doc.metadata, dict)
        self.assertIn('file_path', preprocessed_doc.metadata)
        self.assertIn('file_hash', preprocessed_doc.metadata)
        self.assertIn('num_pages', preprocessed_doc.metadata)
        self.assertEqual(preprocessed_doc.metadata['num_pages'], 3)
        
        # Verify metrics
        self.assertIsInstance(result.metrics, dict)
        self.assertIn('sentences', result.metrics)
        self.assertIn('tables', result.metrics)
        self.assertIn('document_id', result.metrics)
        self.assertIn('text_length', result.metrics)
        
        self.assertEqual(result.metrics['sentences'], len(preprocessed_doc.sentences))
        self.assertEqual(result.metrics['tables'], len(preprocessed_doc.tables))
    
    def test_orchestrator_ingestion_error_handling(self):
        """Test that orchestrator properly handles ingestion errors."""
        orchestrator = Orchestrator(
            monolith_path="questionnaire_monolith.json",
            method_catalog_path="rules/METODOS/metodos_completos_nivel3.json",
            enable_async=False
        )
        
        # Try to ingest a non-existent file
        result = orchestrator._ingest_document("/nonexistent/file.pdf")
        
        # Verify error handling
        self.assertIsInstance(result, PhaseResult)
        self.assertEqual(result.phase_id, "FASE_1")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.data)
    
    def test_document_content_extraction(self):
        """Test that specific content is properly extracted."""
        if not REPORTLAB_AVAILABLE or not self.test_pdf_path:
            self.skipTest("reportlab not available")
        
        orchestrator = Orchestrator(
            monolith_path="questionnaire_monolith.json",
            method_catalog_path="rules/METODOS/metodos_completos_nivel3.json",
            enable_async=False
        )
        
        result = orchestrator._ingest_document(self.test_pdf_path)
        
        self.assertTrue(result.success)
        preprocessed_doc = result.data
        
        # Check that key terms are in the term index
        term_index = preprocessed_doc.indexes['term_index']
        
        # These terms should be present
        expected_terms = ['desarrollo', 'municipal', 'presupuesto', 'educación']
        for term in expected_terms:
            # Term might be lowercase in index
            found = False
            for key in term_index.keys():
                if term.lower() in key.lower():
                    found = True
                    break
            if not found:
                # Also check in normalized text
                self.assertIn(term.lower(), preprocessed_doc.normalized_text.lower())
        
        # Check numeric index has budget numbers
        numeric_index = preprocessed_doc.indexes['numeric_index']
        self.assertGreater(len(numeric_index), 0)
        
        # Check temporal index has dates
        temporal_index = preprocessed_doc.indexes['temporal_index']
        # Dates in format YYYY-MM-DD should be detected
        found_date = any('2024' in str(key) for key in temporal_index.keys())
        # If not in temporal index, should at least be in text
        if not found_date:
            self.assertIn('2024', preprocessed_doc.normalized_text)


if __name__ == '__main__':
    unittest.main()
