#!/usr/bin/env python
"""
Demo script for document ingestion module
==========================================

This script demonstrates how to use the document ingestion module
to process a PDF policy document.
"""

import sys
import tempfile
import os
from pathlib import Path

# Try to create a sample PDF if reportlab is available
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not available, using existing PDF if provided")

from ingestion.document_ingestion import (
    DocumentLoader,
    PreprocessingEngine,
    IngestionError,
)


def create_sample_pdf(output_path: str):
    """Create a sample municipal development plan PDF."""
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is required to create sample PDF")
    
    c = canvas.Canvas(output_path, pagesize=letter)
    c.setTitle("Plan de Desarrollo Municipal - Demo")
    c.setAuthor("Sistema de Análisis")
    
    # Page 1
    c.drawString(100, 750, "PLAN DE DESARROLLO MUNICIPAL")
    c.drawString(100, 730, "Municipio de Ejemplo - 2024-2028")
    c.drawString(100, 700, "")
    c.drawString(100, 680, "1. VISIÓN Y OBJETIVOS")
    c.drawString(100, 660, "")
    c.drawString(100, 640, "Este plan establece la hoja de ruta para el desarrollo")
    c.drawString(100, 620, "municipal durante el periodo 2024-2028.")
    c.drawString(100, 600, "")
    c.drawString(100, 580, "Presupuesto total: 750000000 pesos colombianos.")
    c.drawString(100, 560, "Población beneficiada: 125000 habitantes.")
    
    # Page 2
    c.showPage()
    c.drawString(100, 750, "2. LÍNEAS ESTRATÉGICAS")
    c.drawString(100, 730, "")
    c.drawString(100, 710, "2.1. Infraestructura y Movilidad")
    c.drawString(100, 690, "Meta: Construir 75 km de vías pavimentadas.")
    c.drawString(100, 670, "Presupuesto: 300000000 pesos.")
    c.drawString(100, 650, "Plazo: 2024-01-01 a 2026-12-31.")
    c.drawString(100, 630, "")
    c.drawString(100, 610, "2.2. Educación y Cultura")
    c.drawString(100, 590, "Meta: Mejorar cobertura educativa al 95%.")
    c.drawString(100, 570, "Beneficiarios: 35000 estudiantes.")
    c.drawString(100, 550, "Presupuesto: 250000000 pesos.")
    
    # Page 3
    c.showPage()
    c.drawString(100, 750, "3. INDICADORES Y SEGUIMIENTO")
    c.drawString(100, 730, "")
    c.drawString(100, 710, "Se establecen indicadores trimestrales para cada línea.")
    c.drawString(100, 690, "Responsable: Secretaría de Planeación Municipal.")
    c.drawString(100, 670, "")
    c.drawString(100, 650, "Fechas de evaluación:")
    c.drawString(100, 630, "- Primer trimestre: 2024-03-31")
    c.drawString(100, 610, "- Segundo trimestre: 2024-06-30")
    c.drawString(100, 590, "- Tercer trimestre: 2024-09-30")
    c.drawString(100, 570, "- Cuarto trimestre: 2024-12-31")
    
    c.save()
    print(f"✓ Sample PDF created: {output_path}")


def main():
    """Main demo function."""
    print("=" * 70)
    print("DEMO: Document Ingestion Module")
    print("=" * 70)
    print()
    
    # Determine PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Using provided PDF: {pdf_path}")
    else:
        # Create a sample PDF
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, "plan_desarrollo_demo.pdf")
        
        print("No PDF provided, creating sample...")
        try:
            create_sample_pdf(pdf_path)
        except RuntimeError as e:
            print(f"Error: {e}")
            print()
            print("Usage: python demo_ingestion.py [path/to/pdf]")
            sys.exit(1)
    
    print()
    print("-" * 70)
    print("PHASE 1: Loading PDF")
    print("-" * 70)
    
    try:
        # Step 1: Load PDF
        print("Loading PDF document...")
        raw_document = DocumentLoader.load_pdf(pdf_path)
        
        print(f"✓ PDF loaded successfully")
        print(f"  - File hash: {raw_document.file_hash[:32]}...")
        print(f"  - Number of pages: {raw_document.num_pages}")
        print(f"  - File size: {len(raw_document.pdf_bytes):,} bytes")
        
        if raw_document.metadata:
            print(f"  - Metadata:")
            for key, value in list(raw_document.metadata.items())[:5]:
                if value:
                    print(f"    · {key}: {str(value)[:50]}")
        
        print()
        print("-" * 70)
        print("PHASE 2: Preprocessing Document")
        print("-" * 70)
        
        # Step 2: Preprocess
        print("Preprocessing document...")
        print("  - Extracting text...")
        print("  - Normalizing encoding...")
        print("  - Segmenting into sentences...")
        print("  - Extracting tables...")
        print("  - Building indexes...")
        
        engine = PreprocessingEngine()
        preprocessed_doc = engine.preprocess_document(raw_document)
        
        print()
        print("✓ Document preprocessed successfully")
        print()
        
        print("-" * 70)
        print("RESULTS")
        print("-" * 70)
        
        print(f"Document ID: {preprocessed_doc.document_id}")
        print()
        
        print(f"Text Statistics:")
        print(f"  - Raw text length: {len(preprocessed_doc.raw_text):,} characters")
        print(f"  - Normalized text length: {len(preprocessed_doc.normalized_text):,} characters")
        print(f"  - Number of sentences: {len(preprocessed_doc.sentences)}")
        print(f"  - Number of tables: {len(preprocessed_doc.tables)}")
        print()
        
        print(f"Indexes:")
        print(f"  - Terms indexed: {len(preprocessed_doc.indexes.term_index):,}")
        print(f"  - Numbers indexed: {len(preprocessed_doc.indexes.numeric_index):,}")
        print(f"  - Dates indexed: {len(preprocessed_doc.indexes.temporal_index):,}")
        print(f"  - Table types: {len(preprocessed_doc.indexes.table_index):,}")
        print()
        
        # Show sample sentences
        print("Sample Sentences:")
        for i, sent in enumerate(preprocessed_doc.sentences[:5], 1):
            text_preview = sent.text[:80] + "..." if len(sent.text) > 80 else sent.text
            print(f"  {i}. [{sent.start_offset}:{sent.end_offset}] {text_preview}")
        
        if len(preprocessed_doc.sentences) > 5:
            print(f"  ... ({len(preprocessed_doc.sentences) - 5} more sentences)")
        print()
        
        # Show sample terms from index
        if preprocessed_doc.indexes.term_index:
            print("Sample Indexed Terms:")
            sample_terms = list(preprocessed_doc.indexes.term_index.keys())[:10]
            for term in sample_terms:
                locations = preprocessed_doc.indexes.term_index[term]
                print(f"  - '{term}': {len(locations)} occurrence(s)")
            print()
        
        # Show tables if any
        if preprocessed_doc.tables:
            print("Extracted Tables:")
            for table in preprocessed_doc.tables:
                print(f"  - Table {table.table_index}: {table.table_type} "
                      f"(page {table.page_number}, {len(table.data)} rows, "
                      f"confidence: {table.confidence:.2f})")
            print()
        
        # Show sample numbers
        if preprocessed_doc.indexes.numeric_index:
            print("Sample Indexed Numbers:")
            sample_numbers = list(preprocessed_doc.indexes.numeric_index.keys())[:10]
            for num in sample_numbers:
                locations = preprocessed_doc.indexes.numeric_index[num]
                print(f"  - {num}: {len(locations)} occurrence(s)")
            print()
        
        # Show sample dates
        if preprocessed_doc.indexes.temporal_index:
            print("Sample Indexed Dates:")
            sample_dates = list(preprocessed_doc.indexes.temporal_index.keys())[:10]
            for date in sample_dates:
                locations = preprocessed_doc.indexes.temporal_index[date]
                print(f"  - {date}: {len(locations)} occurrence(s)")
            print()
        
        print("-" * 70)
        print("✓ DEMO COMPLETED SUCCESSFULLY")
        print("-" * 70)
        print()
        print("The preprocessed document is now ready for pipeline processing.")
        print("It can be passed to the Orchestrator for question answering.")
        
    except IngestionError as e:
        print(f"✗ Ingestion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
