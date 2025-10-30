"""
DOCUMENT INGESTION MODULE - ESTRICTAMENTE SEGÚN PSEUDOCÓDIGO
==============================================================
Archivo: document_ingestion.py
Código: DI
Propósito: Carga inicial de documentos PDF y extracción de texto

MÉTODOS (9 EXACTOS):
1. DocumentLoader.load_pdf()
2. DocumentLoader.validate_pdf()
3. DocumentLoader.extract_metadata()
4. TextExtractor.extract_full_text()
5. TextExtractor.extract_by_page()
6. TextExtractor.preserve_structure()
7. PreprocessingEngine.preprocess_document()
8. PreprocessingEngine.normalize_encoding()
9. PreprocessingEngine.detect_language()

INTEGRACIÓN CON MÓDULOS EXISTENTES:
- Usa PP.PolicyTextProcessor.normalize_unicode()
- Usa PP.PolicyTextProcessor.segment_into_sentences()
- Usa FV.PDETMunicipalPlanAnalyzer.extract_tables()
- Usa FV.PDETMunicipalPlanAnalyzer._clean_dataframe()
- Usa FV.PDETMunicipalPlanAnalyzer._classify_tables()

DEPENDENCIAS:
    pip install pdfplumber PyPDF2 spacy langdetect
    python -m spacy download es_core_news_sm
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib

# PDF Processing
import pdfplumber
from PyPDF2 import PdfReader

# Language detection
from langdetect import detect, LangDetectException

# Importar módulos existentes del sistema
# NOTA: Estos imports asumen la estructura existente del proyecto
try:
    from methods.policy_processor import PolicyTextProcessor
    from methods.financiero_viabilidad_tablas import PDETMunicipalPlanAnalyzer
except ImportError:
    # Fallback para testing standalone
    PolicyTextProcessor = None
    PDETMunicipalPlanAnalyzer = None

logger = logging.getLogger(__name__)


# ============================================================================
# DATACLASSES - ESTRUCTURAS DE DATOS INMUTABLES
# ============================================================================

@dataclass(frozen=True)
class RawDocument:
    """
    Documento PDF crudo cargado desde disco.
    Inmutable para garantizar trazabilidad.
    """
    file_path: str
    file_name: str
    num_pages: int
    file_size_bytes: int
    file_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True


@dataclass(frozen=True)
class StructuredText:
    """
    Texto con estructura preservada (secciones, jerarquía).
    """
    full_text: str
    sections: List[Dict[str, Any]] = field(default_factory=list)
    page_boundaries: List[Tuple[int, int]] = field(default_factory=list)  # (start_char, end_char)


@dataclass(frozen=True)
class DocumentIndexes:
    """
    Índices construidos sobre el documento para búsqueda rápida.
    """
    term_index: Dict[str, List[int]] = field(default_factory=dict)  # término -> [sentence_ids]
    numeric_index: Dict[float, List[int]] = field(default_factory=dict)  # número -> [sentence_ids]
    temporal_index: Dict[str, List[int]] = field(default_factory=dict)  # fecha/año -> [sentence_ids]
    entity_index: Dict[str, List[int]] = field(default_factory=dict)  # entidad -> [sentence_ids]


@dataclass(frozen=True)
class PreprocessedDocument:
    """
    Documento completamente preprocesado y listo para evaluación.
    Este objeto se cachea y se distribuye a todos los evaluadores.
    """
    raw_document: RawDocument
    full_text: str
    structured_text: StructuredText
    sentences: List[str]
    sentence_metadata: List[Dict[str, Any]]  # ubicación, página, índices
    tables: List[Dict[str, Any]]  # tablas clasificadas y limpias
    indexes: DocumentIndexes
    language: str
    preprocessing_metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CLASE 1: DocumentLoader
# ============================================================================

class DocumentLoader:
    """
    Carga y valida documentos PDF.
    Responsable de la I/O básica y validación inicial.
    """
    
    def __init__(self):
        self.logger = logger
    
    def load_pdf(self, *, pdf_path: str) -> RawDocument:
        """
        MÉTODO 1: Carga un PDF desde disco (keyword-only params).
        
        ENTRADA: pdf_path (string) - keyword only
        PROCESO:
          - Leer bytes del PDF
          - Validar que es PDF válido
          - Extraer metadata básica (autor, fecha, páginas)
        SALIDA: RawDocument {bytes, metadata, num_pages}
        SYNC
        
        Args:
            pdf_path: Ruta al archivo PDF (keyword-only)
            
        Returns:
            RawDocument con información básica del PDF
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el archivo no es un PDF válido
            TypeError: If pdf_path is not a string
        """
        # Runtime validation at ingress
        if not isinstance(pdf_path, str):
            raise TypeError(
                f"ERR_CONTRACT_MISMATCH[fn=load_pdf, param='pdf_path', "
                f"expected=str, got={type(pdf_path).__name__}, "
                f"producer=caller, consumer=DocumentLoader.load_pdf]"
            )
        file_path = pdf_path
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"El archivo debe ser PDF: {file_path}")
        
        self.logger.info(f"Cargando PDF: {pdf_path.name}")
        
        # Calcular hash del archivo
        file_hash = self._calculate_file_hash(pdf_path)
        
        # Obtener información básica del archivo
        file_stats = pdf_path.stat()
        
        # Extraer metadata con PyPDF2
        try:
            reader = PdfReader(str(pdf_path))
            num_pages = len(reader.pages)
            metadata = self.extract_metadata(reader)
            is_valid = self.validate_pdf_reader(reader)
            
        except Exception as e:
            self.logger.error(f"Error leyendo PDF con PyPDF2: {e}")
            raise ValueError(f"PDF corrupto o inválido: {file_path}") from e
        
        raw_doc = RawDocument(
            file_path=str(pdf_path.absolute()),
            file_name=pdf_path.name,
            num_pages=num_pages,
            file_size_bytes=file_stats.st_size,
            file_hash=file_hash,
            metadata=metadata,
            is_valid=is_valid
        )
        
        self.logger.info(f"✓ PDF cargado: {num_pages} páginas, {file_stats.st_size / 1024:.1f} KB")
        
        return raw_doc
    
    def validate_pdf(self, *, raw_doc: RawDocument) -> bool:
        """
        MÉTODO 2: Valida que el PDF sea procesable.
        
        Verificaciones:
        - Número de páginas > 0
        - Tamaño de archivo razonable (< 500 MB)
        - No está encriptado
        
        Args:
            raw_doc: Documento crudo a validar
            
        Returns:
            True si es válido, False si no
        """
        if raw_doc.num_pages == 0:
            self.logger.error("PDF no tiene páginas")
            return False
        
        # Validar tamaño (500 MB máximo)
        max_size = 500 * 1024 * 1024  # 500 MB
        if raw_doc.file_size_bytes > max_size:
            self.logger.warning(f"PDF muy grande: {raw_doc.file_size_bytes / (1024*1024):.1f} MB")
            return False
        
        # Verificar si está encriptado
        if raw_doc.metadata.get('encrypted', False):
            self.logger.error("PDF está encriptado")
            return False
        
        return True
    
    def validate_pdf_reader(self, reader: PdfReader) -> bool:
        """Valida un PdfReader de PyPDF2."""
        if reader.is_encrypted:
            return False
        if len(reader.pages) == 0:
            return False
        return True
    
    def extract_metadata(self, reader: PdfReader) -> Dict[str, Any]:
        """
        MÉTODO 3: Extrae metadata del PDF.
        
        Args:
            reader: PdfReader de PyPDF2
            
        Returns:
            Diccionario con metadata del PDF
        """
        metadata = {}
        
        try:
            pdf_metadata = reader.metadata
            
            if pdf_metadata:
                metadata = {
                    'author': pdf_metadata.get('/Author', 'Desconocido'),
                    'creator': pdf_metadata.get('/Creator', 'Desconocido'),
                    'producer': pdf_metadata.get('/Producer', 'Desconocido'),
                    'subject': pdf_metadata.get('/Subject', ''),
                    'title': pdf_metadata.get('/Title', ''),
                    'creation_date': str(pdf_metadata.get('/CreationDate', '')),
                    'modification_date': str(pdf_metadata.get('/ModDate', ''))
                }
            
            metadata['encrypted'] = reader.is_encrypted
            metadata['page_count'] = len(reader.pages)
            
        except Exception as e:
            self.logger.warning(f"Error extrayendo metadata: {e}")
        
        return metadata
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcula hash SHA-256 del archivo para trazabilidad."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()


# ============================================================================
# CLASE 2: TextExtractor
# ============================================================================

class TextExtractor:
    """
    Extrae texto de PDFs preservando estructura.
    Usa pdfplumber como método primario.
    """
    
    def __init__(self):
        self.logger = logger
    
    def extract_full_text(self, *, raw_doc: RawDocument) -> str:
        """
        MÉTODO 4: Extrae todo el texto del PDF (keyword-only params).
        
        ENTRADA: RawDocument (keyword only)
        PROCESO:
          - Extraer texto de todas las páginas
          - Preservar estructura (párrafos, secciones)
          - Identificar headers/footers
        SALIDA: string (texto completo)
        SYNC
        
        Args:
            raw_doc: Documento crudo cargado (keyword-only)
            
        Returns:
            Texto completo del documento
        """
        self.logger.info(f"Extrayendo texto completo de: {raw_doc.file_name}")
        
        all_text = []
        
        try:
            with pdfplumber.open(raw_doc.file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text() or ""
                        
                        if text.strip():
                            # Preservar separación entre páginas
                            all_text.append(f"\n--- Página {page_num} ---\n")
                            all_text.append(text)
                    
                    except Exception as e:
                        self.logger.warning(f"Error extrayendo página {page_num}: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Error abriendo PDF con pdfplumber: {e}")
            raise
        
        full_text = "\n".join(all_text)
        
        self.logger.info(f"✓ Texto extraído: {len(full_text)} caracteres")
        
        return full_text
    
    def extract_by_page(self, *, raw_doc: RawDocument, page: int) -> str:
        """
        MÉTODO 5: Extrae texto de una página específica.
        
        Args:
            raw_doc: Documento crudo
            page: Número de página (1-indexed)
            
        Returns:
            Texto de la página especificada
        """
        if page < 1 or page > raw_doc.num_pages:
            raise ValueError(f"Página {page} fuera de rango (1-{raw_doc.num_pages})")
        
        try:
            with pdfplumber.open(raw_doc.file_path) as pdf:
                text = pdf.pages[page - 1].extract_text() or ""
                return text
        
        except Exception as e:
            self.logger.error(f"Error extrayendo página {page}: {e}")
            return ""
    
    def preserve_structure(self, *, text: str) -> StructuredText:
        """
        MÉTODO 6: Preserva estructura del documento.
        
        Detecta:
        - Secciones principales (títulos en mayúsculas)
        - Subsecciones (títulos numerados)
        - Límites de páginas
        
        Args:
            text: Texto completo del documento
            
        Returns:
            StructuredText con jerarquía preservada
        """
        sections = []
        page_boundaries = []
        
        lines = text.split('\n')
        current_position = 0
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detectar marcador de página
            if line_stripped.startswith('--- Página'):
                if current_section:
                    sections.append(current_section)
                    current_section = None
                
                page_num = int(line_stripped.split()[2])
                page_boundaries.append((current_position, current_position + len(line)))
            
            # Detectar título de sección (mayúsculas, > 10 caracteres)
            elif line_stripped.isupper() and len(line_stripped) > 10:
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    'title': line_stripped,
                    'start_char': current_position,
                    'content': ''
                }
            
            # Agregar contenido a sección actual
            elif current_section is not None:
                current_section['content'] += line + '\n'
            
            current_position += len(line) + 1  # +1 por \n
        
        # Agregar última sección
        if current_section:
            sections.append(current_section)
        
        return StructuredText(
            full_text=text,
            sections=sections,
            page_boundaries=page_boundaries
        )


# ============================================================================
# CLASE 3: PreprocessingEngine
# ============================================================================

class PreprocessingEngine:
    """
    Motor de preprocesamiento unificado.
    Coordina la transformación de RawDocument → PreprocessedDocument.
    """
    
    def __init__(self):
        self.logger = logger
        
        # Inicializar procesadores de módulos existentes
        if PolicyTextProcessor:
            self.text_processor = PolicyTextProcessor()
        else:
            self.text_processor = None
            self.logger.warning("PolicyTextProcessor no disponible")
        
        if PDETMunicipalPlanAnalyzer:
            self.table_analyzer = PDETMunicipalPlanAnalyzer()
        else:
            self.table_analyzer = None
            self.logger.warning("PDETMunicipalPlanAnalyzer no disponible")
    
    def preprocess_document(self, *, raw_doc: RawDocument) -> PreprocessedDocument:
        """
        MÉTODO 7: Pipeline completo de preprocesamiento (keyword-only params).
        
        ENTRADA: RawDocument (keyword only)
        PROCESO INTERNO (SYNC pero con llamadas a métodos existentes):
        
          1. Extraer texto completo
          2. Normalizar encoding (usa PP.PolicyTextProcessor.normalize_unicode)
          3. Segmentar en oraciones (usa PP.PolicyTextProcessor.segment_into_sentences)
          4. Extraer tablas (usa FV.PDETMunicipalPlanAnalyzer.extract_tables)
          5. Limpiar y clasificar tablas
          6. Construir índices
          7. Detectar idioma
        
        SALIDA: PreprocessedDocument (inmutable, cacheable)
        SYNC
        
        Args:
            raw_doc: Documento crudo cargado
            
        Returns:
            Documento completamente preprocesado
        """
        self.logger.info(f"Iniciando preprocesamiento: {raw_doc.file_name}")
        
        # PASO 1: Extraer texto completo
        text_extractor = TextExtractor()
        full_text = text_extractor.extract_full_text(raw_doc)
        structured_text = text_extractor.preserve_structure(full_text)
        
        # PASO 2: Normalizar encoding
        normalized_text = self.normalize_encoding(full_text)
        
        # PASO 3: Segmentar en oraciones
        if self.text_processor:
            sentences_data = self.text_processor.segment_into_sentences(normalized_text)
            
            # Extraer lista de oraciones y metadata
            if isinstance(sentences_data, dict):
                sentences = sentences_data.get('sentences', [])
                sentence_metadata = sentences_data.get('metadata', [])
            else:
                sentences = sentences_data
                sentence_metadata = [{'index': i} for i in range(len(sentences))]
        else:
            # Fallback simple
            sentences = [s.strip() for s in normalized_text.split('.') if s.strip()]
            sentence_metadata = [{'index': i} for i in range(len(sentences))]
        
        self.logger.info(f"✓ Segmentado en {len(sentences)} oraciones")
        
        # PASO 4: Extraer tablas
        tables = []
        if self.table_analyzer:
            try:
                raw_tables = self.table_analyzer.extract_tables(raw_doc.file_path)
                
                # PASO 5: Limpiar y clasificar tablas
                if raw_tables:
                    cleaned_tables = [
                        self.table_analyzer._clean_dataframe(table)
                        for table in raw_tables
                    ]
                    tables = self.table_analyzer._classify_tables(cleaned_tables)
                
                self.logger.info(f"✓ Extraídas {len(tables)} tablas")
            
            except Exception as e:
                self.logger.warning(f"Error extrayendo tablas: {e}")
        
        # PASO 6: Construir índices
        indexes = self._build_document_indexes(sentences, tables)
        
        # PASO 7: Detectar idioma
        language = self.detect_language(normalized_text)
        
        # Ensamblar documento preprocesado
        preprocessed_doc = PreprocessedDocument(
            raw_document=raw_doc,
            full_text=normalized_text,
            structured_text=structured_text,
            sentences=sentences,
            sentence_metadata=sentence_metadata,
            tables=tables,
            indexes=indexes,
            language=language,
            preprocessing_metadata={
                'num_sentences': len(sentences),
                'num_tables': len(tables),
                'text_length': len(normalized_text),
                'index_terms': len(indexes.term_index)
            }
        )
        
        self.logger.info(f"✓ Preprocesamiento completado")
        
        return preprocessed_doc
    
    def normalize_encoding(self, *, text: str) -> str:
        """
        MÉTODO 8: Normaliza encoding del texto.
        
        Delega a PP.PolicyTextProcessor.normalize_unicode()
        
        Args:
            text: Texto a normalizar
            
        Returns:
            Texto normalizado
        """
        if self.text_processor:
            return self.text_processor.normalize_unicode(text)
        else:
            # Fallback: normalización básica
            import unicodedata
            return unicodedata.normalize('NFC', text)
    
    def detect_language(self, *, text: str) -> str:
        """
        MÉTODO 9: Detecta el idioma del documento.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Código ISO del idioma ('es', 'en', etc.)
        """
        if not text or len(text.strip()) < 20:
            return 'unknown'
        
        try:
            # Usar muestra del texto
            sample = text[:5000] if len(text) > 5000 else text
            detected_lang = detect(sample)
            
            self.logger.info(f"✓ Idioma detectado: {detected_lang}")
            return detected_lang
        
        except LangDetectException:
            self.logger.warning("No se pudo detectar idioma, asumiendo español")
            return 'es'
        
        except Exception as e:
            self.logger.error(f"Error detectando idioma: {e}")
            return 'unknown'
    
    def _build_document_indexes(
        self,
        sentences: List[str],
        tables: List[Dict[str, Any]]
    ) -> DocumentIndexes:
        """
        Construye índices sobre el documento para búsqueda rápida.
        
        INCLUYE:
        - Índice invertido de términos
        - Índice de números
        - Índice de marcadores temporales
        - Índice de entidades
        
        Args:
            sentences: Lista de oraciones
            tables: Lista de tablas clasificadas
            
        Returns:
            DocumentIndexes con todos los índices
        """
        term_index = {}
        numeric_index = {}
        temporal_index = {}
        entity_index = {}
        
        import re
        
        # Construir índices iterando sobre oraciones
        for sent_idx, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Índice de términos (palabras > 3 caracteres)
            words = re.findall(r'\b\w{4,}\b', sentence_lower)
            for word in set(words):
                if word not in term_index:
                    term_index[word] = []
                term_index[word].append(sent_idx)
            
            # Índice de números
            numbers = re.findall(r'\d+(?:\.\d+)?', sentence)
            for num_str in numbers:
                try:
                    num = float(num_str)
                    if num not in numeric_index:
                        numeric_index[num] = []
                    numeric_index[num].append(sent_idx)
                except ValueError:
                    continue
            
            # Índice temporal (años, fechas)
            years = re.findall(r'\b(20\d{2})\b', sentence)
            for year in years:
                if year not in temporal_index:
                    temporal_index[year] = []
                temporal_index[year].append(sent_idx)
            
            # Índice de entidades (palabras capitalizadas)
            entities = re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b', sentence)
            for entity in set(entities):
                if entity not in entity_index:
                    entity_index[entity] = []
                entity_index[entity].append(sent_idx)
        
        self.logger.info(f"✓ Índices construidos: {len(term_index)} términos, {len(numeric_index)} números")
        
        return DocumentIndexes(
            term_index=term_index,
            numeric_index=numeric_index,
            temporal_index=temporal_index,
            entity_index=entity_index
        )


# ============================================================================
# FUNCIÓN DE CONVENIENCIA
# ============================================================================

def ingest_document(pdf_path: str) -> PreprocessedDocument:
    """
    Función de conveniencia para ejecutar pipeline completo de ingesta.
    
    Args:
        pdf_path: Ruta al archivo PDF
        
    Returns:
        PreprocessedDocument listo para evaluación
    """
    # Paso 1: Cargar PDF
    loader = DocumentLoader()
    raw_doc = loader.load_pdf(pdf_path)
    
    # Paso 2: Validar
    if not loader.validate_pdf(raw_doc):
        raise ValueError(f"PDF no válido: {pdf_path}")
    
    # Paso 3: Preprocesar
    engine = PreprocessingEngine()
    preprocessed_doc = engine.preprocess_document(raw_doc)
    
    return preprocessed_doc


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejemplo de uso
    pdf_path = "plan_desarrollo_municipal.pdf"
    
    try:
        preprocessed_doc = ingest_document(pdf_path)
        
        print(f"\n✅ DOCUMENTO PREPROCESADO:")
        print(f"   - Archivo: {preprocessed_doc.raw_document.file_name}")
        print(f"   - Páginas: {preprocessed_doc.raw_document.num_pages}")
        print(f"   - Oraciones: {len(preprocessed_doc.sentences)}")
        print(f"   - Tablas: {len(preprocessed_doc.tables)}")
        print(f"   - Idioma: {preprocessed_doc.language}")
        print(f"   - Hash: {preprocessed_doc.raw_document.file_hash[:16]}...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
