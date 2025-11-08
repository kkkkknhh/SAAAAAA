"""
SPC (Smart Policy Chunks) Ingestion - Canonical Phase-One
==========================================================

This module provides the canonical phase-one ingestion pipeline for processing
development plans into smart policy chunks with comprehensive analysis.

Main exports:
- CPPIngestionPipeline: Primary ingestion pipeline (for compatibility)
- StrategicChunkingSystem: Core chunking system from smart_policy_chunks_canonic_phase_one

The pipeline performs:
1. Document preprocessing and structural analysis
2. Topic modeling and knowledge graph construction
3. Causal chain extraction
4. Temporal, argumentative, and discourse analysis
5. Smart chunk creation with inter-chunk relationships
6. Quality validation and strategic ranking
"""

from pathlib import Path
import sys

# Add root to path for smart_policy_chunks_canonic_phase_one import
_root = Path(__file__).parent.parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from smart_policy_chunks_canonic_phase_one import StrategicChunkingSystem
except ImportError:
    # Fallback to relative import if needed
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "smart_policy_chunks",
        _root / "smart_policy_chunks_canonic_phase_one.py"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        StrategicChunkingSystem = module.StrategicChunkingSystem
    else:
        raise ImportError("Cannot load smart_policy_chunks_canonic_phase_one module")


class CPPIngestionPipeline:
    """
    Compatibility wrapper for SPC ingestion pipeline.
    
    This class provides backwards compatibility for code expecting CPPIngestionPipeline
    while delegating to the canonical StrategicChunkingSystem implementation.
    """
    
    def __init__(self):
        """Initialize the SPC ingestion pipeline."""
        self.chunking_system = StrategicChunkingSystem()
    
    async def process(self, document_path: Path, document_id: str = None, 
                     title: str = None, max_chunks: int = 50):
        """
        Process a document through the SPC pipeline.
        
        Args:
            document_path: Path to input document
            document_id: Optional document identifier
            title: Optional document title
            max_chunks: Maximum number of chunks to generate
            
        Returns:
            Processed document with smart policy chunks
        """
        # Read document
        with open(document_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        # Prepare metadata
        metadata = {
            'document_id': document_id or str(document_path.stem),
            'title': title or document_path.name,
            'version': 'v3.0'
        }
        
        # Process through chunking system
        chunks = self.chunking_system.process_document(document_text, metadata)
        
        # Return structured result
        return {
            'chunks': chunks,
            'metadata': metadata,
            'document_path': str(document_path)
        }


__all__ = [
    'CPPIngestionPipeline',
    'StrategicChunkingSystem',
]
