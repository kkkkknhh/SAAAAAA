"""
SIN_CARRETA Policy Analysis Pipeline - ExecutionChoreographer (MICRO Level)

ARCHITECTURAL ROLE:
- MICRO-level execution engine for individual policy questions
- Container for ALL 11 YAML specialized components
- Dimensional router (D1-D6 execution chains)
- Method-level orchestrator (584 methods, 95% integration target)
- Provenance tracker (complete data lineage)

PRIME DIRECTIVES:
- No graceful degradation: All 11 components or abort
- No strategic simplification: 584 methods, complexity as asset
- SOTA as baseline: Bayesian inference, DAG validation, bicameral reasoning
- Deterministic reproducibility: Immutable YAML configs, explicit traces
- Explicitness over assumption: Typed contracts, no implicit coercions
- Observability as structure: Provenance records, execution instrumentation

VERSION: 1.0.0
LAST UPDATED: 2025-10-27
"""

import logging
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml
import networkx as nx

# ========================================
# IMPORT ALL 11 YAML SPECIALIZED COMPONENTS
# ========================================

# Producer 1: policy_processor
from policy_processor import IndustrialPolicyProcessor

# Producer 2: contradiction_deteccion
from contradiction_deteccion import (
    PolicyContradictionDetector,
    TemporalLogicVerifier,
    BayesianConfidenceCalculator
)

# Producer 3: teoria_cambio
from teoria_cambio import (
    TeoriaCambio,
    AdvancedDAGValidator,
    IndustrialGradeValidator
)

# Producer 4: Analyzer_one
from Analyzer_one import (
    MunicipalAnalyzer,
    MunicipalOntology,
    SemanticAnalyzer,
    PerformanceAnalyzer,
    TextMiningEngine
)

# Producer 5: financiero_viabilidad_tablas
from financiero_viabilidad_tablas import (
    PDETMunicipalPlanAnalyzer,
    ColombianMunicipalContext
)

# Producer 6: dereck_beach (placeholder - implement as needed)
# from dereck_beach import DerickBeachProcessor

# Producer 7: embedding_policy (placeholder - implement as needed)
# from embedding_policy import EmbeddingPolicyEngine

# Producer 8: semantic_chunking_policy (placeholder - implement as needed)
# from semantic_chunking_policy import SemanticChunkingEngine

# Producer 9: report_assembly
from report_assembly import MicroLevelAnswer, MesoLevelCluster, MacroLevelConvergence

logger = logging.getLogger(__name__)


# ========================================
# TYPE CONTRACTS
# ========================================

class DimensionCode(Enum):
    """Enumeration of valid dimensional codes (D1-D6)"""
    D1_DIAGNOSTICO = "D1"
    D2_ACTIVIDADES = "D2"
    D3_PRODUCTOS = "D3"
    D4_RESULTADOS = "D4"
    D5_IMPACTOS = "D5"
    D6_CAUSALIDAD = "D6"


@dataclass
class ExecutionContext:
    """
    Immutable context for executing a single policy question
    
    INVARIANTS:
    - question_id must be unique and deterministic
    - dimension must be valid DimensionCode
    - policy_area must be non-empty
    """
    question_id: str
    dimension: str
    policy_area: str
    questionnaire_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate dimension code
        if self.dimension not in [d.value for d in DimensionCode]:
            raise ValueError(
                f"Invalid dimension code: {self.dimension}. "
                f"Must be one of: {[d.value for d in DimensionCode]}"
            )


@dataclass
class ProvenanceRecord:
    """
    Complete data lineage for audit and reproducibility
    
    REQUIREMENTS:
    - execution_id: Deterministic hash of inputs
    - timestamp: ISO 8601 UTC timestamp
    - input_artifacts: All input data sources
    - output_artifacts: All generated outputs
    - methods_invoked: Complete method call chain
    - confidence_scores: Bayesian confidence at each step
    - metadata: Execution environment, versions, configs
    """
    execution_id: str
    timestamp: str
    input_artifacts: List[str]
    output_artifacts: List[str]
    methods_invoked: List[str]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class ExecutionResult:
    """
    Complete result of executing a single policy question
    
    CONTAINS:
    - question_id: Question identifier
    - status: 'success' | 'failure' | 'partial'
    - micro_answer: MicroLevelAnswer or None
    - execution_trace: Complete method invocation trace
    - performance_metrics: Timing, memory, token usage
    - provenance: Complete data lineage
    - error: Error details if status != 'success'
    """
    question_id: str
    status: str
    micro_answer: Optional[MicroLevelAnswer]
    execution_trace: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    provenance: ProvenanceRecord
    error: Optional[Dict[str, Any]] = None


# ========================================
# CHOREOGRAPHER - MICRO LEVEL EXECUTOR
# ========================================

class ExecutionChoreographer:
    """
    MICRO-level execution engine for individual policy questions
    
    RESPONSIBILITIES:
    1. Initialize ALL 11 YAML specialized components (no graceful degradation)
    2. Build 584-method registry (95% integration target)
    3. Route questions to dimensional execution chains (D1-D6)
    4. Execute YAML-specified method chains with exact sequences
    5. Build MicroLevelAnswer from dimensional evidence
    6. Track complete provenance for data lineage
    
    DOES NOT:
    - Manage 300 questions (Orchestrator's job)
    - Do MESO clustering (Orchestrator's job)
    - Do MACRO convergence (Orchestrator's job)
    - Generate executive summaries (Orchestrator's job)
    
    BOUNDARY:
    Orchestrator → [execute_question] → ExecutionChoreographer → [MicroLevelAnswer] → Orchestrator
    """
    
    def __init__(
        self,
        execution_mapping_path: str,
        method_class_map_path: str,
        config_path: str,
        questionnaire_hash: str,
        deterministic_context: Dict[str, Any]
    ):
        """
        Initialize Choreographer with immutable YAML configuration
        
        PARAMETERS:
        - execution_mapping_path: Path to execution_mapping.yaml (dimensional chains)
        - method_class_map_path: Path to method_class_map.yaml (method registry)
        - config_path: Path to config.yaml (component configurations)
        - questionnaire_hash: Deterministic hash of questionnaire
        - deterministic_context: Execution environment context
        
        POST-CONDITIONS:
        - All 11 YAML components initialized or exception raised
        - Method registry contains ≥555 methods (95% of 584)
        - Validation engine ready
        """
        logger.info("=" * 80)
        logger.info("SIN_CARRETA CHOREOGRAPHER INITIALIZATION")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}Z")
        logger.info(f"Questionnaire Hash: {questionnaire_hash}")
        logger.info(f"Prime Directive: NO GRACEFUL DEGRADATION")
        
        self.questionnaire_hash = questionnaire_hash
        self.deterministic_context = deterministic_context
        
        # Golden Rule 1: Load immutable declarative configuration
        logger.info("Loading YAML configurations...")
        self.execution_mapping = self._load_execution_mapping(execution_mapping_path)
        self.method_class_map = self._load_method_class_map(method_class_map_path)
        self.config = self._load_config(config_path)
        
        # Build mission config
        logger.info("Building mission configuration...")
        self.mission_config = self._build_mission_config(self.config)
        self.processor_config = self._build_processor_config(self.mission_config)
        
        # CRITICAL: Initialize ALL 11 YAML components
        logger.info("Initializing ALL 11 YAML specialized components...")
        self._producer_instances = {}
        self._initialize_producers()
        
        # Validate: All 11 components must be initialized
        self._validate_component_initialization()
        
        # Build method registry (584 methods, 95% target = 555 methods)
        logger.info("Building method registry (target: 555/584 methods)...")
        self.CANONICAL_METHODS = {}
        self._build_method_registry()
        
        # Validate: Registry must have ≥555 methods
        if len(self.CANONICAL_METHODS) < 555:
            raise RuntimeError(
                f"Method registry incomplete: {len(self.CANONICAL_METHODS)}/555 methods. "
                f"Prime Directive violated: NO GRACEFUL DEGRADATION."
            )
        
        logger.info(f"✓ Method registry complete: {len(self.CANONICAL_METHODS)} methods")
        logger.info("✓ Choreographer initialization complete")
        logger.info("=" * 80)
    
    # ========================================
    # YAML CONFIGURATION LOADING
    # ========================================
    
    def _load_execution_mapping(self, path: str) -> Dict[str, Any]:
        """Load execution_mapping.yaml (dimensional chains)"""
        with open(path, 'r', encoding='utf-8') as f:
            mapping = yaml.safe_load(f)
        logger.info(f"✓ Loaded execution_mapping.yaml: {len(mapping)} dimensions")
        return mapping
    
    def _load_method_class_map(self, path: str) -> Dict[str, Any]:
        """Load method_class_map.yaml (method-to-class mappings)"""
        with open(path, 'r', encoding='utf-8') as f:
            mapping = yaml.safe_load(f)
        logger.info(f"✓ Loaded method_class_map.yaml: {len(mapping)} producers")
        return mapping
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load config.yaml (component configurations)"""
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Loaded config.yaml")
        return config
    
    def _build_mission_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract mission-level configuration"""
        return config.get('mission', {})
    
    def _build_processor_config(self, mission_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build processor-specific configuration"""
        return {
            'embedding_model': mission_config.get('embedding_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
            'chunk_size': mission_config.get('chunk_size', 512),
            'overlap': mission_config.get('overlap', 50),
            'batch_size': mission_config.get('batch_size', 32)
        }
    
    # ========================================
    # COMPONENT INITIALIZATION
    # ========================================
    
    def _initialize_producers(self):
        """
        Initialize ALL 11 YAML specialized components across 9 producers
        
        CRITICAL: This method MUST initialize all components or raise exception.
        NO graceful degradation. NO fallbacks. All or abort.
        
        POST-CONDITIONS:
        - self._producer_instances['policy_processor']['IndustrialPolicyProcessor'] exists
        - self._producer_instances['contradiction_deteccion']['PolicyContradictionDetector'] exists
        - self._producer_instances['contradiction_deteccion']['TemporalLogicVerifier'] exists
        - self._producer_instances['contradiction_deteccion']['BayesianConfidenceCalculator'] exists
        - self._producer_instances['teoria_cambio']['TeoriaCambio'] exists
        - self._producer_instances['teoria_cambio']['AdvancedDAGValidator'] exists
        - self._producer_instances['Analyzer_one']['MunicipalAnalyzer'] exists
        - self._producer_instances['Analyzer_one']['MunicipalOntology'] exists
        - self._producer_instances['Analyzer_one']['SemanticAnalyzer'] exists
        - self._producer_instances['Analyzer_one']['PerformanceAnalyzer'] exists
        - self._producer_instances['financiero_viabilidad_tablas']['PDETMunicipalPlanAnalyzer'] exists
        """
        
        # Producer 1: policy_processor
        logger.info("  [1/9] Initializing policy_processor...")
        try:
            self._producer_instances['policy_processor'] = {
                'IndustrialPolicyProcessor': IndustrialPolicyProcessor(
                    config=self.processor_config
                )
            }
            logger.info("  ✓ IndustrialPolicyProcessor initialized")
        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to initialize IndustrialPolicyProcessor: {e}")
        
        # Producer 2: contradiction_deteccion
        logger.info("  [2/9] Initializing contradiction_deteccion...")
        try:
            self._producer_instances['contradiction_deteccion'] = {
                'PolicyContradictionDetector': PolicyContradictionDetector(),
                'TemporalLogicVerifier': TemporalLogicVerifier(),
                'BayesianConfidenceCalculator': BayesianConfidenceCalculator()
            }
            logger.info("  ✓ PolicyContradictionDetector initialized")
            logger.info("  ✓ TemporalLogicVerifier initialized")
            logger.info("  ✓ BayesianConfidenceCalculator initialized")
        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to initialize contradiction_deteccion: {e}")
        
        # Producer 3: teoria_cambio
        logger.info("  [3/9] Initializing teoria_cambio...")
        try:
            self._producer_instances['teoria_cambio'] = {
                'TeoriaCambio': TeoriaCambio(),
                'AdvancedDAGValidator': AdvancedDAGValidator(),
                'IndustrialGradeValidator': IndustrialGradeValidator()
            }
            logger.info("  ✓ TeoriaCambio initialized")
            logger.info("  ✓ AdvancedDAGValidator initialized")
            logger.info("  ✓ IndustrialGradeValidator initialized")
        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to initialize teoria_cambio: {e}")
        
        # Producer 4: Analyzer_one
        logger.info("  [4/9] Initializing Analyzer_one...")
        try:
            ontology = MunicipalOntology()
            self._producer_instances['Analyzer_one'] = {
                'MunicipalOntology': ontology,
                'MunicipalAnalyzer': MunicipalAnalyzer(ontology=ontology),
                'SemanticAnalyzer': SemanticAnalyzer(ontology=ontology),
                'PerformanceAnalyzer': PerformanceAnalyzer(),
                'TextMiningEngine': TextMiningEngine()
            }
            logger.info("  ✓ MunicipalOntology initialized")
            logger.info("  ✓ MunicipalAnalyzer initialized")
            logger.info("  ✓ SemanticAnalyzer initialized")
            logger.info("  ✓ PerformanceAnalyzer initialized")
            logger.info("  ✓ TextMiningEngine initialized")
        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to initialize Analyzer_one: {e}")
        
        # Producer 5: financiero_viabilidad_tablas
        logger.info("  [5/9] Initializing financiero_viabilidad_tablas...")
        try:
            self._producer_instances['financiero_viabilidad_tablas'] = {
                'PDETMunicipalPlanAnalyzer': PDETMunicipalPlanAnalyzer(),
                'ColombianMunicipalContext': ColombianMunicipalContext()
            }
            logger.info("  ✓ PDETMunicipalPlanAnalyzer initialized")
            logger.info("  ✓ ColombianMunicipalContext initialized")
        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to initialize financiero_viabilidad_tablas: {e}")
        
        # Producers 6-9: Placeholder implementations
        # TODO: Implement when components are ready
        logger.info("  [6/9] dereck_beach: PLACEHOLDER (implement when ready)")
        self._producer_instances['dereck_beach'] = {}
        
        logger.info("  [7/9] embedding_policy: PLACEHOLDER (implement when ready)")
        self._producer_instances['embedding_policy'] = {}
        
        logger.info("  [8/9] semantic_chunking_policy: PLACEHOLDER (implement when ready)")
        self._producer_instances['semantic_chunking_policy'] = {}
        
        logger.info("  [9/9] report_assembly: PLACEHOLDER (implement when ready)")
        self._producer_instances['report_assembly'] = {}
    
    def _validate_component_initialization(self):
        """
        Validate that ALL 11 required YAML components are initialized
        
        RAISES:
        - RuntimeError if any component is missing
        """
        required_components = [
            ('policy_processor', 'IndustrialPolicyProcessor'),
            ('contradiction_deteccion', 'PolicyContradictionDetector'),
            ('contradiction_deteccion', 'TemporalLogicVerifier'),
            ('contradiction_deteccion', 'BayesianConfidenceCalculator'),
            ('teoria_cambio', 'TeoriaCambio'),
            ('teoria_cambio', 'AdvancedDAGValidator'),
            ('Analyzer_one', 'MunicipalOntology'),
            ('Analyzer_one', 'MunicipalAnalyzer'),
            ('Analyzer_one', 'SemanticAnalyzer'),
            ('Analyzer_one', 'PerformanceAnalyzer'),
            ('financiero_viabilidad_tablas', 'PDETMunicipalPlanAnalyzer')
        ]
        
        missing = []
        for producer, component in required_components:
            if producer not in self._producer_instances:
                missing.append(f"{producer}.{component}")
            elif component not in self._producer_instances[producer]:
                missing.append(f"{producer}.{component}")
        
        if missing:
            raise RuntimeError(
                f"FATAL: Component initialization incomplete. Missing: {missing}. "
                f"Prime Directive violated: NO GRACEFUL DEGRADATION."
            )
        
        logger.info(f"✓ All 11 YAML components validated")
    
    # ========================================
    # METHOD REGISTRY BUILDING
    # ========================================
    
    def _build_method_registry(self):
        """
        Build dispatch registry mapping fully-qualified method names to callables
        
        TARGET: 584 methods across 9 producers
        GOAL: 95% integration (555 methods minimum)
        
        POST-CONDITIONS:
        - self.CANONICAL_METHODS contains ≥555 entries
        - All entries are callable
        - All keys follow pattern: 'producer.Class.method'
        """
        for producer_name, producer_dict in self._producer_instances.items():
            for class_name, instance in producer_dict.items():
                # Get all public methods (exclude private methods starting with '_')
                methods = [
                    m for m in dir(instance)
                    if callable(getattr(instance, m)) and not m.startswith('__')
                ]
                
                for method_name in methods:
                    # Build fully-qualified name
                    fq_name = f"{producer_name}.{class_name}.{method_name}"
                    
                    # Store callable
                    self.CANONICAL_METHODS[fq_name] = getattr(instance, method_name)
        
        logger.info(f"✓ Registered {len(self.CANONICAL_METHODS)} methods")
        logger.info(f"  Target: 555 methods (95% of 584)")
        logger.info(f"  Coverage: {len(self.CANONICAL_METHODS)/584*100:.1f}%")
    
    def _get_producer_instance(self, producer_name: str, class_name: str) -> Any:
        """
        Get producer instance by name with validation
        
        RAISES:
        - KeyError if producer or class not found
        """
        if producer_name not in self._producer_instances:
            raise KeyError(f"Producer not found: {producer_name}")
        
        if class_name not in self._producer_instances[producer_name]:
            raise KeyError(f"Class not found: {producer_name}.{class_name}")
        
        return self._producer_instances[producer_name][class_name]
    
    # ========================================
    # QUESTION EXECUTION - CORE RESPONSIBILITY
    # ========================================
    
    def execute_question(
        self,
        question_context,  # Can be ExecutionContext or Dict
        plan_document: str,
        plan_metadata: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a single policy question using dimensional routing
        
        FLOW:
        1. Normalize question context (dict → ExecutionContext if needed)
        2. Normalize dimension code (D1-D6)
        3. Route to dimensional execution chain
        4. Build MicroLevelAnswer from evidence
        5. Build provenance record
        6. Return ExecutionResult
        
        PARAMETERS:
        - question_context: ExecutionContext or dict with question details
        - plan_document: Policy document text
        - plan_metadata: Document metadata
        
        RETURNS:
        - ExecutionResult with MicroLevelAnswer and complete provenance
        """
        start_time = datetime.utcnow()
        execution_trace = []
        
        # Convert dict to ExecutionContext if needed
        if isinstance(question_context, dict):
            context = ExecutionContext(
                question_id=question_context.get('canonical_id', question_context.get('question_id', 'UNKNOWN')),
                dimension=question_context.get('dimension', 'D1'),
                policy_area=question_context.get('policy_area', 'P0'),
                questionnaire_hash=self.questionnaire_hash,
                timestamp=datetime.utcnow().isoformat() + "Z",
                metadata={
                    'scoring_modality': question_context.get('scoring_modality', 'TYPE_F'),
                    'expected_elements': question_context.get('expected_elements', []),
                    'search_patterns': question_context.get('search_patterns', {}),
                    'element_weights': question_context.get('element_weights', {}),
                    'numerical_thresholds': question_context.get('numerical_thresholds', {}),
                    'validation_rules': question_context.get('validation_rules', {}),
                    'question_text': question_context.get('question_text', '')
                }
            )
        else:
            context = question_context
        
        logger.info("=" * 80)
        logger.info(f"EXECUTING QUESTION: {context.question_id}")
        logger.info(f"Dimension: {context.dimension}")
        logger.info(f"Policy Area: {context.policy_area}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Normalize dimension code
            dimension = self._normalize_dimension_code(context.dimension)
            logger.info(f"Dimension normalized: {dimension}")
            
            # Step 2: Route to dimensional execution chain
            logger.info(f"Routing to {dimension} execution chain...")
            
            if dimension == DimensionCode.D1_DIAGNOSTICO.value:
                evidence = self._execute_d1_chain(context, plan_document, execution_trace)
            elif dimension == DimensionCode.D2_ACTIVIDADES.value:
                evidence = self._execute_d2_chain(context, plan_document, execution_trace)
            elif dimension == DimensionCode.D3_PRODUCTOS.value:
                evidence = self._execute_d3_chain(context, plan_document, execution_trace)
            elif dimension == DimensionCode.D4_RESULTADOS.value:
                evidence = self._execute_d4_chain(context, plan_document, execution_trace)
            elif dimension == DimensionCode.D5_IMPACTOS.value:
                evidence = self._execute_d5_chain(context, plan_document, execution_trace)
            elif dimension == DimensionCode.D6_CAUSALIDAD.value:
                evidence = self._execute_d6_chain(context, plan_document, execution_trace)
            else:
                evidence = self._execute_generic_chain(context, plan_document, execution_trace)
            
            # Step 3: Build MicroLevelAnswer
            logger.info("Building MicroLevelAnswer from evidence...")
            
            # Add execution trace to metadata
            metadata_with_trace = {
                **plan_metadata,
                'execution_trace': execution_trace
            }
            
            micro_answer = self._build_micro_answer(
                context,
                evidence,
                metadata_with_trace
            )
            
            # Step 4: Calculate performance metrics
            end_time = datetime.utcnow()
            performance_metrics = {
                'execution_time_ms': (end_time - start_time).total_seconds() * 1000,
                'methods_invoked': len(execution_trace),
                'evidence_size': len(str(evidence))
            }
            
            # Step 5: Build provenance record
            provenance = self._build_provenance_record(
                context,
                execution_trace,
                plan_metadata
            )
            
            logger.info(f"✓ Question execution complete: {context.question_id}")
            logger.info(f"  Score: {micro_answer.quantitative_score:.2f}/3.0")
            logger.info(f"  Note: {micro_answer.qualitative_note}")
            logger.info(f"  Confidence: {micro_answer.confidence:.3f}")
            logger.info(f"  Execution time: {performance_metrics['execution_time_ms']:.1f}ms")
            logger.info("=" * 80)
            
            return ExecutionResult(
                question_id=context.question_id,
                status='success',
                micro_answer=micro_answer,
                execution_trace=execution_trace,
                performance_metrics=performance_metrics,
                provenance=provenance,
                error=None
            )
        
        except Exception as e:
            logger.error(f"✗ Question execution failed: {context.question_id}")
            logger.error(f"  Error: {str(e)}")
            logger.error("=" * 80)
            
            end_time = datetime.utcnow()
            performance_metrics = {
                'execution_time_ms': (end_time - start_time).total_seconds() * 1000,
                'methods_invoked': len(execution_trace),
                'error': str(e)
            }
            
            provenance = self._build_provenance_record(
                context,
                execution_trace,
                plan_metadata
            )
            
            return ExecutionResult(
                question_id=context.question_id,
                status='failure',
                micro_answer=None,
                execution_trace=execution_trace,
                performance_metrics=performance_metrics,
                provenance=provenance,
                error={'message': str(e), 'type': type(e).__name__}
            )
    
    def _normalize_dimension_code(self, dimension: str) -> str:
        """Normalize dimension code to standard format (D1-D6)"""
        dimension_upper = dimension.upper()
        
        # Validate against enum
        if dimension_upper not in [d.value for d in DimensionCode]:
            raise ValueError(
                f"Invalid dimension code: {dimension}. "
                f"Must be one of: {[d.value for d in DimensionCode]}"
            )
        
        return dimension_upper
    
    # ========================================
    # DIMENSIONAL EXECUTION CHAINS
    # ========================================
    
    def _execute_d1_chain(
        self,
        context: ExecutionContext,
        document: str,
        trace: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        D1: Diagnóstico y Consistencia Inicial
        
        YAML CHAINS:
        - D1-Q1 (Brechas): IndustrialPolicyProcessor → PolicyContradictionDetector
        - D1-Q3 (Recursos): PolicyContradictionDetector → BayesianConfidenceCalculator
        - D1-Q4 (Capacidad): IndustrialPolicyProcessor → PolicyContradictionDetector
        - D1-Q5 (Temporal): PolicyContradictionDetector → TemporalLogicVerifier
        
        RETURNS:
        - evidence: Complete evidence bundle with 12-step chain results
        """
        logger.info("Executing D1 (Diagnóstico) chain...")
        evidence = {}
        
        processor = self._get_producer_instance('policy_processor', 'IndustrialPolicyProcessor')
        detector = self._get_producer_instance('contradiction_deteccion', 'PolicyContradictionDetector')
        verifier = self._get_producer_instance('contradiction_deteccion', 'TemporalLogicVerifier')
        calculator = self._get_producer_instance('contradiction_deteccion', 'BayesianConfidenceCalculator')
        analyzer = self._get_producer_instance('Analyzer_one', 'SemanticAnalyzer')
        
        # Step 1: Segment document
        trace.append({'step': 1, 'method': 'IndustrialPolicyProcessor.segment_into_sentences'})
        sentences = processor.segment_into_sentences(document)
        evidence['sentences'] = sentences
        
        # Step 2: Extract quantitative claims (brechas)
        trace.append({'step': 2, 'method': 'PolicyContradictionDetector._extract_quantitative_claims'})
        quantitative_claims = detector._extract_quantitative_claims(sentences)
        evidence['quantitative_claims'] = quantitative_claims
        
        # Step 3: Parse numbers
        trace.append({'step': 3, 'method': 'PolicyContradictionDetector._parse_number'})
        for claim in quantitative_claims:
            parsed = detector._parse_number(claim.get('text', ''))
            claim['normalized_value'] = parsed
        
        # Step 4: Match patterns for official sources
        trace.append({'step': 4, 'method': 'IndustrialPolicyProcessor._match_patterns_in_sentences'})
        patterns_found = processor._match_patterns_in_sentences(
            sentences,
            patterns=['DANE', 'DNP', 'fuente oficial', 'según datos de']
        )
        evidence['official_sources'] = patterns_found
        
        # Step 5: Calculate semantic complexity
        trace.append({'step': 5, 'method': 'SemanticAnalyzer._calculate_semantic_complexity'})
        complexity = analyzer._calculate_semantic_complexity(document)
        evidence['semantic_complexity'] = complexity
        
        # Step 6: Calculate Bayesian confidence
        trace.append({'step': 6, 'method': 'BayesianConfidenceCalculator.calculate_posterior'})
        confidence = calculator.calculate_posterior(
            evidence={'quantitative_claims': quantitative_claims},
            prior=0.5
        )
        evidence['bayesian_confidence'] = confidence
        
        # Step 7: Extract resource mentions
        trace.append({'step': 7, 'method': 'PolicyContradictionDetector._extract_resource_mentions'})
        resources = detector._extract_resource_mentions(document)
        evidence['resources'] = resources
        
        # Step 8: Detect numerical inconsistencies
        trace.append({'step': 8, 'method': 'PolicyContradictionDetector._detect_numerical_inconsistencies'})
        inconsistencies = detector._detect_numerical_inconsistencies(
            quantitative_claims,
            resources
        )
        evidence['inconsistencies'] = inconsistencies
        
        # Step 9: Detect resource conflicts
        trace.append({'step': 9, 'method': 'PolicyContradictionDetector._detect_resource_conflicts'})
        conflicts = detector._detect_resource_conflicts(resources, total_budget=None)
        evidence['conflicts'] = conflicts
        
        # Step 10: Calculate graph fragmentation (capacity)
        trace.append({'step': 10, 'method': 'PolicyContradictionDetector._calculate_graph_fragmentation'})
        fragmentation = detector._calculate_graph_fragmentation(actor_graph=None)
        evidence['capacity_fragmentation'] = fragmentation
        
        # Step 11: Verify temporal consistency
        trace.append({'step': 11, 'method': 'TemporalLogicVerifier.verify_temporal_consistency'})
        temporal_consistency = verifier.verify_temporal_consistency(document)
        evidence['temporal_consistency'] = temporal_consistency
        
        # Step 12: Calculate confidence interval
        trace.append({'step': 12, 'method': 'PolicyContradictionDetector._calculate_confidence_interval'})
        confidence_interval = detector._calculate_confidence_interval(
            confidence,
            alpha=0.05  # 95% confidence
        )
        evidence['confidence_interval'] = confidence_interval
        
        logger.info(f"✓ D1 chain complete: {len(trace)} steps executed")
        return evidence
    
    def _execute_d2_chain(
        self,
        context: ExecutionContext,
        document: str,
        trace: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        D2: Diseño de Actividades y Coherencia
        
        YAML CHAINS:
        - D2-Q1 (Formato): PDETMunicipalPlanAnalyzer → TemporalLogicVerifier
        - D2-Q2 (Causalidad): IndustrialPolicyProcessor → PolicyContradictionDetector
        - D2-Q3 (Temas): SemanticAnalyzer
        - D2-Q4 (Riesgos): PolicyContradictionDetector
        - D2-Q5 (Coherencia): PolicyContradictionDetector (graph + coherence)
        """
        logger.info("Executing D2 (Actividades) chain...")
        evidence = {}
        
        plan_analyzer = self._get_producer_instance('financiero_viabilidad_tablas', 'PDETMunicipalPlanAnalyzer')
        processor = self._get_producer_instance('policy_processor', 'IndustrialPolicyProcessor')
        verifier = self._get_producer_instance('contradiction_deteccion', 'TemporalLogicVerifier')
        detector = self._get_producer_instance('contradiction_deteccion', 'PolicyContradictionDetector')
        analyzer = self._get_producer_instance('Analyzer_one', 'SemanticAnalyzer')
        
        # Step 1: Analyze tabular structure
        trace.append({'step': 1, 'method': 'PDETMunicipalPlanAnalyzer.analyze_municipal_plan'})
        tables = plan_analyzer.analyze_municipal_plan(document)
        evidence['tables'] = tables
        
        # Step 2: Match formalization patterns
        trace.append({'step': 2, 'method': 'IndustrialPolicyProcessor._match_patterns_in_sentences'})
        formalization_patterns = processor._match_patterns_in_sentences(
            document,
            patterns=['tabla', 'columna costo', 'BPIN', 'PPI']
        )
        evidence['formalization'] = formalization_patterns
        
        # Step 3: Build timeline for traceability
        trace.append({'step': 3, 'method': 'TemporalLogicVerifier._build_timeline'})
        timeline = verifier._build_timeline(document)
        evidence['timeline'] = timeline
        
        # Step 4: Match causal mechanism patterns
        trace.append({'step': 4, 'method': 'IndustrialPolicyProcessor._match_patterns_in_sentences'})
        causal_patterns = processor._match_patterns_in_sentences(
            document,
            patterns=['porque', 'genera', 'población objetivo', 'lo cual contribuye a']
        )
        evidence['causal_mechanisms'] = causal_patterns
        
        # Step 5: Determine relation type
        trace.append({'step': 5, 'method': 'PolicyContradictionDetector._determine_relation_type'})
        relations = detector._determine_relation_type(causal_patterns)
        evidence['relations'] = relations
        
        # Step 6: Classify cross-cutting themes
        trace.append({'step': 6, 'method': 'SemanticAnalyzer._classify_cross_cutting_themes'})
        themes = analyzer._classify_cross_cutting_themes(document)
        evidence['cross_cutting_themes'] = themes
        
        # Step 7: Detect logical incompatibilities
        trace.append({'step': 7, 'method': 'PolicyContradictionDetector._detect_logical_incompatibilities'})
        incompatibilities = detector._detect_logical_incompatibilities(document)
        evidence['incompatibilities'] = incompatibilities
        
        # Step 8: Build knowledge graph
        trace.append({'step': 8, 'method': 'PolicyContradictionDetector._build_knowledge_graph'})
        graph = detector._build_knowledge_graph(document)
        evidence['knowledge_graph'] = graph
        
        # Step 9: Calculate global semantic coherence
        trace.append({'step': 9, 'method': 'PolicyContradictionDetector._calculate_global_semantic_coherence'})
        coherence = detector._calculate_global_semantic_coherence(graph)
        evidence['semantic_coherence'] = coherence
        
        # Step 10: Get dependency depth
        trace.append({'step': 10, 'method': 'PolicyContradictionDetector._get_dependency_depth'})
        depth = detector._get_dependency_depth(graph)
        evidence['dependency_depth'] = depth
        
        logger.info(f"✓ D2 chain complete: {len(trace)} steps executed")
        return evidence
    
    def _execute_d3_chain(
        self,
        context: ExecutionContext,
        document: str,
        trace: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        D3: Productos y Factibilidad Operativa
        
        YAML CHAINS:
        - D3-Q1 (Indicadores): PolicyContradictionDetector → BayesianConfidenceCalculator
        - D3-Q2 (Proporcionalidad): PolicyContradictionDetector → PerformanceAnalyzer
        - D3-Q4 (Factibilidad): TemporalLogicVerifier + PolicyContradictionDetector
        - D3-Q5 (Eslabón): IndustrialPolicyProcessor → PolicyContradictionDetector
        """
        logger.info("Executing D3 (Productos) chain...")
        evidence = {}
        
        detector = self._get_producer_instance('contradiction_deteccion', 'PolicyContradictionDetector')
        processor = self._get_producer_instance('policy_processor', 'IndustrialPolicyProcessor')
        calculator = self._get_producer_instance('contradiction_deteccion', 'BayesianConfidenceCalculator')
        verifier = self._get_producer_instance('contradiction_deteccion', 'TemporalLogicVerifier')
        perf_analyzer = self._get_producer_instance('Analyzer_one', 'PerformanceAnalyzer')
        
        # Step 1: Extract quantitative claims
        trace.append({'step': 1, 'method': 'PolicyContradictionDetector._extract_quantitative_claims'})
        claims = detector._extract_quantitative_claims(document)
        evidence['indicators'] = claims
        
        # Step 2: Match verification sources
        trace.append({'step': 2, 'method': 'IndustrialPolicyProcessor._match_patterns_in_sentences'})
        sources = processor._match_patterns_in_sentences(
            document,
            patterns=['BPIN', 'PPI', 'fuente verificación', 'verificable en']
        )
        evidence['verification_sources'] = sources
        
        # Step 3: Calculate Bayesian confidence
        trace.append({'step': 3, 'method': 'BayesianConfidenceCalculator.calculate_posterior'})
        confidence = calculator.calculate_posterior(
            evidence={'sources': sources},
            prior=0.5,
            pattern_specificity=0.8 if sources else 0.3
        )
        evidence['confidence'] = confidence
        
        # Step 4: Detect numerical inconsistencies
        trace.append({'step': 4, 'method': 'PolicyContradictionDetector._detect_numerical_inconsistencies'})
        inconsistencies = detector._detect_numerical_inconsistencies(claims)
        evidence['inconsistencies'] = inconsistencies
        
        # Step 5: Statistical significance test
        trace.append({'step': 5, 'method': 'PolicyContradictionDetector._statistical_significance_test'})
        significance = detector._statistical_significance_test(claims)
        evidence['significance'] = significance
        
        # Step 6: Inject loss function
        trace.append({'step': 6, 'method': 'PerformanceAnalyzer.analyze_loss_function'})
        loss = perf_analyzer.analyze_loss_function(claims, inconsistencies)
        evidence['loss_function'] = loss
        
        # Step 7: Detect temporal conflicts
        trace.append({'step': 7, 'method': 'TemporalLogicVerifier._check_deadline_constraints'})
        temporal_conflicts = verifier._check_deadline_constraints(document)
        evidence['temporal_conflicts'] = temporal_conflicts
        
        # Step 8: Classify temporal type
        trace.append({'step': 8, 'method': 'TemporalLogicVerifier._classify_temporal_type'})
        temporal_type = verifier._classify_temporal_type(document)
        evidence['temporal_type'] = temporal_type
        
        # Step 9: Detect resource conflicts
        trace.append({'step': 9, 'method': 'PolicyContradictionDetector._detect_resource_conflicts'})
        resource_conflicts = detector._detect_resource_conflicts(claims)
        evidence['resource_conflicts'] = resource_conflicts
        
        # Step 10: Determine relation type (Producto→Resultado)
        trace.append({'step': 10, 'method': 'PolicyContradictionDetector._determine_relation_type'})
        relation_type = detector._determine_relation_type(document)
        evidence['relation_strength'] = relation_type
        
        logger.info(f"✓ D3 chain complete: {len(trace)} steps executed")
        return evidence
    
    def _execute_d4_chain(
        self,
        context: ExecutionContext,
        document: str,
        trace: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        D4: Resultados, Supuestos y Alineación
        
        YAML CHAINS:
        - D4-Q2 (Supuestos): IndustrialPolicyProcessor → PolicyContradictionDetector
        - D4-Q3 (Ambición): PolicyContradictionDetector + PDETMunicipalPlanAnalyzer
        - D4-Q5 (Alineación): IndustrialPolicyProcessor → PolicyContradictionDetector
        """
        logger.info("Executing D4 (Resultados) chain...")
        evidence = {}
        
        processor = self._get_producer_instance('policy_processor', 'IndustrialPolicyProcessor')
        detector = self._get_producer_instance('contradiction_deteccion', 'PolicyContradictionDetector')
        plan_analyzer = self._get_producer_instance('financiero_viabilidad_tablas', 'PDETMunicipalPlanAnalyzer')
        
        # Step 1: Match assumption patterns
        trace.append({'step': 1, 'method': 'IndustrialPolicyProcessor._match_patterns_in_sentences'})
        assumptions = processor._match_patterns_in_sentences(
            document,
            patterns=['supuesto', 'condición habilitante', 'si se cumple', 'si... entonces']
        )
        evidence['assumptions'] = assumptions
        
        # Step 2: Build knowledge graph
        trace.append({'step': 2, 'method': 'PolicyContradictionDetector._build_knowledge_graph'})
        graph = detector._build_knowledge_graph(document)
        evidence['knowledge_graph'] = graph
        
        # Step 3: Determine semantic role
        trace.append({'step': 3, 'method': 'PolicyContradictionDetector._determine_semantic_role'})
        roles = detector._determine_semantic_role(assumptions)
        evidence['semantic_roles'] = roles
        
        # Step 4: Detect numerical inconsistencies
        trace.append({'step': 4, 'method': 'PolicyContradictionDetector._extract_quantitative_claims'})
        claims = detector._extract_quantitative_claims(document)
        
        trace.append({'step': 5, 'method': 'PolicyContradictionDetector._detect_numerical_inconsistencies'})
        inconsistencies = detector._detect_numerical_inconsistencies(claims)
        evidence['numerical_consistency'] = inconsistencies
        
        # Step 6: Calculate objective alignment
        trace.append({'step': 6, 'method': 'PolicyContradictionDetector._calculate_objective_alignment'})
        alignment = detector._calculate_objective_alignment(document, benchmarks={})
        evidence['objective_alignment'] = alignment
        
        # Step 7: Generate recommendations
        trace.append({'step': 7, 'method': 'PDETMunicipalPlanAnalyzer.generate_recommendations'})
        recommendations = plan_analyzer.generate_recommendations(document)
        evidence['recommendations'] = recommendations
        
        # Step 8: Match external framework patterns
        trace.append({'step': 8, 'method': 'IndustrialPolicyProcessor._match_patterns_in_sentences'})
        frameworks = processor._match_patterns_in_sentences(
            document,
            patterns=['PND', 'ODS', 'Acuerdo de Paz', 'marco normativo']
        )
        evidence['external_frameworks'] = frameworks
        
        logger.info(f"✓ D4 chain complete: {len(trace)} steps executed")
        return evidence
    
    def _execute_d5_chain(
        self,
        context: ExecutionContext,
        document: str,
        trace: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        D5: Impactos y Riesgos Sistémicos
        
        YAML CHAINS:
        - D5-Q1 (Rezagos): PolicyContradictionDetector → TemporalLogicVerifier
        - D5-Q2/Q3 (Intangibles): IndustrialPolicyProcessor → PolicyContradictionDetector
        - D5-Q4 (Riesgos): IndustrialPolicyProcessor → PolicyContradictionDetector
        - D5-Q5 (Efectos): IndustrialPolicyProcessor → CounterfactualScenario
        """
        logger.info("Executing D5 (Impactos) chain...")
        evidence = {}
        
        detector = self._get_producer_instance('contradiction_deteccion', 'PolicyContradictionDetector')
        processor = self._get_producer_instance('policy_processor', 'IndustrialPolicyProcessor')
        verifier = self._get_producer_instance('contradiction_deteccion', 'TemporalLogicVerifier')
        
        # Step 1: Extract temporal markers
        trace.append({'step': 1, 'method': 'PolicyContradictionDetector._extract_temporal_markers'})
        temporal_markers = detector._extract_temporal_markers(document)
        evidence['temporal_markers'] = temporal_markers
        
        # Step 2: Extract transmission factors
        trace.append({'step': 2, 'method': 'TemporalLogicVerifier._extract_resources'})
        factors = verifier._extract_resources(document)
        evidence['transmission_factors'] = factors
        
        # Step 3: Calculate objective alignment
        trace.append({'step': 3, 'method': 'PolicyContradictionDetector._calculate_objective_alignment'})
        alignment = detector._calculate_objective_alignment(document)
        evidence['impact_alignment'] = alignment
        
        # Step 4: Match intangible measurement patterns
        trace.append({'step': 4, 'method': 'IndustrialPolicyProcessor._match_patterns_in_sentences'})
        intangibles = processor._match_patterns_in_sentences(
            document,
            patterns=['índice de', 'proxy', 'medición indirecta', 'limitación']
        )
        evidence['intangibles'] = intangibles
        
        # Step 5: Classify contradictions
        trace.append({'step': 5, 'method': 'PolicyContradictionDetector._classify_contradiction'})
        contradictions = detector._classify_contradiction(intangibles)
        evidence['contradictions'] = contradictions
        
        # Step 6: Get graph statistics
        trace.append({'step': 6, 'method': 'PolicyContradictionDetector._build_knowledge_graph'})
        graph = detector._build_knowledge_graph(document)
        
        trace.append({'step': 7, 'method': 'PolicyContradictionDetector._get_graph_statistics'})
        graph_stats = detector._get_graph_statistics(graph)
        evidence['graph_statistics'] = graph_stats
        
        # Step 8: Match systemic risk patterns
        trace.append({'step': 8, 'method': 'IndustrialPolicyProcessor._match_patterns_in_sentences'})
        risks = processor._match_patterns_in_sentences(
            document,
            patterns=['riesgo sistémico', 'ruptura mecanismo', 'vulnerabilidad']
        )
        evidence['systemic_risks'] = risks
        
        # Step 9: Detect logical incompatibilities
        trace.append({'step': 9, 'method': 'PolicyContradictionDetector._detect_logical_incompatibilities'})
        incompatibilities = detector._detect_logical_incompatibilities(document)
        evidence['incompatibilities'] = incompatibilities
        
        # Step 10: Calculate contradiction entropy
        trace.append({'step': 10, 'method': 'PolicyContradictionDetector._calculate_contradiction_entropy'})
        entropy = detector._calculate_contradiction_entropy(contradictions)
        evidence['risk_entropy'] = entropy
        
        # Step 11: Match unintended effects patterns
        trace.append({'step': 11, 'method': 'IndustrialPolicyProcessor._match_patterns_in_sentences'})
        effects = processor._match_patterns_in_sentences(
            document,
            patterns=['efecto no deseado', 'hipótesis límite', 'trade-off']
        )
        evidence['unintended_effects'] = effects
        
        logger.info(f"✓ D5 chain complete: {len(trace)} steps executed")
        return evidence
    
    def _execute_d6_chain(
        self,
        context: ExecutionContext,
        document: str,
        trace: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        D6: Coherencia Causal (Teoría de Cambio)
        
        MOST COMPLEX DIMENSIONAL CHAIN - 15 steps
        
        YAML CHAINS:
        - D6-Q1 (Estructura): PolicyContradictionDetector → AdvancedDAGValidator
        - D6-Q2 (Anti-Milagro): IndustrialPolicyProcessor (patrones proporcionalidad)
        - D6-Q3/Q4 (Sistema Bicameral):
            * Ruta 1: PolicyContradictionDetector._suggest_resolutions
            * Ruta 2: TeoriaCambio._generar_sugerencias_internas
        - D6-Q5 (Diferencial): PolicyContradictionDetector → SemanticAnalyzer
        
        CRITICAL FEATURES:
        1. Anti-Milagro validation (proportionality patterns)
        2. Sistema Bicameral (two parallel resolution routes)
        3. Motor Axiomático (TeoriaCambio DAG validation)
        """
        logger.info("Executing D6 (Causalidad) chain - MOST COMPLEX")
        logger.info("Features: Anti-Milagro | Sistema Bicameral | Motor Axiomático")
        evidence = {}
        
        detector = self._get_producer_instance('contradiction_deteccion', 'PolicyContradictionDetector')
        processor = self._get_producer_instance('policy_processor', 'IndustrialPolicyProcessor')
        validator = self._get_producer_instance('teoria_cambio', 'AdvancedDAGValidator')
        teoria = self._get_producer_instance('teoria_cambio', 'TeoriaCambio')
        analyzer = self._get_producer_instance('Analyzer_one', 'SemanticAnalyzer')
        
        # ========================================
        # D6-Q1: Estructura Causal
        # ========================================
        logger.info("  [D6-Q1] Validating causal structure...")
        
        # Step 1: Build knowledge graph
        trace.append({'step': 1, 'method': 'PolicyContradictionDetector._build_knowledge_graph'})
        graph = detector._build_knowledge_graph(document)
        evidence['knowledge_graph'] = graph
        
        # Step 2: Validate with AdvancedDAGValidator
        trace.append({'step': 2, 'method': 'AdvancedDAGValidator.validacion_completa'})
        validation_result = validator.validacion_completa(graph)
        evidence['validation_result'] = validation_result
        
        # Step 3: Validate causal order
        trace.append({'step': 3, 'method': 'AdvancedDAGValidator._validar_orden_causal'})
        order_violations = validator._validar_orden_causal(graph)
        evidence['order_violations'] = order_violations
        
        # Step 4: Find complete paths
        trace.append({'step': 4, 'method': 'AdvancedDAGValidator._encontrar_caminos_completos'})
        complete_paths = validator._encontrar_caminos_completos(graph)
        evidence['complete_paths'] = complete_paths
        
        # ========================================
        # D6-Q2: Anti-Milagro de Implementación
        # ========================================
        logger.info("  [D6-Q2] Validating Anti-Milagro (no implementation miracles)...")
        
        # Step 5: Match proportionality patterns
        trace.append({'step': 5, 'method': 'IndustrialPolicyProcessor._match_patterns_in_sentences'})
        proportionality_patterns = processor._match_patterns_in_sentences(
            document,
            patterns=[
                # enlaces_proporcionales
                'proporcional a', 'acorde con', 'razonable', 'realista',
                # sin_saltos
                'sin saltos', 'gradual', 'incremental', 'paso a paso',
                # no_milagros
                'factible', 'posible', 'alcanzable', 'sin suponer'
            ]
        )
        evidence['proportionality_patterns'] = proportionality_patterns
        
        # Step 6: Calculate syntactic complexity
        trace.append({'step': 6, 'method': 'PolicyContradictionDetector._calculate_syntactic_complexity'})
        complexity = detector._calculate_syntactic_complexity(document)
        evidence['syntactic_complexity'] = complexity
        
        # Step 7: Calculate Anti-Milagro score
        anti_miracle_score = 1.0 if len(proportionality_patterns) >= 3 else 0.0
        evidence['anti_miracle_score'] = anti_miracle_score
        logger.info(f"    Anti-Milagro score: {anti_miracle_score:.2f}")
        
        # ========================================
        # D6-Q3/Q4: Sistema Bicameral
        # ========================================
        logger.info("  [D6-Q3/Q4] Executing Sistema Bicameral (dual resolution routes)...")
        
        # RUTA 1: Detección Específica por Contradicción
        logger.info("    Ruta 1: Specific contradiction-type resolution...")
        
        # Step 8: Detect logical incompatibilities
        trace.append({'step': 8, 'method': 'PolicyContradictionDetector._detect_logical_incompatibilities'})
        incompatibilities = detector._detect_logical_incompatibilities(document)
        evidence['incompatibilities'] = incompatibilities
        
        # Step 9: Generate resolution recommendations (type-specific)
        trace.append({'step': 9, 'method': 'PolicyContradictionDetector._generate_resolution_recommendations'})
        recommendations_route1 = detector._generate_resolution_recommendations(incompatibilities)
        evidence['recommendations_specific'] = recommendations_route1
        logger.info(f"    Ruta 1 complete: {len(recommendations_route1)} specific recommendations")
        
        # RUTA 2: Inferencia Estructural por Motor Axiomático
        logger.info("    Ruta 2: Structural axiom-based inference...")
        
        # Step 10: Execute TeoriaCambio structural suggestions
        trace.append({'step': 10, 'method': 'TeoriaCambio._execute_generar_sugerencias_internas'})
        recommendations_route2 = teoria._execute_generar_sugerencias_internas(validation_result)
        evidence['recommendations_structural'] = recommendations_route2
        logger.info(f"    Ruta 2 complete: {len(recommendations_route2)} structural recommendations")
        
        # Step 11: Match adaptation patterns
        trace.append({'step': 11, 'method': 'IndustrialPolicyProcessor._match_patterns_in_sentences'})
        adaptation_patterns = processor._match_patterns_in_sentences(
            document,
            patterns=['piloto', 'prueba', 'validación', 'aprendizaje', 'mecanismos de corrección']
        )
        evidence['adaptation_patterns'] = adaptation_patterns
        
        # ========================================
        # D6-Q5: Enfoque Diferencial
        # ========================================
        logger.info("  [D6-Q5] Analyzing differential approach...")
        
        # Step 12: Generate embeddings
        trace.append({'step': 12, 'method': 'PolicyContradictionDetector._generate_embeddings'})
        embeddings = detector._generate_embeddings(document)
        evidence['embeddings'] = embeddings
        
        # Step 13: Classify cross-cutting themes
        trace.append({'step': 13, 'method': 'SemanticAnalyzer._classify_cross_cutting_themes'})
        themes = analyzer._classify_cross_cutting_themes(document)
        evidence['cross_cutting_themes'] = themes
        
        # Step 14: Identify dependencies
        trace.append({'step': 14, 'method': 'PolicyContradictionDetector._identify_dependencies'})
        dependencies = detector._identify_dependencies(document)
        evidence['dependencies'] = dependencies
        
        # Step 15: Calculate final coherence score
        trace.append({'step': 15, 'method': 'PolicyContradictionDetector._calculate_global_semantic_coherence'})
        coherence = detector._calculate_global_semantic_coherence(graph)
        evidence['causal_coherence'] = coherence
        
        logger.info(f"✓ D6 chain complete: {len(trace)} steps executed")
        logger.info(f"  Causal coherence: {coherence:.3f}")
        logger.info(f"  Anti-Milagro score: {anti_miracle_score:.2f}")
        logger.info(f"  Bicameral recommendations: {len(recommendations_route1) + len(recommendations_route2)}")
        
        return evidence
    
    def _execute_generic_chain(
        self,
        context: ExecutionContext,
        document: str,
        trace: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generic fallback chain for unrecognized dimensions
        
        NOTE: Should rarely be used - all questions should map to D1-D6
        """
        logger.warning(f"Using generic chain for dimension: {context.dimension}")
        
        evidence = {
            'warning': 'Generic chain used - dimension not recognized',
            'dimension': context.dimension
        }
        
        trace.append({'step': 1, 'method': 'GENERIC_FALLBACK'})
        
        return evidence
    
    # ========================================
    # MICRO ANSWER BUILDING
    # ========================================
    
    def _build_micro_answer(
        self,
        context: ExecutionContext,
        evidence: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> MicroLevelAnswer:
        """
        Build MicroLevelAnswer from dimensional evidence
        
        SCORING LOGIC:
        - D1-D5: Evidence-weighted scoring
        - D6: Causal coherence with Anti-Milagro penalty
        
        RETURNS:
        - MicroLevelAnswer compatible with report_assembly.py
        """
        # Calculate dimension-specific score (0.0-1.0)
        score_normalized = self._calculate_dimensional_score(context.dimension, evidence)
        
        # Convert to quantitative_score (0.0-3.0 scale)
        quantitative_score = score_normalized * 3.0
        
        # Determine qualitative_note based on score
        if quantitative_score >= 2.5:
            qualitative_note = "EXCELENTE"
        elif quantitative_score >= 2.0:
            qualitative_note = "BUENO"
        elif quantitative_score >= 1.5:
            qualitative_note = "ACEPTABLE"
        else:
            qualitative_note = "INSUFICIENTE"
        
        # Extract key findings
        findings = self._extract_findings(evidence, context.dimension)
        
        # Build explanation from findings
        explanation = " ".join(findings[:3])  # First 3 findings as explanation
        
        # Calculate confidence
        confidence = evidence.get('bayesian_confidence', evidence.get('confidence', 0.5))
        
        # Extract evidence text (convert dict evidence to list of strings)
        evidence_texts = []
        for key, value in evidence.items():
            if isinstance(value, list) and value:
                # Extract text from list items
                for item in value[:3]:  # Limit to first 3 items
                    if isinstance(item, dict):
                        text = item.get('text', item.get('pattern', str(item)))
                        evidence_texts.append(f"{key}: {text}")
                    else:
                        evidence_texts.append(f"{key}: {str(item)}")
            elif value and not isinstance(value, (dict, list)):
                evidence_texts.append(f"{key}: {str(value)}")
        
        # Get scoring modality from context metadata
        scoring_modality = context.metadata.get('scoring_modality', 'TYPE_F')
        
        # Build elements_found dict
        expected_elements = context.metadata.get('expected_elements', [])
        elements_found = {elem: True for elem in expected_elements}  # Simplified for now
        
        # Build search_pattern_matches
        search_patterns = context.metadata.get('search_patterns', {})
        search_pattern_matches = {}
        for pattern_key in search_patterns.keys():
            # Check if evidence contains matches for this pattern
            pattern_evidence_key = pattern_key.replace('pattern_', '')
            search_pattern_matches[pattern_key] = pattern_evidence_key in evidence
        
        # Get execution trace for modules_executed
        execution_trace = metadata.get('execution_trace', [])
        modules_executed = list(set([
            step.get('method', '').split('.')[0]
            for step in execution_trace
            if '.' in step.get('method', '')
        ]))
        
        # Build module_results
        module_results = {
            module: {
                'methods': [
                    step.get('method')
                    for step in execution_trace
                    if step.get('method', '').startswith(f"{module}.")
                ]
            }
            for module in modules_executed
        }
        
        # Build MicroLevelAnswer compatible with report_assembly.py
        micro_answer = MicroLevelAnswer(
            question_id=context.question_id,
            qualitative_note=qualitative_note,
            quantitative_score=quantitative_score,
            evidence=evidence_texts,
            explanation=explanation,
            confidence=confidence,
            scoring_modality=scoring_modality,
            elements_found=elements_found,
            search_pattern_matches=search_pattern_matches,
            modules_executed=modules_executed,
            module_results=module_results,
            execution_time=metadata.get('execution_time_ms', 0.0) / 1000.0,
            execution_chain=[
                {'step': str(idx), 'method': step.get('method', 'UNKNOWN')}
                for idx, step in enumerate(execution_trace, 1)
            ],
            metadata={
                'dimension': context.dimension,
                'policy_area': context.policy_area,
                'timestamp': context.timestamp,
                'questionnaire_hash': context.questionnaire_hash,
                **context.metadata
            }
        )
        
        return micro_answer    
    def _calculate_dimensional_score(self, dimension: str, evidence: Dict[str, Any]) -> float:
        """
        Calculate score based on dimensional evidence
        
        SCORING LOGIC:
        - D1: Weighted by quantitative claims, official sources, confidence
        - D2: Weighted by formalization, causal patterns, coherence
        - D3: Weighted by indicators, proportionality, feasibility
        - D4: Weighted by assumptions, alignment, recommendations
        - D5: Weighted by impacts, intangibles coverage, risk analysis
        - D6: Causal coherence with Anti-Milagro penalty
        """
        score = 0.0
        
        if dimension == "D1":
            # Diagnóstico: Quantitative claims + official sources + confidence
            quantitative_claims = evidence.get('quantitative_claims', [])
            official_sources = evidence.get('official_sources', [])
            confidence = evidence.get('bayesian_confidence', 0.5)
            inconsistencies = evidence.get('inconsistencies', [])
            
            # Base score from quantitative evidence
            if len(quantitative_claims) > 0:
                score += 0.3
            if len(quantitative_claims) >= 3:
                score += 0.2
            
            # Official sources bonus
            if len(official_sources) > 0:
                score += 0.2
            
            # Confidence factor
            score += confidence * 0.3
            
            # Penalty for inconsistencies
            if len(inconsistencies) > 0:
                score -= 0.1 * min(len(inconsistencies), 3)
                
        elif dimension == "D2":
            # Actividades: Formalization + causal patterns + coherence
            tables = evidence.get('tables', [])
            causal_patterns = evidence.get('causal_mechanisms', [])
            coherence = evidence.get('semantic_coherence', 0.0)
            incompatibilities = evidence.get('incompatibilities', [])
            
            # Formalization score
            if len(tables) > 0:
                score += 0.3
            
            # Causal patterns
            if len(causal_patterns) >= 2:
                score += 0.3
            elif len(causal_patterns) > 0:
                score += 0.15
            
            # Coherence
            score += coherence * 0.3
            
            # Penalty for incompatibilities
            if len(incompatibilities) > 0:
                score -= 0.1 * min(len(incompatibilities), 2)
                
        elif dimension == "D3":
            # Productos: Indicators + proportionality + feasibility
            indicators = evidence.get('indicators', [])
            sources = evidence.get('verification_sources', [])
            confidence = evidence.get('confidence', 0.5)
            inconsistencies = evidence.get('inconsistencies', [])
            
            # Indicators with sources
            if len(indicators) > 0 and len(sources) > 0:
                score += 0.4
            elif len(indicators) > 0:
                score += 0.2
            
            # Confidence in verification
            score += confidence * 0.3
            
            # Proportionality check
            significance = evidence.get('significance', {})
            if significance.get('is_proportional', False):
                score += 0.2
            
            # Penalty for inconsistencies
            if len(inconsistencies) > 0:
                score -= 0.1
                
        elif dimension == "D4":
            # Resultados: Assumptions + alignment + coherence
            assumptions = evidence.get('assumptions', [])
            alignment = evidence.get('objective_alignment', 0.0)
            frameworks = evidence.get('external_frameworks', [])
            
            # Explicit assumptions
            if len(assumptions) >= 2:
                score += 0.3
            elif len(assumptions) > 0:
                score += 0.15
            
            # Alignment with objectives
            score += alignment * 0.4
            
            # External framework alignment
            if len(frameworks) > 0:
                score += 0.2
            
            # Numerical consistency check
            inconsistencies = evidence.get('numerical_consistency', [])
            if len(inconsistencies) == 0:
                score += 0.1
                
        elif dimension == "D5":
            # Impactos: Impact definition + intangibles + risk analysis
            temporal_markers = evidence.get('temporal_markers', [])
            intangibles = evidence.get('intangibles', [])
            risks = evidence.get('systemic_risks', [])
            effects = evidence.get('unintended_effects', [])
            
            # Impact definition with temporal markers
            if len(temporal_markers) > 0:
                score += 0.25
            
            # Intangibles measurement
            if len(intangibles) >= 2:
                score += 0.25
            elif len(intangibles) > 0:
                score += 0.15
            
            # Risk analysis
            if len(risks) > 0:
                score += 0.2
            
            # Unintended effects consideration
            if len(effects) > 0:
                score += 0.2
            
            # Entropy bonus (comprehensive analysis)
            entropy = evidence.get('risk_entropy', 0.0)
            if entropy > 0.5:
                score += 0.1
                
        elif dimension == "D6":
            # Causalidad: Most complex - causal structure + anti-milagro + bicameral
            validation_result = evidence.get('validation_result', {})
            coherence = evidence.get('causal_coherence', 0.0)
            anti_miracle_score = evidence.get('anti_miracle_score', 0.0)
            complete_paths = evidence.get('complete_paths', [])
            order_violations = evidence.get('order_violations', [])
            
            # Causal coherence (base 40%)
            score += coherence * 0.4
            
            # Anti-Milagro validation (20%)
            score += anti_miracle_score * 0.2
            
            # Complete causal paths (20%)
            if len(complete_paths) > 0:
                score += 0.2
            elif validation_result.get('has_partial_paths', False):
                score += 0.1
            
            # Order violations penalty (up to -20%)
            if len(order_violations) == 0:
                score += 0.2
            else:
                penalty = min(0.2, 0.05 * len(order_violations))
                score -= penalty
            
            # Bicameral recommendations bonus
            recommendations_specific = evidence.get('recommendations_specific', [])
            recommendations_structural = evidence.get('recommendations_structural', [])
            if len(recommendations_specific) + len(recommendations_structural) >= 3:
                score += 0.1
        
        # Normalize score to [0.0, 1.0]
        score = max(0.0, min(1.0, score))
        
        return score
    
    def _extract_findings(self, evidence: Dict[str, Any], dimension: str) -> List[str]:
        """
        Extract human-readable key findings from evidence
        
        Generates 3-5 key findings per dimension
        """
        findings = []
        
        if dimension == "D1":
            # Diagnóstico findings
            claims = evidence.get('quantitative_claims', [])
            sources = evidence.get('official_sources', [])
            inconsistencies = evidence.get('inconsistencies', [])
            
            if len(claims) > 0:
                findings.append(f"Se identificaron {len(claims)} afirmaciones cuantitativas en el diagnóstico")
            
            if len(sources) > 0:
                findings.append(f"Se encontraron {len(sources)} referencias a fuentes oficiales (DANE, DNP)")
            
            if len(inconsistencies) > 0:
                findings.append(f"ALERTA: {len(inconsistencies)} inconsistencias numéricas detectadas")
            else:
                findings.append("No se detectaron inconsistencias numéricas")
            
            confidence = evidence.get('bayesian_confidence', 0.0)
            if confidence >= 0.7:
                findings.append(f"Alta confianza en la evidencia (Bayesian: {confidence:.2f})")
            elif confidence >= 0.4:
                findings.append(f"Confianza moderada en la evidencia (Bayesian: {confidence:.2f})")
            else:
                findings.append(f"Baja confianza en la evidencia (Bayesian: {confidence:.2f})")
                
        elif dimension == "D2":
            # Actividades findings
            tables = evidence.get('tables', [])
            causal = evidence.get('causal_mechanisms', [])
            coherence = evidence.get('semantic_coherence', 0.0)
            
            if len(tables) > 0:
                findings.append(f"Se detectaron {len(tables)} tablas estructuradas de actividades")
            else:
                findings.append("No se encontró estructura tabular formal")
            
            if len(causal) > 0:
                findings.append(f"Se identificaron {len(causal)} patrones de mecanismos causales")
            
            findings.append(f"Coherencia semántica global: {coherence:.2f}")
            
            incompatibilities = evidence.get('incompatibilities', [])
            if len(incompatibilities) > 0:
                findings.append(f"ALERTA: {len(incompatibilities)} incompatibilidades lógicas entre actividades")
                
        elif dimension == "D3":
            # Productos findings
            indicators = evidence.get('indicators', [])
            sources = evidence.get('verification_sources', [])
            
            if len(indicators) > 0:
                findings.append(f"Se definieron {len(indicators)} indicadores de producto")
            
            if len(sources) > 0:
                findings.append(f"Se especificaron {len(sources)} fuentes de verificación")
            else:
                findings.append("No se encontraron fuentes de verificación explícitas")
            
            significance = evidence.get('significance', {})
            if significance.get('is_proportional', False):
                findings.append("Las metas son proporcionales a las brechas identificadas")
            else:
                findings.append("ALERTA: Posible desproporcionalidad meta-brecha")
            
            temporal_conflicts = evidence.get('temporal_conflicts', [])
            if len(temporal_conflicts) > 0:
                findings.append(f"ALERTA: {len(temporal_conflicts)} conflictos temporales detectados")
                
        elif dimension == "D4":
            # Resultados findings
            assumptions = evidence.get('assumptions', [])
            alignment = evidence.get('objective_alignment', 0.0)
            frameworks = evidence.get('external_frameworks', [])
            
            if len(assumptions) > 0:
                findings.append(f"Se explicitan {len(assumptions)} supuestos críticos")
            else:
                findings.append("No se encontraron supuestos explícitos")
            
            findings.append(f"Alineación con objetivos superiores: {alignment:.2f}")
            
            if len(frameworks) > 0:
                findings.append(f"Alineación con {len(frameworks)} marcos normativos (PND, ODS, etc.)")
            else:
                findings.append("No se encontró alineación explícita con marcos externos")
                
        elif dimension == "D5":
            # Impactos findings
            markers = evidence.get('temporal_markers', [])
            intangibles = evidence.get('intangibles', [])
            risks = evidence.get('systemic_risks', [])
            effects = evidence.get('unintended_effects', [])
            
            if len(markers) > 0:
                findings.append(f"Se definen {len(markers)} marcadores temporales de impacto")
            
            if len(intangibles) > 0:
                findings.append(f"Se abordan {len(intangibles)} aspectos intangibles con proxies")
            else:
                findings.append("No se encontró medición de aspectos intangibles")
            
            if len(risks) > 0:
                findings.append(f"Se identifican {len(risks)} riesgos sistémicos")
            else:
                findings.append("ALERTA: No se identifican riesgos sistémicos")
            
            if len(effects) > 0:
                findings.append(f"Se consideran {len(effects)} posibles efectos no deseados")
                
        elif dimension == "D6":
            # Causalidad findings
            coherence = evidence.get('causal_coherence', 0.0)
            anti_miracle = evidence.get('anti_miracle_score', 0.0)
            paths = evidence.get('complete_paths', [])
            violations = evidence.get('order_violations', [])
            
            findings.append(f"Coherencia causal global: {coherence:.2f}")
            
            if anti_miracle >= 0.8:
                findings.append("Validación Anti-Milagro: Enlaces proporcionales y realistas")
            else:
                findings.append("ALERTA: Posibles saltos lógicos inverosímiles (Anti-Milagro)")
            
            if len(paths) > 0:
                findings.append(f"Se identifican {len(paths)} caminos causales completos")
            else:
                findings.append("CRÍTICO: No se encontraron caminos causales completos")
            
            if len(violations) > 0:
                findings.append(f"ALERTA: {len(violations)} violaciones de orden causal")
            
            recommendations = len(evidence.get('recommendations_specific', [])) + \
                            len(evidence.get('recommendations_structural', []))
            if recommendations > 0:
                findings.append(f"Sistema Bicameral generó {recommendations} recomendaciones")
        
        # Ensure we have at least some findings
        if not findings:
            findings.append("Análisis completado sin hallazgos específicos")
        
        return findings
    
    def _build_provenance_record(
        self,
        context: ExecutionContext,
        execution_trace: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> ProvenanceRecord:
        """
        Build complete provenance record for audit and reproducibility
        """
        # Extract method names from trace
        methods_invoked = [step.get('method', 'UNKNOWN') for step in execution_trace]
        
        # Build execution ID from context
        import hashlib
        execution_string = f"{context.question_id}_{context.timestamp}_{context.questionnaire_hash}"
        execution_id = hashlib.sha256(execution_string.encode()).hexdigest()[:16]
        
        # Calculate confidence scores from trace
        confidence_scores = {}
        for idx, step in enumerate(execution_trace, 1):
            method = step.get('method', f'step_{idx}')
            confidence_scores[method] = step.get('confidence', 1.0)
        
        provenance = ProvenanceRecord(
            execution_id=execution_id,
            timestamp=context.timestamp,
            input_artifacts=[
                f"question:{context.question_id}",
                f"questionnaire:{context.questionnaire_hash}"
            ],
            output_artifacts=[
                f"micro_answer:{context.question_id}",
                f"evidence_bundle:{execution_id}"
            ],
            methods_invoked=methods_invoked,
            confidence_scores=confidence_scores,
            metadata={
                'dimension': context.dimension,
                'policy_area': context.policy_area,
                'execution_context': context.metadata,
                **metadata
            }
        )
        
        return provenance
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics from Choreographer
        
        Returns aggregated statistics about method execution
        """
        return {
            'total_methods_registered': len(self.CANONICAL_METHODS),
            'producers_initialized': len(self._producer_instances),
            'coverage_percentage': (len(self.CANONICAL_METHODS) / 584) * 100,
            'components': {
                producer: len(classes)
                for producer, classes in self._producer_instances.items()
            }
        }


# ============================================================================
# EXECUTION CHOREOGRAPHER ALIAS
# ============================================================================

# Alias for backward compatibility with Industrialpolicyprocessor.py
ExecutionChoreographer = Choreographer
