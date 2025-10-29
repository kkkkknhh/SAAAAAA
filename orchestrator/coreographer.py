"""
Choreographer - Single Micro-Question Execution Engine

This module implements ONLY the execution of a SINGLE micro-question as specified
in PSEUDOCODIGO_FLUJO_COMPLETO.md function: process_micro_question()

The choreographer's SOLE responsibility is to execute ONE question's methods.
It does NOT know about:
- Global pipeline (FASE 0-10) 
- Scoring or aggregations
- Other questions
- Report assembly

It ONLY knows about:
- How to execute ONE question's methods via DAG
- Method-level error handling with priorities
- Evidence extraction from one question
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from orchestrator.choreographer_dispatch import (
    ChoreographerDispatcher,
    InvocationContext,
)

logger = logging.getLogger(__name__)


class MethodPriority(Enum):
    """Method execution priority levels."""
    CRITICO = 3      # Must succeed or fail the entire question
    IMPORTANTE = 2   # Log error but continue
    COMPLEMENTARIO = 1  # Skip on error


@dataclass
class DAGNode:
    """Node in the execution DAG."""
    node_id: str
    file_name: str
    class_name: str
    method_names: List[str]
    method_types: List[str]
    priorities: List[int]
    dependencies: List[str] = field(default_factory=list)
    timeout_ms: int = 30000
    max_retries: int = 0


@dataclass
class ExecutionPlan:
    """Complete execution plan for a question."""
    nodes: List[DAGNode]
    parallel_groups: List[List[str]]
    execution_order: List[str]


@dataclass
class MethodResult:
    """Result of executing a single method."""
    method_name: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time_ms: float = 0.0
    retries_used: int = 0


@dataclass
class NodeResult:
    """Result of executing a DAG node."""
    node_id: str
    success: bool
    method_results: List[MethodResult]
    execution_time_ms: float = 0.0


@dataclass
class QuestionResult:
    """Result of processing a single micro question."""
    question_global: int
    base_slot: str
    evidence: Dict[str, Any]
    raw_results: Dict[str, Any]
    execution_time_ms: float = 0.0
    node_results: List[NodeResult] = field(default_factory=list)


@dataclass
class PreprocessedDocument:
    """Preprocessed document passed to choreographer."""
    document_id: str
    raw_text: str
    normalized_text: str
    sentences: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    indexes: Dict[str, Any]
    metadata: Dict[str, Any]


class FlowController:
    """Builds and manages execution DAG from flow specifications."""
    
    @staticmethod
    def build_execution_dag(flow_spec: Optional[str], method_packages: List[Dict[str, Any]]) -> ExecutionPlan:
        """Build execution DAG from flow specification."""
        nodes = []
        
        for idx, package in enumerate(method_packages):
            node = DAGNode(
                node_id=f"node_{idx}",
                file_name=package.get('f', ''),
                class_name=package.get('c', ''),
                method_names=package.get('m', []),
                method_types=package.get('t', []),
                priorities=package.get('pr', [1] * len(package.get('m', []))),
                timeout_ms=package.get('timeout_ms', 30000),
                max_retries=package.get('max_retries', 0)
            )
            nodes.append(node)
        
        execution_order = [node.node_id for node in nodes]
        parallel_groups = []
        
        return ExecutionPlan(
            nodes=nodes,
            parallel_groups=parallel_groups,
            execution_order=execution_order
        )
    
    @staticmethod
    def identify_parallel_branches(execution_plan: ExecutionPlan) -> List[List[str]]:
        """Identify which nodes can be executed in parallel."""
        return execution_plan.parallel_groups


class MethodExecutor:
    """Executes individual methods with retry and timeout logic."""
    
    def __init__(self, dispatcher: ChoreographerDispatcher):
        self.dispatcher = dispatcher
    
    def execute_method(
        self,
        file_name: str,
        class_name: str,
        method_name: str,
        context: InvocationContext,
        priority: int = MethodPriority.COMPLEMENTARIO.value,
        timeout_ms: int = 30000,
        max_retries: int = 0,
    ) -> MethodResult:
        """Execute a single method with retry and timeout logic."""
        start = time.time()
        fqn = f"{class_name}.{method_name}"
        
        retries_used = 0
        last_error = None
        
        while retries_used <= max_retries:
            try:
                result = self.dispatcher.invoke_method(fqn, context)
                execution_time = (time.time() - start) * 1000
                
                if result.success:
                    return MethodResult(
                        method_name=method_name,
                        success=True,
                        result=result.result,
                        execution_time_ms=execution_time,
                        retries_used=retries_used
                    )
                else:
                    last_error = result.error
                    
                    if priority == MethodPriority.CRITICO.value and retries_used < max_retries:
                        retries_used += 1
                        logger.warning(f"Critical method {fqn} failed, retry {retries_used}/{max_retries}")
                        time.sleep(0.1 * retries_used)
                        continue
                    else:
                        break
                        
            except Exception as e:
                last_error = e
                
                if priority == MethodPriority.CRITICO.value and retries_used < max_retries:
                    retries_used += 1
                    logger.warning(f"Critical method {fqn} exception, retry {retries_used}/{max_retries}: {e}")
                    time.sleep(0.1 * retries_used)
                    continue
                else:
                    break
        
        execution_time = (time.time() - start) * 1000
        
        if priority == MethodPriority.CRITICO.value:
            logger.error(f"CRITICAL method {fqn} failed after {retries_used} retries: {last_error}")
            raise RuntimeError(f"Critical method {fqn} failed: {last_error}")
        elif priority == MethodPriority.IMPORTANTE.value:
            logger.warning(f"Important method {fqn} failed: {last_error}")
        else:
            logger.debug(f"Complementary method {fqn} failed: {last_error}")
        
        return MethodResult(
            method_name=method_name,
            success=False,
            error=last_error,
            execution_time_ms=execution_time,
            retries_used=retries_used
        )


class Choreographer:
    """
    Choreographer - Executes a SINGLE micro question.
    
    ONLY responsible for ONE question execution.
    NOT responsible for global pipeline, scoring, aggregation, or reports.
    """
    
    def __init__(
        self,
        dispatcher: Optional[ChoreographerDispatcher] = None,
        default_timeout_ms: int = 30000,
        default_max_retries: int = 2,
    ):
        self.dispatcher = dispatcher or ChoreographerDispatcher()
        self.method_executor = MethodExecutor(self.dispatcher)
        self.default_timeout_ms = default_timeout_ms
        self.default_max_retries = default_max_retries
    
    def _map_question_to_slot(
        self,
        question_global: int,
        monolith: Dict[str, Any],
        method_catalog: Dict[str, Any],
    ) -> tuple:
        """Map question to base_slot and method packages."""
        base_index = (question_global - 1) % 30
        base_slot = f"D{base_index // 5 + 1}-Q{base_index % 5 + 1}"
        
        q_metadata = monolith['blocks']['micro_questions'][question_global - 1]
        
        dimension_index = base_index // 5
        question_in_dimension = base_index % 5
        
        dimensions = method_catalog.get('dimensions', [])
        if dimension_index >= len(dimensions):
            return base_slot, q_metadata, [], None
        
        dimension = dimensions[dimension_index]
        questions = dimension.get('questions', [])
        if question_in_dimension >= len(questions):
            return base_slot, q_metadata, [], None
        
        base_question = questions[question_in_dimension]
        method_packages = base_question.get('p', [])
        flow_spec = base_question.get('flow', None)
        
        return base_slot, q_metadata, method_packages, flow_spec
    
    def _execute_node(self, node: DAGNode, context: InvocationContext) -> NodeResult:
        """Execute all methods in a DAG node."""
        start = time.time()
        method_results = []
        
        for i, method_name in enumerate(node.method_names):
            priority = node.priorities[i] if i < len(node.priorities) else MethodPriority.COMPLEMENTARIO.value
            
            method_result = self.method_executor.execute_method(
                file_name=node.file_name,
                class_name=node.class_name,
                method_name=method_name,
                context=context,
                priority=priority,
                timeout_ms=node.timeout_ms,
                max_retries=node.max_retries if priority == MethodPriority.CRITICO.value else 0
            )
            
            method_results.append(method_result)
            
            if method_result.success and method_result.result is not None:
                cache_key = f"{node.class_name}.{method_name}"
                context.extra_kwargs[cache_key] = method_result.result
        
        execution_time = (time.time() - start) * 1000
        success = any(r.success for r in method_results)
        
        return NodeResult(
            node_id=node.node_id,
            success=success,
            method_results=method_results,
            execution_time_ms=execution_time
        )
    
    def _extract_evidence(
        self,
        all_node_results: List[NodeResult],
        q_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract evidence from node results."""
        total_methods = sum(len(nr.method_results) for nr in all_node_results)
        successful_methods = sum(
            sum(1 for mr in nr.method_results if mr.success)
            for nr in all_node_results
        )
        
        return {
            'total_methods': total_methods,
            'successful_methods': successful_methods,
            'success_rate': successful_methods / total_methods if total_methods > 0 else 0.0,
            'confidence': successful_methods / total_methods if total_methods > 0 else 0.0,
            'patterns_found': [],
        }
    
    def process_micro_question(
        self,
        question_global: int,
        preprocessed_doc: PreprocessedDocument,
        monolith: Dict[str, Any],
        method_catalog: Dict[str, Any],
    ) -> QuestionResult:
        """
        Process a SINGLE micro question (MAIN CHOREOGRAPHER FUNCTION).
        
        This is the ONLY public method. It executes ONE question and returns results.
        """
        start_time = time.time()
        
        # PASO 1: Mapeo
        base_slot, q_metadata, method_packages, flow_spec = self._map_question_to_slot(
            question_global, monolith, method_catalog
        )
        
        # PASO 2: Construcción del DAG
        execution_plan = FlowController.build_execution_dag(flow_spec, method_packages)
        
        # PASO 3: Preparar contexto
        context = InvocationContext(
            text=preprocessed_doc.normalized_text or preprocessed_doc.raw_text,
            data={'preprocessed_doc': preprocessed_doc},
            document=preprocessed_doc,
            questionnaire=monolith,
            question_id=f"Q{question_global:03d}",
            metadata={
                'question_global': question_global,
                'base_slot': base_slot,
            }
        )
        
        # PASO 4: Ejecución topológica
        all_node_results = []
        for node in execution_plan.nodes:
            try:
                node_result = self._execute_node(node, context)
                all_node_results.append(node_result)
            except Exception as e:
                logger.error(f"Node {node.node_id} failed critically: {e}")
                raise
        
        # PASO 5: Extracción de evidencia
        evidence = self._extract_evidence(all_node_results, q_metadata)
        
        raw_results = {}
        for node_result in all_node_results:
            for method_result in node_result.method_results:
                raw_results[method_result.method_name] = method_result.result
        
        execution_time = (time.time() - start_time) * 1000
        
        return QuestionResult(
            question_global=question_global,
            base_slot=base_slot,
            evidence=evidence,
            raw_results=raw_results,
            execution_time_ms=execution_time,
            node_results=all_node_results
        )


__all__ = [
    "Choreographer",
    "QuestionResult",
    "PreprocessedDocument",
    "DAGNode",
    "ExecutionPlan",
    "MethodResult",
    "NodeResult",
    "MethodPriority",
    "FlowController",
    "MethodExecutor",
]
