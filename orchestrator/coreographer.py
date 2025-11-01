"""Lightweight choreographer façade used by the test-suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import MethodExecutor as _CoreMethodExecutor
from . import PreprocessedDocument
from .choreographer_dispatch import ChoreographerDispatcher

class MethodPriority(Enum):
    """Execution priority levels for catalogued methods."""

    CRITICO = 3
    IMPORTANTE = 2
    COMPLEMENTARIO = 1

@dataclass
class DAGNode:
    """Represents a node in the execution DAG."""

    node_id: str
    file_name: str
    class_name: str
    method_names: List[str]
    method_types: List[str]
    priorities: List[int]
    timeout_ms: int = 30000
    max_retries: int = 2

@dataclass
class ExecutionPlan:
    """Simple execution plan describing the orchestrator flow."""

    nodes: List[DAGNode] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)

@dataclass
class MethodResult:
    """Result of an executed method."""

    method_name: str
    success: bool
    result: Any = None
    execution_time_ms: float = 0.0
    retries_used: int = 0

@dataclass
class NodeResult:
    """Aggregated result for a DAG node."""

    node_id: str
    success: bool
    method_results: List[MethodResult]
    execution_time_ms: float = 0.0

@dataclass
class QuestionResult:
    """Aggregated result for a single questionnaire item."""

    question_global: int
    base_slot: str
    evidence: Dict[str, Any]
    raw_results: Dict[str, Any]
    execution_time_ms: float = 0.0

class FlowController:
    """Utilities for constructing and inspecting execution plans."""

    @staticmethod
    def build_execution_dag(
        _monolith: Optional[Dict[str, Any]],
        method_packages: Iterable[Dict[str, Any]],
    ) -> ExecutionPlan:
        nodes: List[DAGNode] = []
        for index, package in enumerate(method_packages, start=1):
            file_name = package.get("f") or package.get("file_name", "unknown.py")
            class_name = package.get("c") or package.get("class", "UnknownClass")
            method_names = list(package.get("m", package.get("methods", [])))
            method_types = list(package.get("t", package.get("types", [])))
            priorities = [int(p) for p in package.get("pr", package.get("priorities", []))]
            node = DAGNode(
                node_id=f"node_{index}",
                file_name=file_name,
                class_name=class_name,
                method_names=method_names,
                method_types=method_types,
                priorities=priorities,
                timeout_ms=int(package.get("timeout_ms", 30000)),
                max_retries=int(package.get("max_retries", 2)),
            )
            nodes.append(node)
        execution_order = [node.node_id for node in nodes]
        return ExecutionPlan(nodes=nodes, parallel_groups=[], execution_order=execution_order)

    @staticmethod
    def identify_parallel_branches(plan: ExecutionPlan) -> List[List[str]]:
        return list(plan.parallel_groups)

class MethodExecutor(_CoreMethodExecutor):
    """Method executor with dispatcher awareness for compatibility tests."""

    def __init__(self, dispatcher: Optional[ChoreographerDispatcher] = None) -> None:
        self.dispatcher = dispatcher or ChoreographerDispatcher()
        super().__init__()

class Choreographer:
    """Minimal choreographer capable of mapping questions to execution slots."""

    def __init__(self, dispatcher: Optional[ChoreographerDispatcher] = None) -> None:
        self.dispatcher = dispatcher or ChoreographerDispatcher()
        self.method_executor = MethodExecutor(self.dispatcher)
        self.default_timeout_ms = 30000
        self.default_max_retries = 2

    @staticmethod
    def _dimension_to_slot(dimension_id: str) -> str:
        digits = "".join(ch for ch in dimension_id if ch.isdigit())
        number = int(digits or "1")
        return f"D{number}"

    def _map_question_to_slot(
        self,
        question_global: int,
        monolith: Dict[str, Any],
        method_catalog: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        question = next(
            (q for q in monolith.get("questions", []) if q.get("question_global") == question_global),
            None,
        )
        if not question:
            raise KeyError(f"Question {question_global} not found in monolith")

        dimension_slot = self._dimension_to_slot(question.get("dimension_id", "DIM01"))
        base_slot = f"{dimension_slot}-Q{question_global}"
        method_packages = method_catalog.get(question.get("question_id"), [])
        flow_spec = {
            "timeout_ms": self.default_timeout_ms,
            "max_retries": self.default_max_retries,
        }
        return base_slot, question, method_packages, flow_spec

__all__ = [
    "Choreographer",
    "ChoreographerDispatcher",
    "DAGNode",
    "ExecutionPlan",
    "FlowController",
    "MethodExecutor",
    "MethodPriority",
    "MethodResult",
    "NodeResult",
    "PreprocessedDocument",
    "QuestionResult",
]
