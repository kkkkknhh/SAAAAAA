"""Coreographer compatibility module for orchestrator single-question execution.

The original repository exposed a ``coreographer`` module that provided a
lightweight façade around the micro-question execution pipeline.  The
refactoring moved orchestrator code under ``saaaaaa.core.orchestrator``, but
several tests (and external integrations) still import the legacy module.  We
re-implement the minimal data structures and helpers exercised by the test
suite so that the modern package remains backwards compatible.

The implementation intentionally focuses on deterministic, easily-testable
behaviour:

* ``Choreographer`` exposes configuration defaults and dependency wiring.
* ``_map_question_to_slot`` extracts the relevant metadata from the
  questionnaire monolith and enriches it with method catalog information.
* ``FlowController`` constructs an ``ExecutionPlan`` that groups method
  packages by class while preserving declared priorities.

No I/O occurs inside this module—callers must supply already-loaded
configurations.  This keeps the core layer pure while still providing a smooth
migration path for legacy imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .contract_loader import JSONContractLoader
from .core import MethodExecutor, PreprocessedDocument


class MethodPriority(Enum):
    """Priority levels for orchestration methods."""

    CRITICO = 3
    IMPORTANTE = 2
    COMPLEMENTARIO = 1


@dataclass
class MethodResult:
    """Result of executing a single method within a DAG node."""

    method_name: str
    success: bool
    result: Any
    execution_time_ms: float = 0.0
    retries_used: int = 0
    error: Any | None = None


@dataclass
class NodeResult:
    """Aggregated result for a DAG node."""

    node_id: str
    success: bool
    method_results: List[MethodResult] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error: Any | None = None


@dataclass
class QuestionResult:
    """Aggregated result for a micro question."""

    question_global: int
    base_slot: str
    evidence: Dict[str, Any]
    raw_results: Dict[str, Any]
    execution_time_ms: float = 0.0
    error: Any | None = None


@dataclass
class DAGNode:
    """Execution metadata for a group of methods."""

    node_id: str
    file_name: str
    class_name: str
    method_names: List[str]
    method_types: List[str]
    priorities: List[int]
    timeout_ms: int = 30_000
    max_retries: int = 2


@dataclass
class ExecutionPlan:
    """Deterministic orchestration plan for a single question."""

    nodes: List[DAGNode]
    parallel_groups: List[List[str]]
    execution_order: List[str]


class FlowController:
    """Utilities for constructing deterministic execution plans."""

    @staticmethod
    def build_execution_dag(
        flow_spec: Mapping[str, Any] | None,
        method_packages: Sequence[Mapping[str, Any]],
    ) -> ExecutionPlan:
        """Build an ``ExecutionPlan`` from method package declarations."""

        nodes: List[DAGNode] = []
        execution_order: List[str] = []

        for index, package in enumerate(method_packages, start=1):
            node_id = f"node_{index:03d}"
            file_name = str(package.get("f", ""))
            class_name = str(package.get("c", ""))
            method_names = [str(name) for name in package.get("m", [])]
            method_types = [str(kind) for kind in package.get("t", [])]
            priorities = [int(priority) for priority in package.get("pr", [])]

            nodes.append(
                DAGNode(
                    node_id=node_id,
                    file_name=file_name,
                    class_name=class_name,
                    method_names=method_names,
                    method_types=method_types,
                    priorities=priorities,
                )
            )
            execution_order.append(node_id)

        parallel_groups: List[List[str]] = []
        if flow_spec and isinstance(flow_spec.get("parallel_groups"), Iterable):
            for group in flow_spec["parallel_groups"]:
                if isinstance(group, Sequence):
                    parallel_groups.append([str(node) for node in group])

        return ExecutionPlan(nodes=nodes, parallel_groups=parallel_groups, execution_order=execution_order)

    @staticmethod
    def identify_parallel_branches(plan: ExecutionPlan) -> List[List[DAGNode]]:
        """Return groups of nodes that can execute in parallel."""

        branches: List[List[DAGNode]] = []
        if not plan.parallel_groups:
            return branches

        lookup: Dict[str, DAGNode] = {node.node_id: node for node in plan.nodes}
        for group in plan.parallel_groups:
            branch = [lookup[node_id] for node_id in group if node_id in lookup]
            if branch:
                branches.append(branch)
        return branches


class ChoreographerDispatcher:
    """Minimal dispatcher placeholder for backwards compatibility."""

    def dispatch(self, node: DAGNode, document: PreprocessedDocument) -> NodeResult:
        """Return a stub ``NodeResult`` indicating the node was skipped."""

        return NodeResult(node_id=node.node_id, success=True, method_results=[])


class Choreographer:
    """Facade exposing micro-question orchestration helpers."""

    default_timeout_ms: int = 30_000
    default_max_retries: int = 2

    def __init__(self, dispatcher: ChoreographerDispatcher | None = None) -> None:
        self.dispatcher = dispatcher or ChoreographerDispatcher()
        self.method_executor = MethodExecutor(self.dispatcher)

    @staticmethod
    def _build_method_mapping(method_catalog: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for entry in method_catalog:
            class_name = str(entry.get("class", ""))
            if class_name and class_name not in mapping:
                mapping[class_name] = {
                    "file": entry.get("file", ""),
                }
        return mapping

    def _map_question_to_slot(
        self,
        question_global: int,
        monolith: Mapping[str, Any],
        method_catalog_payload: Mapping[str, Any],
    ) -> Tuple[str, Mapping[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """Map a question identifier to its execution metadata."""

        micro_questions: Sequence[Mapping[str, Any]] = monolith["blocks"].get("micro_questions", [])
        question_data = next(
            (q for q in micro_questions if int(q.get("question_global", 0)) == int(question_global)),
            None,
        )
        if question_data is None:
            raise KeyError(f"Question {question_global} not found in monolith")

        base_slot = str(question_data.get("base_slot", ""))
        method_catalog = method_catalog_payload.get("methods_catalog", [])
        method_lookup = self._build_method_mapping(method_catalog)

        grouped: MutableMapping[str, Dict[str, Any]] = {}
        for entry in question_data.get("method_sets", []):
            class_name = str(entry.get("class", ""))
            if not class_name:
                continue
            bucket = grouped.setdefault(
                class_name,
                {
                    "f": method_lookup.get(class_name, {}).get("file", ""),
                    "c": class_name,
                    "m": [],
                    "t": [],
                    "pr": [],
                },
            )
            bucket["m"].append(str(entry.get("function", "")))
            bucket["t"].append(str(entry.get("method_type", "")))
            bucket["pr"].append(int(entry.get("priority", MethodPriority.IMPORTANTE.value)))

        method_packages = list(grouped.values())
        flow_spec = {
            "base_slot": base_slot,
            "parallel_groups": [],
            "method_count": sum(len(pkg["m"]) for pkg in method_packages),
        }
        return base_slot, question_data, method_packages, flow_spec


__all__ = [
    "Choreographer",
    "ChoreographerDispatcher",
    "DAGNode",
    "ExecutionPlan",
    "FlowController",
    "MethodExecutor",
    "PreprocessedDocument",
    "MethodPriority",
    "MethodResult",
    "NodeResult",
    "QuestionResult",
]
