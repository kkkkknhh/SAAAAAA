"""Golden-path contract tests for orchestrator executors."""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import importlib
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if "recommendation_engine" not in sys.modules:
    dummy = types.ModuleType("recommendation_engine")

    class _DummyRecommendationEngine:
        def __init__(self, *_, **__):
            self.rules_by_level = {"MICRO": [], "MESO": [], "MACRO": []}

        def generate_all_recommendations(self, **_kwargs):
            return {
                level: types.SimpleNamespace(
                    recommendations=[],
                    to_dict=lambda _level=level: {"level": _level, "recommendations": []},
                )
                for level in ("MICRO", "MESO", "MACRO")
            }

    dummy.RecommendationEngine = _DummyRecommendationEngine
    sys.modules["recommendation_engine"] = dummy

core_contracts = importlib.import_module("saaaaaa.utils.core_contracts")
sys.modules.setdefault("core_contracts", core_contracts)

from saaaaaa.core.orchestrator import executors
from saaaaaa.core.orchestrator.core import PreprocessedDocument
from saaaaaa.core.orchestrator.factory import CoreModuleFactory, construct_policy_processor_input


def _contract_keys(contract: Mapping[str, object]) -> List[str]:
    """Return the declared keys for a TypedDict class."""

    annotations = getattr(contract, "__annotations__", {})
    return sorted(annotations.keys())


CONTRACT_KEY_MAP: Dict[str, Sequence[str]] = {
    "IndustrialPolicyProcessor": _contract_keys(core_contracts.PolicyProcessorOutputContract),
    "PolicyContradictionDetector": _contract_keys(core_contracts.ContradictionDetectorOutputContract),
    "PolicyAnalysisEmbedder": _contract_keys(core_contracts.EmbeddingPolicyOutputContract),
    "AdvancedSemanticChunker": _contract_keys(core_contracts.EmbeddingPolicyOutputContract),
    "SemanticAnalyzer": _contract_keys(core_contracts.SemanticAnalyzerOutputContract),
    "CDAFFramework": _contract_keys(core_contracts.CDAFFrameworkOutputContract),
    "PDETMunicipalPlanAnalyzer": _contract_keys(core_contracts.PDETAnalyzerOutputContract),
    "TeoriaCambio": _contract_keys(core_contracts.TeoriaCambioOutputContract),
    "SemanticChunker": _contract_keys(core_contracts.SemanticChunkingOutputContract),
}


class FakeMethodExecutor:
    """Deterministic stub for :class:`MethodExecutor`."""

    def __init__(self, contract_map: Mapping[str, Sequence[str]]):
        self.contract_map = contract_map
        self.calls: List[Tuple[str, str]] = []
        self.outputs: Dict[str, Dict[str, object]] = {}

    def expected_keys(self, class_name: str) -> Sequence[str]:
        return self.contract_map.get(class_name, ())

    def execute(self, class_name: str, method_name: str, **kwargs):
        self.calls.append((class_name, method_name))
        keys = list(self.expected_keys(class_name))
        if not keys:
            result = {"result": f"{class_name}.{method_name}", "payload": kwargs}
        else:
            result = {key: f"{class_name}:{key}" for key in keys}
        self.outputs[f"{class_name}.{method_name}"] = result
        return result


@pytest.fixture()
def factory(tmp_path: Path) -> CoreModuleFactory:
    """Provide a factory anchored to a temporary directory."""

    tmp_data = tmp_path / "data"
    tmp_data.mkdir()
    questionnaire = tmp_data / "questionnaire_monolith.json"
    questionnaire.write_text("{}", encoding="utf-8")
    return CoreModuleFactory(data_dir=tmp_data)


@pytest.fixture()
def sample_document(factory: CoreModuleFactory, tmp_path: Path) -> Tuple[PreprocessedDocument, Dict[str, object]]:
    """Create a sample document and policy processor input contract."""

    document_path = tmp_path / "plan.txt"
    document_path.write_text("Objetivo general. Línea base.", encoding="utf-8")
    document_data = factory.load_document(document_path)
    policy_input = construct_policy_processor_input(document_data)
    preprocessed = PreprocessedDocument(
        document_id="doc-1",
        raw_text=document_data["raw_text"],
        sentences=list(document_data["sentences"]),
        tables=list(document_data["tables"]),
        metadata={**document_data.get("metadata", {}), "document_id": "doc-1"},
    )
    return preprocessed, policy_input


def _iter_executor_classes() -> Iterable[Tuple[str, type]]:
    for name in dir(executors):
        if not name.endswith("_Executor"):
            continue
        candidate = getattr(executors, name)
        if inspect.isclass(candidate) and candidate is not executors.DataFlowExecutor:
            yield name, candidate


@pytest.mark.parametrize("executor_name, executor_cls", sorted(_iter_executor_classes()))
def test_executor_golden_path_returns_contracts(
    executor_name: str,
    executor_cls: type,
    sample_document: Tuple[PreprocessedDocument, Dict[str, object]],
) -> None:
    document, policy_input = sample_document
    assert set(policy_input.keys()) >= {"data", "text", "sentences", "tables"}

    fake_executor = FakeMethodExecutor(CONTRACT_KEY_MAP)
    instance = executor_cls(fake_executor)
    result_payload = instance.execute(document, fake_executor)

    assert isinstance(result_payload, dict), f"{executor_name} must return a mapping"
    assert fake_executor.calls, f"{executor_name} should invoke MethodExecutor"

    for class_name, method_name in fake_executor.calls:
        result_key = f"{class_name}.{method_name}"
        payload = fake_executor.outputs[result_key]
        assert isinstance(payload, dict), f"{result_key} should return dictionaries"
        expected_keys = set(fake_executor.expected_keys(class_name))
        if expected_keys:
            assert expected_keys <= set(payload.keys()), (
                f"{result_key} must expose contract keys {sorted(expected_keys)}"
            )
