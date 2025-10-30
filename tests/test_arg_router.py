import importlib.util
from pathlib import Path

import pytest


def _load_orchestrator_module():
    module_path = Path(__file__).parent.parent / "orchestrator.py"
    spec = importlib.util.spec_from_file_location("orchestrator_root", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


orchestrator = _load_orchestrator_module()


class _RawDocumentStub:
    def __init__(self):
        self.file_id = "raw-123"
        self.file_name = "fallback.pdf"
        self.file_path = "/tmp/fallback.pdf"


class _PreprocessedStub:
    def __init__(self):
        self.raw_document = _RawDocumentStub()
        self.full_text = "hello world"
        self.sentences = ["hello", "world"]
        self.tables = [{"rows": 1}]
        self.preprocessing_metadata = {"extra": True}
        self.language = "es"


class DummyProcessor:
    def __init__(self):
        self.calls = []

    def process(self, text: str) -> str:
        self.calls.append({"text": text})
        return "ok"


class DummyTextProcessor:
    def __init__(self):
        self.calls = []

    def segment_into_sentences(self, text: str) -> list[str]:
        self.calls.append({"text": text})
        return text.split()


@pytest.fixture
def executor(monkeypatch):
    exec_inst = orchestrator.MethodExecutor()
    exec_inst.instances = {
        "IndustrialPolicyProcessor": DummyProcessor(),
        "PolicyTextProcessor": DummyTextProcessor(),
    }
    monkeypatch.setattr(orchestrator, "MODULES_OK", True, raising=False)
    yield exec_inst
    monkeypatch.setattr(orchestrator, "MODULES_OK", False, raising=False)


def test_orchestrator_document_from_ingestion():
    doc = orchestrator.OrchestratorDocument.from_ingestion(_PreprocessedStub())

    assert doc.document_id == "raw-123"
    assert doc.raw_text == "hello world"
    assert doc.metadata["language"] == "es"
    assert doc.metadata["source_path"].endswith("fallback.pdf")


def test_arg_router_registry_contains_primary_routes():
    assert (
        "IndustrialPolicyProcessor",
        "process",
    ) in orchestrator.ARG_ROUTER
    assert (
        "PolicyTextProcessor",
        "segment_into_sentences",
    ) in orchestrator.ARG_ROUTER


def test_executor_uses_router_with_context(executor):
    document = orchestrator.OrchestratorDocument(
        document_id="doc-1",
        raw_text="texto",
        sentences=("uno", "dos"),
        tables=(),
        metadata={"source_path": "demo.pdf"},
    )

    result = executor.execute(
        "IndustrialPolicyProcessor",
        "process",
        context=document,
    )

    assert result == "ok"
    processor = executor.instances["IndustrialPolicyProcessor"]
    assert processor.calls[-1] == {"text": "texto"}


def test_executor_upgrades_legacy_kwargs(executor):
    result = executor.execute(
        "PolicyTextProcessor",
        "segment_into_sentences",
        text="hola mundo",
        sentences=["hola", "mundo"],
        tables=[],
    )

    assert result == ["hola", "mundo"]
    text_processor = executor.instances["PolicyTextProcessor"]
    assert text_processor.calls[-1] == {"text": "hola mundo"}
