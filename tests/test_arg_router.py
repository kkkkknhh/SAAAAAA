"""Unit tests for the orchestrator argument router."""
from __future__ import annotations

import pytest

from orchestrator.arg_router import ArgRouter, ArgumentValidationError
def _load_orchestrator_module():
    module_path = Path(__file__).parent.parent / "orchestrator.py"
    spec = importlib.util.spec_from_file_location("orchestrator_root", module_path)
    if spec is None:
        raise ImportError(f"Cannot create a module spec for '{module_path}'.")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
    spec.loader.exec_module(module)
    return module
        return x, y


@pytest.fixture()
def router() -> ArgRouter:
    return ArgRouter({"SampleExecutor": SampleExecutor})


def test_route_honors_signature(router: ArgRouter) -> None:
    args, kwargs = router.route(
        "SampleExecutor", "compute", {"x": 4, "y": 2, "flag": True}
    )
    assert args == (4, 2)
    assert kwargs == {"flag": True}


def test_missing_argument_raises(router: ArgRouter) -> None:
    with pytest.raises(ArgumentValidationError) as excinfo:
        router.route("SampleExecutor", "compute", {"x": 4})
    assert excinfo.value.missing == {"y"}


def test_unexpected_argument_raises(router: ArgRouter) -> None:
    with pytest.raises(ArgumentValidationError) as excinfo:
        router.route("SampleExecutor", "compute", {"x": 1, "y": 2, "extra": 5})
    assert excinfo.value.unexpected == {"extra"}


def test_optional_parameters_use_sentinel(router: ArgRouter) -> None:
    args, kwargs = router.route("SampleExecutor", "optional", {"x": 3})
    assert args == (3,)
    assert kwargs == {}


def test_type_mismatch_raises(router: ArgRouter) -> None:
    with pytest.raises(ArgumentValidationError) as excinfo:
        router.route("SampleExecutor", "compute", {"x": "bad", "y": 2})
    type_msgs = excinfo.value.type_mismatches
    assert "x" in type_msgs
    assert "expected int" in type_msgs["x"]
@pytest.fixture
def executor(monkeypatch):
    original_init = getattr(DummyProcessor, "__init__", None)

    def _dp_init(self, *args, **kwargs):
        if original_init is not None:
            original_init(self, *args, **kwargs)
        self.calls = []

    monkeypatch.setattr(DummyProcessor, "__init__", _dp_init, raising=False)

    exec_inst = orchestrator.MethodExecutor()
    exec_inst.instances = {
        "IndustrialPolicyProcessor": DummyProcessor(),
        "PolicyTextProcessor": DummyTextProcessor(),
    }
    monkeypatch.setattr(orchestrator, "MODULES_OK", True, raising=False)
    yield exec_inst
    monkeypatch.setattr(orchestrator, "MODULES_OK", False, raising=False)
        "IndustrialPolicyProcessor": industrial_proc,
        "PolicyTextProcessor": text_proc,
    }
    monkeypatch.setattr(orchestrator, "MODULES_OK", True, raising=False)
    yield exec_inst
    monkeypatch.setattr(orchestrator, "MODULES_OK", False, raising=False)