"""Unit tests for the orchestrator argument router."""

from __future__ import annotations

import inspect
from typing import Any, Dict

import pytest

from orchestrator.arg_router import ArgRouter, ArgumentValidationError


class SampleExecutor:
    """Simple class used to exercise argument routing."""

    def compute(self, x: int, y: int, *, flag: bool = False) -> int:
        return x + y if flag else x - y

    def optional(self, x: int, y: int | None = None) -> int:
        return x if y is None else x + y

    def accepts_kwargs(self, *, name: str, **extras: Any) -> Dict[str, Any]:
        payload = {"name": name}
        payload.update(extras)
        return payload


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


def test_var_keyword_arguments_allowed(router: ArgRouter) -> None:
    payload = router.route(
        "SampleExecutor",
        "accepts_kwargs",
        {"name": "router", "feature": "kwargs"},
    )
    assert payload == ((), {"name": "router", "feature": "kwargs"})


def test_expected_arguments_matches_signature(router: ArgRouter) -> None:
    expected = router.expected_arguments("SampleExecutor", "compute")
    signature = inspect.signature(SampleExecutor.compute)
    assert set(expected) == {param.name for param in signature.parameters.values()}
