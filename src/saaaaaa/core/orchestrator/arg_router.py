"""Argument routing and validation utilities for orchestrated method calls."""
from __future__ import annotations

import inspect
import logging
import os
import random
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Union,
)
from typing import get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)

MISSING: object = object()


class ArgRouterError(RuntimeError):
    """Base exception for routing and validation issues."""


class ArgumentValidationError(ArgRouterError):
    """Raised when the provided payload does not match the method signature."""

    def __init__(
        self,
        class_name: str,
        method_name: str,
        *,
        missing: Optional[Iterable[str]] = None,
        unexpected: Optional[Iterable[str]] = None,
        type_mismatches: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.class_name = class_name
        self.method_name = method_name
        self.missing = set(missing or ())
        self.unexpected = set(unexpected or ())
        self.type_mismatches = dict(type_mismatches or {})
        detail = []
        if self.missing:
            detail.append(f"missing={sorted(self.missing)}")
        if self.unexpected:
            detail.append(f"unexpected={sorted(self.unexpected)}")
        if self.type_mismatches:
            detail.append(f"type_mismatches={self.type_mismatches}")
        message = (
            f"Invalid payload for {class_name}.{method_name}"
            + (f" ({'; '.join(detail)})" if detail else "")
        )
        super().__init__(message)


@dataclass(frozen=True)
class _ParameterSpec:
    name: str
    kind: inspect._ParameterKind
    default: Any
    annotation: Any

    @property
    def required(self) -> bool:
        return self.default is MISSING


@dataclass(frozen=True)
class MethodSpec:
    class_name: str
    method_name: str
    positional: Tuple[_ParameterSpec, ...]
    keyword_only: Tuple[_ParameterSpec, ...]
    has_var_keyword: bool
    has_var_positional: bool

    @property
    def required_arguments(self) -> Tuple[str, ...]:
        required = tuple(
            spec.name
            for spec in (*self.positional, *self.keyword_only)
            if spec.required
        )
        return required

    @property
    def accepted_arguments(self) -> Tuple[str, ...]:
        accepted = tuple(spec.name for spec in (*self.positional, *self.keyword_only))
        return accepted


class ArgRouter:
    """Resolve method call payloads based on inspected signatures."""

    def __init__(self, class_registry: Mapping[str, type]) -> None:
        self._class_registry = dict(class_registry)
        self._spec_cache: Dict[Tuple[str, str], MethodSpec] = {}

    def describe(self, class_name: str, method_name: str) -> MethodSpec:
        """Return the cached method specification, building it if necessary."""
        key = (class_name, method_name)
        if key not in self._spec_cache:
            self._spec_cache[key] = self._build_spec(class_name, method_name)
        return self._spec_cache[key]

    def route(
        self,
        class_name: str,
        method_name: str,
        payload: MutableMapping[str, Any],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Validate and split a payload into positional and keyword arguments."""
        spec = self.describe(class_name, method_name)
        provided_keys = set(payload.keys())
        required = set(spec.required_arguments)
        accepted = set(spec.accepted_arguments)

        missing = required - provided_keys
        unexpected = provided_keys - accepted
        if unexpected and spec.has_var_keyword:
            unexpected = set()

        if missing or unexpected:
            raise ArgumentValidationError(
                class_name,
                method_name,
                missing=missing,
                unexpected=unexpected,
            )

        args: list[Any] = []
        kwargs: Dict[str, Any] = {}
        type_mismatches: Dict[str, str] = {}

        remaining = dict(payload)

        for param in spec.positional:
            if param.name not in remaining:
                if param.required:
                    missing = {param.name}
                    raise ArgumentValidationError(
                        class_name,
                        method_name,
                        missing=missing,
                    )
                continue
            value = remaining.pop(param.name)
            if not self._matches_annotation(value, param.annotation):
                expected = self._describe_annotation(param.annotation)
                type_mismatches[param.name] = expected
            args.append(value)

        for param in spec.keyword_only:
            if param.name not in remaining:
                if param.required:
                    raise ArgumentValidationError(
                        class_name,
                        method_name,
                        missing={param.name},
                    )
                continue
            value = remaining.pop(param.name)
            if not self._matches_annotation(value, param.annotation):
                expected = self._describe_annotation(param.annotation)
                type_mismatches[param.name] = expected
            kwargs[param.name] = value

        if spec.has_var_keyword and remaining:
            kwargs.update(remaining)
            remaining = {}

        if remaining:
            raise ArgumentValidationError(
                class_name,
                method_name,
                unexpected=set(remaining.keys()),
            )

        if type_mismatches:
            raise ArgumentValidationError(
                class_name,
                method_name,
                type_mismatches={
                    name: f"expected {expected}; received {type(payload[name]).__name__}"
                    for name, expected in type_mismatches.items()
                },
            )

        return tuple(args), kwargs

    def expected_arguments(self, class_name: str, method_name: str) -> Tuple[str, ...]:
        spec = self.describe(class_name, method_name)
        return spec.accepted_arguments

    def _build_spec(self, class_name: str, method_name: str) -> MethodSpec:
        try:
            cls = self._class_registry[class_name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ArgRouterError(f"Unknown class '{class_name}'") from exc

        try:
            method = getattr(cls, method_name)
        except AttributeError as exc:
            raise ArgRouterError(f"Class '{class_name}' has no method '{method_name}'") from exc

        signature = inspect.signature(method)
        try:
            type_hints = get_type_hints(method)
        except Exception:
            type_hints = {}
        positional: list[_ParameterSpec] = []
        keyword_only: list[_ParameterSpec] = []
        has_var_keyword = False
        has_var_positional = False

        for parameter in signature.parameters.values():
            if parameter.name == "self":
                continue
            default = (
                parameter.default
                if parameter.default is not inspect._empty
                else MISSING
            )
            annotation = type_hints.get(parameter.name, parameter.annotation)
            param_spec = _ParameterSpec(
                name=parameter.name,
                kind=parameter.kind,
                default=default,
                annotation=annotation,
            )
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional.append(param_spec)
            elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                keyword_only.append(param_spec)
            elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                has_var_keyword = True
            elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                has_var_positional = True

        return MethodSpec(
            class_name=class_name,
            method_name=method_name,
            positional=tuple(positional),
            keyword_only=tuple(keyword_only),
            has_var_keyword=has_var_keyword,
            has_var_positional=has_var_positional,
        )

    @staticmethod
    def _matches_annotation(value: Any, annotation: Any) -> bool:
        if annotation in (inspect._empty, Any):
            return True
        origin = get_origin(annotation)
        if origin is None:
            if isinstance(annotation, type):
                return isinstance(value, annotation)
            return True
        args = get_args(annotation)
        if origin is tuple:
            if not isinstance(value, tuple):
                return False
            if not args:
                return True
            if len(args) == 2 and args[1] is Ellipsis:
                return all(ArgRouter._matches_annotation(item, args[0]) for item in value)
            if len(args) != len(value):
                return False
            return all(
                ArgRouter._matches_annotation(item, arg_type)
                for item, arg_type in zip(value, args)
            )
        if origin in (list, List):
            if not isinstance(value, list):
                return False
            if not args:
                return True
            return all(ArgRouter._matches_annotation(item, args[0]) for item in value)
        if origin in (set, Set):
            if not isinstance(value, set):
                return False
            if not args:
                return True
            return all(ArgRouter._matches_annotation(item, args[0]) for item in value)
        if origin in (dict, Dict):
            if not isinstance(value, dict):
                return False
            if len(args) != 2:
                return True
            key_type, value_type = args
            return all(
                ArgRouter._matches_annotation(k, key_type)
                and ArgRouter._matches_annotation(v, value_type)
                for k, v in value.items()
            )
        if origin is Union:
            return any(ArgRouter._matches_annotation(value, arg) for arg in args)
        return True

    @staticmethod
    def _describe_annotation(annotation: Any) -> str:
        if annotation in (inspect._empty, Any):
            return "Any"
        origin = get_origin(annotation)
        if origin is None:
            if isinstance(annotation, type):
                return annotation.__name__
            return str(annotation)
        args = get_args(annotation)
        if origin is tuple:
            return f"Tuple[{', '.join(ArgRouter._describe_annotation(arg) for arg in args)}]"
        if origin in (list, List):
            return f"List[{ArgRouter._describe_annotation(args[0])}]" if args else "List[Any]"
        if origin in (set, Set):
            return f"Set[{ArgRouter._describe_annotation(args[0])}]" if args else "Set[Any]"
        if origin in (dict, Dict):
            if len(args) == 2:
                return (
                    f"Dict[{ArgRouter._describe_annotation(args[0])}, "
                    f"{ArgRouter._describe_annotation(args[1])}]"
                )
            return "Dict[Any, Any]"
        if origin is Union:
            return " | ".join(ArgRouter._describe_annotation(arg) for arg in args)
        return str(annotation)


class PayloadDriftMonitor:
    """Sampling validator for ingress/egress payloads."""

    CRITICAL_KEYS = {
        "content": str,
        "pdq_context": (dict, type(None)),
    }

    def __init__(self, *, sample_rate: float, enabled: bool) -> None:
        self.sample_rate = max(0.0, min(sample_rate, 1.0))
        self.enabled = enabled and self.sample_rate > 0.0

    @classmethod
    def from_env(cls) -> "PayloadDriftMonitor":
        enabled = os.getenv("ORCHESTRATOR_SAMPLING_VALIDATION", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            sample_rate = float(os.getenv("ORCHESTRATOR_SAMPLING_RATE", "0.05"))
        except ValueError:
            sample_rate = 0.05
        return cls(sample_rate=sample_rate, enabled=enabled)

    def maybe_validate(self, payload: Mapping[str, Any], *, producer: str, consumer: str) -> None:
        if not self.enabled:
            return
        if random.random() > self.sample_rate:
            return
        if not isinstance(payload, Mapping):
            return
        keys = set(payload.keys())
        if not keys.intersection(self.CRITICAL_KEYS):
            return

        missing = [key for key in self.CRITICAL_KEYS if key not in payload]
        type_mismatches = {
            key: self._expected_type_name(expected)
            for key, expected in self.CRITICAL_KEYS.items()
            if key in payload and not isinstance(payload[key], expected)
        }
        if missing or type_mismatches:
            logger.error(
                "Payload drift detected [%s -> %s]: missing=%s type_mismatches=%s",
                producer,
                consumer,
                missing,
                type_mismatches,
            )
        else:
            logger.debug(
                "Payload validation OK [%s -> %s]", producer, consumer
            )

    @staticmethod
    def _expected_type_name(expected: Any) -> str:
        return ", ".join(getattr(t, "__name__", str(t)) for t in expected)
        if hasattr(expected, "__name__"):
            return expected.__name__
        return str(expected)
