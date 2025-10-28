"""
Choreographer Dispatch System with QMCM Integration

This module implements a FQN-based method invocation system that:
1. Resolves fully-qualified method names (FQN) to callables via canonical registry
2. Intercepts all method calls through QMCM for tracking and quality assurance
3. Provides context-aware argument binding and dependency injection
4. Maintains full traceability for all method invocations

Architecture:
- FQN-based dispatch: Replace direct calls with _invoke_method("Class.method")
- QMCM interceptor: All invocations logged for quality monitoring
- Evidence recording: Optional evidence capture for provenance tracking
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field

from orchestrator.canonical_registry import CANONICAL_METHODS
from qmcm_hooks import get_global_recorder, QMCMRecorder

logger = logging.getLogger(__name__)


@dataclass
class InvocationContext:
    """Context for method invocation with dependency injection support."""
    
    # Core data context
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    document: Optional[Any] = None
    
    # Questionnaire/metadata context
    questionnaire: Optional[Dict[str, Any]] = None
    question_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Instance pool for dependency injection
    instances: Dict[str, Any] = field(default_factory=dict)
    
    # Additional kwargs
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def get_instance(self, class_name: str) -> Optional[Any]:
        """Get or create instance for a class."""
        return self.instances.get(class_name)
    
    def set_instance(self, class_name: str, instance: Any) -> None:
        """Register an instance for dependency injection."""
        self.instances[class_name] = instance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            "has_text": self.text is not None,
            "has_data": self.data is not None,
            "has_document": self.document is not None,
            "question_id": self.question_id,
            "instance_pool": list(self.instances.keys()),
            "extra_kwargs": list(self.extra_kwargs.keys()),
        }


@dataclass
class InvocationResult:
    """Result of a method invocation with full traceability."""
    
    fqn: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time_ms: float = 0.0
    context_summary: Dict[str, Any] = field(default_factory=dict)
    
    # QMCM tracking
    qmcm_recorded: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fqn": self.fqn,
            "success": self.success,
            "result_type": type(self.result).__name__ if self.result is not None else None,
            "error": str(self.error) if self.error else None,
            "execution_time_ms": self.execution_time_ms,
            "context_summary": self.context_summary,
            "qmcm_recorded": self.qmcm_recorded,
        }


class ChoreographerDispatcher:
    """
    Central dispatcher for FQN-based method invocation with QMCM integration.
    
    This class replaces direct component calls with a traceable, intercepted
    invocation mechanism that ensures:
    - All methods are resolved through canonical registry
    - All invocations are recorded via QMCM
    - Full context and provenance tracking
    - Consistent error handling and reporting
    """
    
    def __init__(
        self,
        registry: Optional[Dict[str, Callable]] = None,
        qmcm_recorder: Optional[QMCMRecorder] = None,
        enable_evidence_recording: bool = False,
    ):
        """
        Initialize choreographer dispatcher.
        
        Args:
            registry: Canonical method registry (defaults to CANONICAL_METHODS)
            qmcm_recorder: QMCM recorder instance (defaults to global recorder)
            enable_evidence_recording: Enable evidence capture for provenance
        """
        self.registry = registry if registry is not None else CANONICAL_METHODS
        self.qmcm_recorder = qmcm_recorder or get_global_recorder()
        self.enable_evidence_recording = enable_evidence_recording
        
        # Evidence recording (if enabled)
        self.evidence_records: List[Dict[str, Any]] = []
        
        logger.info(
            f"ChoreographerDispatcher initialized with {len(self.registry)} methods, "
            f"evidence_recording={'enabled' if enable_evidence_recording else 'disabled'}"
        )
    
    def _resolve_fqn(self, fqn: str) -> Callable:
        """
        Resolve fully-qualified method name to callable.
        
        Args:
            fqn: Fully-qualified name like "IndustrialPolicyProcessor.process"
            
        Returns:
            Callable method
            
        Raises:
            ValueError: If FQN cannot be resolved
        """
        if fqn not in self.registry:
            raise ValueError(
                f"Method '{fqn}' not found in canonical registry. "
                f"Available methods: {len(self.registry)}"
            )
        
        return self.registry[fqn]
    
    def _inspect_method_signature(
        self,
        method: Callable
    ) -> Dict[str, Any]:
        """
        Inspect method signature for parameter binding.
        
        Args:
            method: Callable to inspect
            
        Returns:
            Dictionary with signature information
        """
        try:
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            
            # Check if bound method
            is_bound = inspect.ismethod(method) and hasattr(method, "__self__")
            is_unbound = inspect.isfunction(method) and params and params[0] == "self"
            
            return {
                "params": params,
                "is_bound": is_bound,
                "is_unbound": is_unbound,
                "needs_instance": is_unbound and not is_bound,
            }
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to inspect method signature: {e}")
            return {
                "params": [],
                "is_bound": False,
                "is_unbound": False,
                "needs_instance": False,
            }
    
    def _bind_arguments(
        self,
        method: Callable,
        context: InvocationContext,
        sig_info: Dict[str, Any],
    ) -> tuple:
        """
        Bind arguments from context to method parameters.
        
        Args:
            method: Callable to invoke
            context: Invocation context
            sig_info: Signature information from _inspect_method_signature
            
        Returns:
            Tuple of (args, kwargs) for method invocation
        """
        params = sig_info["params"]
        args = []
        kwargs = {}
        
        # Skip 'self' parameter if present (handled by bound method or instance)
        param_index = 1 if (params and params[0] == "self") else 0
        remaining_params = params[param_index:]
        
        # Try to bind common parameters
        for param_name in remaining_params:
            if param_name == "text" and context.text is not None:
                kwargs["text"] = context.text
            elif param_name == "data" and context.data is not None:
                kwargs["data"] = context.data
            elif param_name == "document" and context.document is not None:
                kwargs["document"] = context.document
            elif param_name == "questionnaire" and context.questionnaire is not None:
                kwargs["questionnaire"] = context.questionnaire
            elif param_name == "question_id" and context.question_id is not None:
                kwargs["question_id"] = context.question_id
            elif param_name in context.extra_kwargs:
                kwargs[param_name] = context.extra_kwargs[param_name]
        
        return tuple(args), kwargs
    
    def _get_or_create_instance(
        self,
        class_name: str,
        context: InvocationContext,
    ) -> Any:
        """
        Get or create instance for unbound method invocation.
        
        Args:
            class_name: Name of the class to instantiate
            context: Invocation context with instance pool
            
        Returns:
            Instance of the class
            
        Raises:
            ValueError: If instance cannot be created
        """
        # Check instance pool first
        instance = context.get_instance(class_name)
        if instance is not None:
            return instance
        
        # Try to create new instance (basic implementation)
        # In production, this would use a proper factory pattern
        logger.warning(
            f"Instance for {class_name} not found in context pool. "
            f"Creating basic instance (may fail if class requires args)."
        )
        
        # This is a placeholder - real implementation would need a factory
        raise ValueError(
            f"Cannot create instance of {class_name}. "
            f"Instance must be provided in context.instances"
        )
    
    def invoke_method(
        self,
        fqn: str,
        context: Optional[InvocationContext] = None,
        **kwargs
    ) -> InvocationResult:
        """
        Invoke method by fully-qualified name with QMCM interception.
        
        This is the primary dispatch method that:
        1. Resolves FQN to callable via registry
        2. Binds arguments from context
        3. Invokes method
        4. Records invocation via QMCM
        5. Captures evidence (if enabled)
        
        Args:
            fqn: Fully-qualified method name (e.g., "IndustrialPolicyProcessor.process")
            context: Invocation context with data and instances
            **kwargs: Additional keyword arguments
            
        Returns:
            InvocationResult with execution details
            
        Example:
            >>> dispatcher = ChoreographerDispatcher()
            >>> ctx = InvocationContext(text="policy document...")
            >>> result = dispatcher.invoke_method("PolicyTextProcessor.segment_into_sentences", ctx)
        """
        # Initialize context if not provided
        if context is None:
            context = InvocationContext()
        
        # Add kwargs to context
        context.extra_kwargs.update(kwargs)
        
        start_time = time.time()
        
        try:
            # Step 1: Resolve FQN to callable
            method = self._resolve_fqn(fqn)
            
            # Step 2: Inspect method signature
            sig_info = self._inspect_method_signature(method)
            
            # Step 3: Handle unbound methods (need instance)
            if sig_info["needs_instance"]:
                class_name = fqn.split(".")[0]
                instance = self._get_or_create_instance(class_name, context)
                # Bind method to instance
                method = getattr(instance, fqn.split(".")[1])
                # Re-inspect as bound method
                sig_info = self._inspect_method_signature(method)
            
            # Step 4: Bind arguments from context
            args, bound_kwargs = self._bind_arguments(method, context, sig_info)
            
            # Step 5: Invoke method
            result = method(*args, **bound_kwargs)
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Step 6: Record via QMCM
            qmcm_recorded = False
            try:
                self.qmcm_recorder.record_call(
                    method_name=fqn,
                    input_types={k: type(v).__name__ for k, v in bound_kwargs.items()},
                    output_type=type(result).__name__,
                    execution_status="success",
                    execution_time_ms=execution_time_ms,
                )
                qmcm_recorded = True
            except Exception as e:
                logger.warning(f"Failed to record QMCM invocation: {e}")
            
            # Step 7: Capture evidence (if enabled)
            if self.enable_evidence_recording:
                self._record_evidence(fqn, context, result, execution_time_ms)
            
            # Return result
            return InvocationResult(
                fqn=fqn,
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
                context_summary=context.to_dict(),
                qmcm_recorded=qmcm_recorded,
            )
            
        except Exception as error:
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Record error via QMCM
            qmcm_recorded = False
            try:
                self.qmcm_recorder.record_call(
                    method_name=fqn,
                    input_types={},
                    output_type="error",
                    execution_status="error",
                    execution_time_ms=execution_time_ms,
                )
                qmcm_recorded = True
            except Exception as e:
                logger.warning(f"Failed to record QMCM error: {e}")
            
            # Log error
            logger.error(f"Method invocation failed: {fqn} - {error}")
            
            # Return error result
            return InvocationResult(
                fqn=fqn,
                success=False,
                error=error,
                execution_time_ms=execution_time_ms,
                context_summary=context.to_dict() if context else {},
                qmcm_recorded=qmcm_recorded,
            )
    
    def _record_evidence(
        self,
        fqn: str,
        context: InvocationContext,
        result: Any,
        execution_time_ms: float,
    ) -> None:
        """
        Record evidence for provenance tracking.
        
        Args:
            fqn: Method FQN
            context: Invocation context
            result: Method result
            execution_time_ms: Execution time
        """
        evidence = {
            "timestamp": time.time(),
            "fqn": fqn,
            "context_summary": context.to_dict(),
            "result_type": type(result).__name__,
            "execution_time_ms": execution_time_ms,
        }
        
        self.evidence_records.append(evidence)
    
    def get_invocation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about method invocations.
        
        Returns:
            Dictionary with invocation statistics
        """
        return {
            "registry_size": len(self.registry),
            "qmcm_stats": self.qmcm_recorder.get_statistics(),
            "evidence_records": len(self.evidence_records) if self.enable_evidence_recording else 0,
        }


# Global dispatcher instance
_global_dispatcher: Optional[ChoreographerDispatcher] = None


def get_global_dispatcher() -> ChoreographerDispatcher:
    """Get or create global choreographer dispatcher."""
    global _global_dispatcher
    if _global_dispatcher is None:
        _global_dispatcher = ChoreographerDispatcher()
    return _global_dispatcher


def invoke_method(fqn: str, context: Optional[InvocationContext] = None, **kwargs) -> InvocationResult:
    """
    Convenience function to invoke method via global dispatcher.
    
    Args:
        fqn: Fully-qualified method name
        context: Invocation context
        **kwargs: Additional keyword arguments
        
    Returns:
        InvocationResult
        
    Example:
        >>> from orchestrator.choreographer_dispatch import invoke_method, InvocationContext
        >>> ctx = InvocationContext(text="policy text...")
        >>> result = invoke_method("PolicyTextProcessor.segment_into_sentences", ctx)
    """
    dispatcher = get_global_dispatcher()
    return dispatcher.invoke_method(fqn, context, **kwargs)


__all__ = [
    "ChoreographerDispatcher",
    "InvocationContext",
    "InvocationResult",
    "get_global_dispatcher",
    "invoke_method",
]
