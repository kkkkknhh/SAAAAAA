#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 Integration Hook for PolicyAnalysisOrchestrator.

This module provides integration between the D2 Activities Orchestrator
and the main PolicyAnalysisOrchestrator, enabling method concurrence
validation as part of the main analysis pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from orchestrator.d2_activities_orchestrator import (
    D2ActivitiesOrchestrator,
    D2Question,
    OrchestrationError,
    QuestionOrchestrationResult,
)

logger = logging.getLogger(__name__)


class D2IntegrationHook:
    """
    Integration hook for D2 method concurrence validation.
    
    This class provides a clean interface for the PolicyAnalysisOrchestrator
    to validate D2 method concurrence before or during question execution.
    """
    
    def __init__(
        self,
        strict_mode: bool = False,
        trace_execution: bool = True,
        validate_on_init: bool = False,
    ):
        """
        Initialize D2 integration hook.
        
        Args:
            strict_mode: If True, enforce strict SIN_CARRETA doctrine
            trace_execution: If True, maintain execution traces
            validate_on_init: If True, validate all methods on initialization
        """
        self.orchestrator = D2ActivitiesOrchestrator(
            strict_mode=strict_mode,
            trace_execution=trace_execution
        )
        
        self.validation_results: Optional[Dict[str, QuestionOrchestrationResult]] = None
        self.validation_report: Optional[Dict[str, Any]] = None
        
        logger.info(
            f"D2 Integration Hook initialized "
            f"(strict={strict_mode}, trace={trace_execution})"
        )
        
        if validate_on_init:
            self.validate_all()
    
    def validate_all(self) -> bool:
        """
        Validate all D2 questions.
        
        Returns:
            True if all validations pass, False otherwise
            
        Raises:
            OrchestrationError: If strict_mode is True and validation fails
        """
        logger.info("Validating all D2 questions...")
        
        try:
            self.validation_results = self.orchestrator.validate_all_d2_questions()
            self.validation_report = self.orchestrator.generate_validation_report(
                self.validation_results
            )
            
            success = self.validation_report["summary"]["overall_success"]
            
            if success:
                logger.info("✓ D2 validation: All methods resolved successfully")
            else:
                failed = self.validation_report["summary"]["methods_failed"]
                total = self.validation_report["summary"]["total_methods"]
                logger.warning(
                    f"⚠ D2 validation: {failed}/{total} methods failed resolution"
                )
            
            return success
            
        except OrchestrationError as e:
            logger.error(f"D2 validation failed: {e}")
            raise
    
    def validate_question(self, question_id: str) -> bool:
        """
        Validate a specific D2 question.
        
        Args:
            question_id: Question ID (e.g., "D2-Q1", "D2-Q2", etc.)
            
        Returns:
            True if validation passes, False otherwise
        """
        # Map question_id to D2Question enum
        question_map = {
            "D2-Q1": D2Question.Q1_FORMATO_TABULAR,
            "D2-Q2": D2Question.Q2_CAUSALIDAD_ACTIVIDADES,
            "D2-Q3": D2Question.Q3_CLASIFICACION_TEMATICA,
            "D2-Q4": D2Question.Q4_RIESGOS_MITIGACION,
            "D2-Q5": D2Question.Q5_COHERENCIA_ESTRATEGICA,
        }
        
        if question_id not in question_map:
            logger.warning(f"Question {question_id} is not a D2 question")
            return True  # Not a D2 question, so validation passes trivially
        
        question = question_map[question_id]
        
        try:
            result = self.orchestrator.validate_method_existence(question)
            
            if result.success:
                logger.info(f"✓ {question_id}: All {result.total_methods} methods validated")
            else:
                logger.warning(
                    f"⚠ {question_id}: {result.failed_methods}/{result.total_methods} methods failed"
                )
            
            return result.success
            
        except OrchestrationError as e:
            logger.error(f"Validation failed for {question_id}: {e}")
            return False
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of validation results.
        
        Returns:
            Dictionary with validation summary, or empty dict if not yet validated
        """
        if self.validation_report is None:
            return {
                "validated": False,
                "message": "D2 validation has not been run yet"
            }
        
        return {
            "validated": True,
            "overall_success": self.validation_report["summary"]["overall_success"],
            "total_methods": self.validation_report["summary"]["total_methods"],
            "methods_resolved": self.validation_report["summary"]["methods_resolved"],
            "methods_failed": self.validation_report["summary"]["methods_failed"],
            "success_rate": (
                self.validation_report["summary"]["methods_resolved"] /
                self.validation_report["summary"]["total_methods"]
                if self.validation_report["summary"]["total_methods"] > 0
                else 0.0
            ),
            "questions": {
                qid: {
                    "success": qdata["success"],
                    "methods_resolved": qdata["executed_methods"] - qdata["failed_methods"],
                    "total_methods": qdata["total_methods"],
                }
                for qid, qdata in self.validation_report["questions"].items()
            }
        }
    
    def save_validation_report(self, output_path: Path) -> None:
        """
        Save validation report to file.
        
        Args:
            output_path: Path to save the report
        """
        if self.validation_report is None:
            logger.warning("No validation report to save")
            return
        
        self.orchestrator.generate_validation_report(
            self.validation_results,
            output_path=output_path
        )
        
        logger.info(f"D2 validation report saved to {output_path}")
    
    def pre_execution_check(
        self,
        question_id: str,
        abort_on_failure: bool = False
    ) -> bool:
        """
        Perform a pre-execution check for a question.
        
        This can be called by the orchestrator before executing a D2 question
        to ensure all required methods are available.
        
        Args:
            question_id: Question ID to check
            abort_on_failure: If True, raise exception on failure
            
        Returns:
            True if check passes, False otherwise
            
        Raises:
            OrchestrationError: If abort_on_failure is True and check fails
        """
        success = self.validate_question(question_id)
        
        if not success and abort_on_failure:
            raise OrchestrationError(
                f"Pre-execution check failed for {question_id}. "
                f"Required methods are missing or unresolvable."
            )
        
        return success


def integrate_d2_validation(
    orchestrator_instance,
    strict_mode: bool = False,
    validate_on_init: bool = True,
    save_report_path: Optional[Path] = None
) -> D2IntegrationHook:
    """
    Integrate D2 validation into a PolicyAnalysisOrchestrator instance.
    
    This is a convenience function to attach D2 validation to an existing
    orchestrator instance.
    
    Args:
        orchestrator_instance: Instance of PolicyAnalysisOrchestrator
        strict_mode: Enable strict SIN_CARRETA doctrine
        validate_on_init: Validate all D2 questions on initialization
        save_report_path: Optional path to save validation report
        
    Returns:
        D2IntegrationHook instance
        
    Example:
        >>> orchestrator = PolicyAnalysisOrchestrator(config)
        >>> d2_hook = integrate_d2_validation(
        ...     orchestrator,
        ...     strict_mode=True,
        ...     save_report_path=Path("d2_validation.json")
        ... )
        >>> if d2_hook.get_validation_summary()["overall_success"]:
        ...     # Proceed with analysis
        ...     results = orchestrator.execute()
    """
    hook = D2IntegrationHook(
        strict_mode=strict_mode,
        trace_execution=True,
        validate_on_init=validate_on_init
    )
    
    # Attach hook to orchestrator instance
    orchestrator_instance.d2_validation = hook
    
    # Save report if requested
    if save_report_path and hook.validation_report:
        hook.save_validation_report(save_report_path)
    
    logger.info("D2 validation integrated into orchestrator")
    
    return hook
